"""
Data acquisition for the noise-residual-reactivity experiment.

Single-dataset constraint: real and AI images MUST come from the same
Hugging Face dataset (CIFAKE), so both classes share identical origin
and processing. We do NOT assemble the two classes from different
sources. If CIFAKE is unreachable, we stop and print instructions
rather than substituting a different source per class.

Caching: once ./data/real and ./data/ai each contain >= N_PER_CLASS
images, re-runs skip the download entirely.
"""

import os
import sys
import random

import numpy as np

N_PER_CLASS = 500
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
REAL_DIR = os.path.join(DATA_DIR, "real")
AI_DIR = os.path.join(DATA_DIR, "ai")
SEED = 1234

# Candidate HF dataset repos to try, in order. All are CIFAKE (same
# underlying image origin: CIFAR-10 reals + Stable-Diffusion-generated
# fakes), so using any of these still satisfies the single-dataset,
# shared-provenance constraint.
CIFAKE_CANDIDATES = [
    "dragonintelligence/CIFAKE-image-dataset",
    "yanbax/CIFAKE_autotrain_compatible",
    "Hemg/cifake-real-and-ai-generated-synthetic-images",
    "kaustubh202/cifake_new_prompts",
    "batgre/CIFAKE",
]

# Full-resolution alternative: a single dataset of real photos + AI
# (deepfake/generator) images sharing one origin/label scheme, used to
# check whether the CIFAKE findings (native 32x32, heavily upsampled)
# generalize to native full-resolution images. Verified at inspection
# time to have internally-consistent per-class resolutions (AI images
# are generator-typical square sizes e.g. 1024x1024/1536x1536; real
# images have varied real-camera resolutions/aspect ratios) rather than
# being a chaotic blend of unrelated sources.
FULLRES_CANDIDATES = [
    "itsLeen/deepfake_vs_real_image_detection",
]

# A second, independent full-resolution single-source dataset (different
# content domain - stylized/art images rather than general photos) used
# to cross-check whether a signal found on FULLRES_CANDIDATES generalizes
# or was specific to that dataset's own artifacts (exactly the kind of
# confound the CIFAKE vs. fullres comparison exposed).
FULLRES2_CANDIDATES = [
    "itsLeen/deepfake_vs_real_image",
]


def already_cached(real_dir=REAL_DIR, ai_dir=AI_DIR):
    if not (os.path.isdir(real_dir) and os.path.isdir(ai_dir)):
        return False
    n_real = len([f for f in os.listdir(real_dir) if f.lower().endswith(".png")])
    n_ai = len([f for f in os.listdir(ai_dir) if f.lower().endswith(".png")])
    return n_real > 50 and n_ai > 50


def find_label_mapping(features):
    """
    Inspect a HF dataset `features` schema and figure out which column
    is the image column and which column carries the real/fake label,
    and how the label values map to 'real' / 'ai'. Returns
    (image_col, label_col, value_to_class) or None if ambiguous.
    """
    from datasets import Image as HFImage, ClassLabel

    image_col = None
    label_col = None
    value_to_class = None

    for col, feat in features.items():
        if isinstance(feat, HFImage):
            image_col = col

    for col, feat in features.items():
        if isinstance(feat, ClassLabel):
            names_lower = [str(n).lower() for n in feat.names]
            # Look for tokens indicating real vs fake/ai in the class names.
            real_tokens = {"real"}
            fake_tokens = {"fake", "ai", "generated", "synthetic", "gan", "diffusion"}
            mapping = {}
            for idx, name in enumerate(names_lower):
                if any(t in name for t in real_tokens):
                    mapping[idx] = "real"
                elif any(t in name for t in fake_tokens):
                    mapping[idx] = "ai"
            if len(mapping) == 2 and set(mapping.values()) == {"real", "ai"}:
                label_col = col
                value_to_class = mapping
                break

    if image_col is None or label_col is None or value_to_class is None:
        return None
    return image_col, label_col, value_to_class


def try_cifake(repo_id, log, real_dir=REAL_DIR, ai_dir=AI_DIR, n_per_class=N_PER_CLASS):
    from datasets import load_dataset_builder, load_dataset

    log(f"  Inspecting schema of {repo_id} ...")
    try:
        builder = load_dataset_builder(repo_id)
    except Exception as e:
        log(f"  -> could not load builder: {e}")
        return None

    features = builder.info.features
    mapping = find_label_mapping(features)
    if mapping is None:
        log(f"  -> ambiguous/unknown schema, features = {features}")
        log("  -> refusing to guess label mapping; skipping this candidate.")
        return None

    image_col, label_col, value_to_class = mapping
    log(f"  -> image column: '{image_col}', label column: '{label_col}', "
        f"mapping: {value_to_class}")

    split = "train" if "train" in builder.info.splits else list(builder.info.splits)[0]
    log(f"  Loading split '{split}' (streaming) ...")
    try:
        ds = load_dataset(repo_id, split=split, streaming=True)
    except Exception as e:
        log(f"  -> failed to stream dataset: {e}")
        return None

    # Reservoir-free approach: since these splits are label-sorted or
    # shuffled depending on repo, just shuffle the stream and take the
    # first N per class we encounter.
    ds = ds.shuffle(seed=SEED, buffer_size=2000)

    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(ai_dir, exist_ok=True)
    counts = {"real": 0, "ai": 0}
    saved = {"real": [], "ai": []}

    try:
        for ex in ds:
            if counts["real"] >= n_per_class and counts["ai"] >= n_per_class:
                break
            raw_label = ex[label_col]
            cls = value_to_class.get(raw_label)
            if cls is None:
                continue
            if counts[cls] >= n_per_class:
                continue
            img = ex[image_col]
            if img is None:
                continue
            img = img.convert("RGB")
            # Skip degenerate images (e.g. all-black/constant frames from a
            # corrupted source record or truncated download) - these break
            # denoising (zero-variance -> NaN) and are not representative
            # photos/generations anyway.
            if np.asarray(img.convert("L"), dtype=np.float64).std() < 1e-6:
                continue
            out_dir = real_dir if cls == "real" else ai_dir
            idx = counts[cls]
            out_path = os.path.join(out_dir, f"{cls}_{idx:04d}.png")
            img.save(out_path)
            counts[cls] += 1
            saved[cls].append(out_path)
    except Exception as e:
        log(f"  -> error while streaming/saving: {e}")
        if counts["real"] < 50 or counts["ai"] < 50:
            return None

    log(f"  -> collected real={counts['real']}, ai={counts['ai']}")
    if counts["real"] > 50 and counts["ai"] > 50:
        return {
            "source": repo_id,
            "n_real": counts["real"],
            "n_ai": counts["ai"],
            "examples_real": saved["real"][:3],
            "examples_ai": saved["ai"][:3],
        }
    return None


def print_manual_instructions(real_dir=REAL_DIR, ai_dir=AI_DIR):
    print("""
================================================================================
AUTOMATIC DOWNLOAD FAILED FOR ALL SOURCES
================================================================================
No reachable Hugging Face dataset candidate was found (offline, gated, or
schema changed). Per the experiment's rules, we will NOT substitute a
different source per class (that would break shared provenance). Please
populate the data directories manually with a SINGLE shared-origin
real/fake dataset:

Option 1 - Kaggle CLI (CIFAKE: Real and AI-Generated Synthetic Images):
    pip install kaggle
    kaggle datasets download -d birdy654/cifake-real-and-ai-generated-synthetic-images
    unzip cifake-real-and-ai-generated-synthetic-images.zip -d cifake_raw
    # then copy up to 500 images from the REAL class into:
    #   {real_dir}
    # and up to 500 images from the FAKE class into:
    #   {ai_dir}

Option 2 - manually place any shared-origin real/fake PNG/JPG files:
    {real_dir}/*.png   (real photos)
    {ai_dir}/*.png     (AI-generated images)

Then re-run this script / the main experiment; it will detect the cached
files and skip downloading.
================================================================================
""".format(real_dir=real_dir, ai_dir=ai_dir))


def acquire_data(log=print, candidates=None, real_dir=None, ai_dir=None, n_per_class=None):
    """Download (or reuse cached) real+AI images from a single shared-origin
    HF dataset. Defaults to the original CIFAKE candidate list/dirs; pass
    `candidates=FULLRES_CANDIDATES` plus different `real_dir`/`ai_dir` to
    run the same acquisition logic against the full-resolution dataset.
    """
    candidates = candidates if candidates is not None else CIFAKE_CANDIDATES
    real_dir = real_dir if real_dir is not None else REAL_DIR
    ai_dir = ai_dir if ai_dir is not None else AI_DIR
    n_per_class = n_per_class if n_per_class is not None else N_PER_CLASS

    if already_cached(real_dir, ai_dir):
        n_real = len([f for f in os.listdir(real_dir) if f.lower().endswith(".png")])
        n_ai = len([f for f in os.listdir(ai_dir) if f.lower().endswith(".png")])
        log(f"Found cached data: n_real={n_real}, n_ai={n_ai}. Skipping download.")
        ex_real = sorted(os.path.join(real_dir, f) for f in os.listdir(real_dir))[:3]
        ex_ai = sorted(os.path.join(ai_dir, f) for f in os.listdir(ai_dir))[:3]
        return {
            "source": "cache (previous download)",
            "n_real": n_real,
            "n_ai": n_ai,
            "examples_real": ex_real,
            "examples_ai": ex_ai,
        }

    log("No sufficient cached data found. Attempting automatic download ...")
    log(f"Trying candidates in order: {candidates}")

    for repo_id in candidates:
        log(f"Trying {repo_id} ...")
        result = try_cifake(repo_id, log, real_dir=real_dir, ai_dir=ai_dir, n_per_class=n_per_class)
        if result is not None:
            log(f"SUCCESS using source: {repo_id}")
            return result
        log(f"  -> {repo_id} did not yield a usable real+ai split; trying next.")

    print_manual_instructions(real_dir=real_dir, ai_dir=ai_dir)
    return None


if __name__ == "__main__":
    info = acquire_data()
    if info is None:
        sys.exit(1)
    print("\nData acquisition summary:")
    for k, v in info.items():
        print(f"  {k}: {v}")
