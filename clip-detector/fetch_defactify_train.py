"""
Fetches TRAINING data from Rajarshi-Roy-research/Defactify_Image_Dataset's
TRAIN split (disjoint from the TEST split used by
fetch_generator_diagnostic.py / evaluate_generator_diagnostic.py - that
diagnostic set stays untouched and valid as a before/after comparison).

Samples across all 5 generators (SD2.1, SDXL, SD3, DALL-E 3,
Midjourney 6), saved into the standard real/ai layout (ai/ only) so
train_classifier.py's DATASETS list can pick it up like any other
source - this is specifically meant to close the generator-diversity
gap found by the diagnostic evaluation (near-chance accuracy on
SD2.1/SD3/Midjourney6, weak on DALL-E3).

Deliberately does NOT fetch this dataset's real (Label_B=0) images:
they're sourced from MS COCO, and the organizer's WildFake
demonstration-only benchmark uses COCO val2017 as its non-AIGC side.
Defactify's real images almost certainly come from the much larger
train2017 split rather than val2017, so overlap is unlikely - but
real-photo diversity is already covered by data_fullres/data_sid_set,
so there's no reason to carry even a small, unconfirmed risk of
training on images the organizer reserved for demonstration.

Usage:
    python fetch_defactify_train.py
"""

import os

from PIL import Image

REPO_ID = "Rajarshi-Roy-research/Defactify_Image_Dataset"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_defactify_train")
AI_DIR = os.path.join(DATA_DIR, "ai")
N_PER_CLASS = 150  # per generator (5 generators); real (Label_B=0) is skipped, see module docstring
SEED = 1234

# Label_B: 0=real (skipped - see module docstring), 1=SD21, 2=SDXL, 3=SD3, 4=DALLE3, 5=Midjourney6
GENERATOR_NAMES = {1: "sd21", 2: "sdxl", 3: "sd3", 4: "dalle3", 5: "midjourney6"}


def already_cached(min_n=50):
    if not os.path.isdir(AI_DIR):
        return False
    n_ai = len([f for f in os.listdir(AI_DIR) if f.endswith(".png")])
    return n_ai >= min_n * 5  # ai pool spans 5 generators


def fetch(n_per_class=N_PER_CLASS, log=print):
    if already_cached(min_n=min(50, n_per_class)):
        n_ai = len([f for f in os.listdir(AI_DIR) if f.endswith(".png")])
        log(f"Defactify train set already cached: ai={n_ai} (real intentionally not fetched)")
        return {"n_ai": n_ai}

    from datasets import load_dataset

    os.makedirs(AI_DIR, exist_ok=True)

    log(f"Streaming {REPO_ID} (train split, no shuffle - sequential, memory-safe)...")
    ds = load_dataset(REPO_ID, split="train", streaming=True)

    # target n_per_class for EACH of the 5 generators (real is skipped
    # entirely - see module docstring); "ai" bucket ends up with
    # n_per_class * 5 images
    counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    ai_idx = 0
    for ex in ds:
        if all(v >= n_per_class for v in counts.values()):
            break
        lbl = ex["Label_B"]
        if lbl not in counts or counts[lbl] >= n_per_class:
            continue
        img = ex["Image"]
        if img is None:
            continue
        img = img.convert("RGB")
        w, h = img.size
        if max(w, h) > 512:
            scale = 512 / max(w, h)
            img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.BICUBIC)

        gen_name = GENERATOR_NAMES[lbl]
        img.save(os.path.join(AI_DIR, f"ai_{gen_name}_{ai_idx:04d}.png"))
        ai_idx += 1
        counts[lbl] += 1

    log("Collected per generator:")
    for k, v in counts.items():
        log(f"  {GENERATOR_NAMES[k]}: {v}")
    log(f"Total ai={ai_idx} (real intentionally not fetched, see module docstring)")
    return {"n_ai": ai_idx}


if __name__ == "__main__":
    fetch()
