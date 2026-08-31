"""
Streaming sampler for saberzl/SID_Set (organizer-recommended dataset),
following the same cache-to-PNG-then-skip-on-rerun pattern as
fetch_data.py. SID_Set isn't schema-detectable by fetch_data's
find_label_mapping (its label column is a plain int64, not a
ClassLabel), so it gets its own small loader.

Confirmed schema via sampling (see plan): label 0 = real (varied
resolution/aspect, no mask), label 1 = fully AI-generated (1024x1024,
no mask), label 2 = locally edited/inpainted (1024x1024, WITH a mask).
We only pull labels 0 and 1 for training (binary real-vs-AI); label 2
is left for later error-analysis/explainability material, not used
here.

Usage:
    python sid_set_fetch.py                # download/cache
"""

import os

from PIL import Image

REPO_ID = "saberzl/SID_Set"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_sid_set")
REAL_DIR = os.path.join(DATA_DIR, "real")
AI_DIR = os.path.join(DATA_DIR, "ai")
N_PER_CLASS = 600
SEED = 1234


def already_cached(real_dir=REAL_DIR, ai_dir=AI_DIR, min_n=200):
    if not (os.path.isdir(real_dir) and os.path.isdir(ai_dir)):
        return False
    n_real = len([f for f in os.listdir(real_dir) if f.lower().endswith(".png")])
    n_ai = len([f for f in os.listdir(ai_dir) if f.lower().endswith(".png")])
    return n_real >= min_n and n_ai >= min_n


def fetch_sid_set(real_dir=REAL_DIR, ai_dir=AI_DIR, n_per_class=N_PER_CLASS, log=print):
    if already_cached(real_dir, ai_dir, min_n=min(200, n_per_class)):
        n_real = len([f for f in os.listdir(real_dir) if f.lower().endswith(".png")])
        n_ai = len([f for f in os.listdir(ai_dir) if f.lower().endswith(".png")])
        log(f"SID_Set: found cached data n_real={n_real}, n_ai={n_ai}, skipping download.")
        return {"n_real": n_real, "n_ai": n_ai}

    from datasets import load_dataset

    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(ai_dir, exist_ok=True)

    log(f"SID_Set: streaming {REPO_ID} (train split)...")
    ds = load_dataset(REPO_ID, split="train", streaming=True)
    # NOTE: deliberately NOT calling .shuffle() here. Even a small
    # buffer_size caused this machine to OOM against SID_Set's large
    # (1024x1024) images - the memory pressure comes from shard-level
    # buffering in the underlying reader, not primarily our own
    # per-example objects, so shrinking buffer_size alone didn't fix it.
    # We take examples in on-disk order instead; label 0/1 selection
    # below still gives an unbiased (if not globally shuffled) sample.

    counts = {0: 0, 1: 0}
    for ex in ds:
        if counts[0] >= n_per_class and counts[1] >= n_per_class:
            break
        lbl = ex["label"]
        if lbl not in (0, 1):
            continue  # skip label 2 (locally-edited) for the binary training set
        if counts[lbl] >= n_per_class:
            continue
        img = ex["image"]
        if img is None:
            continue
        img = img.convert("RGB")
        # Source images are up to 1024x1024; downsize the long side to 512
        # (still well above CLIP's 224x224 input and our transform grid's
        # needs) to keep disk/decode costs down for ~1800 saved images.
        w, h = img.size
        if max(w, h) > 512:
            scale = 512 / max(w, h)
            img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.BICUBIC)
        out_dir = real_dir if lbl == 0 else ai_dir
        idx = counts[lbl]
        img.save(os.path.join(out_dir, f"{'real' if lbl == 0 else 'ai'}_{idx:04d}.png"))
        counts[lbl] += 1

    log(f"SID_Set: collected real={counts[0]}, ai={counts[1]}")
    return {"n_real": counts[0], "n_ai": counts[1]}


if __name__ == "__main__":
    info = fetch_sid_set()
    print(info)
