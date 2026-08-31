"""
Fetches a small, generator-labeled DIAGNOSTIC sample from
Rajarshi-Roy-research/Defactify_Image_Dataset's TEST split, to measure
our already-trained production model's accuracy broken down by which
AI generator produced each image (Real, SD2.1, SDXL, SD3, DALL-E 3,
Midjourney 6). This directly tests for per-generator blind spots (the
DALL-E 3 miss the user found) without touching training data at all -
strictly a held-out evaluation set.

Usage:
    python fetch_generator_diagnostic.py
"""

import os

from PIL import Image

REPO_ID = "Rajarshi-Roy-research/Defactify_Image_Dataset"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_generator_diagnostic")
N_PER_CLASS = 60
SEED = 1234

LABEL_B_NAMES = {0: "real", 1: "sd21", 2: "sdxl", 3: "sd3", 4: "dalle3", 5: "midjourney6"}


def already_cached(min_n=20):
    if not os.path.isdir(DATA_DIR):
        return False
    for name in LABEL_B_NAMES.values():
        d = os.path.join(DATA_DIR, name)
        if not os.path.isdir(d) or len([f for f in os.listdir(d) if f.endswith(".png")]) < min_n:
            return False
    return True


def fetch(n_per_class=N_PER_CLASS, log=print):
    if already_cached(min_n=min(20, n_per_class)):
        log("Generator diagnostic set already cached, skipping download.")
        for name in LABEL_B_NAMES.values():
            d = os.path.join(DATA_DIR, name)
            log(f"  {name}: {len([f for f in os.listdir(d) if f.endswith('.png')])} images")
        return

    from datasets import load_dataset

    for name in LABEL_B_NAMES.values():
        os.makedirs(os.path.join(DATA_DIR, name), exist_ok=True)

    log(f"Streaming {REPO_ID} (test split, no shuffle - sequential, memory-safe)...")
    ds = load_dataset(REPO_ID, split="test", streaming=True)

    counts = {k: 0 for k in LABEL_B_NAMES}
    for ex in ds:
        if all(counts[k] >= n_per_class for k in LABEL_B_NAMES):
            break
        lbl = ex["Label_B"]
        if lbl not in LABEL_B_NAMES or counts[lbl] >= n_per_class:
            continue
        img = ex["Image"]
        if img is None:
            continue
        img = img.convert("RGB")
        w, h = img.size
        if max(w, h) > 512:
            scale = 512 / max(w, h)
            img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.BICUBIC)
        name = LABEL_B_NAMES[lbl]
        out_path = os.path.join(DATA_DIR, name, f"{name}_{counts[lbl]:04d}.png")
        img.save(out_path)
        counts[lbl] += 1

    log("Collected:")
    for k, name in LABEL_B_NAMES.items():
        log(f"  {name}: {counts[k]}")


if __name__ == "__main__":
    fetch()
