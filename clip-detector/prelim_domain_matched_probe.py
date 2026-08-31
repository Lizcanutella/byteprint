"""
Follow-up to prelim_integration_test.py: that test used a single fixed
probe (jpeg_q50) for every domain, and found ~zero gain specifically on
the "jpeg" domain (+0.0004) - unsurprising, since re-applying a JPEG
probe to an already-JPEG-degraded image adds little new information.
This tests whether a DIFFERENT probe (blur_s1.0 or noise_s0.05) does
better specifically for the "jpeg" domain's specialist, using the same
degraded train/test images for all three probes so the comparison is
apples-to-apples (only the probe changes, not the underlying data).

Does NOT touch model/ or any production file - purely diagnostic.

Usage:
    python prelim_domain_matched_probe.py
"""

import json
import os
import random
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from clip_features import embed_images_preproj, load_image_capped
from domain_classifier import apply_group_transform
from train_classifier import SEED, compute_split
from transforms import gaussian_blur, gaussian_noise, jpeg_compress

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TARGET_GROUP = "jpeg"
N_SUBSAMPLE = 500
CHUNK_SIZE = 100

CANDIDATE_PROBES = {
    "jpeg_q50 (original, redundant)": lambda img: jpeg_compress(img, 50),
    "blur_s1.0": lambda img: gaussian_blur(img, 1.0),
    "noise_s0.05": lambda img: gaussian_noise(img, 0.05),
}


def build_group_images(paths, labels, indices, group, rng, chunk_size=CHUNK_SIZE):
    """Apply `group`'s transform once to every image in `indices`, embed
    the base (degraded) images, and return (degraded_pil_images, X_base, y)
    so multiple probes can be tried on the SAME degraded images."""
    all_imgs, all_base, all_y = [], [], []
    for start in range(0, len(indices), chunk_size):
        chunk = indices[start:start + chunk_size]
        degraded_imgs, ys = [], []
        for i in chunk:
            img = load_image_capped(paths[i])
            degraded_imgs.append(apply_group_transform(img, group, rng))
            ys.append(labels[i])
        emb_base = embed_images_preproj(degraded_imgs, batch_size=32)
        all_imgs.extend(degraded_imgs)
        all_base.append(emb_base)
        all_y.extend(ys)
    return all_imgs, np.concatenate(all_base, axis=0), np.array(all_y)


def embed_probe(imgs, probe_fn, chunk_size=CHUNK_SIZE):
    embs = []
    for start in range(0, len(imgs), chunk_size):
        chunk = [probe_fn(img) for img in imgs[start:start + chunk_size]]
        embs.append(embed_images_preproj(chunk, batch_size=32))
    return np.concatenate(embs, axis=0)


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    paths, labels, sources, train_idx, test_idx = compute_split()
    labels_arr = np.array(labels)

    sub_idx, _ = train_test_split(
        train_idx, train_size=N_SUBSAMPLE, random_state=SEED, stratify=labels_arr[train_idx]
    )
    print(f"Training subsample: {len(sub_idx)} of {len(train_idx)} originals; group={TARGET_GROUP}")

    rng_train = random.Random(SEED + 100)  # dedicated seed for this focused test
    rng_test = random.Random(SEED + 101)

    print("Building jpeg-domain TRAIN images + base embeddings...")
    t0 = time.time()
    train_imgs, Xb_train, y_train = build_group_images(paths, labels, sub_idx, TARGET_GROUP, rng_train)
    print(f"  done in {time.time() - t0:.1f}s")

    print("Building jpeg-domain TEST images + base embeddings...")
    t0 = time.time()
    test_imgs, Xb_test, y_test = build_group_images(paths, labels, test_idx, TARGET_GROUP, rng_test)
    print(f"  done in {time.time() - t0:.1f}s")

    baseline_clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    baseline_clf.fit(Xb_train, y_train)
    auc_baseline = float(roc_auc_score(y_test, baseline_clf.predict_proba(Xb_test)[:, 1]))
    print(f"\nBaseline (absolute embedding only, no delta): AUROC={auc_baseline:.4f}")

    results = {"baseline_no_delta": auc_baseline}
    for probe_name, probe_fn in CANDIDATE_PROBES.items():
        print(f"\n=== Probe: {probe_name} ===")
        t0 = time.time()
        Xp_train = embed_probe(train_imgs, probe_fn)
        Xp_test = embed_probe(test_imgs, probe_fn)
        print(f"  embedded probed copies in {time.time() - t0:.1f}s")

        Xd_train = Xp_train - Xb_train
        Xd_test = Xp_test - Xb_test

        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
        X_train_ext = np.concatenate([Xb_train, Xd_train], axis=1)
        X_test_ext = np.concatenate([Xb_test, Xd_test], axis=1)
        clf.fit(X_train_ext, y_train)
        auc = float(roc_auc_score(y_test, clf.predict_proba(X_test_ext)[:, 1]))
        results[probe_name] = auc
        print(f"  extended AUROC={auc:.4f}  delta vs baseline={auc - auc_baseline:+.4f}")

    out_path = os.path.join(BASE_DIR, "results_detector", "prelim_domain_matched_probe.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")

    print("\n" + "=" * 70)
    print("SUMMARY (jpeg domain, best substitute probe search)")
    print("=" * 70)
    print(f"baseline (no delta): {auc_baseline:.4f}")
    for probe_name, probe_fn in CANDIDATE_PROBES.items():
        print(f"  + {probe_name}: {results[probe_name]:.4f}  ({results[probe_name] - auc_baseline:+.4f})")


if __name__ == "__main__":
    main()
