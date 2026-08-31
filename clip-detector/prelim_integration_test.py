"""
Preliminary "does the jpeg_q50 reactivity-delta feature actually help
when trained/evaluated the same way the REAL production specialists
are" check - a step up from diagnostic_reactivity_delta.py /
diagnostic_reactivity_comprehensive.py (which used a simplified,
non-domain-routed CV harness), before committing to the full ~1hr
production integration + re-validation.

Trains, for each of the 5 real domain groups (clean/jpeg/spatial/noise/
colorjitter), TWO specialists on a SUBSAMPLE (500 of the 2506 training
originals, to keep this under production-scale runtime) of the exact
same balanced per-domain training data the real specialists use:
  - baseline: absolute CLIP pre-projection embedding only (768-dim)
  - extended: baseline + jpeg_q50-probe embedding delta (1536-dim)

Evaluates both against the FULL 443-image held-out test set (not
subsampled - cheap and gives the real answer), with that same domain's
transform applied to the test images (mirroring robustness_eval.py's
per-cell evaluation), for an apples-to-apples baseline-vs-extended
comparison per domain.

Does NOT touch model/ or any production file - purely diagnostic.

Usage:
    python prelim_integration_test.py
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
from transforms import GROUP_NAMES, jpeg_compress

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROBE_FN = lambda img: jpeg_compress(img, 50)
N_SUBSAMPLE = 500
CHUNK_SIZE = 100


def build_domain_batch(paths, labels, indices, group, rng, chunk_size=CHUNK_SIZE):
    """For `group`, apply that domain's transform to every image in
    `indices`, then embed both the plain degraded image AND its
    jpeg_q50-probed copy. Returns (X_base, X_delta, y)."""
    all_base, all_delta, all_y = [], [], []
    for start in range(0, len(indices), chunk_size):
        chunk = indices[start:start + chunk_size]
        degraded_imgs, ys = [], []
        for i in chunk:
            img = load_image_capped(paths[i])
            degraded_imgs.append(apply_group_transform(img, group, rng))
            ys.append(labels[i])
        probed_imgs = [PROBE_FN(img) for img in degraded_imgs]

        emb_base = embed_images_preproj(degraded_imgs, batch_size=32)
        emb_probed = embed_images_preproj(probed_imgs, batch_size=32)
        all_base.append(emb_base)
        all_delta.append(emb_probed - emb_base)
        all_y.extend(ys)
    return np.concatenate(all_base, axis=0), np.concatenate(all_delta, axis=0), np.array(all_y)


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    paths, labels, sources, train_idx, test_idx = compute_split()
    labels_arr = np.array(labels)
    print(f"Full split: train={len(train_idx)} test={len(test_idx)}")

    sub_idx, _ = train_test_split(
        train_idx, train_size=N_SUBSAMPLE, random_state=SEED, stratify=labels_arr[train_idx]
    )
    print(f"Training subsample: {len(sub_idx)} of {len(train_idx)} originals")

    rng_train = random.Random(SEED)
    rng_test = random.Random(SEED + 1)  # different draw than training, still deterministic

    results = {}
    for group in GROUP_NAMES:
        print(f"\n{'=' * 70}\nDOMAIN: {group}\n{'=' * 70}")

        print(f"Building training batch ({len(sub_idx)} images, group={group})...")
        t0 = time.time()
        Xb_train, Xd_train, y_train = build_domain_batch(paths, labels, sub_idx, group, rng_train)
        print(f"  done in {time.time() - t0:.1f}s")

        print(f"Building held-out test batch (443 images, group={group})...")
        t0 = time.time()
        Xb_test, Xd_test, y_test = build_domain_batch(paths, labels, test_idx, group, rng_test)
        print(f"  done in {time.time() - t0:.1f}s")

        baseline_clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
        baseline_clf.fit(Xb_train, y_train)
        auc_baseline = float(roc_auc_score(y_test, baseline_clf.predict_proba(Xb_test)[:, 1]))

        extended_clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
        X_train_ext = np.concatenate([Xb_train, Xd_train], axis=1)
        X_test_ext = np.concatenate([Xb_test, Xd_test], axis=1)
        extended_clf.fit(X_train_ext, y_train)
        auc_extended = float(roc_auc_score(y_test, extended_clf.predict_proba(X_test_ext)[:, 1]))

        results[group] = {
            "baseline_auroc": auc_baseline,
            "extended_with_delta_auroc": auc_extended,
            "delta_improvement": auc_extended - auc_baseline,
            "n_train": len(sub_idx),
            "n_test": len(test_idx),
        }
        print(f"  baseline={auc_baseline:.4f}  extended(+delta)={auc_extended:.4f}  "
              f"delta={auc_extended - auc_baseline:+.4f}")

    out_path = os.path.join(BASE_DIR, "results_detector", "prelim_integration_test.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")

    print("\n" + "=" * 70)
    print("SUMMARY (baseline vs extended-with-jpeg_q50-delta, per domain)")
    print("=" * 70)
    mean_base = np.mean([r["baseline_auroc"] for r in results.values()])
    mean_ext = np.mean([r["extended_with_delta_auroc"] for r in results.values()])
    for group, r in results.items():
        print(f"{group:<14} baseline={r['baseline_auroc']:.4f}  extended={r['extended_with_delta_auroc']:.4f}  "
              f"delta={r['delta_improvement']:+.4f}")
    print(f"{'MEAN':<14} baseline={mean_base:.4f}  extended={mean_ext:.4f}  delta={mean_ext - mean_base:+.4f}")


if __name__ == "__main__":
    main()
