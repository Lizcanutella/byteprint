"""
Tests the specific, refined hypothesis from the crop-vs-whole-image
GPU experiment that was never actually checked: does native-resolution
texture-crop embedding beat whole-image embedding specifically on
DEGRADED domains (where fine detail is scarcer and BYTEPRINT's
crop-preserves-evidence argument should matter most), even though it
lost on the CLEAN domain (where the whole-image approach was already
near-ceiling at 0.999 AUROC)?

For each of two domains - "spatial" (blur/resize/crop, hypothesized to
benefit most from native detail) and "jpeg" (hypothesized to show no
benefit, as a control, since jpeg was already near-ceiling in
production) - trains a whole-image specialist and a crop-based
specialist on the SAME degraded training subsample, then evaluates both
on the full held-out test set under that same domain's degradation.

Runs entirely on CPU with a modest subsample - no GPU needed for this
scoped comparison, unlike the full production-scale crop retrain.

Usage:
    python diagnostic_crops_vs_whole_by_domain.py
"""

import json
import random
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from clip_features import (
    embed_images_preproj,
    embed_images_preproj_crops,
    load_image_capped,
    load_image_native,
)
from domain_classifier import apply_group_transform
from train_classifier import SEED, compute_split

N_SUBSAMPLE = 400
TOP_K = 3
DOMAINS_TO_TEST = ["spatial", "jpeg"]  # hypothesized "hard" vs "easy" control


def build_domain_set(paths, labels, indices, group, rng, loader, embed_fn):
    imgs, ys = [], []
    for i in indices:
        img = loader(paths[i])
        imgs.append(apply_group_transform(img, group, rng))
        ys.append(labels[i])
    X = embed_fn(imgs)
    return X, np.array(ys)


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

    results = {}
    for group in DOMAINS_TO_TEST:
        print(f"\n{'=' * 70}\nDOMAIN: {group}\n{'=' * 70}")
        rng_train = random.Random(SEED + hash(group) % 1000)
        rng_test = random.Random(SEED + 1 + hash(group) % 1000)

        print("Building whole-image (production-style) train/test...")
        t0 = time.time()
        Xw_train, y_train = build_domain_set(
            paths, labels, sub_idx, group, rng_train, load_image_capped,
            lambda imgs: embed_images_preproj(imgs, batch_size=32),
        )
        Xw_test, y_test = build_domain_set(
            paths, labels, test_idx, group, rng_test, load_image_capped,
            lambda imgs: embed_images_preproj(imgs, batch_size=32),
        )
        print(f"  done in {time.time() - t0:.1f}s")

        print(f"Building native-crop (top_k={TOP_K}) train/test...")
        t0 = time.time()
        Xc_train, _ = build_domain_set(
            paths, labels, sub_idx, group, random.Random(SEED + hash(group) % 1000), load_image_native,
            lambda imgs: embed_images_preproj_crops(imgs, top_k=TOP_K, batch_size=32),
        )
        Xc_test, _ = build_domain_set(
            paths, labels, test_idx, group, random.Random(SEED + 1 + hash(group) % 1000), load_image_native,
            lambda imgs: embed_images_preproj_crops(imgs, top_k=TOP_K, batch_size=32),
        )
        print(f"  done in {time.time() - t0:.1f}s")

        whole_clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
        whole_clf.fit(Xw_train, y_train)
        auc_whole = float(roc_auc_score(y_test, whole_clf.predict_proba(Xw_test)[:, 1]))

        crop_clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
        crop_clf.fit(Xc_train, y_train)
        auc_crop = float(roc_auc_score(y_test, crop_clf.predict_proba(Xc_test)[:, 1]))

        results[group] = {"whole_image_auroc": auc_whole, "native_crop_auroc": auc_crop,
                           "crop_advantage": auc_crop - auc_whole}
        print(f"  whole-image AUROC={auc_whole:.4f}  native-crop AUROC={auc_crop:.4f}  "
              f"crop_advantage={auc_crop - auc_whole:+.4f}")

    with open("results_detector/crops_vs_whole_by_domain.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for group, r in results.items():
        print(f"{group:<10} whole={r['whole_image_auroc']:.4f}  crop={r['native_crop_auroc']:.4f}  "
              f"advantage={r['crop_advantage']:+.4f}")


if __name__ == "__main__":
    main()
