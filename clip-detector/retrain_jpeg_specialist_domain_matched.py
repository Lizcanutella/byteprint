"""
Retrains ONLY the "jpeg" domain specialist using a domain-matched probe
(blur_s1.0) instead of the universal jpeg_q50 probe used by the other 4
specialists - re-JPEG-probing an already-JPEG-degraded image is
redundant by construction (prelim_domain_matched_probe.py measured
+0.0078 AUROC for this domain with blur vs +0.0004 with jpeg).

The other 4 specialists (clean/spatial/noise/colorjitter) are untouched
- they keep the jpeg_q50 probe, which already works well for them. Only
the jpeg specialist's training data + probe changes.

Usage:
    python retrain_jpeg_specialist_domain_matched.py
"""

import json
import pickle
import random
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from clip_features import embed_images_preproj, load_image_capped
from domain_classifier import apply_group_transform
from train_classifier import SEED, compute_split
from transforms import gaussian_blur

PROBE_FN = lambda img: gaussian_blur(img, 1.0)
CHUNK_SIZE = 250


def build_jpeg_domain_with_delta(paths, labels, indices, rng, chunk_size=CHUNK_SIZE):
    all_base, all_delta, all_y = [], [], []
    for start in range(0, len(indices), chunk_size):
        chunk = indices[start:start + chunk_size]
        imgs, ys = [], []
        for i in chunk:
            img = load_image_capped(paths[i])
            imgs.append(apply_group_transform(img, "jpeg", rng))
            ys.append(labels[i])
        probed_imgs = [PROBE_FN(img) for img in imgs]
        base = embed_images_preproj(imgs, batch_size=32)
        probed = embed_images_preproj(probed_imgs, batch_size=32)
        all_base.append(base)
        all_delta.append(probed - base)
        all_y.extend(ys)
        print(f"  {start + len(chunk)}/{len(indices)} embedded")
    return np.concatenate(all_base, axis=0), np.concatenate(all_delta, axis=0), np.array(all_y)


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    paths, labels, sources, train_idx, test_idx = compute_split()
    print(f"train={len(train_idx)} test={len(test_idx)}")

    rng_train = random.Random(SEED)
    print("Building jpeg-domain train set (blur_s1.0 probe)...")
    t0 = time.time()
    Xb_train, Xd_train, y_train = build_jpeg_domain_with_delta(paths, labels, train_idx, rng_train)
    print(f"  done in {time.time() - t0:.1f}s")
    X_train = np.concatenate([Xb_train, Xd_train], axis=1)

    rng_test = random.Random(SEED + 1)
    print("Building jpeg-domain test set (blur_s1.0 probe)...")
    t0 = time.time()
    Xb_test, Xd_test, y_test = build_jpeg_domain_with_delta(paths, labels, test_idx, rng_test)
    print(f"  done in {time.time() - t0:.1f}s")
    X_test = np.concatenate([Xb_test, Xd_test], axis=1)

    print("Training jpeg specialist (domain-matched blur probe)...")
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    clf.fit(X_train, y_train)
    auc = float(roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]))
    print(f"jpeg specialist (domain-matched) held-out AUROC: {auc:.4f}")

    # compare against current production jpeg specialist (jpeg_q50 probe) on the SAME test images
    from transforms import jpeg_compress
    old_probe = lambda img: jpeg_compress(img, 50)
    old_probed_test = [old_probe(img) for img in
                        [apply_group_transform(load_image_capped(paths[i]), "jpeg", random.Random(SEED + 1))
                         for i in tqdm(test_idx, desc="rebuild-for-old-probe-comparison")]]
    # NOTE: this rebuild uses a fresh rng matching rng_test's seed sequence, so the degraded
    # base images match what Xb_test was built from
    Xd_test_old_probe = embed_images_preproj(old_probed_test, batch_size=32) - Xb_test
    X_test_old_probe = np.concatenate([Xb_test, Xd_test_old_probe], axis=1)

    with open("model/specialists.pkl", "rb") as f:
        current_specialists = pickle.load(f)
    old_auc = float(roc_auc_score(y_test, current_specialists["jpeg"].predict_proba(X_test_old_probe)[:, 1]))
    print(f"Current production jpeg specialist (jpeg_q50 probe) on same test images: AUROC={old_auc:.4f}")

    result = {
        "domain_matched_jpeg_specialist_auroc": auc,
        "current_production_jpeg_q50_probe_auroc_on_same_data": old_auc,
        "improvement": auc - old_auc,
    }
    with open("results_detector/jpeg_domain_matched_retrain_result.json", "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))

    with open("model/experiments_jpeg_specialist_domain_matched.pkl", "wb") as f:
        pickle.dump(clf, f)
    print("Saved candidate specialist to model/experiments_jpeg_specialist_domain_matched.pkl")


if __name__ == "__main__":
    main()
