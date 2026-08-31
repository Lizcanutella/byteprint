"""
Same idea as train_domain_approaches.py, but using CLIP's PRE-projection
(768-dim) features instead of post-projection (512-dim) - testing
whether domain-specialist routing (Approach B) recovers the accuracy
lost when naively stacking balanced-augmentation + pre-projection
features as a single generalist (experiment_balanced_preproj.py found
0.9433, worse than either individual improvement alone).

Reuses the EXISTING domain_classifier.pkl (classical IQA features,
independent of which CLIP feature space the downstream heads use) for
routing - no need to retrain the router.

Usage:
    python train_domain_approaches_preproj.py
"""

import json
import os
import pickle
import random
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from clip_features import embed_images_preproj, load_image_capped
from train_classifier import SEED, compute_split
from train_domain_approaches import embed_balanced_set
from transforms import GROUP_NAMES

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_DIR = os.path.join(BASE_DIR, "results_detector", "experiments")


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    os.makedirs(EXPERIMENT_DIR, exist_ok=True)

    paths, labels, sources, train_idx, test_idx = compute_split()
    print(f"Collected {len(paths)} images; train={len(train_idx)} test={len(test_idx)}")

    rng = random.Random(SEED)
    print("Building + embedding BALANCED train set with PRE-projection features...")
    t0 = time.time()
    X_train, y_train, train_groups = embed_balanced_set(
        paths, labels, train_idx, rng, embed_fn=embed_images_preproj
    )
    print(f"  done in {time.time() - t0:.1f}s  shape={X_train.shape}")

    print("Building + embedding held-out test split (clean only, pre-projection)...")
    t0 = time.time()
    test_imgs = [load_image_capped(paths[i]) for i in tqdm(test_idx, desc="load-test")]
    y_test = np.array([labels[i] for i in test_idx])
    X_test = embed_images_preproj(test_imgs, batch_size=32)
    print(f"  done in {time.time() - t0:.1f}s")

    print("\nTraining Approach A (balanced generalist, pre-projection)...")
    clf_a = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    clf_a.fit(X_train, y_train)
    auc_a = float(roc_auc_score(y_test, clf_a.predict_proba(X_test)[:, 1]))
    acc_a = float(accuracy_score(y_test, clf_a.predict(X_test)))
    print(f"  Held-out clean test AUROC: {auc_a:.4f}  accuracy: {acc_a:.4f}")

    print("\nTraining Approach B (per-domain specialists, pre-projection)...")
    specialists = {}
    for group in GROUP_NAMES:
        mask = train_groups == group
        Xg, yg = X_train[mask], y_train[mask]
        clf_g = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
        clf_g.fit(Xg, yg)
        specialists[group] = clf_g
        print(f"  [{group}] trained on {int(mask.sum())} images")

    with open(os.path.join(EXPERIMENT_DIR, "classifier_head_balanced_preproj_v2.pkl"), "wb") as f:
        pickle.dump(clf_a, f)
    with open(os.path.join(EXPERIMENT_DIR, "specialists_preproj.pkl"), "wb") as f:
        pickle.dump(specialists, f)

    result = {
        "approach_a_balanced_preproj_generalist": {"test_auroc": auc_a, "test_accuracy": acc_a},
        "approach_b_specialists_preproj": {
            "group_train_sizes": {g: int((train_groups == g).sum()) for g in GROUP_NAMES}
        },
    }
    result_path = os.path.join(EXPERIMENT_DIR, "domain_approaches_preproj_train_result.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved models + result to {result_path}")


if __name__ == "__main__":
    main()
