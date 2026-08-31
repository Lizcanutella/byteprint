"""
Experiment: does training on CLIP's PRE-projection vision features
(768-dim, before the visual_projection layer that aligns images with
text) outperform the POST-projection features (512-dim, what
train_classifier.py currently uses)? The projection is trained for
image-text alignment and may discard purely-visual information (subtle
generation artifacts) that isn't relevant to matching captions.

Uses the exact same train/test split as train_classifier.py (via
train_classifier.compute_split/build_split, same seed) so the
comparison is apples-to-apples against model/model_meta.json's
test_auroc=0.9458, test_accuracy=0.8656.

Usage:
    python experiment_preproj.py
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

from clip_features import embed_images_preproj
from train_classifier import SEED, build_split, compute_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_DIR = os.path.join(BASE_DIR, "results_detector", "experiments")


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    os.makedirs(EXPERIMENT_DIR, exist_ok=True)

    paths, labels, sources, train_idx, test_idx = compute_split()
    print(f"Collected {len(paths)} images; train={len(train_idx)} test={len(test_idx)} "
          f"(identical split to train_classifier.py)")

    rng = random.Random(SEED)
    print("Building train split (with robustness-grid augmentation, same as main run)...")
    train_imgs, y_train, _ = build_split(paths, labels, sources, train_idx, True, rng)
    print(f"  {len(train_imgs)} images (incl. augmented copies)")

    print("Building held-out test split (clean only)...")
    test_imgs, y_test, _ = build_split(paths, labels, sources, test_idx, False, rng)
    print(f"  {len(test_imgs)} images")

    print("Extracting PRE-projection CLIP embeddings for train split...")
    t0 = time.time()
    X_train = embed_images_preproj(train_imgs, batch_size=32)
    print(f"  done in {time.time() - t0:.1f}s  shape={X_train.shape}")

    print("Extracting PRE-projection CLIP embeddings for held-out test split...")
    t0 = time.time()
    X_test = embed_images_preproj(test_imgs, batch_size=32)
    print(f"  done in {time.time() - t0:.1f}s")

    print("Training logistic regression head on pre-projection features...")
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    clf.fit(X_train, y_train)

    train_auc = float(roc_auc_score(y_train, clf.predict_proba(X_train)[:, 1]))
    test_auc = float(roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]))
    test_acc = float(accuracy_score(y_test, clf.predict(X_test)))

    baseline_test_auc, baseline_test_acc = 0.9458272612455066, 0.8656387665198237
    print(f"\nTrain AUROC: {train_auc:.4f}")
    print(f"Held-out test AUROC: {test_auc:.4f}  accuracy: {test_acc:.4f}")
    print(f"Baseline (post-projection, 512-dim): AUROC {baseline_test_auc:.4f}  accuracy {baseline_test_acc:.4f}")
    delta = test_auc - baseline_test_auc
    print(f"Delta vs. baseline: {delta:+.4f} AUROC")

    model_path = os.path.join(EXPERIMENT_DIR, "classifier_head_preproj.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(clf, f)

    result = {
        "experiment": "preprojection_features",
        "feature_dim": int(X_train.shape[1]),
        "train_auroc": train_auc,
        "test_auroc": test_auc,
        "test_accuracy": test_acc,
        "baseline_test_auroc_postproj": baseline_test_auc,
        "baseline_test_accuracy_postproj": baseline_test_acc,
        "delta_auroc_vs_baseline": delta,
        "n_train": len(train_imgs),
        "n_test": len(test_imgs),
    }
    result_path = os.path.join(EXPERIMENT_DIR, "preproj_result.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved model to {model_path}")
    print(f"Saved result to {result_path}")


if __name__ == "__main__":
    main()
