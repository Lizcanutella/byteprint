"""
Experiment: stack the two independent wins found so far -
balanced-domain-coverage augmentation (Approach A, +0.0092 mean AUROC
across the 16-cell grid) and CLIP pre-projection features (+0.0048 on
clean test) - by training the balanced generalist on 768-dim
PRE-projection embeddings instead of the 512-dim post-projection ones.

Uses the exact same train/test split as train_classifier.py and the
exact same balanced-augmentation scheme as train_domain_approaches.py's
Approach A, so this is a clean two-way comparison, not a new dataset.

Usage:
    python experiment_balanced_preproj.py
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
    print("Building + embedding BALANCED train set with PRE-projection features...")
    t0 = time.time()
    X_train, y_train, train_groups = embed_balanced_set(
        paths, labels, train_idx, rng, embed_fn=embed_images_preproj
    )
    print(f"  done in {time.time() - t0:.1f}s  shape={X_train.shape}")

    print("Building + embedding held-out test split (clean only, PRE-projection)...")
    t0 = time.time()
    test_imgs = [load_image_capped(paths[i]) for i in tqdm(test_idx, desc="load-test")]
    y_test = np.array([labels[i] for i in test_idx])
    X_test = embed_images_preproj(test_imgs, batch_size=32)
    print(f"  done in {time.time() - t0:.1f}s")

    print("Training logistic regression head (balanced + pre-projection)...")
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    clf.fit(X_train, y_train)

    train_auc = float(roc_auc_score(y_train, clf.predict_proba(X_train)[:, 1]))
    test_auc = float(roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]))
    test_acc = float(accuracy_score(y_test, clf.predict(X_test)))

    references = {
        "baseline_postproj_single_random_aug": 0.9458272612455066,
        "balanced_postproj": 0.9449,  # from train_domain_approaches.py Approach A
        "preproj_single_random_aug": 0.9506,  # from experiment_preproj.py
    }
    print(f"\nTrain AUROC: {train_auc:.4f}")
    print(f"Held-out clean test AUROC: {test_auc:.4f}  accuracy: {test_acc:.4f}")
    for name, val in references.items():
        print(f"  vs. {name}: {val:.4f}  (delta {test_auc - val:+.4f})")

    model_path = os.path.join(EXPERIMENT_DIR, "classifier_head_balanced_preproj.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(clf, f)

    result = {
        "experiment": "balanced_augmentation_plus_preprojection_features",
        "feature_dim": int(X_train.shape[1]),
        "train_auroc": train_auc,
        "test_auroc": test_auc,
        "test_accuracy": test_acc,
        "reference_scores": references,
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
    }
    result_path = os.path.join(EXPERIMENT_DIR, "balanced_preproj_result.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved model to {model_path}")
    print(f"Saved result to {result_path}")


if __name__ == "__main__":
    main()
