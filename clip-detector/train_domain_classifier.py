"""
Trains the domain classifier (routing mechanism for the production
pipeline, see production_pipeline.py): a RandomForest over classical
no-reference features (domain_classifier.py) that predicts which of 5
degradation groups (clean/jpeg/spatial/noise/colorjitter) an image is
in. Trained on synthetic degradations of a subset of the training pool
with known ground truth - content-agnostic (it never sees real/AI
labels), so it doesn't need the same data-quality scrutiny as the
real-vs-AI classifier heads, but is retrained here from the current
(clean) train_classifier.compute_split() for full consistency.

Usage:
    python train_domain_classifier.py
"""

import os
import pickle
import random

from clip_features import load_image_capped
from domain_classifier import build_training_data, train_domain_classifier
from train_classifier import SEED, compute_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_DIR = os.path.join(BASE_DIR, "results_detector", "experiments")
N_DOMAIN_CLF_IMAGES = 200
N_SAMPLES_PER_IMAGE = 6


def main():
    paths, labels, sources, train_idx, test_idx = compute_split()
    rng = random.Random(SEED)
    train_paths = rng.sample([paths[i] for i in train_idx], min(N_DOMAIN_CLF_IMAGES, len(train_idx)))
    train_imgs = [load_image_capped(p) for p in train_paths]

    X, y = build_training_data(train_imgs, n_per_image=N_SAMPLES_PER_IMAGE)
    clf, metrics = train_domain_classifier(X, y)
    print(f"Domain classifier held-out accuracy: {metrics['accuracy']:.4f}")
    print(f"Confusion matrix labels: {metrics['confusion_matrix_labels']}")
    for row in metrics["confusion_matrix"]:
        print(" ", row)

    os.makedirs(EXPERIMENT_DIR, exist_ok=True)
    out_path = os.path.join(EXPERIMENT_DIR, "domain_classifier.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(clf, f)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
