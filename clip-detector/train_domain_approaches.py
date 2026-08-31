"""
Builds and trains TWO alternative approaches to the baseline generalist
classifier (model/classifier_head.pkl), motivated by the fact that the
organizers' robustness test images are each degraded in exactly ONE
domain at a time (never stacked):

  Approach A ("balanced"): one generalist logistic-regression head, but
  trained with BALANCED augmentation - every training image contributes
  one sample from EVERY domain group (clean + jpeg + spatial + noise +
  colorjitter), instead of the baseline's one-random-pick-at-50%.

  Approach B ("specialists"): a domain classifier (domain_classifier.py,
  cheap classical features, no CLIP needed) predicts which domain group
  an image is in, then a domain-SPECIALIZED logistic-regression head
  (trained only on that group's augmented images) makes the real/AI call.

Both share the same CLIP embedding extraction pass (the balanced
augmented training set built here IS exactly what each specialist head
needs too), so this only costs one round of embedding extraction, not two.

Uses the exact same train/test split as train_classifier.py.

Usage:
    python train_domain_approaches.py
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

from clip_features import embed_images, load_image_capped
from domain_classifier import build_training_data as build_domain_training_data
from domain_classifier import extract_color_iqa_features, train_domain_classifier
from train_classifier import SEED, compute_split
from transforms import GROUP_NAMES

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_DIR = os.path.join(BASE_DIR, "results_detector", "experiments")
N_DOMAIN_CLF_IMAGES = 200
N_DOMAIN_CLF_SAMPLES_PER_IMAGE = 6


def embed_balanced_set(paths, labels, indices, rng, chunk_size=250, embed_fn=embed_images):
    """For each index, one clean copy + one randomly-parameterized sample
    from EACH non-clean group - balanced domain coverage per image,
    unlike train_classifier.build_split's single random pick.

    Processes `indices` in chunks and embeds each chunk immediately
    rather than materializing all ~5x augmented PIL images in memory at
    once (holding e.g. 2571 x 5 decoded 512x512 RGB images ~= 9.5GB
    OOM'd this machine - only the resulting embeddings, ~500 bytes/row,
    need to survive past each chunk)."""
    from domain_classifier import apply_group_transform

    all_embs, all_ys, all_groups = [], [], []
    for start in range(0, len(indices), chunk_size):
        chunk = indices[start:start + chunk_size]
        imgs, ys, groups = [], [], []
        for i in chunk:
            path, label = paths[i], labels[i]
            img = load_image_capped(path)
            imgs.append(img)
            ys.append(label)
            groups.append("clean")
            for group in GROUP_NAMES:
                if group == "clean":
                    continue
                degraded = apply_group_transform(img, group, rng)
                imgs.append(degraded)
                ys.append(label)
                groups.append(group)
        embs = embed_fn(imgs, batch_size=32)
        all_embs.append(embs)
        all_ys.extend(ys)
        all_groups.extend(groups)
        print(f"  chunk {start // chunk_size + 1}/"
              f"{(len(indices) + chunk_size - 1) // chunk_size}: "
              f"{len(chunk)} originals -> {len(imgs)} augmented images embedded")
    return np.concatenate(all_embs, axis=0), np.array(all_ys), np.array(all_groups)


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    os.makedirs(EXPERIMENT_DIR, exist_ok=True)

    paths, labels, sources, train_idx, test_idx = compute_split()
    print(f"Collected {len(paths)} images; train={len(train_idx)} test={len(test_idx)} "
          f"(identical split to train_classifier.py)")

    rng = random.Random(SEED)
    print("Building + embedding BALANCED train set (clean + one sample per domain "
          "group, per image), in chunks to bound memory...")
    t0 = time.time()
    X_train, y_train, train_groups = embed_balanced_set(paths, labels, train_idx, rng)
    print(f"  done in {time.time() - t0:.1f}s  shape={X_train.shape} "
          f"({len(train_idx)} originals x {len(GROUP_NAMES)} groups incl. clean)")

    print("Building + embedding held-out test split (clean only)...")
    t0 = time.time()
    test_imgs = [load_image_capped(paths[i]) for i in tqdm(test_idx, desc="load-test")]
    y_test = np.array([labels[i] for i in test_idx])
    X_test = embed_images(test_imgs, batch_size=32)
    print(f"  done in {time.time() - t0:.1f}s")

    # --- Approach A: balanced generalist ---
    print("\nTraining Approach A (balanced generalist)...")
    clf_a = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    clf_a.fit(X_train, y_train)
    auc_a = float(roc_auc_score(y_test, clf_a.predict_proba(X_test)[:, 1]))
    acc_a = float(accuracy_score(y_test, clf_a.predict(X_test)))
    print(f"  Held-out clean test AUROC: {auc_a:.4f}  accuracy: {acc_a:.4f}")

    # --- Approach B: per-domain specialists ---
    print("\nTraining Approach B (per-domain specialists)...")
    specialists = {}
    for group in GROUP_NAMES:
        mask = train_groups == group
        Xg, yg = X_train[mask], y_train[mask]
        clf_g = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
        clf_g.fit(Xg, yg)
        specialists[group] = clf_g
        print(f"  [{group}] trained on {mask.sum()} images")

    # --- Domain classifier (routing mechanism for Approach B) ---
    print("\nTraining domain classifier (cheap classical features, routes Approach B)...")
    domain_train_paths = rng.sample(list(paths[i] for i in train_idx),
                                     min(N_DOMAIN_CLF_IMAGES, len(train_idx)))
    domain_train_imgs = [load_image_capped(p) for p in domain_train_paths]
    Xd, yd = build_domain_training_data(domain_train_imgs, n_per_image=N_DOMAIN_CLF_SAMPLES_PER_IMAGE)
    domain_clf, domain_metrics = train_domain_classifier(Xd, yd)
    print(f"  Domain classifier held-out accuracy: {domain_metrics['accuracy']:.4f}")
    print(f"  Confusion matrix labels: {domain_metrics['confusion_matrix_labels']}")
    for row in domain_metrics["confusion_matrix"]:
        print(f"    {row}")

    # --- Save everything ---
    with open(os.path.join(EXPERIMENT_DIR, "classifier_head_balanced.pkl"), "wb") as f:
        pickle.dump(clf_a, f)
    with open(os.path.join(EXPERIMENT_DIR, "specialists.pkl"), "wb") as f:
        pickle.dump(specialists, f)
    with open(os.path.join(EXPERIMENT_DIR, "domain_classifier.pkl"), "wb") as f:
        pickle.dump(domain_clf, f)

    result = {
        "approach_a_balanced_generalist": {
            "test_auroc": auc_a, "test_accuracy": acc_a,
            "n_train": len(train_imgs), "n_test": len(test_imgs),
        },
        "approach_b_specialists": {
            "group_train_sizes": {g: int((train_groups == g).sum()) for g in GROUP_NAMES},
        },
        "domain_classifier_metrics": domain_metrics,
        "baseline_test_auroc_for_reference": 0.9458272612455066,
        "baseline_test_accuracy_for_reference": 0.8656387665198237,
    }
    result_path = os.path.join(EXPERIMENT_DIR, "domain_approaches_train_result.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved models + result to {EXPERIMENT_DIR}/")
    print(f"Saved training summary to {result_path}")


if __name__ == "__main__":
    main()
