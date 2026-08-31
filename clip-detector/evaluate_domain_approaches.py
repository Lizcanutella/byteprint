"""
Compares three approaches on the SAME held-out test set, across the full
16-cell transform grid:

  Baseline:    model/classifier_head.pkl (single generalist, one random
               transform per augmented training image - the original run)
  Approach A:  results_detector/experiments/classifier_head_balanced.pkl
               (single generalist, but every training image contributes
               one sample from EVERY domain group)
  Approach B:  domain classifier routes each test image to a
               domain-SPECIALIZED head (results_detector/experiments/
               specialists.pkl), based on the classical-feature domain
               classifier's prediction (not ground truth - a realistic
               routing accuracy is part of what's being tested here).

Usage:
    python evaluate_domain_approaches.py
"""

import json
import os
import pickle

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

from clip_features import embed_images, load_image_capped
from domain_classifier import extract_color_iqa_features
from transforms import TRANSFORM_GRID, TRANSFORM_TO_GROUP

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
EXPERIMENT_DIR = os.path.join(BASE_DIR, "results_detector", "experiments")


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def effective_auroc_raw(y_true, scores):
    return float(roc_auc_score(y_true, scores))


def main():
    baseline_clf = load_pickle(os.path.join(MODEL_DIR, "classifier_head.pkl"))
    balanced_clf = load_pickle(os.path.join(EXPERIMENT_DIR, "classifier_head_balanced.pkl"))
    specialists = load_pickle(os.path.join(EXPERIMENT_DIR, "specialists.pkl"))
    domain_clf = load_pickle(os.path.join(EXPERIMENT_DIR, "domain_classifier.pkl"))

    with open(os.path.join(MODEL_DIR, "test_manifest.json")) as f:
        manifest = json.load(f)
    print(f"Held-out test set: {len(manifest)} images")

    y = np.array([m["label"] for m in manifest])

    rows = []
    for name, fn in TRANSFORM_GRID:
        true_group = TRANSFORM_TO_GROUP[name]
        print(f"[{name}] (true group={true_group}) applying transform + predicting...")

        imgs = [fn(load_image_capped(m["path"])) for m in manifest]
        embs = embed_images(imgs, batch_size=32)

        baseline_probs = baseline_clf.predict_proba(embs)[:, 1]
        balanced_probs = balanced_clf.predict_proba(embs)[:, 1]

        color_feats = np.array([extract_color_iqa_features(img) for img in imgs])
        predicted_groups = domain_clf.predict(color_feats)
        routing_acc = float((predicted_groups == true_group).mean())

        specialist_probs = np.zeros(len(imgs))
        for i, g in enumerate(predicted_groups):
            clf_g = specialists.get(g, baseline_clf)
            specialist_probs[i] = clf_g.predict_proba(embs[i:i + 1])[:, 1][0]

        def metrics(probs):
            auc = effective_auroc_raw(y, probs)
            preds = (probs >= 0.5).astype(int)
            acc = float(accuracy_score(y, preds))
            return auc, acc

        base_auc, base_acc = metrics(baseline_probs)
        bal_auc, bal_acc = metrics(balanced_probs)
        spec_auc, spec_acc = metrics(specialist_probs)

        print(f"  baseline AUROC={base_auc:.4f} acc={base_acc:.4f} | "
              f"balanced AUROC={bal_auc:.4f} acc={bal_acc:.4f} | "
              f"specialists AUROC={spec_auc:.4f} acc={spec_acc:.4f} | "
              f"routing_acc={routing_acc:.4f}")

        rows.append({
            "transform": name, "true_group": true_group, "routing_accuracy": routing_acc,
            "baseline_auroc": base_auc, "baseline_accuracy": base_acc,
            "balanced_auroc": bal_auc, "balanced_accuracy": bal_acc,
            "specialists_auroc": spec_auc, "specialists_accuracy": spec_acc,
        })

    out_path = os.path.join(EXPERIMENT_DIR, "domain_approaches_eval_result.json")
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)

    print("\n" + "=" * 110)
    print("COMPARISON: baseline vs. balanced-generalist vs. domain-specialists")
    print("=" * 110)
    header = (f"{'transform':<20}{'group':<12}{'route_acc':<11}{'base_auc':<10}"
              f"{'bal_auc':<10}{'spec_auc':<10}")
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['transform']:<20}{r['true_group']:<12}{r['routing_accuracy']:<11.4f}"
              f"{r['baseline_auroc']:<10.4f}{r['balanced_auroc']:<10.4f}{r['specialists_auroc']:<10.4f}")

    mean_base = np.mean([r["baseline_auroc"] for r in rows])
    mean_bal = np.mean([r["balanced_auroc"] for r in rows])
    mean_spec = np.mean([r["specialists_auroc"] for r in rows])
    mean_route = np.mean([r["routing_accuracy"] for r in rows])
    print(f"\nMean across all 16 cells: baseline={mean_base:.4f}  balanced={mean_bal:.4f}  "
          f"specialists={mean_spec:.4f}  routing_acc={mean_route:.4f}")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
