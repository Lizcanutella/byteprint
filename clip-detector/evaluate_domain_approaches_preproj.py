"""
Compares, on the pre-projection CLIP feature space, across the full
16-cell transform grid:
  - balanced generalist (preproj)
  - domain-specialist routing (preproj specialists + the EXISTING
    classical-feature domain_classifier.pkl - routing doesn't depend on
    which CLIP feature space the downstream heads use)

against the post-projection baselines already measured (mean AUROC:
baseline=0.9316, balanced=0.9408, specialists=0.9362) and the naive
balanced+preproj stack (0.9433 on clean test, worse than either
individual improvement).

Usage:
    python evaluate_domain_approaches_preproj.py
"""

import json
import os
import pickle

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

from clip_features import embed_images_preproj, load_image_capped
from domain_classifier import extract_color_iqa_features
from transforms import TRANSFORM_GRID, TRANSFORM_TO_GROUP

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
EXPERIMENT_DIR = os.path.join(BASE_DIR, "results_detector", "experiments")


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def main():
    balanced_preproj_clf = load_pickle(
        os.path.join(EXPERIMENT_DIR, "classifier_head_balanced_preproj_v2.pkl")
    )
    specialists_preproj = load_pickle(os.path.join(EXPERIMENT_DIR, "specialists_preproj.pkl"))
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
        embs = embed_images_preproj(imgs, batch_size=32)

        bal_probs = balanced_preproj_clf.predict_proba(embs)[:, 1]

        color_feats = np.array([extract_color_iqa_features(img) for img in imgs])
        predicted_groups = domain_clf.predict(color_feats)
        routing_acc = float((predicted_groups == true_group).mean())

        spec_probs = np.zeros(len(imgs))
        for i, g in enumerate(predicted_groups):
            clf_g = specialists_preproj.get(g, balanced_preproj_clf)
            spec_probs[i] = clf_g.predict_proba(embs[i:i + 1])[:, 1][0]

        def metrics(probs):
            auc = float(roc_auc_score(y, probs))
            preds = (probs >= 0.5).astype(int)
            acc = float(accuracy_score(y, preds))
            return auc, acc

        bal_auc, bal_acc = metrics(bal_probs)
        spec_auc, spec_acc = metrics(spec_probs)

        print(f"  balanced-preproj AUROC={bal_auc:.4f} acc={bal_acc:.4f} | "
              f"specialists-preproj AUROC={spec_auc:.4f} acc={spec_acc:.4f} | "
              f"routing_acc={routing_acc:.4f}")

        rows.append({
            "transform": name, "true_group": true_group, "routing_accuracy": routing_acc,
            "balanced_preproj_auroc": bal_auc, "balanced_preproj_accuracy": bal_acc,
            "specialists_preproj_auroc": spec_auc, "specialists_preproj_accuracy": spec_acc,
        })

    out_path = os.path.join(EXPERIMENT_DIR, "domain_approaches_preproj_eval_result.json")
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)

    print("\n" + "=" * 100)
    print("PRE-PROJECTION: balanced-generalist vs. domain-specialists")
    print("=" * 100)
    header = f"{'transform':<20}{'group':<12}{'route_acc':<11}{'bal_auc':<10}{'spec_auc':<10}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['transform']:<20}{r['true_group']:<12}{r['routing_accuracy']:<11.4f}"
              f"{r['balanced_preproj_auroc']:<10.4f}{r['specialists_preproj_auroc']:<10.4f}")

    mean_bal = np.mean([r["balanced_preproj_auroc"] for r in rows])
    mean_spec = np.mean([r["specialists_preproj_auroc"] for r in rows])
    print(f"\nMean across all 16 cells: balanced-preproj={mean_bal:.4f}  "
          f"specialists-preproj={mean_spec:.4f}")
    print("\nFor reference (post-projection, from earlier experiments):")
    print("  baseline=0.9316  balanced=0.9408  specialists=0.9362")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
