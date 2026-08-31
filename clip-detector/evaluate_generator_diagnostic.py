"""
Evaluates the production pipeline against the generator-labeled
diagnostic set (fetch_generator_diagnostic.py: Real, SD2.1, SDXL, SD3,
DALL-E 3, Midjourney 6 - from Rajarshi-Roy-research/Defactify_Image_Dataset's
TEST split, never used in training). Reports accuracy broken down by
generator, to find per-generator blind spots (motivated by a user-found
miss on a ChatGPT/DALL-E 3 image).

Usage:
    python evaluate_generator_diagnostic.py
"""

import json
import os

import numpy as np
from sklearn.metrics import roc_auc_score

from clip_features import load_image_capped
from production_pipeline import predict_proba

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data_generator_diagnostic")
RESULTS_DIR = os.path.join(BASE_DIR, "results_detector")

CLASSES = ["real", "sd21", "sdxl", "sd3", "dalle3", "midjourney6"]


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    records = []
    for cls in CLASSES:
        d = os.path.join(DATA_DIR, cls)
        for f in sorted(os.listdir(d)):
            records.append({"path": os.path.join(d, f), "class": cls, "label": 0 if cls == "real" else 1})

    print(f"Loaded {len(records)} diagnostic images across {CLASSES}")

    imgs = [load_image_capped(r["path"]) for r in records]
    probs, groups = predict_proba(imgs, batch_size=32)

    for r, p, g in zip(records, probs, groups):
        r["pred"] = float(p)
        r["routed_domain"] = str(g)

    y = np.array([r["label"] for r in records])
    all_probs = np.array([r["pred"] for r in records])
    overall_auc = float(roc_auc_score(y, all_probs))

    print(f"\nOverall AUROC (real vs. all generators pooled): {overall_auc:.4f}\n")

    print(f"{'class':<14}{'n':<6}{'mean_pred':<12}{'accuracy(@0.5)':<16}")
    print("-" * 48)
    per_class = {}
    for cls in CLASSES:
        cls_records = [r for r in records if r["class"] == cls]
        preds = np.array([r["pred"] for r in cls_records])
        if cls == "real":
            acc = float((preds < 0.5).mean())  # correct = predicted real
        else:
            acc = float((preds >= 0.5).mean())  # correct = predicted AI
        per_class[cls] = {"n": len(cls_records), "mean_pred": float(preds.mean()), "accuracy": acc}
        print(f"{cls:<14}{len(cls_records):<6}{preds.mean():<12.4f}{acc:<16.4f}")

    out = {
        "overall_auroc_real_vs_all_generators": overall_auc,
        "per_class": per_class,
    }
    out_path = os.path.join(RESULTS_DIR, "generator_diagnostic_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    per_image_path = os.path.join(RESULTS_DIR, "generator_diagnostic_per_image.json")
    with open(per_image_path, "w") as f:
        json.dump(records, f, indent=2)

    print(f"\nSaved: {out_path}")
    print(f"Saved: {per_image_path}")


if __name__ == "__main__":
    main()
