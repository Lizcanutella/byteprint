"""
Robustness evaluation: runs the PRODUCTION detector (production_pipeline.py
- domain classifier + CLIP-pre-projection specialists) against the FULL
hackathon transform grid (transforms.py) on the held-out test set saved
by train_classifier.py (train_classifier.py sets this split aside
BEFORE augmentation, so it is never seen in training in any form).
Produces the required "compact table or visual summary comparing
performance on clean images versus transformed images."

Usage:
    python robustness_eval.py
"""

import json
import os
import time

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from clip_features import load_image_capped
from production_pipeline import predict_proba
from transforms import TRANSFORM_GRID

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
RESULTS_DIR = os.path.join(BASE_DIR, "results_detector")


def load_test_manifest():
    with open(os.path.join(MODEL_DIR, "test_manifest.json")) as f:
        return json.load(f)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    manifest = load_test_manifest()
    print(f"Held-out test set: {len(manifest)} images")

    y = np.array([m["label"] for m in manifest])

    rows = []
    per_image_records = {}
    for name, fn in TRANSFORM_GRID:
        print(f"[{name}] applying transform + predicting...")
        t0 = time.time()
        imgs = [fn(load_image_capped(m["path"])) for m in manifest]
        probs, groups = predict_proba(imgs, batch_size=32)
        preds = (probs >= 0.5).astype(int)

        auc = float(roc_auc_score(y, probs))
        acc = float(accuracy_score(y, preds))
        f1 = float(f1_score(y, preds))
        n_real = int((y == 0).sum())
        n_ai = int((y == 1).sum())
        fpr = int(((y == 0) & (preds == 1)).sum()) / max(n_real, 1)
        fnr = int(((y == 1) & (preds == 0)).sum()) / max(n_ai, 1)

        print(f"  AUROC={auc:.4f} acc={acc:.4f} f1={f1:.4f} FPR={fpr:.4f} FNR={fnr:.4f} "
              f"({time.time() - t0:.1f}s)")

        rows.append({
            "transform": name, "auroc": auc, "accuracy": acc, "f1": f1,
            "n": len(manifest), "false_positive_rate": fpr, "false_negative_rate": fnr,
        })
        per_image_records[name] = [
            {"path": m["path"], "label": int(m["label"]), "source": m["source"],
             "pred_prob": float(p), "routed_group": str(g)}
            for m, p, g in zip(manifest, probs, groups)
        ]

    table_path = os.path.join(RESULTS_DIR, "robustness_table.json")
    with open(table_path, "w") as f:
        json.dump(rows, f, indent=2)

    per_image_path = os.path.join(RESULTS_DIR, "robustness_per_image.json")
    with open(per_image_path, "w") as f:
        json.dump(per_image_records, f, indent=2)

    names = [r["transform"] for r in rows]
    aucs = [r["auroc"] for r in rows]
    accs = [r["accuracy"] for r in rows]

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(names))
    ax.bar(x - 0.2, aucs, width=0.4, label="AUROC", color="#1f77b4")
    ax.bar(x + 0.2, accs, width=0.4, label="Accuracy", color="#ff7f0e")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=60, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.set_title("Robustness: clean vs. transformed (held-out test set)")
    ax.legend()
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "robustness_plot.png")
    plt.savefig(plot_path, dpi=120)
    plt.close()

    print("\n" + "=" * 90)
    print("ROBUSTNESS TABLE")
    print("=" * 90)
    header = f"{'transform':<20}{'AUROC':<10}{'acc':<10}{'f1':<10}{'FPR':<10}{'FNR':<10}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['transform']:<20}{r['auroc']:<10.4f}{r['accuracy']:<10.4f}{r['f1']:<10.4f}"
              f"{r['false_positive_rate']:<10.4f}{r['false_negative_rate']:<10.4f}")
    print(f"\nSaved table: {table_path}")
    print(f"Saved per-image predictions: {per_image_path}")
    print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()
