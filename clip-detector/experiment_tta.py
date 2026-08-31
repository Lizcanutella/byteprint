"""
Experiment: test-time augmentation (TTA) with the EXISTING trained
model (model/classifier_head.pkl, post-projection CLIP features - no
retraining). For each test image (under each transform-grid condition),
average the predicted probability over a few non-destructive "views" of
that same image - identity, horizontal flip, and a very mild 95%
center-crop - rather than a single embedding. Non-destructive views are
used deliberately: further degrading an already-degraded image (e.g.
also blurring a blur_s2.0 image) would confound "does TTA help" with
"does more degradation hurt", which isn't the question here.

Compares against results_detector/robustness_table.json (no-TTA
baseline) per transform-grid cell.

Usage:
    python experiment_tta.py
"""

import json
import os
import pickle
import time

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from clip_features import embed_images, load_image_capped
from transforms import TRANSFORM_GRID, center_crop, hflip

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
RESULTS_DIR = os.path.join(BASE_DIR, "results_detector")
EXPERIMENT_DIR = os.path.join(RESULTS_DIR, "experiments")

TTA_VIEWS = [
    ("identity", lambda img: img),
    ("hflip", hflip),
    ("crop95", lambda img: center_crop(img, 0.95)),
]


def load_model():
    with open(os.path.join(MODEL_DIR, "classifier_head.pkl"), "rb") as f:
        return pickle.load(f)


def load_test_manifest():
    with open(os.path.join(MODEL_DIR, "test_manifest.json")) as f:
        return json.load(f)


def load_baseline_table():
    with open(os.path.join(RESULTS_DIR, "robustness_table.json")) as f:
        rows = json.load(f)
    return {r["transform"]: r for r in rows}


def main():
    os.makedirs(EXPERIMENT_DIR, exist_ok=True)
    clf = load_model()
    manifest = load_test_manifest()
    baseline = load_baseline_table()
    print(f"Held-out test set: {len(manifest)} images, TTA views: {[n for n, _ in TTA_VIEWS]}")

    y = np.array([m["label"] for m in manifest])

    rows = []
    for name, fn in TRANSFORM_GRID:
        print(f"[{name}] applying transform, {len(TTA_VIEWS)} TTA views + predicting...")
        t0 = time.time()
        base_imgs = [fn(load_image_capped(m["path"])) for m in manifest]

        view_probs = []
        for view_name, view_fn in TTA_VIEWS:
            view_imgs = [view_fn(img) for img in base_imgs]
            embs = embed_images(view_imgs, batch_size=32)
            probs = clf.predict_proba(embs)[:, 1]
            view_probs.append(probs)

        tta_probs = np.mean(view_probs, axis=0)
        preds = (tta_probs >= 0.5).astype(int)

        auc = float(roc_auc_score(y, tta_probs))
        acc = float(accuracy_score(y, preds))
        f1 = float(f1_score(y, preds))
        n_real, n_ai = int((y == 0).sum()), int((y == 1).sum())
        fpr = int(((y == 0) & (preds == 1)).sum()) / max(n_real, 1)
        fnr = int(((y == 1) & (preds == 0)).sum()) / max(n_ai, 1)

        base_row = baseline.get(name, {})
        base_auc = base_row.get("auroc")
        delta = (auc - base_auc) if base_auc is not None else None

        print(f"  TTA AUROC={auc:.4f} acc={acc:.4f}  (no-TTA baseline={base_auc}, "
              f"delta={delta:+.4f})" if delta is not None else
              f"  TTA AUROC={auc:.4f} acc={acc:.4f}  ({time.time()-t0:.1f}s)")

        rows.append({
            "transform": name, "tta_auroc": auc, "tta_accuracy": acc, "tta_f1": f1,
            "tta_false_positive_rate": fpr, "tta_false_negative_rate": fnr,
            "no_tta_auroc": base_auc, "delta_auroc": delta, "n": len(manifest),
        })

    out_path = os.path.join(EXPERIMENT_DIR, "tta_result.json")
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)

    print("\n" + "=" * 90)
    print("TTA vs NO-TTA")
    print("=" * 90)
    header = f"{'transform':<20}{'no_tta_auroc':<14}{'tta_auroc':<12}{'delta':<10}{'tta_acc':<10}"
    print(header)
    print("-" * len(header))
    mean_delta = []
    for r in rows:
        d = r["delta_auroc"]
        if d is not None:
            mean_delta.append(d)
        print(f"{r['transform']:<20}{(r['no_tta_auroc'] or 0):<14.4f}{r['tta_auroc']:<12.4f}"
              f"{(d or 0):<+10.4f}{r['tta_accuracy']:<10.4f}")
    if mean_delta:
        print(f"\nMean AUROC delta across all cells: {np.mean(mean_delta):+.4f}")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
