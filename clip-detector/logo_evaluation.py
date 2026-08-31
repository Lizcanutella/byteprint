"""
True leave-one-generator-out (LOGO) evaluation, restricted to the
Defactify-labeled portion of the data (the only source with per-
generator subclass labels - fullres/sid_set's AI images don't carry
generator identity).

Note on what this can and can't be: a fully faithful LOGO would retrain
the entire multi-domain specialist architecture 5 times (once per
held-out generator), which is too expensive to redo here. Instead, this
trains a single classifier per fold using production's EXACT feature
extraction (production_pipeline._extended_features: CLIP pre-projection
embedding + jpeg_q50 reactivity-delta, the same 1536-dim feature the
real 'clean' domain specialist uses) on the held-out generator-
diagnostic set (360 images, 60/generator + 60 real, all disjoint from
training) - a true K-fold LOGO on production's actual feature
representation, just not on the full routed architecture.

Usage:
    python logo_evaluation.py
"""

import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from clip_features import load_image_capped
from production_pipeline import _extended_features

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data_generator_diagnostic")
GEN_CLASSES = ["real", "sd21", "sdxl", "sd3", "dalle3", "midjourney6"]
AI_CLASSES = ["sd21", "sdxl", "sd3", "dalle3", "midjourney6"]
SEED = 1234


def main():
    records = []
    for cls in GEN_CLASSES:
        d = os.path.join(DATA_DIR, cls)
        for f in sorted(os.listdir(d)):
            records.append({"path": os.path.join(d, f), "cls": cls, "label": 0 if cls == "real" else 1})
    print(f"Loaded {len(records)} images across {GEN_CLASSES} (all disjoint from training)")

    imgs = [load_image_capped(r["path"]) for r in records]
    print("Extracting production's exact feature (CLIP preproj embedding + jpeg_q50 delta)...")
    feats = _extended_features(imgs, batch_size=32)
    print(f"  feature shape: {feats.shape}")

    y = np.array([r["label"] for r in records])
    cls_arr = np.array([r["cls"] for r in records])

    rng = np.random.RandomState(SEED)
    real_idx = np.where(cls_arr == "real")[0]
    real_fold = np.full(len(records), -1)
    shuffled_real = rng.permutation(real_idx)
    for i, idx in enumerate(shuffled_real):
        real_fold[idx] = i % len(AI_CLASSES)

    fold_groups = np.empty(len(records), dtype=object)
    for i, cls in enumerate(cls_arr):
        fold_groups[i] = f"fold{real_fold[i]}" if cls == "real" else f"fold{AI_CLASSES.index(cls)}"

    logo = LeaveOneGroupOut()
    per_fold_auc = {}
    for train_mask_idx, test_mask_idx in logo.split(feats, y, fold_groups):
        fold_name = fold_groups[test_mask_idx[0]]  # e.g. "fold2" -> AI_CLASSES[2]
        held_out_gen = AI_CLASSES[int(fold_name.replace("fold", ""))]
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
        clf.fit(feats[train_mask_idx], y[train_mask_idx])
        preds = clf.predict_proba(feats[test_mask_idx])[:, 1]
        auc = float(roc_auc_score(y[test_mask_idx], preds))
        per_fold_auc[held_out_gen] = auc
        print(f"  held-out generator = {held_out_gen:<12} AUROC = {auc:.4f}  (n={len(test_mask_idx)})")

    logo_mean = float(np.mean(list(per_fold_auc.values())))
    print(f"\nLOGO mean AUROC (production feature, true leave-one-generator-out): {logo_mean:.4f}")

    result = {
        "per_generator_auroc": per_fold_auc,
        "logo_mean_auroc": logo_mean,
        "feature": "production_pipeline._extended_features (CLIP preproj + jpeg_q50 delta), 1536-dim",
        "note": "single LogisticRegression per fold, not the full routed multi-domain specialist "
                "architecture - a full LOGO retrain of that architecture (5x) was too expensive to run "
                "in the remaining time. This uses production's exact feature representation on a true "
                "K-fold leave-one-generator-out split.",
    }
    with open("results_detector/logo_evaluation.json", "w") as f:
        json.dump(result, f, indent=2)
    print("Saved to results_detector/logo_evaluation.json")


if __name__ == "__main__":
    main()
