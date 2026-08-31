"""
Follow-up to diagnostic_reactivity_delta.py (which found that a JPEG-q50
probe's CLIP-embedding shift carries real/AI signal beyond the absolute
embedding, under leave-one-source-out CV: 0.92 vs 0.72 baseline). Before
deciding whether to integrate this into production, two harder questions
that first diagnostic did NOT answer:

1. Does it generalize to entirely UNSEEN GENERATORS, not just unseen
   *sources* within the training-side datasets? (data_generator_diagnostic/
   - Defactify TEST split, SD2.1/SDXL/SD3/DALL-E3/Midjourney6, the same
   gold-standard cross-generator check used throughout this project.)
   Evaluated via leave-one-generator-out CV.

2. Does the signal survive when the input is already degraded by some
   OTHER, unknown transform - the realistic production scenario - rather
   than starting from a clean image? Simulated by applying the probe
   (jpeg_q50) on top of an already-blurred (blur_s1.0) version of each
   held-out test image, then checking if the resulting delta still
   separates real/AI via the same leave-one-source-out CV as before.

Does NOT touch model/ or any production file - purely diagnostic.

Usage:
    python diagnostic_reactivity_comprehensive.py
"""

import json
import os
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from clip_features import embed_images_preproj, load_image_capped
from train_classifier import compute_split
from transforms import gaussian_blur, jpeg_compress

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GEN_DIAG_DIR = os.path.join(BASE_DIR, "data_generator_diagnostic")
GEN_CLASSES = ["real", "sd21", "sdxl", "sd3", "dalle3", "midjourney6"]
AI_CLASSES = ["sd21", "sdxl", "sd3", "dalle3", "midjourney6"]

PROBE_NAME = "jpeg_q50"
PROBE_FN = lambda img: jpeg_compress(img, 50)
BASE_TRANSFORM_NAME = "blur_s1.0"
BASE_TRANSFORM_FN = lambda img: gaussian_blur(img, 1.0)


def cv_auroc_stratified(X, y, n_splits=5):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1234)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    preds = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, preds))


def cv_auroc_groups(X, y, groups, group_values=None):
    """Leave-one-group-out CV. If group_values given, only iterate those
    (each held-out fold = that group's rows + non-grouped rows already
    folded in via `groups` array)."""
    from sklearn.model_selection import LeaveOneGroupOut
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    splits = list(LeaveOneGroupOut().split(X, y, groups))
    preds = cross_val_predict(clf, X, y, cv=splits, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, preds))


def test1_cross_generator():
    print("=" * 70)
    print("TEST 1: cross-generator generalization (leave-one-generator-out)")
    print("=" * 70)
    records = []
    for cls in GEN_CLASSES:
        d = os.path.join(GEN_DIAG_DIR, cls)
        for f in sorted(os.listdir(d)):
            records.append({"path": os.path.join(d, f), "cls": cls, "label": 0 if cls == "real" else 1})
    print(f"Loaded {len(records)} images across {GEN_CLASSES}")

    imgs = [load_image_capped(r["path"]) for r in records]
    print("Embedding clean...")
    t0 = time.time()
    emb_clean = embed_images_preproj(imgs, batch_size=32)
    print(f"  {time.time() - t0:.1f}s")

    print(f"Applying probe {PROBE_NAME} and embedding...")
    t0 = time.time()
    probed_imgs = [PROBE_FN(img) for img in imgs]
    emb_probed = embed_images_preproj(probed_imgs, batch_size=32)
    print(f"  {time.time() - t0:.1f}s")

    delta = emb_probed - emb_clean
    y = np.array([r["label"] for r in records])
    cls_arr = np.array([r["cls"] for r in records])

    # assign real images a rotating fold id 0..4 (aligned with AI_CLASSES
    # index) so each leave-one-generator-out fold has some held-out real
    # images too, not just the held-out generator's AI images
    rng = np.random.RandomState(1234)
    real_idx = np.where(cls_arr == "real")[0]
    real_fold = np.full(len(records), -1)
    shuffled_real = rng.permutation(real_idx)
    for i, idx in enumerate(shuffled_real):
        real_fold[idx] = i % len(AI_CLASSES)

    groups = np.empty(len(records), dtype=object)
    for i, cls in enumerate(cls_arr):
        if cls == "real":
            groups[i] = f"fold{real_fold[i]}"
        else:
            groups[i] = cls
    # map each AI class's group name to the same fold label as intended
    # (fold index = position in AI_CLASSES) so LeaveOneGroupOut pulls out
    # exactly {that generator} + {that fold's real images} together
    fold_groups = np.empty(len(records), dtype=object)
    for i, cls in enumerate(cls_arr):
        if cls == "real":
            fold_groups[i] = f"fold{real_fold[i]}"
        else:
            fold_groups[i] = f"fold{AI_CLASSES.index(cls)}"

    results = {
        "absolute_embedding_only": cv_auroc_groups(emb_clean, y, fold_groups),
        "delta_only": cv_auroc_groups(delta, y, fold_groups),
        "combined": cv_auroc_groups(np.concatenate([emb_clean, delta], axis=1), y, fold_groups),
    }
    print(json.dumps(results, indent=2))
    return results


def test2_already_degraded():
    print("=" * 70)
    print(f"TEST 2: does the {PROBE_NAME} delta survive on already-{BASE_TRANSFORM_NAME}-degraded inputs?")
    print("=" * 70)
    paths, labels, sources, train_idx, test_idx = compute_split()
    labels = np.array(labels)
    sources = np.array(sources)
    y = labels[test_idx]
    src = sources[test_idx]

    print("Loading + applying base transform (simulated unknown degradation)...")
    t0 = time.time()
    clean_imgs = [load_image_capped(paths[i]) for i in test_idx]
    base_imgs = [BASE_TRANSFORM_FN(img) for img in clean_imgs]
    print(f"  {time.time() - t0:.1f}s")

    print("Embedding already-degraded (base) images...")
    t0 = time.time()
    emb_base = embed_images_preproj(base_imgs, batch_size=32)
    print(f"  {time.time() - t0:.1f}s")

    print(f"Applying {PROBE_NAME} probe on top of the already-degraded images...")
    t0 = time.time()
    probed_imgs = [PROBE_FN(img) for img in base_imgs]
    emb_probed = embed_images_preproj(probed_imgs, batch_size=32)
    print(f"  {time.time() - t0:.1f}s")

    delta = emb_probed - emb_base

    results = {
        "absolute_embedding_of_degraded_input_only": {
            "stratified_5fold": cv_auroc_stratified(emb_base, y),
            "leave_one_source_out": cv_auroc_groups(emb_base, y, src),
        },
        "delta_only": {
            "stratified_5fold": cv_auroc_stratified(delta, y),
            "leave_one_source_out": cv_auroc_groups(delta, y, src),
        },
        "combined": {
            "stratified_5fold": cv_auroc_stratified(np.concatenate([emb_base, delta], axis=1), y),
            "leave_one_source_out": cv_auroc_groups(np.concatenate([emb_base, delta], axis=1), y, src),
        },
    }
    print(json.dumps(results, indent=2))
    return results


def main():
    out = {}
    out["test1_cross_generator_leave_one_out"] = test1_cross_generator()
    out["test2_reactivity_on_already_degraded_input"] = test2_already_degraded()

    out_path = os.path.join(BASE_DIR, "results_detector", "reactivity_delta_comprehensive.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    t1 = out["test1_cross_generator_leave_one_out"]
    print(f"Test 1 (leave-one-generator-out): absolute={t1['absolute_embedding_only']:.4f} "
          f"delta={t1['delta_only']:.4f} combined={t1['combined']:.4f}")
    t2 = out["test2_reactivity_on_already_degraded_input"]
    print(f"Test 2 (reactivity on already-blurred input), LOSO: "
          f"absolute={t2['absolute_embedding_of_degraded_input_only']['leave_one_source_out']:.4f} "
          f"delta={t2['delta_only']['leave_one_source_out']:.4f} "
          f"combined={t2['combined']['leave_one_source_out']:.4f}")


if __name__ == "__main__":
    main()
