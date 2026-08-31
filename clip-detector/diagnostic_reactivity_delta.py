"""
Diagnostic (NOT part of the production pipeline): tests whether "how much
a probe transform shifts an image's CLIP embedding" (reactivity) carries
any real/AI signal beyond what the absolute embedding already gives.

This is a CLIP-embedding-space variant of the classical-pixel-statistic
"noise-residual reactivity" signals from Part 2 of this project, which
were tried and rigorously falsified (best ~0.62 AUROC). Motivation for
re-testing in CLIP space: CLIP's embedding might capture reactivity
differences the hand-crafted pixel features missed. This script only
measures whether the idea has any legs at all - it does NOT touch
model/ or any production file.

Uses the held-out test split (443 images, never trained on, from
train_classifier.compute_split()) so results are clean of train/test
leakage. For each image: embed the clean version, embed 3 probed
versions (one representative probe per domain family: blur_s1.0,
noise_s0.05, jpeg_q50), compute delta = emb_probed - emb_clean, and
evaluate delta's predictive power for real-vs-AI three ways:
  (a) delta_norm alone (univariate AUROC)
  (b) full delta vector, 5-fold stratified CV logistic regression
  (c) full delta vector, leave-one-source-out CV (generalization check)
  (d) delta concatenated with the absolute clean embedding, vs. the
      absolute embedding alone - does reactivity add anything on top?

Usage:
    python diagnostic_reactivity_delta.py
"""

import json
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from clip_features import embed_images_preproj, load_image_capped
from train_classifier import compute_split
from transforms import gaussian_blur, gaussian_noise, jpeg_compress

PROBES = {
    "blur_s1.0": lambda img: gaussian_blur(img, 1.0),
    "noise_s0.05": lambda img: gaussian_noise(img, 0.05),
    "jpeg_q50": lambda img: jpeg_compress(img, 50),
}


def cv_auroc(X, y, groups=None, n_splits=5):
    if groups is None:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1234)
        splitter = cv.split(X, y)
    else:
        from sklearn.model_selection import LeaveOneGroupOut
        splitter = LeaveOneGroupOut().split(X, y, groups)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    preds = cross_val_predict(clf, X, y, cv=list(splitter), method="predict_proba")[:, 1]
    return float(roc_auc_score(y, preds))


def main():
    paths, labels, sources, train_idx, test_idx = compute_split()
    labels = np.array(labels)
    sources = np.array(sources)
    y = labels[test_idx]
    src = sources[test_idx]
    print(f"Held-out diagnostic set: {len(test_idx)} images, sources={sorted(set(src))}")

    print("Loading clean images...")
    t0 = time.time()
    clean_imgs = [load_image_capped(paths[i]) for i in test_idx]
    print(f"  {time.time() - t0:.1f}s")

    print("Embedding clean images (pre-projection CLIP)...")
    t0 = time.time()
    emb_clean = embed_images_preproj(clean_imgs, batch_size=32)
    print(f"  {time.time() - t0:.1f}s")

    results = {}
    results["baseline_absolute_embedding_only"] = {
        "stratified_5fold_auroc": cv_auroc(emb_clean, y),
        "leave_one_source_out_auroc": cv_auroc(emb_clean, y, groups=src),
    }
    print(f"Baseline (absolute embedding only): {results['baseline_absolute_embedding_only']}")

    for probe_name, probe_fn in PROBES.items():
        print(f"\n=== Probe: {probe_name} ===")
        t0 = time.time()
        probed_imgs = [probe_fn(img) for img in clean_imgs]
        print(f"  applied probe in {time.time() - t0:.1f}s")

        t0 = time.time()
        emb_probed = embed_images_preproj(probed_imgs, batch_size=32)
        print(f"  embedded in {time.time() - t0:.1f}s")

        delta = emb_probed - emb_clean
        delta_norm = np.linalg.norm(delta, axis=1)

        auroc_norm = float(roc_auc_score(y, delta_norm))
        auroc_delta_5fold = cv_auroc(delta, y)
        auroc_delta_loso = cv_auroc(delta, y, groups=src)

        combined = np.concatenate([emb_clean, delta], axis=1)
        auroc_combined_5fold = cv_auroc(combined, y)
        auroc_combined_loso = cv_auroc(combined, y, groups=src)

        results[probe_name] = {
            "delta_norm_univariate_auroc": auroc_norm,
            "delta_vector_stratified_5fold_auroc": auroc_delta_5fold,
            "delta_vector_leave_one_source_out_auroc": auroc_delta_loso,
            "combined_absolute_plus_delta_stratified_5fold_auroc": auroc_combined_5fold,
            "combined_absolute_plus_delta_leave_one_source_out_auroc": auroc_combined_loso,
        }
        print(json.dumps(results[probe_name], indent=2))

    out_path = "results_detector/reactivity_delta_diagnostic.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    base_5fold = results["baseline_absolute_embedding_only"]["stratified_5fold_auroc"]
    base_loso = results["baseline_absolute_embedding_only"]["leave_one_source_out_auroc"]
    print(f"Absolute embedding alone: 5fold={base_5fold:.4f} LOSO={base_loso:.4f}")
    for probe_name in PROBES:
        r = results[probe_name]
        print(f"{probe_name}: delta_only 5fold={r['delta_vector_stratified_5fold_auroc']:.4f} "
              f"LOSO={r['delta_vector_leave_one_source_out_auroc']:.4f} | "
              f"combined 5fold={r['combined_absolute_plus_delta_stratified_5fold_auroc']:.4f} "
              f"LOSO={r['combined_absolute_plus_delta_leave_one_source_out_auroc']:.4f}")


if __name__ == "__main__":
    main()
