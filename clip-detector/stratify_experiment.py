"""
Follow-up experiment: does knowing an image's (predicted) NATIVE
degradation state - before our own provenance-control JPEG re-encode -
help explain or improve the noise-residual reactivity signal?

Two things happen here that the original main.py experiment didn't do:

1. CONFOUND CHECK: predict each image's native blur/JPEG-quality (as it
   was before we touched it) using a small regressor trained on
   synthetic degradations with known ground truth. Check whether this
   differs systematically between real and AI images - if so, that's a
   confound the original AUROC could partly reflect, independent of
   scene busyness.

2. STRATIFY + COMBINE: split Delta_energy AUROC by native-quality
   tercile (does the residual signal only work on already-clean
   images?), and test whether a small classifier using
   [Delta_energy, Delta_energy_norm, native_blur, native_quality,
   busyness] beats Delta_energy alone under cross-validation.

Usage:
    python stratify_experiment.py --profile fullres   (default)
    python stratify_experiment.py --profile cifake
"""

import argparse
import json
import os
import random
import time

import numpy as np
from scipy.stats import mannwhitneyu, pearsonr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from degrade_model import build_training_data, extract_iqa_features, train_degradation_models
from main import PROFILES, PROVENANCE_Q, set_seeds, SEED
from pipeline import (
    load_resize_crop,
    jpeg_reencode,
    to_float,
    compute_deltas,
    busyness,
    DEFAULT_DENOISER,
    DEFAULT_PROBE,
)

N_TRAIN_IMAGES = 150
N_SYNTH_PER_IMAGE = 4


def effective_auroc(y_true, scores):
    raw = float(roc_auc_score(y_true, scores))
    return raw if raw >= 0.5 else 1.0 - raw


def main(profile_name="fullres"):
    profile = PROFILES[profile_name]
    real_dir = os.path.join(profile["data_dir"], "real")
    ai_dir = os.path.join(profile["data_dir"], "ai")
    results_dir = profile["results_dir"]
    set_seeds()

    print("=" * 80)
    print(f"DEGRADATION-AWARE STRATIFICATION EXPERIMENT  [profile: {profile_name}]")
    print("=" * 80)

    real_files = sorted(
        os.path.join(real_dir, f) for f in os.listdir(real_dir) if f.lower().endswith(".png")
    )
    ai_files = sorted(
        os.path.join(ai_dir, f) for f in os.listdir(ai_dir) if f.lower().endswith(".png")
    )
    all_paths = [(p, 0) for p in real_files] + [(p, 1) for p in ai_files]
    print(f"Loaded {len(all_paths)} images (n_real={len(real_files)}, n_ai={len(ai_files)})")

    # -----------------------------------------------------------------
    # Step 1: load base images - resize/crop only, BEFORE our own
    # provenance-control JPEG re-encode. This is each image's "native"
    # state as it arrived from the dataset.
    # -----------------------------------------------------------------
    print("\nLoading base (pre-provenance) grayscale images...")
    t0 = time.time()
    base_imgs = {}
    for path, _ in tqdm(all_paths, desc="load"):
        base_imgs[path] = to_float(load_resize_crop(path))
    print(f"  done in {time.time() - t0:.1f}s")

    # -----------------------------------------------------------------
    # Step 2: train the degradation regressor on synthetic degradations
    # of a random subset of base images (content-agnostic: mixes real
    # and AI images, predicts degradation params, never the class label).
    # -----------------------------------------------------------------
    rng = random.Random(SEED)
    train_paths = rng.sample(list(base_imgs.keys()), min(N_TRAIN_IMAGES, len(base_imgs)))
    print(f"\nTraining degradation regressor on {len(train_paths)} images x "
          f"{N_SYNTH_PER_IMAGE} synthetic samples each...")
    t0 = time.time()
    X, y_blur, y_qual = build_training_data(
        [base_imgs[p] for p in train_paths], n_per_image=N_SYNTH_PER_IMAGE
    )
    blur_model, qual_model, reg_metrics = train_degradation_models(X, y_blur, y_qual)
    print(f"  done in {time.time() - t0:.1f}s")
    print(f"  Regressor validation (held-out synthetic samples):")
    print(f"    blur_sigma:    R^2={reg_metrics['blur_r2']:.3f}  MAE={reg_metrics['blur_mae']:.3f}")
    print(f"    jpeg_quality:  R^2={reg_metrics['quality_r2']:.3f}  MAE={reg_metrics['quality_mae']:.2f}")

    # -----------------------------------------------------------------
    # Step 3: predict NATIVE degradation for every image.
    # -----------------------------------------------------------------
    print("\nPredicting native (pre-provenance) degradation for all images...")
    native_blur = {}
    native_qual = {}
    for path in tqdm(base_imgs, desc="predict-native"):
        feat = extract_iqa_features(base_imgs[path]).reshape(1, -1)
        native_blur[path] = float(blur_model.predict(feat)[0])
        native_qual[path] = float(qual_model.predict(feat)[0])

    # -----------------------------------------------------------------
    # Step 4: run the standard Test-A pipeline (provenance-controlled,
    # q95) to get Delta_energy / Delta_energy_norm per image.
    # -----------------------------------------------------------------
    print(f"\nRunning Test-A residual pipeline (JPEG q{PROVENANCE_Q} provenance control)...")
    t0 = time.time()
    records = []
    for path, label in tqdm(all_paths, desc="pipeline"):
        gray_q95 = jpeg_reencode(load_resize_crop(path), PROVENANCE_Q)
        img_q95 = to_float(gray_q95)
        res = compute_deltas(img_q95, denoiser=DEFAULT_DENOISER, probe=DEFAULT_PROBE)
        vals = [res["delta_energy"], res["delta_energy_norm"]]
        if not all(np.isfinite(v) for v in vals):
            print(f"  WARNING: non-finite result on {path}, skipping.")
            continue
        records.append({
            "path": path,
            "label": label,
            "delta_energy": res["delta_energy"],
            "delta_energy_norm": res["delta_energy_norm"],
            "busyness": busyness(img_q95),
            "native_blur": native_blur[path],
            "native_quality": native_qual[path],
        })
    print(f"  done in {time.time() - t0:.1f}s ({len(records)} images)")

    y = np.array([r["label"] for r in records])
    delta_energy = np.array([r["delta_energy"] for r in records])
    delta_energy_norm = np.array([r["delta_energy_norm"] for r in records])
    native_q = np.array([r["native_quality"] for r in records])
    native_b = np.array([r["native_blur"] for r in records])
    busy = np.array([r["busyness"] for r in records])

    n_real = int((y == 0).sum())
    n_ai = int((y == 1).sum())

    # -----------------------------------------------------------------
    # Step 5: confound check - does native predicted degradation differ
    # by class?
    # -----------------------------------------------------------------
    print("\n" + "-" * 80)
    print("CONFOUND CHECK: native (pre-provenance) predicted degradation, by class")
    print("-" * 80)
    q_real, q_ai = native_q[y == 0], native_q[y == 1]
    b_real, b_ai = native_b[y == 0], native_b[y == 1]
    u_q, p_q = mannwhitneyu(q_real, q_ai)
    u_b, p_b = mannwhitneyu(b_real, b_ai)
    print(f"  Native predicted JPEG quality: real mean={q_real.mean():.1f} "
          f"(median={np.median(q_real):.1f}), ai mean={q_ai.mean():.1f} "
          f"(median={np.median(q_ai):.1f})  Mann-Whitney p={p_q:.2e}")
    print(f"  Native predicted blur sigma:   real mean={b_real.mean():.3f} "
          f"(median={np.median(b_real):.3f}), ai mean={b_ai.mean():.3f} "
          f"(median={np.median(b_ai):.3f})  Mann-Whitney p={p_b:.2e}")
    quality_confound = bool(p_q < 0.01)
    blur_confound = bool(p_b < 0.01)
    if quality_confound or blur_confound:
        print("  -> FLAGGED: native degradation state differs significantly by class. "
              "Part of any Delta_energy separation could reflect 'how heavily was this "
              "image already compressed/blurred before we touched it' rather than "
              "real-vs-AI reactivity.")
    else:
        print("  -> OK: no strong native-degradation confound detected by class.")

    auroc_native_quality = effective_auroc(y, native_q)
    auroc_native_blur = effective_auroc(y, native_b)
    print(f"  (native_quality alone as a real/ai classifier: eff. AUROC = {auroc_native_quality:.4f})")
    print(f"  (native_blur alone as a real/ai classifier:    eff. AUROC = {auroc_native_blur:.4f})")

    # -----------------------------------------------------------------
    # Step 6: stratify Delta_energy AUROC by native-quality tercile.
    # -----------------------------------------------------------------
    print("\n" + "-" * 80)
    print("STRATIFIED AUROC: Delta_energy by native-quality tercile")
    print("-" * 80)
    terciles = np.quantile(native_q, [1 / 3, 2 / 3])
    bins = np.digitize(native_q, terciles)  # 0=most-degraded-looking, 2=cleanest-looking
    strat_table = []
    for b in [0, 1, 2]:
        mask = bins == b
        n_bin = int(mask.sum())
        if n_bin < 20 or len(set(y[mask])) < 2:
            print(f"  bin {b}: n={n_bin} - too small/one-sided, skipped")
            continue
        auc = effective_auroc(y[mask], delta_energy[mask])
        strat_table.append({
            "bin": int(b),
            "n": n_bin,
            "native_quality_range": [float(native_q[mask].min()), float(native_q[mask].max())],
            "delta_energy_auroc_effective": auc,
        })
        print(f"  bin {b} (native quality {native_q[mask].min():.0f}-{native_q[mask].max():.0f}, "
              f"n={n_bin}): Delta_energy eff. AUROC = {auc:.4f}")

    # -----------------------------------------------------------------
    # Step 7: combined classifier (degradation-aware features) vs
    # Delta_energy alone, cross-validated.
    # -----------------------------------------------------------------
    print("\n" + "-" * 80)
    print("COMBINED CLASSIFIER: Delta_energy alone vs. + degradation features")
    print("-" * 80)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    def cv_effective_auroc(X_feat, y_labels):
        pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        scores = cross_val_score(pipe, X_feat, y_labels, cv=skf, scoring="roc_auc")
        return float(np.mean(np.maximum(scores, 1 - scores))), [float(s) for s in scores]

    baseline_auc, baseline_folds = cv_effective_auroc(delta_energy.reshape(-1, 1), y)
    combined_X = np.column_stack([delta_energy, delta_energy_norm, native_q, native_b, busy])
    combined_auc, combined_folds = cv_effective_auroc(combined_X, y)

    # Ablations, to tell apart "degradation-awareness rescues the residual
    # signal" from "native_quality/busyness are themselves a shortcut that
    # has nothing to do with residual reactivity" (the same kind of
    # confound the leakage check caught earlier, one level up).
    reactivity_only_X = np.column_stack([delta_energy, delta_energy_norm])
    reactivity_only_auc, _ = cv_effective_auroc(reactivity_only_X, y)
    shortcut_only_X = np.column_stack([native_q, native_b, busy])
    shortcut_only_auc, _ = cv_effective_auroc(shortcut_only_X, y)

    print(f"  Baseline (Delta_energy only):                         CV eff. AUROC = {baseline_auc:.4f}")
    print(f"  Reactivity only (Delta_energy + Delta_energy_norm):   CV eff. AUROC = {reactivity_only_auc:.4f}")
    print(f"  Shortcut only (native_quality + native_blur")
    print(f"                 + busyness, NO residual stats at all): CV eff. AUROC = {shortcut_only_auc:.4f}")
    print(f"  Combined (all five features):                         CV eff. AUROC = {combined_auc:.4f}")
    improved = combined_auc > baseline_auc + 0.01
    shortcut_dominates = shortcut_only_auc >= combined_auc - 0.02
    if improved and shortcut_dominates:
        print(f"  -> Combined score IS higher (+{combined_auc - baseline_auc:.4f}), but the "
              f"shortcut-only score ({shortcut_only_auc:.4f}) nearly matches it on its own - "
              f"most of the gain is native-quality/busyness acting as a class shortcut for "
              f"THIS dataset, not degradation-awareness rescuing genuine residual reactivity.")
    elif improved:
        print(f"  -> Degradation-awareness IMPROVES the classifier "
              f"(+{combined_auc - baseline_auc:.4f} AUROC) beyond what the shortcut features "
              f"alone explain ({shortcut_only_auc:.4f}).")
    else:
        print(f"  -> Degradation-awareness does NOT meaningfully improve the classifier "
              f"({combined_auc - baseline_auc:+.4f} AUROC).")

    # -----------------------------------------------------------------
    # Plots
    # -----------------------------------------------------------------
    os.makedirs(results_dir, exist_ok=True)

    plt.figure(figsize=(6, 5))
    plt.scatter(native_q[y == 0], delta_energy[y == 0], alpha=0.4, s=10, label="real", color="#1f77b4")
    plt.scatter(native_q[y == 1], delta_energy[y == 1], alpha=0.4, s=10, label="ai", color="#d62728")
    plt.yscale("log")
    plt.xlabel("native predicted JPEG quality")
    plt.ylabel("Delta_energy (log scale)")
    plt.title(f"Delta_energy vs. native predicted quality [{profile_name}]")
    plt.legend()
    plt.tight_layout()
    scatter_path = os.path.join(results_dir, "stratify_scatter_quality_vs_delta.png")
    plt.savefig(scatter_path, dpi=120)
    plt.close()

    plt.figure(figsize=(6, 4))
    bins_plot = 30
    plt.hist(q_real, bins=bins_plot, alpha=0.6, label="real", color="#1f77b4", density=True)
    plt.hist(q_ai, bins=bins_plot, alpha=0.6, label="ai", color="#d62728", density=True)
    plt.xlabel("native predicted JPEG quality")
    plt.ylabel("density")
    plt.title(f"Native predicted quality by class [{profile_name}]")
    plt.legend()
    plt.tight_layout()
    hist_path = os.path.join(results_dir, "stratify_hist_native_quality.png")
    plt.savefig(hist_path, dpi=120)
    plt.close()

    plt.figure(figsize=(6, 4))
    if strat_table:
        xs = [f"bin {r['bin']}\n({r['native_quality_range'][0]:.0f}-{r['native_quality_range'][1]:.0f})"
              for r in strat_table]
        ys = [r["delta_energy_auroc_effective"] for r in strat_table]
        plt.bar(xs, ys, color="#2ca02c")
        plt.axhline(0.65, color="gray", linestyle="--", linewidth=1, label="0.65 threshold")
    plt.ylabel("Delta_energy effective AUROC")
    plt.title(f"Delta_energy AUROC by native-quality tercile [{profile_name}]")
    plt.legend()
    plt.tight_layout()
    bar_path = os.path.join(results_dir, "stratify_bar_auroc_by_bin.png")
    plt.savefig(bar_path, dpi=120)
    plt.close()

    # -----------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------
    out = {
        "profile": profile_name,
        "n_real": n_real,
        "n_ai": n_ai,
        "degradation_regressor_validation": reg_metrics,
        "confound_check": {
            "native_quality_real_mean": float(q_real.mean()),
            "native_quality_ai_mean": float(q_ai.mean()),
            "native_quality_mannwhitney_p": float(p_q),
            "native_blur_real_mean": float(b_real.mean()),
            "native_blur_ai_mean": float(b_ai.mean()),
            "native_blur_mannwhitney_p": float(p_b),
            "quality_confound_flagged": quality_confound,
            "blur_confound_flagged": blur_confound,
            "native_quality_alone_auroc_effective": auroc_native_quality,
            "native_blur_alone_auroc_effective": auroc_native_blur,
        },
        "stratified_auroc_by_native_quality_tercile": strat_table,
        "combined_classifier": {
            "baseline_delta_energy_only_cv_auroc": baseline_auc,
            "baseline_folds": baseline_folds,
            "reactivity_only_cv_auroc": reactivity_only_auc,
            "shortcut_only_cv_auroc": shortcut_only_auc,
            "combined_with_degradation_features_cv_auroc": combined_auc,
            "combined_folds": combined_folds,
            "improved": improved,
            "shortcut_dominates": shortcut_dominates,
        },
        "plots": {
            "scatter_quality_vs_delta": scatter_path,
            "hist_native_quality_by_class": hist_path,
            "bar_auroc_by_bin": bar_path,
        },
    }
    out_path = os.path.join(results_dir, "stratify_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Confound: native quality differs by class: {quality_confound} (p={p_q:.2e})")
    print(f"Confound: native blur differs by class:    {blur_confound} (p={p_b:.2e})")
    print(f"Delta_energy alone -> combined:  {baseline_auc:.4f} -> {combined_auc:.4f}  (improved={improved})")
    print(f"Shortcut-only (no residual stats at all): {shortcut_only_auc:.4f}"
          f"  (shortcut_dominates={shortcut_dominates})")
    print(f"Results saved to: {out_path}")
    for p in [scatter_path, hist_path, bar_path]:
        print(f"Plot saved to: {p}")
    print("=" * 80)

    # Log to the shared cross-experiment leaderboard (deferred import to
    # avoid a circular import - harness.py imports PROFILES etc. from
    # main.py, not from this module, so this is safe).
    try:
        from harness import append_leaderboard, BASE_DIR as HARNESS_BASE_DIR
        append_leaderboard([{
            "signal": "degradation_features", "feature": "combined_shortcut_aware",
            "profile": profile_name, "n_real": n_real, "n_ai": n_ai,
            "raw_auroc_testA": None, "effective_auroc_testA": combined_auc,
            "raw_auroc_testB": None, "effective_auroc_testB": None,
            "leakage_corr_vs_busyness": None, "leakage_flagged": bool(quality_confound or blur_confound),
            "cv_auroc_feature_only": reactivity_only_auc, "cv_auroc_shortcut_only": shortcut_only_auc,
            "cv_auroc_combined": combined_auc, "shortcut_dominates": shortcut_dominates,
            "hist_path": hist_path, "scores_path": out_path,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }])
        print(f"Logged 1 entry to shared leaderboard ({os.path.join(HARNESS_BASE_DIR, 'results', 'leaderboard.json')})")
    except Exception as e:
        print(f"  (leaderboard logging skipped: {e})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(PROFILES.keys()), default="fullres")
    args = parser.parse_args()
    main(profile_name=args.profile)
