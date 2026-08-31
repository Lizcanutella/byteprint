"""
Main experiment: differential noise-residual reactivity as an AI-image
detector signal.

Usage:
    python main.py                    # CIFAKE (32x32-native), default
    python main.py --profile fullres  # full-resolution single-source dataset

Both profiles run the exact same pipeline (pipeline.py) and acquisition
logic (fetch_data.py) against different single-source HF datasets, with
separate data/results directories so runs don't clobber each other.

See README.md for how to swap the denoiser/probe or the data source.
"""

import argparse
import json
import os
import random
import time

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fetch_data import acquire_data, CIFAKE_CANDIDATES, FULLRES_CANDIDATES, FULLRES2_CANDIDATES
from pipeline import (
    load_resize_crop,
    jpeg_reencode,
    to_float,
    compute_deltas,
    busyness,
    DEFAULT_DENOISER,
    DEFAULT_PROBE,
    LOCAL_ACTIVITY_WINDOW,
    LOCAL_ACTIVITY_EPS,
    local_activity_map,
)

SEED = 1234
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROVENANCE_Q = 95
DEGRADE_Q = 50

PROFILES = {
    "cifake": {
        "candidates": CIFAKE_CANDIDATES,
        "data_dir": os.path.join(BASE_DIR, "data"),
        "results_dir": os.path.join(BASE_DIR, "results"),
        "n_per_class": 500,
    },
    "fullres": {
        "candidates": FULLRES_CANDIDATES,
        "data_dir": os.path.join(BASE_DIR, "data_fullres"),
        "results_dir": os.path.join(BASE_DIR, "results_fullres"),
        "n_per_class": 500,
    },
    "fullres2": {
        "candidates": FULLRES2_CANDIDATES,
        "data_dir": os.path.join(BASE_DIR, "data_fullres2"),
        "results_dir": os.path.join(BASE_DIR, "results_fullres2"),
        "n_per_class": 500,
    },
}


def set_seeds(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)


def process_image(path, label, keep_visual=False):
    """Run the full pipeline (provenance control + Test A + Test B) on a
    single image file. `label` is 0=real, 1=ai.
    Returns a dict of per-image results. Full residual/image arrays are
    only retained when `keep_visual=True` (used for a handful of
    leakage-check example images), to keep memory bounded across ~1000
    images.
    """
    # Step 1: load, RGB, resize short side 512, center-crop 512x512, grayscale.
    gray = load_resize_crop(path)

    # Provenance control: re-encode JPEG q95 + reload, applied identically
    # to both classes so file-format/compression history cannot leak.
    gray_q95 = jpeg_reencode(gray, PROVENANCE_Q)
    img_q95 = to_float(gray_q95)

    # --- Test A: clean (q95-normalized) ---
    resA = compute_deltas(img_q95, denoiser=DEFAULT_DENOISER, probe=DEFAULT_PROBE)
    busy = busyness(img_q95)

    # --- Test B: pre-degraded (simulate internet laundering: q95 -> q50) ---
    gray_q50 = jpeg_reencode(gray_q95, DEGRADE_Q)
    img_q50 = to_float(gray_q50)
    resB = compute_deltas(img_q50, denoiser=DEFAULT_DENOISER, probe=DEFAULT_PROBE)

    record = {
        "path": path,
        "label": label,
        "busyness": busy,
        "A_delta_energy": resA["delta_energy"],
        "A_delta_spectral": resA["delta_spectral"],
        "A_delta_energy_norm": resA["delta_energy_norm"],
        "B_delta_energy": resB["delta_energy"],
        "B_delta_spectral": resB["delta_spectral"],
        "B_delta_energy_norm": resB["delta_energy_norm"],
    }
    if keep_visual:
        record["A_R0"] = resA["R0"]
        record["A_R1"] = resA["R1"]
        record["A_img"] = img_q95
    return record


def auroc_with_direction(y_true, scores):
    """Return (raw_auc, effective_auc, flipped_bool)."""
    raw = float(roc_auc_score(y_true, scores))
    if raw < 0.5:
        return raw, 1.0 - raw, True
    return raw, raw, False


def save_histogram(real_vals, ai_vals, title, xlabel, out_path):
    plt.figure(figsize=(6, 4))
    bins = 40
    plt.hist(real_vals, bins=bins, alpha=0.6, label="real", color="#1f77b4", density=True)
    plt.hist(ai_vals, bins=bins, alpha=0.6, label="ai", color="#d62728", density=True)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def save_leakage_examples(records, out_dir):
    reals = [r for r in records if r["label"] == 0 and "A_R0" in r][:3]
    ais = [r for r in records if r["label"] == 1 and "A_R0" in r][:3]
    paths = []
    for i, rec in enumerate(reals + ais):
        cls = "real" if rec["label"] == 0 else "ai"
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(rec["A_img"], cmap="gray", vmin=0, vmax=1)
        axes[0].set_title(f"{cls} - original (q95)")
        axes[0].axis("off")

        R0 = rec["A_R0"]
        vmax = np.percentile(np.abs(R0), 99) + 1e-8
        axes[1].imshow(R0, cmap="seismic", vmin=-vmax, vmax=vmax)
        axes[1].set_title("R0 (noise residual)")
        axes[1].axis("off")

        # Content-normalized reactivity map: (R1-R0)^2 / local_activity.
        # Shows whether the raw statistic's high values track scene edges
        # (leakage) and whether normalizing flattens that dependence.
        diff_sq = (rec["A_R1"] - R0) ** 2
        activity = local_activity_map(rec["A_img"])
        norm_map = diff_sq / (activity + LOCAL_ACTIVITY_EPS)
        vmax_norm = np.percentile(norm_map, 99) + 1e-12
        axes[2].imshow(norm_map, cmap="magma", vmin=0, vmax=vmax_norm)
        axes[2].set_title("(R1-R0)^2 / local activity")
        axes[2].axis("off")

        fname = os.path.join(out_dir, f"leakage_example_{cls}_{i}.png")
        plt.tight_layout()
        plt.savefig(fname, dpi=120)
        plt.close()
        paths.append(fname)
    return paths


def main(profile_name="cifake"):
    profile = PROFILES[profile_name]
    real_dir = os.path.join(profile["data_dir"], "real")
    ai_dir = os.path.join(profile["data_dir"], "ai")
    results_dir = profile["results_dir"]

    set_seeds()
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 80)
    print(f"DIFFERENTIAL NOISE-RESIDUAL REACTIVITY EXPERIMENT  [profile: {profile_name}]")
    print("=" * 80)

    # ---------------------------------------------------------------
    # 1. Data acquisition
    # ---------------------------------------------------------------
    t0 = time.time()
    info = acquire_data(
        log=print,
        candidates=profile["candidates"],
        real_dir=real_dir,
        ai_dir=ai_dir,
        n_per_class=profile["n_per_class"],
    )
    if info is None:
        print("\nAborting: no data available (see instructions above).")
        return
    t1 = time.time()
    print(f"\nData acquisition done in {t1 - t0:.1f}s")
    print(f"  Source: {info['source']}")
    print(f"  n_real: {info['n_real']}, n_ai: {info['n_ai']}")
    print(f"  example real files: {info['examples_real']}")
    print(f"  example ai files:   {info['examples_ai']}")

    real_files = sorted(
        os.path.join(real_dir, f) for f in os.listdir(real_dir) if f.lower().endswith(".png")
    )
    ai_files = sorted(
        os.path.join(ai_dir, f) for f in os.listdir(ai_dir) if f.lower().endswith(".png")
    )

    print(f"\nProvenance control: every image (both classes) is re-encoded to "
          f"JPEG q{PROVENANCE_Q} and reloaded before any measurement, so real "
          f"and AI images share identical final compression provenance.")

    # ---------------------------------------------------------------
    # 2. Run pipeline on every image
    # ---------------------------------------------------------------
    all_paths = [(p, 0) for p in real_files] + [(p, 1) for p in ai_files]
    print(f"\nProcessing {len(all_paths)} images through the pipeline "
          f"(provenance control + Test A clean + Test B pre-degraded)...")

    visual_paths = set(real_files[:3]) | set(ai_files[:3])

    records = []
    t2 = time.time()
    for path, label in tqdm(all_paths, desc="pipeline"):
        try:
            rec = process_image(path, label, keep_visual=path in visual_paths)
            scalar_vals = [rec["busyness"], rec["A_delta_energy"], rec["A_delta_spectral"],
                           rec["A_delta_energy_norm"], rec["B_delta_energy"],
                           rec["B_delta_spectral"], rec["B_delta_energy_norm"]]
            if not all(np.isfinite(v) for v in scalar_vals):
                print(f"  WARNING: non-finite result on {path} (degenerate/corrupt "
                      f"image?), skipping.")
                continue
            records.append(rec)
        except Exception as e:
            print(f"  WARNING: failed on {path}: {e}")
    t3 = time.time()
    print(f"Pipeline finished in {t3 - t2:.1f}s "
          f"({(t3 - t2) / max(1, len(records)):.3f}s/image)")

    n_real_done = sum(1 for r in records if r["label"] == 0)
    n_ai_done = sum(1 for r in records if r["label"] == 1)
    print(f"Successfully processed: n_real={n_real_done}, n_ai={n_ai_done}")

    y = np.array([r["label"] for r in records])
    busy = np.array([r["busyness"] for r in records])

    # ---------------------------------------------------------------
    # 3. Content-leakage sanity check
    # ---------------------------------------------------------------
    print("\nRunning content-leakage sanity check...")
    leakage_dir = results_dir
    example_paths = save_leakage_examples(records, leakage_dir)

    energyA = np.array([r["A_delta_energy"] for r in records])
    corr_A = float(pearsonr(energyA, busy)[0])
    energyB = np.array([r["B_delta_energy"] for r in records])
    corr_B = float(pearsonr(energyB, busy)[0])

    energyA_norm = np.array([r["A_delta_energy_norm"] for r in records])
    corr_A_norm = float(pearsonr(energyA_norm, busy)[0])
    energyB_norm = np.array([r["B_delta_energy_norm"] for r in records])
    corr_B_norm = float(pearsonr(energyB_norm, busy)[0])

    leakage_flag_A = bool(abs(corr_A) > 0.5)
    leakage_flag_B = bool(abs(corr_B) > 0.5)
    leakage_flagged = bool(leakage_flag_A or leakage_flag_B)

    leakage_flag_A_norm = bool(abs(corr_A_norm) > 0.5)
    leakage_flag_B_norm = bool(abs(corr_B_norm) > 0.5)
    leakage_flagged_norm = bool(leakage_flag_A_norm or leakage_flag_B_norm)

    print(f"  corr(Delta_energy, busyness) Test A: {corr_A:.3f} "
          f"{'[FLAGGED]' if leakage_flag_A else '[ok]'}")
    print(f"  corr(Delta_energy, busyness) Test B: {corr_B:.3f} "
          f"{'[FLAGGED]' if leakage_flag_B else '[ok]'}")
    print(f"  corr(Delta_energy_norm, busyness) Test A: {corr_A_norm:.3f} "
          f"{'[FLAGGED]' if leakage_flag_A_norm else '[ok]'}  (content-normalized variant)")
    print(f"  corr(Delta_energy_norm, busyness) Test B: {corr_B_norm:.3f} "
          f"{'[FLAGGED]' if leakage_flag_B_norm else '[ok]'}  (content-normalized variant)")
    print(f"  saved {len(example_paths)} leakage example PNGs to {leakage_dir}/")

    # ---------------------------------------------------------------
    # 4. AUROC metrics for both tests, all statistics
    # ---------------------------------------------------------------
    stats_map = {
        "A": {
            "energy": np.array([r["A_delta_energy"] for r in records]),
            "spectral": np.array([r["A_delta_spectral"] for r in records]),
            "energy_norm": energyA_norm,
        },
        "B": {
            "energy": np.array([r["B_delta_energy"] for r in records]),
            "spectral": np.array([r["B_delta_spectral"] for r in records]),
            "energy_norm": energyB_norm,
        },
    }

    results_table = []
    hist_paths = []
    for test in ["A", "B"]:
        for stat in ["energy", "spectral", "energy_norm"]:
            scores = stats_map[test][stat]
            raw_auc, eff_auc, flipped = auroc_with_direction(y, scores)
            results_table.append({
                "test": test,
                "statistic": stat,
                "raw_auroc": raw_auc,
                "effective_auroc": eff_auc,
                "direction_flipped": flipped,
                "n_real": n_real_done,
                "n_ai": n_ai_done,
            })

            real_vals = scores[y == 0]
            ai_vals = scores[y == 1]
            hist_path = os.path.join(results_dir, f"hist_test{test}_{stat}.png")
            save_histogram(
                real_vals, ai_vals,
                title=f"Test {test} - Delta_{stat} (real vs ai)",
                xlabel=f"Delta_{stat}",
                out_path=hist_path,
            )
            hist_paths.append(hist_path)

    print("\nResults table:")
    header = f"{'test':<6}{'stat':<10}{'raw_auroc':<12}{'eff_auroc':<12}{'flipped':<10}{'n_real':<8}{'n_ai':<8}"
    print(header)
    print("-" * len(header))
    for row in results_table:
        print(f"{row['test']:<6}{row['statistic']:<10}{row['raw_auroc']:<12.4f}"
              f"{row['effective_auroc']:<12.4f}{str(row['direction_flipped']):<10}"
              f"{row['n_real']:<8}{row['n_ai']:<8}")

    # ---------------------------------------------------------------
    # 5. Verdict (based on Delta_energy AUROCs)
    # ---------------------------------------------------------------
    energyA_eff = next(r["effective_auroc"] for r in results_table
                        if r["test"] == "A" and r["statistic"] == "energy")
    energyB_eff = next(r["effective_auroc"] for r in results_table
                        if r["test"] == "B" and r["statistic"] == "energy")

    if energyA_eff >= 0.65 and energyB_eff >= 0.60:
        verdict = "UPSIDE"
    elif energyA_eff >= 0.65 and energyB_eff < 0.60:
        verdict = "PREDICTED / clean-only"
    else:
        verdict = "DEAD"

    # ---------------------------------------------------------------
    # 5b. Secondary verdict for the content-normalized statistic
    # (experimental extension, same thresholds, on Delta_energy_norm)
    # ---------------------------------------------------------------
    energyA_norm_eff = next(r["effective_auroc"] for r in results_table
                             if r["test"] == "A" and r["statistic"] == "energy_norm")
    energyB_norm_eff = next(r["effective_auroc"] for r in results_table
                             if r["test"] == "B" and r["statistic"] == "energy_norm")

    if energyA_norm_eff >= 0.65 and energyB_norm_eff >= 0.60:
        verdict_norm = "UPSIDE"
    elif energyA_norm_eff >= 0.65 and energyB_norm_eff < 0.60:
        verdict_norm = "PREDICTED / clean-only"
    else:
        verdict_norm = "DEAD"

    # ---------------------------------------------------------------
    # 6. Save results.json
    # ---------------------------------------------------------------
    results_json = {
        "profile": profile_name,
        "data_source": info["source"],
        "n_real": n_real_done,
        "n_ai": n_ai_done,
        "provenance_control": f"JPEG q{PROVENANCE_Q} re-encode+reload applied to all images",
        "degrade_quality_test_B": DEGRADE_Q,
        "results_table": results_table,
        "leakage_check": {
            "corr_testA_energy_vs_busyness": corr_A,
            "corr_testB_energy_vs_busyness": corr_B,
            "flagged": leakage_flagged,
        },
        "verdict": verdict,
        "delta_energy_auroc_effective": {"A": energyA_eff, "B": energyB_eff},
        "content_normalized_variant": {
            "description": (
                "delta_energy_norm: pixel-wise (R1-R0)^2 divided by a local "
                f"scene-activity map (box-filter local variance, window="
                f"{LOCAL_ACTIVITY_WINDOW}, eps={LOCAL_ACTIVITY_EPS}) before "
                "averaging, to make the reactivity statistic content-"
                "invariant by construction rather than correcting for "
                "busyness after the fact."
            ),
            "leakage_check": {
                "corr_testA_energy_norm_vs_busyness": corr_A_norm,
                "corr_testB_energy_norm_vs_busyness": corr_B_norm,
                "flagged": leakage_flagged_norm,
            },
            "delta_energy_norm_auroc_effective": {"A": energyA_norm_eff, "B": energyB_norm_eff},
            "verdict": verdict_norm,
        },
        "timing_seconds": {
            "data_acquisition": t1 - t0,
            "pipeline": t3 - t2,
        },
        "example_files": {
            "real": info["examples_real"],
            "ai": info["examples_ai"],
        },
    }
    results_json_path = os.path.join(results_dir, "results.json")
    with open(results_json_path, "w") as f:
        json.dump(results_json, f, indent=2)

    # ---------------------------------------------------------------
    # 7. Final report
    # ---------------------------------------------------------------
    print("\n" + "=" * 80)
    print("FINAL REPORT")
    print("=" * 80)
    print(f"Data source: {info['source']}")
    print(f"Counts: n_real={n_real_done}, n_ai={n_ai_done}")
    print(f"Provenance control applied: JPEG q{PROVENANCE_Q} re-encode+reload "
          f"on all images (both classes share identical compression provenance).")
    print("\nResults table:")
    print(header)
    print("-" * len(header))
    for row in results_table:
        print(f"{row['test']:<6}{row['statistic']:<10}{row['raw_auroc']:<12.4f}"
              f"{row['effective_auroc']:<12.4f}{str(row['direction_flipped']):<10}"
              f"{row['n_real']:<8}{row['n_ai']:<8}")

    print("\nVERDICT:", verdict)
    print(f"  (Test A Delta_energy eff. AUROC = {energyA_eff:.4f}, "
          f"Test B Delta_energy eff. AUROC = {energyB_eff:.4f})")

    explanation = {
        "UPSIDE": (
            "Delta_energy separates real from AI images both on cleanly "
            "processed images and on images that have already been "
            "JPEG-degraded (the realistic, internet-laundered case). This "
            "is a genuinely promising signal for an AI-image detector: it "
            "survives a common form of real-world image handling, so it "
            "could be used as one feature (ideally combined with others) "
            "in a practical detection pipeline."
        ),
        "PREDICTED / clean-only": (
            "Delta_energy separates real from AI images on cleanly "
            "processed inputs, but the signal collapses once images are "
            "additionally JPEG-degraded, which is what happens whenever "
            "content moves through social media, messaging apps, or "
            "re-hosting. That means this exact signal is unlikely to be "
            "useful as a standalone real-world AI-image detector, though "
            "it may still be usable on freshly-generated, unlaundered "
            "content or as a weak feature among many."
        ),
        "DEAD": (
            "Delta_energy does not meaningfully separate real from AI "
            "images even on cleanly processed inputs, so this particular "
            "noise-residual-reactivity signal is not a viable basis for an "
            "AI-image detector as tested here. A different denoiser, "
            "probe, or statistic might behave differently, but this "
            "configuration shows no usable effect."
        ),
    }
    print("\n" + explanation[verdict])

    if leakage_flagged:
        print(
            "\nLEAKAGE CHECK: FLAGGED - the correlation between Delta_energy "
            "and image busyness exceeds 0.5 in at least one test, so part of "
            "the observed separation may reflect scene content/edge "
            "structure rather than a genuine real-vs-AI 'reactivity' "
            "difference. Treat the AUROC numbers above with caution."
        )
    else:
        print(
            "\nLEAKAGE CHECK: PASSED - Delta_energy does not strongly "
            "correlate with image busyness, so the residual signal looks "
            "like noise-domain reactivity rather than leaked scene content."
        )

    print("\n" + "-" * 80)
    print("CONTENT-NORMALIZED VARIANT (Delta_energy_norm) - experimental extension")
    print("-" * 80)
    print(
        "Delta_energy_norm divides the pixel-wise squared residual-difference "
        "by a local scene-activity map (box-filter local variance, window="
        f"{LOCAL_ACTIVITY_WINDOW}) BEFORE averaging, so busy/edge pixels don't "
        "dominate the statistic regardless of real-vs-AI origin - a "
        "content-invariant-by-construction fix for the leakage seen above."
    )
    print(f"  corr(Delta_energy_norm, busyness) Test A: {corr_A_norm:.3f} "
          f"{'[FLAGGED]' if leakage_flag_A_norm else '[ok]'}")
    print(f"  corr(Delta_energy_norm, busyness) Test B: {corr_B_norm:.3f} "
          f"{'[FLAGGED]' if leakage_flag_B_norm else '[ok]'}")
    print(f"  Test A Delta_energy_norm eff. AUROC = {energyA_norm_eff:.4f}")
    print(f"  Test B Delta_energy_norm eff. AUROC = {energyB_norm_eff:.4f}")
    print(f"  Normalized-variant verdict: {verdict_norm}  (primary verdict was: {verdict})")
    if verdict_norm != verdict:
        print(
            f"  -> The verdict CHANGES under content normalization "
            f"({verdict} -> {verdict_norm}). This means the raw Delta_energy "
            f"result above was substantially confounded by scene content, "
            f"and the normalized statistic gives a materially different "
            f"read on whether this is a usable detector signal."
        )
    else:
        print(
            f"  -> The verdict is unchanged under content normalization "
            f"({verdict}). Content leakage does not appear to be the "
            f"deciding factor here."
        )

    print("\nOutput files:")
    print(f"  results.json: {results_json_path}")
    for p in hist_paths:
        print(f"  histogram: {p}")
    for p in example_paths:
        print(f"  leakage example: {p}")
    print("=" * 80)

    # ---------------------------------------------------------------
    # 8. Log to the shared cross-experiment leaderboard (deferred import:
    # harness.py imports from this module, so importing it back here at
    # call time - not module load time - avoids a circular import).
    # ---------------------------------------------------------------
    try:
        from harness import append_leaderboard
        raw_A = next(r["raw_auroc"] for r in results_table if r["test"] == "A" and r["statistic"] == "energy")
        raw_B = next(r["raw_auroc"] for r in results_table if r["test"] == "B" and r["statistic"] == "energy")
        raw_A_norm = next(r["raw_auroc"] for r in results_table if r["test"] == "A" and r["statistic"] == "energy_norm")
        raw_B_norm = next(r["raw_auroc"] for r in results_table if r["test"] == "B" and r["statistic"] == "energy_norm")
        ts = time.strftime("%Y-%m-%dT%H:%M:%S")
        append_leaderboard([
            {
                "signal": "noise_residual", "feature": "delta_energy", "profile": profile_name,
                "n_real": n_real_done, "n_ai": n_ai_done,
                "raw_auroc_testA": raw_A, "effective_auroc_testA": energyA_eff,
                "raw_auroc_testB": raw_B, "effective_auroc_testB": energyB_eff,
                "leakage_corr_vs_busyness": corr_A, "leakage_flagged": leakage_flag_A,
                "cv_auroc_feature_only": None, "cv_auroc_shortcut_only": None,
                "cv_auroc_combined": None, "shortcut_dominates": None,
                "hist_path": os.path.join(results_dir, "hist_testA_energy.png"),
                "scores_path": None, "timestamp": ts,
            },
            {
                "signal": "noise_residual", "feature": "delta_energy_norm", "profile": profile_name,
                "n_real": n_real_done, "n_ai": n_ai_done,
                "raw_auroc_testA": raw_A_norm, "effective_auroc_testA": energyA_norm_eff,
                "raw_auroc_testB": raw_B_norm, "effective_auroc_testB": energyB_norm_eff,
                "leakage_corr_vs_busyness": corr_A_norm, "leakage_flagged": leakage_flag_A_norm,
                "cv_auroc_feature_only": None, "cv_auroc_shortcut_only": None,
                "cv_auroc_combined": None, "shortcut_dominates": None,
                "hist_path": os.path.join(results_dir, "hist_testA_energy_norm.png"),
                "scores_path": None, "timestamp": ts,
            },
        ])
        print(f"Logged 2 entries to shared leaderboard ({os.path.join(BASE_DIR, 'results', 'leaderboard.json')})")
    except Exception as e:
        print(f"  (leaderboard logging skipped: {e})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile", choices=sorted(PROFILES.keys()), default="cifake",
        help="Which single-source dataset/config to run: 'cifake' (32x32-native, "
             "default) or 'fullres' (native full-resolution real/AI dataset).",
    )
    args = parser.parse_args()
    main(profile_name=args.profile)
