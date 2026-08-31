"""
Shared rigor harness for running and comparing many candidate real-vs-AI
signals under the same validation gauntlet, so a new signal gets exactly
the same scrutiny that caught the earlier false positives (CIFAKE's
32x32-upsampling artifact, the native-JPEG-quality shortcut):

  - Test A (clean, JPEG q95 provenance-controlled) and Test B
    (pre-degraded to JPEG q50) AUROC.
  - Leakage check: correlation of the feature vs. scene busyness.
  - Shortcut-ablation check: does a classifier using ONLY
    [native_quality, native_blur, busyness] (no real signal at all)
    already explain the feature's apparent AUROC?
  - Per-image scores are cached to disk so `leaderboard_report.py` can
    do a genuine cross-dataset consistency check once a signal has been
    run on more than one profile.

A new signal is just a function `feature_fn(img_float) -> dict[str, float]`
plugged into `run_signal_experiment`. See `example_signal_spectral_slope.py`
for a template.
"""

import json
import os
import pickle
import random
import time

import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fetch_data import acquire_data
from main import PROFILES, PROVENANCE_Q, DEGRADE_Q, SEED, set_seeds, save_histogram
from pipeline import load_resize_crop, jpeg_reencode, to_float, busyness
from degrade_model import build_training_data, extract_iqa_features, train_degradation_models

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LEADERBOARD_PATH = os.path.join(BASE_DIR, "results", "leaderboard.json")
N_TRAIN_IMAGES = 150
N_SYNTH_PER_IMAGE = 4
LEAKAGE_THRESHOLD = 0.5
SHORTCUT_MARGIN = 0.02


def _ensure_data(profile_name):
    profile = PROFILES[profile_name]
    real_dir = os.path.join(profile["data_dir"], "real")
    ai_dir = os.path.join(profile["data_dir"], "ai")
    info = acquire_data(
        log=print, candidates=profile["candidates"], real_dir=real_dir,
        ai_dir=ai_dir, n_per_class=profile["n_per_class"],
    )
    if info is None:
        raise RuntimeError(f"No data available for profile '{profile_name}'")
    real_files = sorted(
        os.path.join(real_dir, f) for f in os.listdir(real_dir) if f.lower().endswith(".png")
    )
    ai_files = sorted(
        os.path.join(ai_dir, f) for f in os.listdir(ai_dir) if f.lower().endswith(".png")
    )
    return real_files, ai_files, profile


def get_degradation_models(profile_name, base_imgs_gray):
    """Train (or load cached) native-degradation regressors for a profile.
    `base_imgs_gray`: dict path -> grayscale float [0,1] array, used only
    if the cache doesn't already exist.
    """
    profile = PROFILES[profile_name]
    cache_path = os.path.join(profile["data_dir"], "degradation_models.pkl")
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    rng = random.Random(SEED)
    train_paths = rng.sample(list(base_imgs_gray.keys()), min(N_TRAIN_IMAGES, len(base_imgs_gray)))
    X, y_blur, y_qual = build_training_data(
        [base_imgs_gray[p] for p in train_paths], n_per_image=N_SYNTH_PER_IMAGE
    )
    blur_model, qual_model, metrics = train_degradation_models(X, y_blur, y_qual)
    with open(cache_path, "wb") as f:
        pickle.dump((blur_model, qual_model, metrics), f)
    return blur_model, qual_model, metrics


def effective_auroc(y_true, scores):
    raw = float(roc_auc_score(y_true, scores))
    return raw if raw >= 0.5 else 1.0 - raw


def cv_effective_auroc(X_feat, y_labels, seed=SEED):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    scores = cross_val_score(pipe, X_feat, y_labels, cv=skf, scoring="roc_auc")
    return float(np.mean(np.maximum(scores, 1 - scores)))


def append_leaderboard(rows):
    os.makedirs(os.path.dirname(LEADERBOARD_PATH), exist_ok=True)
    existing = []
    if os.path.exists(LEADERBOARD_PATH):
        with open(LEADERBOARD_PATH) as f:
            existing = json.load(f)
    existing.extend(rows)
    with open(LEADERBOARD_PATH, "w") as f:
        json.dump(existing, f, indent=2)


def run_signal_experiment(name, feature_fn, profile_name, needs_color=False, degrade_test=True):
    """
    feature_fn(img_float) -> dict[str, float]. `img_float` is a grayscale
    (H,W) or RGB (H,W,3) float [0,1] array depending on `needs_color`.
    Runs the full validation gauntlet and appends one leaderboard row per
    named feature `feature_fn` returns.
    """
    set_seeds()
    real_files, ai_files, profile = _ensure_data(profile_name)
    all_paths = [(p, 0) for p in real_files] + [(p, 1) for p in ai_files]
    results_dir = profile["results_dir"]
    os.makedirs(results_dir, exist_ok=True)

    print(f"[{name}/{profile_name}] loading base (pre-provenance) images ({len(all_paths)})...")
    t0 = time.time()
    base_imgs_gray = {}
    for path, _ in tqdm(all_paths, desc="load"):
        base_imgs_gray[path] = to_float(load_resize_crop(path, to_gray=True))
    print(f"  done in {time.time() - t0:.1f}s")

    print(f"[{name}/{profile_name}] getting/training degradation regressor (for shortcut ablation)...")
    blur_model, qual_model, reg_metrics = get_degradation_models(profile_name, base_imgs_gray)

    print(f"[{name}/{profile_name}] extracting feature(s) + Test A/B...")
    per_image = []
    t0 = time.time()
    for path, label in tqdm(all_paths, desc=f"{name}-extract"):
        gray_float = base_imgs_gray[path]
        native_feat = extract_iqa_features(gray_float).reshape(1, -1)
        native_blur = float(blur_model.predict(native_feat)[0])
        native_qual = float(qual_model.predict(native_feat)[0])
        busy = busyness(gray_float)

        pil_img = load_resize_crop(path, to_gray=not needs_color)
        pil_q95 = jpeg_reencode(pil_img, PROVENANCE_Q)
        img_q95 = to_float(pil_q95)
        feats_A = feature_fn(img_q95)

        feats_B = None
        if degrade_test:
            pil_q50 = jpeg_reencode(pil_q95, DEGRADE_Q)
            img_q50 = to_float(pil_q50)
            feats_B = feature_fn(img_q50)

        row = {"path": path, "label": label, "busyness": busy,
               "native_blur": native_blur, "native_quality": native_qual}
        bad = False
        for k, v in feats_A.items():
            if not np.isfinite(v):
                bad = True
            row[f"A_{k}"] = v
        if feats_B is not None:
            for k, v in feats_B.items():
                if not np.isfinite(v):
                    bad = True
                row[f"B_{k}"] = v
        if bad:
            print(f"  WARNING: non-finite feature on {path}, skipping.")
            continue
        per_image.append(row)
    print(f"  done in {time.time() - t0:.1f}s ({len(per_image)} images)")

    y = np.array([r["label"] for r in per_image])
    busy = np.array([r["busyness"] for r in per_image])
    native_b = np.array([r["native_blur"] for r in per_image])
    native_q = np.array([r["native_quality"] for r in per_image])
    n_real = int((y == 0).sum())
    n_ai = int((y == 1).sum())

    feature_names = sorted({k[2:] for k in per_image[0] if k.startswith("A_")})
    leaderboard_rows = []

    for fname in feature_names:
        scores_A = np.array([r[f"A_{fname}"] for r in per_image])
        raw_A = float(roc_auc_score(y, scores_A))
        eff_A = raw_A if raw_A >= 0.5 else 1.0 - raw_A

        raw_B = eff_B = None
        if degrade_test:
            scores_B = np.array([r[f"B_{fname}"] for r in per_image])
            raw_B = float(roc_auc_score(y, scores_B))
            eff_B = raw_B if raw_B >= 0.5 else 1.0 - raw_B

        corr = float(pearsonr(scores_A, busy)[0])
        leakage_flagged = bool(abs(corr) > LEAKAGE_THRESHOLD)

        feature_only_auc = cv_effective_auroc(scores_A.reshape(-1, 1), y)
        shortcut_X = np.column_stack([native_q, native_b, busy])
        shortcut_only_auc = cv_effective_auroc(shortcut_X, y)
        combined_X = np.column_stack([scores_A, native_q, native_b, busy])
        combined_auc = cv_effective_auroc(combined_X, y)
        shortcut_dominates = bool(shortcut_only_auc >= combined_auc - SHORTCUT_MARGIN)

        print(f"  [{fname}] Test A eff.AUROC={eff_A:.4f}"
              + (f"  Test B eff.AUROC={eff_B:.4f}" if eff_B is not None else "")
              + f"  leakage_corr={corr:.3f}{'[FLAGGED]' if leakage_flagged else ''}"
              + f"  shortcut_only={shortcut_only_auc:.4f}"
              + f"{' [SHORTCUT-DOMINATES]' if shortcut_dominates else ''}")

        hist_path = os.path.join(results_dir, f"signal_{name}_{fname}_hist.png")
        save_histogram(
            scores_A[y == 0], scores_A[y == 1],
            title=f"{name}/{fname} Test A (real vs ai) [{profile_name}]",
            xlabel=fname, out_path=hist_path,
        )

        scores_path = os.path.join(results_dir, f"signal_{name}_{fname}_scores.npz")
        np.savez(scores_path, y=y, scores_A=scores_A,
                 scores_B=scores_B if degrade_test else np.array([]))

        leaderboard_rows.append({
            "signal": name,
            "feature": fname,
            "profile": profile_name,
            "n_real": n_real,
            "n_ai": n_ai,
            "raw_auroc_testA": raw_A,
            "effective_auroc_testA": eff_A,
            "raw_auroc_testB": raw_B,
            "effective_auroc_testB": eff_B,
            "leakage_corr_vs_busyness": corr,
            "leakage_flagged": leakage_flagged,
            "cv_auroc_feature_only": feature_only_auc,
            "cv_auroc_shortcut_only": shortcut_only_auc,
            "cv_auroc_combined": combined_auc,
            "shortcut_dominates": shortcut_dominates,
            "hist_path": hist_path,
            "scores_path": scores_path,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })

    append_leaderboard(leaderboard_rows)
    print(f"[{name}/{profile_name}] logged {len(leaderboard_rows)} feature row(s) to {LEADERBOARD_PATH}")
    return leaderboard_rows
