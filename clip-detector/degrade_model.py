"""
Lightweight "degradation classifier": predicts how blurred and how
JPEG-compressed an image is, using classical no-reference features and a
small regressor trained on SYNTHETIC degradations (so ground truth is
exact by construction).

This is deliberately not a deep CNN: on CPU, with ~150 base images x a
handful of synthetic samples each, a small RandomForest over a few
classical IQA-style features trains in seconds and is easy to validate
(the same style of approach used in blind image-quality-assessment /
forensics literature for blur-kernel and JPEG-quality estimation).
"""

import io
import random

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, laplace
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from pipeline import high_freq_energy

SEED = 1234


# ---------------------------------------------------------------------------
# Synthetic degradation (ground truth for training)
# ---------------------------------------------------------------------------

def apply_synthetic_degradation(img_float, blur_sigma, jpeg_quality):
    """Apply a known blur + JPEG-quality degradation to a float [0,1]
    grayscale image and return the resulting float image."""
    if blur_sigma > 0:
        img_float = gaussian_filter(img_float, sigma=blur_sigma)
    arr8 = np.clip(img_float * 255.0, 0, 255).astype(np.uint8)
    im = Image.fromarray(arr8, mode="L")
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=int(round(jpeg_quality)))
    buf.seek(0)
    im2 = Image.open(buf).convert("L")
    im2.load()
    return np.asarray(im2, dtype=np.float64) / 255.0


# ---------------------------------------------------------------------------
# Classical no-reference features
# ---------------------------------------------------------------------------

def blockiness_score(img_float, block=8):
    """Proxy for JPEG blocking artifacts: how much bigger the average
    pixel-to-pixel jump is exactly at 8x8 block-grid boundaries compared
    to interior (non-boundary) jumps. Higher = more visible blocking =
    lower JPEG quality."""
    v_diffs = np.abs(np.diff(img_float, axis=1))
    cols = np.arange(v_diffs.shape[1])
    v_mask = ((cols + 1) % block == 0)
    v_score = v_diffs[:, v_mask].mean() - v_diffs[:, ~v_mask].mean()

    h_diffs = np.abs(np.diff(img_float, axis=0))
    rows = np.arange(h_diffs.shape[0])
    h_mask = ((rows + 1) % block == 0)
    h_score = h_diffs[h_mask, :].mean() - h_diffs[~h_mask, :].mean()

    return float((v_score + h_score) / 2.0)


def highfreq_ratio(img_float, frac=0.5):
    """High-frequency FFT power as a fraction of total power. More
    content-scale-invariant than the raw high_freq_energy used elsewhere,
    which matters here since the regressor must generalize across many
    different scenes."""
    f = np.fft.fftshift(np.fft.fft2(img_float))
    power = np.abs(f) ** 2
    total = float(power.sum()) + 1e-12
    return high_freq_energy(img_float, frac=frac) / total


def extract_iqa_features(img_float):
    """Feature vector used by the degradation regressors:
    [laplacian_variance, highfreq_power_ratio, blockiness, contrast_std]."""
    lap_var = float(laplace(img_float).var())
    hf_ratio = float(highfreq_ratio(img_float))
    block = blockiness_score(img_float)
    contrast = float(img_float.std())
    return np.array([lap_var, hf_ratio, block, contrast])


FEATURE_NAMES = ["laplacian_var", "highfreq_ratio", "blockiness", "contrast"]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def build_training_data(base_images, n_per_image=4, blur_range=(0.0, 2.5),
                         quality_range=(30, 100), seed=SEED):
    """base_images: list of float [0,1] grayscale arrays (content only,
    no class labels needed - this is a content-agnostic regressor)."""
    rng = random.Random(seed)
    X, y_blur, y_qual = [], [], []
    for img_float in base_images:
        for _ in range(n_per_image):
            sigma = rng.uniform(*blur_range)
            quality = rng.uniform(*quality_range)
            degraded = apply_synthetic_degradation(img_float, sigma, quality)
            X.append(extract_iqa_features(degraded))
            y_blur.append(sigma)
            y_qual.append(quality)
    return np.array(X), np.array(y_blur), np.array(y_qual)


def train_degradation_models(X, y_blur, y_qual, seed=SEED):
    Xtr, Xte, yb_tr, yb_te, yq_tr, yq_te = train_test_split(
        X, y_blur, y_qual, test_size=0.25, random_state=seed
    )

    blur_model = RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=-1)
    blur_model.fit(Xtr, yb_tr)
    qual_model = RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=-1)
    qual_model.fit(Xtr, yq_tr)

    blur_pred = blur_model.predict(Xte)
    qual_pred = qual_model.predict(Xte)
    metrics = {
        "n_train": len(Xtr),
        "n_test": len(Xte),
        "blur_r2": float(r2_score(yb_te, blur_pred)),
        "blur_mae": float(mean_absolute_error(yb_te, blur_pred)),
        "quality_r2": float(r2_score(yq_te, qual_pred)),
        "quality_mae": float(mean_absolute_error(yq_te, qual_pred)),
    }
    return blur_model, qual_model, metrics


def predict_degradation(img_float, blur_model, qual_model):
    feat = extract_iqa_features(img_float).reshape(1, -1)
    return float(blur_model.predict(feat)[0]), float(qual_model.predict(feat)[0])
