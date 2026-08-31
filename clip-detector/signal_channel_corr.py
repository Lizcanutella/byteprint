"""
Tier-1 candidate signal: cross-channel noise-residual correlation.

Real camera pipelines demosaic RGB from a single-channel Bayer CFA
sensor, which leaves a characteristic correlation structure between the
R/G/B noise residuals (each channel's "missing" samples are interpolated
from its neighbors, including cross-channel information). Many
generative models produce channels through a more independent or
differently-smoothed process. Feature: per-channel wavelet noise
residual, then pairwise correlation between channels.

Requires color (needs_color=True in the harness).

Run via the harness:
    python signal_channel_corr.py --profile fullres
"""

import argparse

import numpy as np
from skimage.restoration import denoise_wavelet

from main import PROFILES
from harness import run_signal_experiment


def channel_residual_correlation_features(img_float_rgb):
    residuals = []
    for c in range(3):
        channel = img_float_rgb[:, :, c]
        denoised = denoise_wavelet(channel, rescale_sigma=True)
        residuals.append((channel - denoised).ravel())
    r, g, b = residuals
    corr_rg = float(np.corrcoef(r, g)[0, 1])
    corr_rb = float(np.corrcoef(r, b)[0, 1])
    corr_gb = float(np.corrcoef(g, b)[0, 1])
    mean_corr = float(np.mean([corr_rg, corr_rb, corr_gb]))
    return {
        "corr_rg": corr_rg,
        "corr_rb": corr_rb,
        "corr_gb": corr_gb,
        "mean_channel_corr": mean_corr,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(PROFILES.keys()), default="fullres")
    args = parser.parse_args()
    run_signal_experiment(
        "channel_corr", channel_residual_correlation_features, args.profile, needs_color=True
    )
