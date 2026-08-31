"""
Tier-1 candidate signal: radial power-spectrum slope + spectral-peak
anomaly.

Natural photos have a smooth ~1/f^2 radial power-spectrum falloff.
Many GAN/diffusion decoders use transposed convolutions or repeated
upsampling that leave periodic checkerboard artifacts - these show up
as an anomalous peak that breaks the otherwise-smooth radial trend.

Two features per image:
  - radial_slope: fitted slope of log(power) vs log(radius) over a
    mid-frequency band (excludes DC and the noisiest extreme corners).
  - peak_zscore: z-score of the largest positive deviation from that
    smooth fit - a proxy for periodic/checkerboard artifacts.

Run via the harness:
    python signal_spectral_slope.py --profile fullres
"""

import argparse

import numpy as np

from main import PROFILES
from harness import run_signal_experiment


def _radial_profile(power, center, nbins=100):
    h, w = power.shape
    yy, xx = np.indices((h, w))
    r = np.sqrt((xx - center[0]) ** 2 + (yy - center[1]) ** 2)
    r_max = r.max()
    bin_edges = np.linspace(0, r_max, nbins + 1)
    bin_idx = np.clip(np.digitize(r.ravel(), bin_edges) - 1, 0, nbins - 1)
    sums = np.bincount(bin_idx, weights=power.ravel(), minlength=nbins)
    counts = np.bincount(bin_idx, minlength=nbins)
    radial_mean = sums / np.maximum(counts, 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return bin_centers, radial_mean


def spectral_slope_features(img_float):
    f = np.fft.fftshift(np.fft.fft2(img_float))
    power = np.abs(f) ** 2
    h, w = img_float.shape
    center = (w / 2.0, h / 2.0)
    r, radial_power = _radial_profile(power, center, nbins=100)

    r_max = r.max()
    mask = (r > 0.03 * r_max) & (r < 0.9 * r_max) & (radial_power > 0)
    log_r = np.log(r[mask])
    log_p = np.log(radial_power[mask])
    slope, intercept = np.polyfit(log_r, log_p, 1)

    fitted = slope * log_r + intercept
    residual = log_p - fitted
    resid_std = residual.std() + 1e-12
    peak_zscore = float(residual.max() / resid_std)

    return {"radial_slope": float(slope), "peak_zscore": peak_zscore}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(PROFILES.keys()), default="fullres")
    args = parser.parse_args()
    run_signal_experiment("spectral_slope", spectral_slope_features, args.profile)
