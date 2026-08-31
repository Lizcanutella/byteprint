"""
Tier-1 candidate signal: per-8x8-block DCT statistics.

Classical JPEG steganalysis/forensics features, computed on our own
pixel-domain DCT (via scipy.fft.dctn - no new dependency; this is an
approximation of the true JPEG-encoder DCT coefficients, but since every
image already goes through our own JPEG provenance-control re-encode,
that approximation is applied identically to both classes):

  - ac_kurtosis: excess kurtosis of the AC-coefficient distribution.
    Real-photo DCT AC coefficients are famously close to a Laplacian
    distribution; a different generative process could shift this.
  - benford_deviation: L1 deviation of the AC coefficients' first
    significant digit distribution from Benford's law - a standard
    JPEG-forensics / double-compression feature.

Run via the harness:
    python signal_dct_stats.py --profile fullres
"""

import argparse

import numpy as np
from scipy.fft import dctn
from scipy.stats import kurtosis

from main import PROFILES
from harness import run_signal_experiment


def _block_dct_ac_coeffs(img_float, block=8):
    h, w = img_float.shape
    h2, w2 = (h // block) * block, (w // block) * block
    img = img_float[:h2, :w2]
    blocks = (
        img.reshape(h2 // block, block, w2 // block, block)
        .transpose(0, 2, 1, 3)
        .reshape(-1, block, block)
    )
    dct_blocks = dctn(blocks, type=2, norm="ortho", axes=(1, 2))
    flat = dct_blocks.reshape(dct_blocks.shape[0], -1)
    return flat[:, 1:]  # drop the DC term (index 0) from every block


def dct_stats_features(img_float):
    ac = _block_dct_ac_coeffs(img_float).ravel()
    ac = ac[np.abs(ac) > 1e-8]

    ac_kurt = float(kurtosis(ac, fisher=True))

    abs_vals = np.abs(ac)
    abs_vals = abs_vals[abs_vals > 1e-6]
    first_digits = np.clip(
        np.floor(abs_vals / 10.0 ** np.floor(np.log10(abs_vals))).astype(int), 1, 9
    )
    counts = np.bincount(first_digits, minlength=10)[1:10]
    freqs = counts / max(counts.sum(), 1)
    benford_expected = np.log10(1 + 1 / np.arange(1, 10))
    benford_dev = float(np.sum(np.abs(freqs - benford_expected)))

    return {"ac_kurtosis": ac_kurt, "benford_deviation": benford_dev}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(PROFILES.keys()), default="fullres")
    args = parser.parse_args()
    run_signal_experiment("dct_stats", dct_stats_features, args.profile)
