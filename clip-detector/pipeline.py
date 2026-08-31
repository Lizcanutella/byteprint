"""
Core image pipeline for the "differential noise-residual reactivity"
experiment.

All per-image transforms are exposed as small, swappable functions so a
different denoiser or probe can be plugged in without touching the rest
of the pipeline (see README.md).
"""

import io

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, laplace, uniform_filter
from skimage.restoration import denoise_wavelet

TARGET_SIZE = 512


# ---------------------------------------------------------------------------
# Loading / geometric normalization
# ---------------------------------------------------------------------------

def load_resize_crop(path, size=TARGET_SIZE, to_gray=True):
    """Load an image, force RGB, resize short side to `size` (keep aspect),
    center-crop to size x size. Returns a PIL 'L' mode image by default
    (`to_gray=True`, the original behavior); pass `to_gray=False` to get
    the cropped 'RGB' image instead, for signals that need color (e.g.
    cross-channel residual correlation).
    """
    img = Image.open(path).convert("RGB")
    w, h = img.size
    short = min(w, h)
    scale = size / short
    new_w = max(size, round(w * scale))
    new_h = max(size, round(h * scale))
    img = img.resize((new_w, new_h), Image.BICUBIC)

    left = (new_w - size) // 2
    top = (new_h - size) // 2
    img = img.crop((left, top, left + size, top + size))

    return img.convert("L") if to_gray else img


def jpeg_reencode(img, quality):
    """Re-encode a PIL image (mode 'L' or 'RGB') to JPEG at `quality` and
    reload it, to normalize / control compression provenance. Preserves
    the input's mode.
    """
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    reloaded = Image.open(buf).convert(img.mode)
    reloaded.load()
    return reloaded


def to_float(img):
    """PIL 'L' or 'RGB' image -> float64 array in [0, 1] (shape (H,W) or
    (H,W,3) respectively)."""
    return np.asarray(img, dtype=np.float64) / 255.0


# ---------------------------------------------------------------------------
# Swappable denoiser / probe
# ---------------------------------------------------------------------------

def denoiser_wavelet(img_float):
    """Default denoiser: skimage wavelet denoising."""
    return denoise_wavelet(img_float, rescale_sigma=True)


def probe_gaussian_blur(img_float, sigma=0.8):
    """Default probe: mild Gaussian blur."""
    return gaussian_filter(img_float, sigma=sigma)


DEFAULT_DENOISER = denoiser_wavelet
DEFAULT_PROBE = probe_gaussian_blur


# ---------------------------------------------------------------------------
# Residual statistics
# ---------------------------------------------------------------------------

def high_freq_energy(img_float, frac=0.5):
    """Energy of the 2D FFT power spectrum outside a central disk covering
    `frac` of the max radius (i.e. the high-frequency band)."""
    f = np.fft.fftshift(np.fft.fft2(img_float))
    power = np.abs(f) ** 2
    h, w = img_float.shape
    cy, cx = h / 2.0, w / 2.0
    yy, xx = np.ogrid[:h, :w]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    rmax = np.sqrt(cy ** 2 + cx ** 2)
    mask = r > (frac * rmax)
    return float(power[mask].sum())


def busyness(img_float):
    """Scene 'busyness' proxy: variance of the Laplacian of the image
    (classic sharpness/edge-content measure)."""
    return float(laplace(img_float).var())


LOCAL_ACTIVITY_WINDOW = 15
LOCAL_ACTIVITY_EPS = 1e-4


def local_activity_map(img_float, size=LOCAL_ACTIVITY_WINDOW):
    """Local scene-content 'activity' map: a sliding-window variance of the
    image (box filter, via the identity Var = E[x^2] - E[x]^2). High near
    edges/texture, near zero in flat regions.

    Used to make the reactivity statistic content-invariant BY
    CONSTRUCTION: dividing the pixel-wise squared residual-difference by
    this map before averaging down-weights exactly the busy/edge pixels
    that would otherwise dominate Delta_energy regardless of real-vs-AI
    origin (this is the standard trick from PRNU/sensor-noise forensics,
    where local scene variance is divided out before comparing noise
    statistics across images of different content).
    """
    mean = uniform_filter(img_float, size=size)
    mean_sq = uniform_filter(img_float ** 2, size=size)
    return np.clip(mean_sq - mean ** 2, 0.0, None)


def compute_deltas(img_float, denoiser=DEFAULT_DENOISER, probe=DEFAULT_PROBE):
    """Run the residual-reactivity pipeline on a single float grayscale
    image. Returns a dict with delta_energy, delta_spectral, a
    content-normalized delta_energy_norm, and the two residual maps R0, R1
    (for visualization / leakage checks).
    """
    denoised0 = denoiser(img_float)
    R0 = img_float - denoised0

    probed = probe(img_float)
    denoised1 = denoiser(probed)
    R1 = probed - denoised1

    diff_sq = (R1 - R0) ** 2
    delta_energy = float(np.mean(diff_sq))
    delta_spectral = abs(high_freq_energy(R1) - high_freq_energy(R0))

    # Content-invariant-by-construction variant: normalize each pixel's
    # contribution by local scene activity before pooling, instead of
    # correcting the pooled scalar after the fact.
    activity = local_activity_map(img_float)
    delta_energy_norm = float(np.mean(diff_sq / (activity + LOCAL_ACTIVITY_EPS)))

    return {
        "delta_energy": delta_energy,
        "delta_spectral": delta_spectral,
        "delta_energy_norm": delta_energy_norm,
        "R0": R0,
        "R1": R1,
    }
