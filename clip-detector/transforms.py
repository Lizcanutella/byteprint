"""
The hackathon's robustness-transform grid, as pure functions on PIL RGB
images. Used both for training-time augmentation (train_classifier.py)
and for the robustness evaluation grid (robustness_eval.py), so the two
can never silently drift apart.

Transform          Parameters                          Real-world analog
JPEG Compression    quality = 90, 70, 50, 30             social-media re-encode
Gaussian Blur       sigma = 0.5, 1.0, 2.0                 out-of-focus
Resize              scale 0.5x / 0.25x then upscale back  thumbnail generation
Gaussian Noise      sigma = 0.02, 0.05, 0.10 (on [0,1])   low-light sensor noise
Color Jitter        brightness/contrast/sat. +/-20%       filter apps, auto-enhance
Center Crop         crop 80% (then resized back)          profile-pic cropping
"""

import io

import numpy as np
from PIL import Image, ImageEnhance

JPEG_QUALITIES = [90, 70, 50, 30]
BLUR_SIGMAS = [0.5, 1.0, 2.0]
RESIZE_SCALES = [0.5, 0.25]
NOISE_SIGMAS = [0.02, 0.05, 0.10]
COLOR_JITTER_FACTORS = [0.8, 1.2]  # -20% / +20%
CROP_FRACTION = 0.8


def jpeg_compress(img, quality):
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    out = Image.open(buf).convert("RGB")
    out.load()
    return out


def gaussian_blur(img, sigma):
    from scipy.ndimage import gaussian_filter
    arr = np.asarray(img.convert("RGB"), dtype=np.float64) / 255.0
    blurred = np.stack([gaussian_filter(arr[..., c], sigma=sigma) for c in range(3)], axis=-1)
    return Image.fromarray(np.clip(blurred * 255.0, 0, 255).astype(np.uint8), mode="RGB")


def resize_roundtrip(img, scale):
    w, h = img.size
    small = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.BICUBIC)
    return small.resize((w, h), Image.BICUBIC)


def gaussian_noise(img, sigma):
    arr = np.asarray(img.convert("RGB"), dtype=np.float64) / 255.0
    noisy = arr + np.random.normal(0, sigma, arr.shape)
    return Image.fromarray(np.clip(noisy * 255.0, 0, 255).astype(np.uint8), mode="RGB")


def color_jitter(img, factor):
    """factor: 1.0 = no change; 0.8/1.2 = the hackathon's +/-20% grid."""
    out = img.convert("RGB")
    out = ImageEnhance.Brightness(out).enhance(factor)
    out = ImageEnhance.Contrast(out).enhance(factor)
    out = ImageEnhance.Color(out).enhance(factor)  # saturation
    return out


def hflip(img):
    from PIL import ImageOps
    return ImageOps.mirror(img.convert("RGB"))


def center_crop(img, fraction=CROP_FRACTION):
    w, h = img.size
    cw, ch = int(w * fraction), int(h * fraction)
    left, top = (w - cw) // 2, (h - ch) // 2
    cropped = img.crop((left, top, left + cw, top + ch))
    return cropped.resize((w, h), Image.BICUBIC)  # back to a consistent size for the backbone


# ---------------------------------------------------------------------------
# The full named grid: (name, fn) pairs, "clean" = no-op baseline.
# ---------------------------------------------------------------------------

def build_transform_grid():
    grid = [("clean", lambda img: img)]
    for q in JPEG_QUALITIES:
        grid.append((f"jpeg_q{q}", lambda img, q=q: jpeg_compress(img, q)))
    for s in BLUR_SIGMAS:
        grid.append((f"blur_s{s}", lambda img, s=s: gaussian_blur(img, s)))
    for sc in RESIZE_SCALES:
        grid.append((f"resize_{sc}", lambda img, sc=sc: resize_roundtrip(img, sc)))
    for s in NOISE_SIGMAS:
        grid.append((f"noise_s{s}", lambda img, s=s: gaussian_noise(img, s)))
    for f in COLOR_JITTER_FACTORS:
        sign = "up" if f > 1 else "down"
        grid.append((f"colorjitter_{sign}20", lambda img, f=f: color_jitter(img, f)))
    grid.append(("crop80", lambda img: center_crop(img, CROP_FRACTION)))
    return grid


TRANSFORM_GRID = build_transform_grid()

# Coarse domain groups, used for the domain-specialist experiment: the
# organizers' robustness test images are each degraded in exactly ONE
# domain at a time (never stacked), so a classifier that first detects
# *which* domain applies and routes to a domain-specialized head is a
# well-posed idea here (it wouldn't be if degradations could combine).
# Blur/resize/crop are grouped together ("spatial") because they're all
# resampling/detail-loss operations that look similar in classical
# no-reference features (Laplacian variance, high-freq ratio) - trying
# to split them further isn't reliably separable with cheap features.
DOMAIN_GROUPS = {
    "clean": ["clean"],
    "jpeg": [f"jpeg_q{q}" for q in JPEG_QUALITIES],
    "spatial": (
        [f"blur_s{s}" for s in BLUR_SIGMAS]
        + [f"resize_{sc}" for sc in RESIZE_SCALES]
        + ["crop80"]
    ),
    "noise": [f"noise_s{s}" for s in NOISE_SIGMAS],
    "colorjitter": ["colorjitter_down20", "colorjitter_up20"],
}
TRANSFORM_TO_GROUP = {t: g for g, ts in DOMAIN_GROUPS.items() for t in ts}
GROUP_NAMES = list(DOMAIN_GROUPS.keys())


def random_transform(img, rng):
    """Pick one random non-clean transform from the grid, for training-time
    augmentation."""
    non_clean = TRANSFORM_GRID[1:]
    name, fn = non_clean[rng.randrange(len(non_clean))]
    return name, fn(img)
