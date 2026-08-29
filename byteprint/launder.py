"""Image laundering: the corruptions an image picks up on its way to a viewer.

Used twice. During extraction it is *augmentation* -- the single highest-leverage
trick in this literature, since a detector that never saw JPEG has only learned
about images that no longer exist by the time anyone shares them. During
evaluation it is the *ladder*: report a score per rung, because one averaged
number hides which transformation is the one that breaks you.

A spec is a pipe-separated chain of ``op:value`` stages, e.g. ``scale:0.5|jpeg:30``.

The six operations and the parameters in :data:`OFFICIAL_LADDER` are fixed by
section 5.2 of the competition brief (``docs/competition-brief.md``). They are
transcribed here verbatim and pinned by a test. Do not add rungs to
:data:`OFFICIAL_LADDER` -- put exploratory chains in :data:`STRESS_LADDER`,
which is reported separately and clearly marked as beyond the brief.

Note the unit convention, which is the brief's and *not* the obvious one:
``noise`` sigma is a fraction of full scale, so ``noise:0.10`` means sigma =
25.5 of 255 levels. Values above 1.0 are rejected rather than reinterpreted.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

NO_OP = "none"


def _jpeg(image: np.ndarray, quality: float, rng: np.random.Generator) -> np.ndarray:
    if not 1 <= quality <= 100:
        raise ValueError(f"jpeg quality must be in [1, 100], got {quality}")
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="JPEG", quality=int(quality))
    buffer.seek(0)
    return np.asarray(Image.open(buffer).convert("RGB"))


def _scale(image: np.ndarray, factor: float, rng: np.random.Generator) -> np.ndarray:
    """Downscale then restore the original size -- a low-pass filter with extra steps."""
    if not 0 < factor <= 1:
        raise ValueError(f"scale factor must be in (0, 1], got {factor}")
    height, width = image.shape[:2]
    small = (max(1, round(width * factor)), max(1, round(height * factor)))
    pil = Image.fromarray(image).resize(small, Image.Resampling.BICUBIC)
    return np.asarray(pil.resize((width, height), Image.Resampling.BICUBIC))


def _blur(image: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    if sigma < 0:
        raise ValueError(f"blur sigma must be >= 0, got {sigma}")
    blurred = Image.fromarray(image).filter(ImageFilter.GaussianBlur(radius=float(sigma)))
    return np.asarray(blurred)


def _noise(image: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Additive Gaussian noise. Sigma is normalised: 0.10 means 25.5 of 255 levels."""
    if not 0 <= sigma <= 1:
        raise ValueError(
            f"noise sigma is normalised to [0, 1] (the brief's 0.02/0.05/0.10), got {sigma}"
            + (f" -- did you mean {sigma / 255:.4f}?" if sigma > 1 else "")
        )
    noisy = image.astype(np.float64) + rng.normal(0.0, float(sigma) * 255.0, size=image.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def _jitter(image: np.ndarray, amount: float, rng: np.random.Generator) -> np.ndarray:
    """Brightness, contrast and saturation each scaled by 1 +- ``amount``.

    Each factor is drawn independently, so the rung is a *distribution* of
    filter-app edits rather than one fixed recolour -- reproducible because the
    draw comes from the caller's seeded generator.
    """
    if not 0 <= amount < 1:
        raise ValueError(f"jitter amount must be in [0, 1), got {amount}")
    if amount == 0:
        return image

    pil = Image.fromarray(image)
    for enhancer in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
        factor = 1.0 + float(rng.uniform(-amount, amount))
        pil = enhancer(pil).enhance(factor)
    return np.asarray(pil.convert("RGB"))


def _crop(image: np.ndarray, fraction: float, rng: np.random.Generator) -> np.ndarray:
    """Centre crop to ``fraction`` of each side. The image gets *smaller*.

    Unlike ``scale`` the brief does not ask for an upscale back to the original
    size, and it should not: reframing is what a profile picture actually
    suffers, and crop selection downstream works at native resolution anyway.
    """
    if not 0 < fraction <= 1:
        raise ValueError(f"crop fraction must be in (0, 1], got {fraction}")
    if fraction == 1:
        return image
    height, width = image.shape[:2]
    keep_h, keep_w = max(1, round(height * fraction)), max(1, round(width * fraction))
    top, left = (height - keep_h) // 2, (width - keep_w) // 2
    return np.ascontiguousarray(image[top : top + keep_h, left : left + keep_w])


_OPS = {
    "jpeg": _jpeg,
    "blur": _blur,
    "scale": _scale,
    "noise": _noise,
    "jitter": _jitter,
    "crop": _crop,
}


def apply(image: np.ndarray, spec: str, seed: int | None = None) -> np.ndarray:
    """Apply a laundering chain to an RGB uint8 image."""
    image = np.asarray(image)
    if spec == NO_OP:
        return image

    rng = np.random.default_rng(seed)
    for stage in spec.split("|"):
        name, _, raw_value = stage.partition(":")
        if name not in _OPS:
            raise ValueError(f"unknown laundering operation {name!r} in spec {spec!r}")
        if not raw_value:
            raise ValueError(f"operation {name!r} needs a value, e.g. '{name}:0.5' (got {spec!r})")
        try:
            value = float(raw_value)
        except ValueError as exc:
            raise ValueError(f"non-numeric value in laundering spec {spec!r}") from exc
        image = _OPS[name](image, value, rng)

    return image


# The brief's section 5.2, transcribed. Pinned by a test: changing this list
# changes what every robustness number in the report means.
OFFICIAL_LADDER: tuple[str, ...] = (
    NO_OP,
    "jpeg:90",      # social-media re-encode
    "jpeg:70",
    "jpeg:50",
    "jpeg:30",
    "blur:0.5",     # out of focus
    "blur:1.0",
    "blur:2.0",
    "scale:0.5",    # thumbnail generation, then upscaled back
    "scale:0.25",
    "noise:0.02",   # low-light sensor noise
    "noise:0.05",
    "noise:0.10",
    "jitter:0.2",   # filter apps, auto-enhance
    "crop:0.8",     # profile-picture framing
)

# Beyond the brief: compositions, which is what an image reposted twice
# actually looks like. Reported separately so the official numbers stay
# comparable with everyone else's.
STRESS_LADDER: tuple[str, ...] = (
    NO_OP,
    "scale:0.5|jpeg:50",
    "scale:0.25|jpeg:30",
    "crop:0.8|scale:0.5|jpeg:70",
    "jitter:0.2|jpeg:50",
    "blur:1.0|noise:0.05|jpeg:70",
    "scale:0.5|blur:1.0|noise:0.02|jpeg:30",
)

LADDERS: dict[str, tuple[str, ...]] = {
    "official": OFFICIAL_LADDER,
    "stress": STRESS_LADDER,
    "all": OFFICIAL_LADDER + tuple(r for r in STRESS_LADDER if r != NO_OP),
}


def ladder(name: str = "official") -> list[str]:
    """The evaluation rungs, from untouched to heavily degraded.

    ``official`` is the brief's fixed list and the default. ``stress`` is the
    composed-transform set; ``all`` is both.
    """
    if name not in LADDERS:
        raise ValueError(f"unknown ladder {name!r}; expected one of {sorted(LADDERS)}")
    return list(LADDERS[name])


def sample_spec(rng: np.random.Generator) -> str:
    """Draw one random augmentation chain for training-time extraction.

    Ranges span the official parameters rather than matching them exactly:
    training on the precise test rungs teaches the probe those constants, not
    the damage. Stages compose in the order an image really suffers them --
    reframed, resized, softened, recoloured, sensor noise, then re-encoded.
    """
    if rng.random() < 0.20:
        return NO_OP

    stages = []
    if rng.random() < 0.20:
        stages.append(f"crop:{round(float(rng.uniform(0.75, 1.0)), 3)}")
    if rng.random() < 0.40:
        stages.append(f"scale:{round(float(rng.uniform(0.25, 0.95)), 3)}")
    if rng.random() < 0.30:
        stages.append(f"blur:{round(float(rng.uniform(0.3, 2.0)), 3)}")
    if rng.random() < 0.30:
        stages.append(f"jitter:{round(float(rng.uniform(0.05, 0.25)), 3)}")
    if rng.random() < 0.30:
        stages.append(f"noise:{round(float(rng.uniform(0.01, 0.12)), 4)}")
    if rng.random() < 0.70:
        stages.append(f"jpeg:{int(rng.integers(30, 96))}")

    return "|".join(stages) if stages else NO_OP
