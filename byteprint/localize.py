"""Crop strategies that look for the edited region, not the busy one.

`texture` spends the backbone's budget on the most high-frequency windows in the
frame, which is right when the *whole* image is synthetic and wrong when only a
region of it is. A tampered image is authentic everywhere except one patch, and
that patch has no reason to be the busiest thing in the picture -- on SID_Set the
consequence is AUC 0.851 for tampered images against 0.954 for fully synthetic
ones.

Worse than uninformative, the heuristic is often anti-correlated. A region
produced by a decoder rather than captured through a lens tends to be *cleaner*
than its surroundings: the sensor grain that covers the rest of the frame is
absent from it. Ranking by high-frequency energy therefore sorts the edited
region last.

So rank windows by how far each one's high-frequency statistics sit from the
*rest of the same image*, and take the outliers.

Two properties make that a better instrument than it first appears:

- **It is a within-image contrast**, so a transform applied uniformly to the
  whole frame largely cancels out of it -- it hits the odd region and its
  surroundings alike. That holds while the transform merely *degrades* the grain.
  It stops holding once a transform destroys the grain outright everywhere, which
  removes the contrast along with it; `blur:2.0` and `scale:0.25` do exactly that
  and the cue goes blind on them.
- **It never compares across images**, so it cannot learn the corpus-level
  compression confound that the JPEG-95 control was built to rule out.

`ela` is registered alongside as the absolute cue to measure against. It was
expected to lose under recompression and does not: ranked as an outlier rather
than by raw energy, it holds up across the ladder and beats this cue outright on
the two rungs above, where the compression response of a smoothed region still
differs from its surroundings after the grain is gone. The two fail on different
rungs rather than together, which is the same argument this project already makes
for fusing two experts, and the reason both stay registered.

The failure case that would matter most is a *uniformly* generated image, where
no window disagrees with any other and the ranking degenerates into noise --
which would put the 0.954 full-synthetic number at risk to buy back the 0.851
one. So when no window stands out, `anomaly` defers to `texture` and today's
behaviour is preserved exactly.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

from byteprint.crops import as_rgb, random_origins, register_crop_mode

Origin = tuple[int, int]

# Below this many MAD units, the most unusual window in the image is not unusual
# enough to be worth chasing and `anomaly` hands over to `texture`. Set from the
# measurement in tests/test_localize.py: on uniform images the observed maximum
# sits an order of magnitude below it, and on images carrying a planted edit an
# order of magnitude above, so it sits in an empty gap rather than on a slope.
Z_FLOOR = 6.0

# A floor on the spread used to normalise deviations, in log-response units.
# Without it, an image whose windows agree almost exactly divides by ~0 and every
# window looks like a wild outlier -- the precise case we want to fall back on.
_MAD_FLOOR = 0.05


def band_map(window: np.ndarray) -> np.ndarray:
    """Per-pixel responses in two frequency bands, as ``(h-2, w-2, 2)``.

    Channel 0 is the luma Laplacian response; channel 1 is the luma gradient
    magnitude, a coarser band. Two bands rather than one because what separates
    a decoded region from a captured one is the *shape* of its high-frequency
    content, not only how much of it there is.

    The Laplacian is written as shifted slices rather than as a convolution. It
    is the same kernel as ``crops.LAPLACIAN`` -- pinned by a test against
    ``convolve2d_valid`` -- but a few array additions instead of an einsum over a
    strided window view, which matters because this runs on every candidate
    window of every image and crop selection is the pipeline's bottleneck.

    Luma rather than per colour channel, also for cost: three channels triple the
    most expensive step for no measurable gain on the planted-edit fixture. Real
    chroma inconsistency -- a spliced region whose demosaicing differs -- is a
    genuine cue this therefore cannot see, and is worth revisiting against real
    tampered images rather than against a proxy that cannot distinguish them.
    """
    window = as_rgb(np.asarray(window)).astype(np.float64)
    height, width = window.shape[:2]
    if min(height, width) < 3:
        return np.zeros((0, 0, 2))

    luma = window.mean(axis=2)
    laplacian = (
        luma[1:-1, 2:] + luma[1:-1, :-2] + luma[2:, 1:-1] + luma[:-2, 1:-1] - 4.0 * luma[1:-1, 1:-1]
    )
    gradient = np.hypot(np.diff(luma, axis=0)[:, :-1], np.diff(luma, axis=1)[:-1, :])

    return np.stack([laplacian, gradient[: height - 2, : width - 2]], axis=-1)


def window_stats(
    image: np.ndarray, origins: list[Origin], *, crop_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """Per-window band fingerprints ``(n, 2)`` and texture scores ``(n,)``.

    Both come out of one pass over each window, because they are the same
    convolution: a window's texture score is the variance of its luma Laplacian,
    which is band 0. Computing them together is what keeps this mode cheaper than
    paying for the fingerprint and then paying for `texture` again on the
    fallback path -- which is most images, since most images are uniform.

    Fingerprints are logs because what distinguishes a decoded region from a
    captured one is the *ratio* of its high-frequency energy to its surroundings,
    not the difference: a bright window and a dark one differ in absolute
    response for reasons that have nothing to do with provenance.
    """
    origins = list(origins)
    if not origins:
        return np.zeros((0, 2)), np.zeros(0)

    fingerprints = np.empty((len(origins), 2), dtype=np.float64)
    textures = np.empty(len(origins), dtype=np.float64)
    for index, (top, left) in enumerate(origins):
        bands = band_map(image[top : top + crop_size, left : left + crop_size])
        if bands.size == 0:
            fingerprints[index], textures[index] = 0.0, 0.0
        else:
            fingerprints[index] = np.log(bands.std(axis=(0, 1)) + 1e-6)
            textures[index] = bands[:, :, 0].var()
    return fingerprints, textures


def window_fingerprints(
    image: np.ndarray, origins: list[Origin], *, crop_size: int
) -> np.ndarray:
    """Log spread of each band inside each candidate window, as ``(n, 2)``."""
    return window_stats(image, origins, crop_size=crop_size)[0]


def anomaly_z(fingerprints: np.ndarray) -> np.ndarray:
    """How far each window sits from the image's own centre, in MAD units.

    Median and MAD rather than mean and standard deviation, deliberately: if a
    third of the sampled windows land inside the edited region, a mean is dragged
    towards the very thing it is supposed to make stand out.
    """
    values = np.asarray(fingerprints, dtype=np.float64)
    if values.size == 0:
        return np.zeros(len(values))

    deviation = np.abs(values - np.median(values, axis=0))
    scale = np.maximum(np.median(deviation, axis=0), _MAD_FLOOR)
    return (deviation / scale).max(axis=1)


def _by_texture(origins: list[Origin], textures: np.ndarray) -> list[Origin]:
    """The `texture` ordering, from scores already in hand.

    Ties keep their original order under both a stable argsort and the stable
    ``sorted(..., reverse=True)`` that `texture` itself uses, so this is the same
    sequence that mode would have produced, not merely a similar one.
    """
    return [origins[i] for i in np.argsort(-textures, kind="stable")]


def rank_origins(
    image: np.ndarray,
    origins: list[Origin],
    *,
    crop_size: int,
    z_floor: float = Z_FLOOR,
) -> tuple[list[Origin], bool]:
    """Order candidate windows most-anomalous first.

    Returns the ordered origins and whether the texture fallback was used. The
    fallback is not a safety net bolted on afterwards -- it is what keeps a
    uniformly generated image, where this ranking has nothing to say, scoring
    exactly as well as it does today.
    """
    image = as_rgb(np.asarray(image))
    origins = list(origins)
    if not origins:
        return [], True

    fingerprints, textures = window_stats(image, origins, crop_size=crop_size)
    z = anomaly_z(fingerprints)
    if float(z.max(initial=0.0)) < z_floor:
        return _by_texture(origins, textures), True

    return [origins[i] for i in np.argsort(-z, kind="stable")], False


def ela_residual(image: np.ndarray, *, quality: int = 90) -> np.ndarray:
    """``|image - jpeg(image)|``: how much a fresh encode changes each pixel."""
    image = as_rgb(np.asarray(image))
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    with Image.open(buffer) as reopened:
        recompressed = np.asarray(reopened.convert("RGB"))
    return np.abs(image.astype(np.float64) - recompressed.astype(np.float64))


def ela_rank_origins(
    image: np.ndarray, origins: list[Origin], *, crop_size: int, quality: int = 90
) -> list[Origin]:
    """Order candidate windows by how far their error level sits from the image's.

    Deviation from the median rather than raw highest energy, in both directions.
    That is what error-level analysis is in practice -- an analyst looks for the
    region that responds to a re-encode *unlike* its surroundings, and a pasted
    region can respond either more or less than its host. Ranking by raw energy
    instead would make this control anti-correlated on any edit that is smoother
    than what surrounds it, and lose the comparison for a reason that has nothing
    to do with the cue it is here to represent.
    """
    origins = list(origins)
    if not origins:
        return []

    residual = ela_residual(image, quality=quality)
    energy = np.asarray(
        [float(residual[t : t + crop_size, l : l + crop_size].mean()) for t, l in origins]
    )
    deviation = np.abs(energy - np.median(energy))
    return [origins[i] for i in np.argsort(-deviation, kind="stable")]


@register_crop_mode("anomaly")
def _anomaly(image, *, crop_size, top_k, candidates, rng) -> list[np.ndarray]:
    """Keep the windows least like the rest of their own image.

    Same candidate pool as `texture`, drawn the same way from the same seed --
    only the ranking differs, so the comparison between the two isolates *where
    it looked* rather than what it sampled.
    """
    origins = random_origins(image, crop_size, max(candidates, top_k), rng)
    ranked, _ = rank_origins(image, origins, crop_size=crop_size)
    return [image[t : t + crop_size, l : l + crop_size] for t, l in ranked[:top_k]]


@register_crop_mode("ela")
def _ela(image, *, crop_size, top_k, candidates, rng) -> list[np.ndarray]:
    """The absolute-cue control: rank by response to a fresh JPEG encode.

    Classic error-level analysis. Registered so the claim that a within-image
    contrast survives laundering better than a compression cue can be measured
    rather than asserted -- and, as it turns out, so that it can be corrected.
    """
    origins = random_origins(image, crop_size, max(candidates, top_k), rng)
    ranked = ela_rank_origins(image, origins, crop_size=crop_size)
    return [image[t : t + crop_size, l : l + crop_size] for t, l in ranked[:top_k]]
