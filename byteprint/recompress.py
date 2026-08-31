"""Put every class through the same encoder, so compression history cannot classify.

SID_Set's reals come from OpenImages as JPEG; its fully-synthetic images are
PNG. :mod:`byteprint.sid_set` already equalises the *container* by writing all
three classes as PNG -- but a PNG of a decoded JPEG still carries that JPEG's
quantisation artifacts in its pixels, and a PNG of a diffusion sample does not.
A detector can learn "has 8x8 block structure" and score 0.90 without ever
looking at a generator fingerprint.

This module is the control for that. It re-encodes an existing split through
one lossy encoder, so both classes carry the same kind of damage, and the run
can be repeated end to end and compared against the baseline.

**What the control does and does not establish.** After it, the presence of
JPEG artifacts is no longer a free discriminator. It does *not* make the two
classes' compression histories identical: the reals are now JPEG -> JPEG
(double-compressed) while the synthetics are PNG -> JPEG (single). Double-JPEG
is itself detectable, so a score that stays high is evidence against the
simplest shortcut, not proof of none. Getting further would mean matching the
reals' original quantisation tables, which SID_Set does not record and which
vary image to image.

The split is rewritten rather than edited in place: keeping both trees is what
lets the two runs be compared, and an in-place pass that dies halfway leaves a
corpus in two states with no way to tell which images are which.
"""

from __future__ import annotations

import io
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from byteprint.data import scan_split

#: The containers this is allowed to write. Anything else is refused rather
#: than guessed -- a control run's encoder is part of what the result means.
SUFFIXES: dict[str, str] = {"JPEG": ".jpg", "PNG": ".png"}

DEFAULT_ENCODING = "jpeg:95"


def parse_encoding(spec: str) -> tuple[str, int | None]:
    """``"jpeg:95"`` -> ``("JPEG", 95)``; ``"png"`` -> ``("PNG", None)``.

    A lossy format must name its quality. Defaulting it would make two runs
    quietly incomparable, which is the one thing a control cannot afford.
    """
    name, _, raw_quality = spec.strip().lower().partition(":")
    fmt = {"jpeg": "JPEG", "jpg": "JPEG", "png": "PNG"}.get(name)
    if fmt is None:
        raise ValueError(f"unknown encoding {spec!r}; expected 'png' or 'jpeg:<quality>'")

    if fmt == "PNG":
        if raw_quality:
            raise ValueError(f"png is lossless and takes no quality, got {spec!r}")
        return fmt, None

    if not raw_quality:
        raise ValueError(f"jpeg needs an explicit quality, e.g. 'jpeg:95' (got {spec!r})")
    try:
        quality = int(raw_quality)
    except ValueError as exc:
        raise ValueError(f"non-numeric jpeg quality in {spec!r}") from exc
    if not 1 <= quality <= 100:
        raise ValueError(f"jpeg quality must be in [1, 100], got {quality}")
    return fmt, quality


def encode_image(payload: bytes, encoding: str = DEFAULT_ENCODING) -> tuple[bytes, str]:
    """Decode an image of any container and re-encode it as RGB in ``encoding``.

    Returns the bytes and the suffix they should be written under, because the
    two must never disagree -- a ``.png`` holding JPEG bytes is exactly the kind
    of quiet inconsistency this module exists to remove.
    """
    fmt, quality = parse_encoding(encoding)
    with Image.open(io.BytesIO(payload)) as handle:
        rgb = handle.convert("RGB")
        buffer = io.BytesIO()
        if quality is None:
            rgb.save(buffer, format=fmt)
        else:
            # subsampling=0 keeps full chroma resolution: at quality 95 the
            # default 4:2:0 would throw away colour detail that the reals'
            # original encode may well have kept, adding a difference between
            # the classes rather than removing one.
            rgb.save(buffer, format=fmt, quality=quality, subsampling=0)
    return buffer.getvalue(), SUFFIXES[fmt]


@dataclass(slots=True)
class RecompressStats:
    """What a recompression pass wrote, skipped and refused."""

    encoding: str = DEFAULT_ENCODING
    written: int = 0
    skipped: int = 0
    failed: int = 0
    #: ``(generator, original container) -> count``. The imbalance this reports
    #: is the reason the pass exists, so a reviewer can confirm it was needed.
    source_formats: dict[tuple[str, str], int] = field(default_factory=dict)

    def render(self) -> str:
        formats = ", ".join(
            f"{generator}/{fmt}: {count}"
            for (generator, fmt), count in sorted(self.source_formats.items())
        )
        return (
            f"re-encoded {self.written} images as {self.encoding} "
            f"({self.skipped} already present, {self.failed} failed)\n"
            f"  source containers: {formats or 'none'}"
        )


def _destination(source: Path, src_root: Path, dst_root: Path, suffix: str) -> Path:
    return (dst_root / source.relative_to(src_root)).with_suffix(suffix)


def recompress_split(
    src_root: Path | str,
    dst_root: Path | str,
    *,
    encoding: str = DEFAULT_ENCODING,
    workers: int = 1,
) -> RecompressStats:
    """Re-encode every image of a ``real/`` + ``fake/<generator>/`` split.

    The label tree is preserved exactly, so the output is a split the engine
    reads unchanged and holds the same images as the input -- a control that
    also changed *which* images are in the corpus would confound what it is
    trying to isolate.

    Images already present in the destination are skipped, so a pass killed by
    a wall-clock limit resumes instead of repeating.
    """
    src_root, dst_root = Path(src_root), Path(dst_root)
    samples = scan_split(src_root)  # raises if the split does not exist
    _, quality = parse_encoding(encoding)
    stats = RecompressStats(encoding=encoding)
    formats: Counter[tuple[str, str]] = Counter()

    def work(sample) -> tuple[str, str] | None:
        """Re-encode one image. Returns its source format, or None if skipped."""
        payload = sample.path.read_bytes()
        with Image.open(io.BytesIO(payload)) as handle:
            source_format = handle.format or "unknown"
        encoded, suffix = encode_image(payload, encoding)
        destination = _destination(sample.path, src_root, dst_root, suffix)
        if destination.exists():
            return None
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(encoded)
        return sample.generator, source_format

    def account(sample, outcome: tuple[str, str] | None) -> None:
        if outcome is None:
            stats.skipped += 1
        else:
            stats.written += 1
            formats[outcome] += 1

    if workers > 1:
        # Encoding is PIL, which releases the GIL, so threads genuinely overlap.
        # Results are consumed in submission order: the outputs are independent
        # files, but the counters must not depend on thread scheduling.
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [(sample, pool.submit(work, sample)) for sample in samples]
            for sample, future in futures:
                try:
                    account(sample, future.result())
                except (UnidentifiedImageError, OSError, ValueError):
                    stats.failed += 1
    else:
        for sample in samples:
            try:
                account(sample, work(sample))
            except (UnidentifiedImageError, OSError, ValueError):
                stats.failed += 1

    stats.source_formats = dict(formats)
    return stats
