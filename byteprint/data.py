"""Dataset discovery over the on-disk folder layout.

A split directory looks like::

    train/
      real/                 # any nesting; every image is label 0
        a.jpg
      fake/
        sdxl/               # one directory per generator
          c.png
        flux/
          e.png

Images placed directly under ``fake/`` are kept with the generator name
``unknown`` so a quick unlabelled dump still trains, just without
per-generator reporting.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"})

REAL = 0
FAKE = 1

REAL_GENERATOR = "real"
UNKNOWN_GENERATOR = "unknown"


@dataclass(frozen=True, slots=True)
class Sample:
    """One image on disk, with the two labels we can derive from its path."""

    path: Path
    label: int
    generator: str


def _images_under(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)


def scan_images(directory: Path | str) -> list[Path]:
    """Every image under ``directory``, recursively, in a stable order.

    Unlike :func:`scan_split` this derives no labels -- it is the inference
    path, where a directory is just a pile of images to score.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"image directory does not exist: {directory}")
    return _images_under(directory)


def scan_split(root: Path | str) -> list[Sample]:
    """Return every image under ``root``, labelled by its position in the tree."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"split directory does not exist: {root}")

    samples = [
        Sample(path=path, label=REAL, generator=REAL_GENERATOR)
        for path in _images_under(root / "real")
    ]

    fake_root = root / "fake"
    for path in _images_under(fake_root):
        relative = path.relative_to(fake_root)
        generator = relative.parts[0] if len(relative.parts) > 1 else UNKNOWN_GENERATOR
        samples.append(Sample(path=path, label=FAKE, generator=generator))

    return sorted(samples, key=lambda s: s.path)


def generators(samples: list[Sample]) -> list[str]:
    """Sorted names of the fake generators present in ``samples``."""
    return sorted({s.generator for s in samples if s.label == FAKE})


def leave_one_generator_out(
    samples: list[Sample], held_out: str
) -> tuple[list[Sample], list[Sample]]:
    """Split into (train, held-out) where no fake from ``held_out`` is in train.

    Real images stay in train: holding out a generator tests transfer to unseen
    synthesis, not to unseen photographs.
    """
    available = generators(samples)
    if held_out not in available:
        raise ValueError(
            f"unknown generator {held_out!r}; dataset has {available or ['<none>']}"
        )

    train = [s for s in samples if s.generator != held_out]
    held = [s for s in samples if s.generator == held_out]
    return train, held
