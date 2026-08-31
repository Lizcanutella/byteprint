"""The competition's required interface: an image directory in, a JSON file out.

Section 5.5 of the brief asks for *a script that takes an image directory and
outputs a confidence score per image -- a JSON file with ``image_path`` and
``pred`` for each image*, where ``pred`` is the likelihood the image is
AI-generated. This module is that, and `byteprint score` is its command line.

Two decisions worth stating, because both are visible in the output file:

**Every discovered image gets exactly one entry.** A directory of ten thousand
images will contain a truncated download or a mislabelled ``.png`` that is
really HTML, and a scorer that dies on the first one has scored nothing. A file
that cannot be read is reported with ``pred`` = :data:`UNSCORABLE` (0.5, maximum
uncertainty) and an ``error`` field saying why, so the output stays complete and
the failure stays visible rather than being quietly dropped. ``strict=True``
turns the first failure back into an exception.

**The scorer is a protocol, not a class hierarchy.** :class:`ProbeScorer` maps
one image to one probability using the frozen backbone and the trained head;
anything else with a ``score_image`` method -- a fused two-expert scorer, an
ensemble over crops -- drops in without touching the directory walk, the error
handling or the JSON writer.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

from byteprint.cache import ExtractConfig
from byteprint.crops import select_crops
from byteprint.data import scan_images
from byteprint.pipeline import load_image
from byteprint.prefetch import ordered_prefetch
from byteprint.probe import LinearProbe

log = logging.getLogger(__name__)

# The score given to an image that could not be read. Maximum uncertainty: it
# adds no evidence either way, which is the honest answer when we never saw it.
UNSCORABLE = 0.5


@dataclass(frozen=True, slots=True)
class Prediction:
    """One row of the output file."""

    image_path: str
    pred: float
    error: str | None = None

    def to_record(self) -> dict:
        record = {"image_path": self.image_path, "pred": self.pred}
        if self.error is not None:
            record["error"] = self.error
        return record


class ImageScorer(Protocol):
    """Maps a batch of RGB uint8 images to one AIGC likelihood each."""

    def score_images(self, images: Sequence[np.ndarray]) -> np.ndarray: ...


class ProbeScorer:
    """Frozen backbone + trained head, pooled over crops.

    Crops are taken with exactly the settings the probe was trained under --
    the config travels inside the saved probe, because scoring at a different
    crop size than you trained at degrades quietly rather than loudly.
    """

    def __init__(
        self,
        *,
        backbone,
        probe: LinearProbe,
        config: ExtractConfig,
        workers: int = 1,
    ) -> None:
        if backbone.dim != config.dim:
            raise ValueError(
                f"backbone {backbone.name!r} produces width {backbone.dim}, "
                f"but this probe was trained on width {config.dim}"
            )
        if workers < 1:
            raise ValueError(f"workers must be at least 1, got {workers}")
        self.backbone = backbone
        self.probe = probe
        self.config = config
        self.workers = workers

    def score_images(self, images: Sequence[np.ndarray]) -> np.ndarray:
        """One probability per image, pooling that image's crops before scoring."""
        if not images:
            return np.zeros((0,), dtype=np.float64)

        # Crop every image first, then embed the whole chunk in one call: the
        # backbone is far more efficient on a full batch than on four crops.
        #
        # Choosing crops costs more than decoding the image did -- scoring 32
        # candidate windows by Laplacian variance is most of the CPU time here --
        # so it runs on a pool. The seed stays tied to the image's position in
        # the batch, so the crops chosen do not depend on the worker count.
        def crops_for(item: tuple[int, np.ndarray]) -> list[np.ndarray]:
            index, image = item
            return select_crops(
                image,
                crop_size=self.config.crop_size,
                top_k=self.config.crops_per_image,
                mode=self.config.crop_mode,
                seed=self.config.seed + index,
            )

        crops_per_image = [
            prepared.result()
            for _, prepared in ordered_prefetch(
                enumerate(images), crops_for, workers=self.workers
            )
        ]

        flat = [crop for crops in crops_per_image for crop in crops]
        embedded = self.backbone.embed(flat)

        pooled, start = [], 0
        for crops in crops_per_image:
            pooled.append(embedded[start : start + len(crops)].mean(axis=0))
            start += len(crops)

        return np.asarray(self.probe.score(np.stack(pooled)), dtype=np.float64)


def score_directory(
    directory: Path | str,
    *,
    scorer: ImageScorer,
    relative: bool = False,
    chunk_size: int = 8,
    strict: bool = False,
    log_every: int = 200,
    workers: int = 1,
) -> list[Prediction]:
    """Score every image under ``directory``, one :class:`Prediction` each.

    ``relative`` reports paths relative to ``directory`` rather than absolute,
    for harnesses whose ground truth is keyed on bare filenames.

    ``workers`` decodes images ahead of the scorer on a thread pool. The output
    file is identical whatever it is set to -- see the comment on chunking
    below -- so it trades only wall clock.
    """
    directory = Path(directory).resolve()
    paths = scan_images(directory)
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    if workers < 1:
        raise ValueError(f"workers must be at least 1, got {workers}")

    def name_of(path: Path) -> str:
        return str(path.relative_to(directory)) if relative else str(path)

    def score_chunk(loaded: list[tuple[Path, np.ndarray]]) -> list[Prediction]:
        try:
            scores = scorer.score_images([image for _, image in loaded])
        except (OSError, ValueError) as exc:
            if strict:
                raise
            # Fall back to one at a time so a single bad image costs one
            # score, not the whole chunk.
            scores = []
            for path, image in loaded:
                try:
                    scores.append(float(scorer.score_images([image])[0]))
                except (OSError, ValueError) as inner:
                    log.warning("cannot score %s: %s", path, inner)
                    scores.append(None)
            del exc

        scored: list[Prediction] = []
        for (path, _), score in zip(loaded, scores):
            if score is None:
                scored.append(
                    Prediction(name_of(path), UNSCORABLE, error="could not be scored")
                )
            else:
                scored.append(Prediction(name_of(path), float(score)))
        return scored

    # Decoding runs ahead of the scorer on a pool, in path order. Chunks are
    # still cut at the same places -- every `chunk_size` *paths*, failures
    # included -- because an image's crops are seeded by its position within
    # its chunk, and a prediction must not depend on the worker count.
    predictions: list[Prediction] = []
    loaded: list[tuple[Path, np.ndarray]] = []
    failures: list[Prediction] = []

    def flush() -> None:
        if loaded:
            predictions.extend(score_chunk(loaded))
        predictions.extend(failures)
        loaded.clear()
        failures.clear()

        if log_every and predictions and len(predictions) % log_every < chunk_size:
            log.info("scored %d of %d images", len(predictions), len(paths))

    stream = ordered_prefetch(paths, load_image, workers=workers)
    for offset, (path, decoded) in enumerate(stream):
        # A file that cannot be decoded must not take down the batch it
        # happened to share.
        try:
            loaded.append((path, decoded.result()))
        except (OSError, ValueError) as exc:
            if strict:
                raise
            log.warning("cannot read %s: %s", path, exc)
            failures.append(Prediction(name_of(path), UNSCORABLE, error=str(exc)))

        if (offset + 1) % chunk_size == 0:
            flush()

    flush()

    # Stable, path-sorted output: a diff between two runs should show score
    # changes, not reordering.
    return sorted(predictions, key=lambda p: p.image_path)


def write_predictions(predictions: Sequence[Prediction], path: Path | str) -> Path:
    """Write the JSON file the brief asks for. Returns the path written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([p.to_record() for p in predictions], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path
