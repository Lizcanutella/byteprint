"""Extraction: images on disk -> pooled embeddings in the cache.

This is the only expensive stage, so it is the only one that resumes. Every
image is keyed by content signature plus laundering spec, so interrupting a run
and restarting it picks up where it left off, and adding a new generator to the
dataset only costs the new images.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol, Sequence

import numpy as np
from PIL import Image

from byteprint.cache import EmbeddingStore, key_for
from byteprint.crops import select_crops
from byteprint.data import Sample
from byteprint.launder import NO_OP, sample_spec

log = logging.getLogger(__name__)


class Backbone(Protocol):
    name: str
    dim: int

    def embed(self, crops: Sequence[np.ndarray]) -> np.ndarray: ...


@dataclass(frozen=True, slots=True)
class ExtractStats:
    added: int = 0
    skipped: int = 0
    failed: int = 0

    def render(self) -> str:
        return f"extracted {self.added}, skipped {self.skipped} cached, {self.failed} failed"


def load_image(path: Path | str) -> np.ndarray:
    """Read an image as an RGB uint8 array."""
    with Image.open(path) as handle:
        return np.asarray(handle.convert("RGB"))


def _specs_for(specs: Sequence[str], augment: int, rng: np.random.Generator) -> list[str]:
    """The laundering views to build for one image.

    Augmentation draws *distinct* chains: two identical views would collide on
    the same cache key, silently costing the caller a view they asked for.
    """
    if augment <= 0:
        return list(specs)

    drawn: list[str] = []
    seen: set[str] = set()
    for _ in range(augment * 20):
        if len(drawn) == augment:
            break
        candidate = sample_spec(rng)
        if candidate not in seen:
            seen.add(candidate)
            drawn.append(candidate)
    return drawn


def extract(
    samples: Iterable[Sample],
    *,
    backbone: Backbone,
    store: EmbeddingStore,
    specs: Sequence[str] = (NO_OP,),
    augment: int = 0,
    seed: int = 0,
    skip_errors: bool = True,
    log_every: int = 200,
) -> ExtractStats:
    """Embed every sample under every laundering spec, writing into ``store``.

    ``augment=n`` replaces the fixed ``specs`` with n randomly drawn laundering
    chains per image -- the training-time path. A fixed ``specs`` list is the
    evaluation path, where each rung of the ladder is scored separately.
    """
    config = store.config
    if backbone.dim != config.dim:
        raise ValueError(
            f"backbone {backbone.name!r} produces width {backbone.dim}, "
            f"but this cache stores width {config.dim}"
        )

    from byteprint.launder import apply as launder

    rng = np.random.default_rng(seed)
    added = skipped = failed = 0

    for index, sample in enumerate(samples):
        for spec in _specs_for(specs, augment, rng):
            try:
                key = key_for(sample.path, spec)
            except OSError:
                failed += 1
                if not skip_errors:
                    raise
                continue

            if store.has(key):
                skipped += 1
                continue

            try:
                image = load_image(sample.path)
                if spec != NO_OP:
                    image = launder(image, spec, seed=seed + index)
                crops = select_crops(
                    image,
                    crop_size=config.crop_size,
                    top_k=config.crops_per_image,
                    mode=config.crop_mode,
                    seed=config.seed + index,
                )
                features = backbone.embed(crops)
            except (OSError, ValueError) as exc:
                failed += 1
                if not skip_errors:
                    raise
                log.warning("skipping %s (%s): %s", sample.path, spec, exc)
                continue

            store.add(
                key,
                features,
                path=sample.path,
                label=sample.label,
                generator=sample.generator,
                spec=spec,
            )
            added += 1

        if log_every and added and added % log_every == 0:
            log.info("extracted %d embeddings", added)

    return ExtractStats(added=added, skipped=skipped, failed=failed)
