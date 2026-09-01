"""On-disk embedding cache.

Because the backbone is frozen, features are a pure function of (image,
laundering spec, extraction config). Computing them is the only expensive part
of this pipeline; training the probe on top is seconds. So features are
extracted once and written here -- after which sweeping the probe, recalibrating
a threshold or running leave-one-generator-out costs nothing.

**Schema 2 stores an image's crops individually.** The previous format averaged
them on the way in, which made pooling the only swappable part of this pipeline
that was not a hyperparameter -- the crop embeddings were gone by the time
anything could ask a question about them. Storing the bag and reducing it on
read costs ``crops_per_image`` times the disk and buys a pooling sweep over one
extraction. See ``byteprint/pooling.py`` and ``docs/design-crop-pooling.md``.

Layout on disk::

    cache/
      config.json     extraction settings + schema; a mismatch invalidates it
      records.jsonl   one line per stored image, in row order, with n_crops
      features.npy    (total_crops, dim) float32

An image's rows are ``features[offset : offset + n_crops]``, where the offsets
are a cumulative sum over the records. There is deliberately no second index
file: row order keeps one source of truth.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

# 1: one mean-pooled row per image. 2: one row per crop, plus n_crops per record.
SCHEMA_VERSION = 2


class StaleCacheError(RuntimeError):
    """Raised when a cache on disk was built with different extraction settings."""


@dataclass(frozen=True, slots=True)
class ExtractConfig:
    """Everything that changes the value of an embedding."""

    backbone: str
    dim: int
    crop_size: int
    crops_per_image: int
    crop_mode: str
    seed: int


def read_config(root: Path | str) -> ExtractConfig:
    """The extraction settings a cache was built with, schema key stripped."""
    stored = json.loads((Path(root) / "config.json").read_text())
    stored.pop("schema", None)
    return ExtractConfig(**stored)


def key_for(path: Path | str, spec: str) -> str:
    """Cache key for one image under one laundering spec.

    Uses size and modification time rather than a content hash: reading every
    byte of a large corpus just to decide what to skip defeats the point.
    """
    path = Path(path)
    stat = path.stat()
    return f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}|{spec}"


class EmbeddingStore:
    """Append-only store of per-crop embeddings, resumable across runs."""

    def __init__(self, root: Path, config: ExtractConfig) -> None:
        self.root = Path(root)
        self.config = config
        # One entry per image: the (n_crops, dim) block that image contributed.
        self._bags: list[np.ndarray] = []
        self._records: list[dict] = []
        self._keys: set[str] = set()

    # -- lifecycle ---------------------------------------------------------

    @classmethod
    def open(
        cls, root: Path | str, config: ExtractConfig, *, rebuild: bool = False
    ) -> "EmbeddingStore":
        """Open a cache directory, resuming its contents when the config matches."""
        root = Path(root)
        store = cls(root, config)
        config_path = root / "config.json"

        if not config_path.exists():
            return store

        stored = json.loads(config_path.read_text())
        schema = stored.pop("schema", 1)

        if schema != SCHEMA_VERSION:
            if not rebuild:
                raise StaleCacheError(
                    f"cache at {root} stores one pooled row per image "
                    f"(schema {schema}); pooling needs per-crop rows "
                    f"(schema {SCHEMA_VERSION}). Re-run `byteprint extract`, "
                    f"or pass rebuild=True to discard it"
                )
            return store

        if stored != asdict(config):
            if not rebuild:
                differing = sorted(
                    k for k, v in asdict(config).items() if stored.get(k) != v
                )
                raise StaleCacheError(
                    f"cache at {root} was built with different settings "
                    f"({', '.join(differing)} changed); pass rebuild=True to discard it"
                )
            return store

        features_path = root / "features.npy"
        records_path = root / "records.jsonl"
        if features_path.exists() and records_path.exists():
            matrix = np.load(features_path).astype(np.float32)
            store._records = [
                json.loads(line) for line in records_path.read_text().splitlines() if line
            ]
            counts = [int(record["n_crops"]) for record in store._records]
            if sum(counts) != len(matrix):
                raise StaleCacheError(
                    f"cache at {root} is corrupt: its records account for "
                    f"{sum(counts)} crop rows but features.npy holds {len(matrix)}"
                )
            store._bags = list(np.split(matrix, np.cumsum(counts)[:-1]))
            store._keys = {record["key"] for record in store._records}

        return store

    def flush(self) -> None:
        """Persist everything added so far."""
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "config.json").write_text(
            json.dumps({**asdict(self.config), "schema": SCHEMA_VERSION}, indent=2)
        )
        np.save(self.root / "features.npy", self.crop_matrix())
        with (self.root / "records.jsonl").open("w") as handle:
            for record in self._records:
                handle.write(json.dumps(record) + "\n")

    # -- writing -----------------------------------------------------------

    def has(self, key: str) -> bool:
        return key in self._keys

    def add(
        self,
        key: str,
        crop_features: np.ndarray,
        *,
        path: Path | str,
        label: int,
        generator: str,
        spec: str,
    ) -> None:
        """Store one image's crop embeddings, keeping each crop as its own row."""
        if key in self._keys:
            raise ValueError(f"key {key!r} is already in the cache")

        crop_features = np.atleast_2d(np.asarray(crop_features, dtype=np.float32))
        if crop_features.shape[1] != self.config.dim:
            raise ValueError(
                f"expected embeddings of width {self.config.dim}, got {crop_features.shape[1]}"
            )
        if crop_features.shape[0] < 1:
            raise ValueError(f"{key!r} has no crops; an image contributes at least one row")

        self._bags.append(crop_features)
        self._records.append(
            {
                "key": key,
                "path": str(path),
                "label": int(label),
                "generator": generator,
                "spec": spec,
                "n_crops": int(crop_features.shape[0]),
            }
        )
        self._keys.add(key)

    # -- reading -----------------------------------------------------------

    def __len__(self) -> int:
        return len(self._records)

    def crop_matrix(self) -> np.ndarray:
        """Every crop of every image, ``(total_crops, dim)``, in record order."""
        if not self._bags:
            return np.zeros((0, self.config.dim), dtype=np.float32)
        return np.concatenate(self._bags).astype(np.float32)

    def crop_counts(self) -> np.ndarray:
        """How many rows of :meth:`crop_matrix` belong to each record."""
        return np.asarray([r["n_crops"] for r in self._records], dtype=np.int64)

    def matrix(self) -> np.ndarray:
        """One mean-pooled row per image.

        Kept, with its meaning unchanged, because it is what `fusion`, `logo`
        and the pooling-free paths ask for. The averaging simply happens on
        read now instead of on write, where it could not be undone.
        """
        if not self._bags:
            return np.zeros((0, self.config.dim), dtype=np.float32)
        return np.stack([bag.mean(axis=0) for bag in self._bags]).astype(np.float32)

    def labels(self) -> np.ndarray:
        return np.asarray([r["label"] for r in self._records], dtype=np.int64)

    def generators(self) -> list[str]:
        return [r["generator"] for r in self._records]

    def specs(self) -> list[str]:
        return [r["spec"] for r in self._records]

    def paths(self) -> list[str]:
        return [r["path"] for r in self._records]

    def keys(self) -> list[str]:
        """Cache keys in row order, for joining two caches of the same images."""
        return [r["key"] for r in self._records]
