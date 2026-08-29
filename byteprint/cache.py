"""On-disk embedding cache.

Because the backbone is frozen, features are a pure function of (image,
laundering spec, extraction config). Computing them is the only expensive part
of this pipeline; training the probe on top is seconds. So features are
extracted once, pooled per image, and written here -- after which sweeping the
probe, recalibrating a threshold or running leave-one-generator-out costs
nothing.

Layout on disk::

    cache/
      config.json     extraction settings; a mismatch invalidates the cache
      records.jsonl   one line per stored image, in row order
      features.npy    (n_records, dim) float32, row i belongs to line i
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


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


def key_for(path: Path | str, spec: str) -> str:
    """Cache key for one image under one laundering spec.

    Uses size and modification time rather than a content hash: reading every
    byte of a large corpus just to decide what to skip defeats the point.
    """
    path = Path(path)
    stat = path.stat()
    return f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}|{spec}"


class EmbeddingStore:
    """Append-only store of pooled embeddings, resumable across runs."""

    def __init__(self, root: Path, config: ExtractConfig) -> None:
        self.root = Path(root)
        self.config = config
        self._features: list[np.ndarray] = []
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
            matrix = np.load(features_path)
            store._features = [row for row in matrix.astype(np.float32)]
            store._records = [
                json.loads(line) for line in records_path.read_text().splitlines() if line
            ]
            store._keys = {record["key"] for record in store._records}

        return store

    def flush(self) -> None:
        """Persist everything added so far."""
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "config.json").write_text(json.dumps(asdict(self.config), indent=2))
        np.save(self.root / "features.npy", self.matrix())
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
        """Mean-pool one image's crop embeddings into a single cached row."""
        if key in self._keys:
            raise ValueError(f"key {key!r} is already in the cache")

        crop_features = np.atleast_2d(np.asarray(crop_features, dtype=np.float32))
        if crop_features.shape[1] != self.config.dim:
            raise ValueError(
                f"expected embeddings of width {self.config.dim}, got {crop_features.shape[1]}"
            )

        self._features.append(crop_features.mean(axis=0))
        self._records.append(
            {
                "key": key,
                "path": str(path),
                "label": int(label),
                "generator": generator,
                "spec": spec,
            }
        )
        self._keys.add(key)

    # -- reading -----------------------------------------------------------

    def __len__(self) -> int:
        return len(self._records)

    def matrix(self) -> np.ndarray:
        if not self._features:
            return np.zeros((0, self.config.dim), dtype=np.float32)
        return np.stack(self._features).astype(np.float32)

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
