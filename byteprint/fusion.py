"""Score-level fusion of the two experts.

The DINOv2 probe and the AEROBLADE reconstruction expert fail on different
inputs -- the probe leans on learned texture statistics, the reconstruction
expert on proximity to a decoder's output manifold -- so combining them buys
coverage rather than decimal places. Fusing at the *score* level keeps each
expert independently trainable, independently evaluable, and swappable, and it
means the ablation (probe only / recon only / fused) falls out for free.

The two caches are keyed identically but are not row-aligned: extraction runs
happen at different times, images can fail in one pass and not the other, and
augmentation draws different specs. So they are joined on key, never zipped.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from byteprint.cache import EmbeddingStore
from byteprint.metrics import threshold_at_fpr
from byteprint.probe import LinearProbe
from byteprint.recon import aeroblade_score


@dataclass(frozen=True, slots=True)
class JoinedCaches:
    """Rows present in both caches, aligned by key."""

    keys: list[str]
    dino: np.ndarray
    recon: np.ndarray
    labels: np.ndarray
    generators: list[str]
    specs: list[str]

    @property
    def n(self) -> int:
        return len(self.keys)


def join_caches(dino_store: EmbeddingStore, recon_store: EmbeddingStore) -> JoinedCaches:
    """Inner-join two caches on their shared keys."""
    dino_rows = {key: index for index, key in enumerate(dino_store.keys())}
    recon_rows = {key: index for index, key in enumerate(recon_store.keys())}
    shared = sorted(dino_rows.keys() & recon_rows.keys())

    dino_matrix, recon_matrix = dino_store.matrix(), recon_store.matrix()
    dino_labels, recon_labels = dino_store.labels(), recon_store.labels()
    generators, specs = dino_store.generators(), dino_store.specs()

    for key in shared:
        if dino_labels[dino_rows[key]] != recon_labels[recon_rows[key]]:
            raise ValueError(f"caches disagree on the label for {key!r}")

    dino_index = [dino_rows[key] for key in shared]
    recon_index = [recon_rows[key] for key in shared]

    return JoinedCaches(
        keys=shared,
        dino=dino_matrix[dino_index] if shared else np.zeros((0, dino_store.config.dim)),
        recon=recon_matrix[recon_index] if shared else np.zeros((0, recon_store.config.dim)),
        labels=dino_labels[dino_index] if shared else np.zeros((0,), dtype=np.int64),
        generators=[generators[i] for i in dino_index],
        specs=[specs[i] for i in dino_index],
    )


class FusedDetector:
    """A logistic regression over the two experts' scalar outputs."""

    def __init__(self, probe: LinearProbe, *, seed: int = 0) -> None:
        self.probe = probe
        self.seed = seed
        self.threshold: float = 0.5
        self.target_fpr: float | None = None
        self._model: LogisticRegression | None = None

    def _stack(self, dino: np.ndarray, recon: np.ndarray) -> np.ndarray:
        probe_scores = self.probe.score(dino)
        recon_scores = aeroblade_score(recon)
        if len(probe_scores) != len(recon_scores):
            raise ValueError(
                f"expert outputs must be the same length, got {len(probe_scores)} and "
                f"{len(recon_scores)}"
            )
        return np.column_stack([probe_scores, recon_scores])

    def fit(self, dino: np.ndarray, recon: np.ndarray, labels: np.ndarray) -> "FusedDetector":
        y = np.asarray(labels, dtype=np.int64)
        if len(dino) != len(y):
            raise ValueError(
                f"features and labels must be the same length, got {len(dino)} and {len(y)}"
            )
        if len(np.unique(y)) < 2:
            raise ValueError("fusion needs both classes present; got only one")

        # Standardising is unnecessary here: two columns, both already well scaled.
        self._model = LogisticRegression(max_iter=5000, random_state=self.seed)
        self._model.fit(self._stack(dino, recon), y)
        return self

    def score(self, dino: np.ndarray, recon: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("fused detector is not fitted; call fit() first")
        return self._model.predict_proba(self._stack(dino, recon))[:, 1]

    def component_scores(self, dino: np.ndarray, recon: np.ndarray) -> dict[str, np.ndarray]:
        """Each expert alone plus the fusion, for the ablation table."""
        return {
            "probe": self.probe.score(dino),
            "recon": aeroblade_score(recon),
            "fused": self.score(dino, recon),
        }

    def calibrate(
        self,
        dino: np.ndarray,
        recon: np.ndarray,
        labels: np.ndarray,
        *,
        target_fpr: float = 0.01,
    ) -> "FusedDetector":
        self.threshold = threshold_at_fpr(labels, self.score(dino, recon), target_fpr)
        self.target_fpr = target_fpr
        return self

    def predict(self, dino: np.ndarray, recon: np.ndarray) -> np.ndarray:
        return (self.score(dino, recon) >= self.threshold).astype(np.int64)

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "probe": self.probe,
                "model": self._model,
                "threshold": self.threshold,
                "target_fpr": self.target_fpr,
                "seed": self.seed,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path | str) -> "FusedDetector":
        payload = joblib.load(Path(path))
        detector = cls(payload["probe"], seed=payload["seed"])
        detector._model = payload["model"]
        detector.threshold = payload["threshold"]
        detector.target_fpr = payload["target_fpr"]
        return detector
