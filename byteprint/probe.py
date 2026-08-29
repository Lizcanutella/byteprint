"""The probe: a trained head over frozen features.

The default is a logistic regression, deliberately: it is deterministic, has
no training loop to debug, fits in under a second, and produces probabilities
that can be calibrated. At this scale a torch head buys nothing but variance.

The head is not hard-wired, though. ``ProbeConfig.head`` names an entry in
:data:`byteprint.heads.HEADS`, so the training objective -- log loss, hinge loss,
a small MLP, or something a teammate registers from their own module -- is one
flag, and everything downstream (calibration, the ladder, fusion) is unchanged.

Calibration is a first-class step rather than an afterthought. Probe scores
shift systematically between generators and between laundering paths, so a
threshold of 0.5 is close to meaningless; the operating point has to be fitted
on held-out data at the false-positive rate the deployment can actually afford.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler

from byteprint.heads import DEFAULT_HEAD, build_head
from byteprint.metrics import threshold_at_fpr


@dataclass(frozen=True, slots=True)
class ProbeConfig:
    head: str = DEFAULT_HEAD
    C: float = 1.0
    max_iter: int = 5000
    l2_normalize: bool = True
    class_weight: str | None = "balanced"
    seed: int = 0


class LinearProbe:
    """A trained head over frozen backbone embeddings, with a fitted threshold."""

    def __init__(self, config: ProbeConfig | None = None) -> None:
        self.config = config or ProbeConfig()
        self.threshold: float = 0.5
        self.target_fpr: float | None = None
        # The extraction settings the training features were built with. Carried
        # so a saved probe is self-contained: scoring a directory needs the same
        # backbone, crop size and crop mode, and asking the caller to remember
        # them is how you get a detector silently scored at the wrong resolution.
        self.extract_config = None
        self._pipeline: Pipeline | None = None

    def _build(self) -> Pipeline:
        steps = []
        if self.config.l2_normalize:
            # Row-wise unit norm first: embedding magnitude carries little signal
            # and varies with crop content.
            steps.append(("normalize", Normalizer(norm="l2")))
        steps.append(("scale", StandardScaler()))
        steps.append(("head", build_head(self.config)))
        return Pipeline(steps)

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "LinearProbe":
        X = np.asarray(features, dtype=np.float64)
        y = np.asarray(labels, dtype=np.int64)
        if len(X) != len(y):
            raise ValueError(f"features and labels must be the same length, got {len(X)} and {len(y)}")
        if len(np.unique(y)) < 2:
            raise ValueError("training needs both classes present; got only one")

        self._pipeline = self._build().fit(X, y)
        return self

    def score(self, features: np.ndarray) -> np.ndarray:
        """Probability that each row is synthetic."""
        if self._pipeline is None:
            raise RuntimeError("probe is not fitted; call fit() first")
        X = np.asarray(features, dtype=np.float64)
        return self._pipeline.predict_proba(X)[:, 1]

    def calibrate(
        self, features: np.ndarray, labels: np.ndarray, *, target_fpr: float = 0.01
    ) -> "LinearProbe":
        """Set the decision threshold from held-out data at a false-positive budget."""
        scores = self.score(features)
        self.threshold = threshold_at_fpr(labels, scores, target_fpr)
        self.target_fpr = target_fpr
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        return (self.score(features) >= self.threshold).astype(np.int64)

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "config": self.config,
                "pipeline": self._pipeline,
                "threshold": self.threshold,
                "target_fpr": self.target_fpr,
                "extract_config": self.extract_config,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path | str) -> "LinearProbe":
        payload = joblib.load(Path(path))
        probe = cls(payload["config"])
        probe._pipeline = payload["pipeline"]
        probe.threshold = payload["threshold"]
        probe.target_fpr = payload["target_fpr"]
        probe.extract_config = payload.get("extract_config")
        return probe
