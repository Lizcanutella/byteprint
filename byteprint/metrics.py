"""Evaluation metrics.

AUC answers "does the detector rank synthetic above real?" -- useful, but not
the deployment question. On a real platform authentic images outnumber
synthetic ones enormously, so the number that decides whether a detector is
usable is the true-positive rate at a false-positive rate you can actually
afford. Both are reported here, plus a per-generator breakdown, because an
average over generators hides the one you cannot detect at all.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

REAL = 0
FAKE = 1


def _as_arrays(labels: Iterable[int], scores: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(list(labels), dtype=np.int64)
    s = np.asarray(list(scores), dtype=np.float64)
    if y.shape != s.shape:
        raise ValueError(f"labels and scores must be the same length, got {y.size} and {s.size}")
    if y.size == 0:
        raise ValueError("no samples to evaluate")
    return y, s


def roc_auc(labels: Iterable[int], scores: Iterable[float]) -> float:
    """Area under the ROC curve, via the rank-sum identity (ties share mid-ranks)."""
    y, s = _as_arrays(labels, scores)
    n_pos = int((y == FAKE).sum())
    n_neg = int((y == REAL).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUC needs both classes present; got only one")

    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(s.size, dtype=np.float64)
    ranks[order] = np.arange(1, s.size + 1, dtype=np.float64)

    # Average the ranks inside each tied group so ties count as half-credit.
    sorted_scores = s[order]
    start = 0
    for end in range(1, s.size + 1):
        if end == s.size or sorted_scores[end] != sorted_scores[start]:
            if end - start > 1:
                ranks[order[start:end]] = ranks[order[start:end]].mean()
            start = end

    return float((ranks[y == FAKE].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def threshold_at_fpr(labels: Iterable[int], scores: Iterable[float], target_fpr: float) -> float:
    """Lowest decision threshold whose false-positive rate stays within budget.

    Samples are called synthetic when ``score >= threshold``.
    """
    if not 0.0 <= target_fpr <= 1.0:
        raise ValueError(f"target_fpr must be in [0, 1], got {target_fpr}")
    y, s = _as_arrays(labels, scores)
    negatives = np.sort(s[y == REAL])
    if negatives.size == 0:
        raise ValueError("cannot calibrate a threshold without real samples")

    allowed = int(math.floor(target_fpr * negatives.size + 1e-9))
    if allowed >= negatives.size:
        return float(-np.inf)
    # Keep the (allowed) highest-scoring negatives; exclude every one below them.
    cutoff = negatives[negatives.size - allowed - 1]
    return float(np.nextafter(cutoff, np.inf))


def tpr_at_fpr(labels: Iterable[int], scores: Iterable[float], target_fpr: float) -> float:
    """True-positive rate once the threshold is set to respect ``target_fpr``."""
    y, s = _as_arrays(labels, scores)
    positives = s[y == FAKE]
    if positives.size == 0:
        raise ValueError("cannot measure TPR without synthetic samples")
    return float((positives >= threshold_at_fpr(y, s, target_fpr)).mean())


@dataclass(frozen=True, slots=True)
class GeneratorScore:
    """How the detector does on one generator, judged against all real images."""

    auc: float
    n_fake: int


@dataclass(frozen=True, slots=True)
class Report:
    auc: float
    n_real: int
    n_fake: int
    tpr_at_fpr: dict[float, float] = field(default_factory=dict)
    per_generator: dict[str, GeneratorScore] = field(default_factory=dict)
    label: str = ""

    def render(self) -> str:
        """A compact text table, for the CLI and for logs."""
        head = f"{self.label or 'overall'}: AUC {self.auc:.4f}"
        head += f"  ({self.n_real} real / {self.n_fake} fake)"
        lines = [head]
        for target in sorted(self.tpr_at_fpr, reverse=True):
            lines.append(f"  TPR @ {target:.1%} FPR   {self.tpr_at_fpr[target]:.4f}")
        if self.per_generator:
            width = max(len(name) for name in self.per_generator)
            lines.append(f"  {'generator'.ljust(width)}   AUC      n")
            for name in sorted(self.per_generator, key=lambda k: self.per_generator[k].auc):
                score = self.per_generator[name]
                lines.append(f"  {name.ljust(width)}   {score.auc:.4f}   {score.n_fake}")
        return "\n".join(lines)


def evaluate(
    labels: Iterable[int],
    scores: Iterable[float],
    *,
    generators: Sequence[str] | None = None,
    fpr_targets: Sequence[float] = (0.01, 0.001),
    label: str = "",
) -> Report:
    """Full evaluation: overall AUC, TPR at each budget, and a per-generator split."""
    y, s = _as_arrays(labels, scores)

    per_generator: dict[str, GeneratorScore] = {}
    if generators is not None:
        names = np.asarray(list(generators))
        if names.shape != y.shape:
            raise ValueError("labels and generators must be the same length")
        is_real = y == REAL
        for name in sorted(set(names[y == FAKE].tolist())):
            subset = is_real | (names == name)
            per_generator[name] = GeneratorScore(
                auc=roc_auc(y[subset], s[subset]),
                n_fake=int((names == name).sum()),
            )

    return Report(
        auc=roc_auc(y, s),
        n_real=int((y == REAL).sum()),
        n_fake=int((y == FAKE).sum()),
        tpr_at_fpr={float(t): tpr_at_fpr(y, s, t) for t in fpr_targets},
        per_generator=per_generator,
        label=label,
    )
