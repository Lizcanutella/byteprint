"""How one image's crops become one score.

An image is embedded as a *bag* of crops, and something has to reduce that bag
to a single number. For most of this project's life that reduction was one line
inside ``EmbeddingStore.add``::

    self._features.append(crop_features.mean(axis=0))

which made pooling the only swappable part of the pipeline that was not a
hyperparameter: the crop embeddings were gone by the time anything could ask a
question about them, so changing the reduction meant re-extracting the corpus.
The reduction now happens here, at train and eval time, over a cache that stores
crops individually -- after which mean, max and top-k are a sweep rather than
three extractions.

Why it matters is in ``docs/results-crop-localisation.md``. A fully-synthetic
image is synthetic in every crop, so averaging costs nothing. A *tampered* image
is a real photograph with one edited region: its evidence is localised by
construction, and averaging it against three crops of authentic content is the
wrong operator. Pointing crops at the edited region without fixing the pooling
made the tampered number *worse*, from two independent cues at once.

Poolings reduce in one of two spaces, and the distinction is the point rather
than an implementation detail:

``feature``
    reduce the crop embeddings, then call the head once. ``mean`` is this, and
    it is exactly what the cache used to store.
``score``
    call the head on every crop, then reduce the probabilities. A localised
    signal survives this and does not survive the other.

Registered like every other swappable part, so a teammate adds one from their
own module::

    @register_pooling("noisy-or", space="score")
    def _build(arg):
        return lambda scores: 1.0 - np.prod(1.0 - scores, axis=0)

    byteprint train --plugin myteam.pooling --pooling noisy-or
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from byteprint.registry import Registry

# A reduction over axis 0 of a bag: (n_crops, dim) -> (dim,) in feature space,
# (n_crops,) -> scalar in score space.
Reduce = Callable[[np.ndarray], np.ndarray]

FEATURE_SPACE = "feature"
SCORE_SPACE = "score"


@dataclass(frozen=True, slots=True)
class Pooling:
    """A resolved reduction, with the argument its name carried applied."""

    name: str
    space: str
    reduce: Reduce


@dataclass(frozen=True, slots=True)
class PoolingKind:
    """A registered template. ``build`` applies the ``name:arg`` argument."""

    name: str
    space: str
    build: Callable[[str | None], Reduce]
    usage: str


POOLINGS: Registry[PoolingKind] = Registry("pooling")

DEFAULT_POOLING = "mean"

# How training rows are formed from bags. Not a registry: there are two
# coherent answers and no reason to expect a third.
TRAIN_POOLINGS = ("mean", "crop")
DEFAULT_TRAIN_POOLING = "mean"


def register_pooling(
    name: str, *, space: str, usage: str | None = None, replace: bool = False
):
    """Decorator: register a factory taking the ``name:arg`` argument, or None.

    ``space`` decides *where* the reduction happens, and so how the head is
    used: ``feature`` pools embeddings and calls the head once per image,
    ``score`` calls the head per crop and pools the probabilities.
    """
    if space not in (FEATURE_SPACE, SCORE_SPACE):
        raise ValueError(f"space must be {FEATURE_SPACE!r} or {SCORE_SPACE!r}, got {space!r}")

    def decorator(build: Callable[[str | None], Reduce]) -> Callable[[str | None], Reduce]:
        POOLINGS.register(
            name,
            PoolingKind(name=name, space=space, build=build, usage=usage or name),
            replace=replace,
        )
        return build

    return decorator


def resolve_pooling(spec: str) -> Pooling:
    """Resolve ``"max"`` or ``"topk:2"`` into a :class:`Pooling`.

    The ``name:arg`` form is the one laundering specs already use (``jpeg:90``),
    borrowed rather than invented so there is one convention to learn.
    """
    name, _, argument = str(spec).partition(":")
    kind = POOLINGS.resolve(name)
    return Pooling(name=str(spec), space=kind.space, reduce=kind.build(argument or None))


# -- the reductions --------------------------------------------------------


@register_pooling("mean", space=FEATURE_SPACE)
def _mean(argument: str | None) -> Reduce:
    """Average the crop embeddings. The behaviour this project shipped with."""
    _reject_argument("mean", argument)
    return lambda values: values.mean(axis=0)


@register_pooling("mean-score", space=SCORE_SPACE)
def _mean_score(argument: str | None) -> Reduce:
    """Average the per-crop probabilities.

    Not the same operator as ``mean``: a logistic head is monotone but not
    affine, so averaging before it and averaging after it differ. Registered so
    a run can separate the *space* the reduction happens in from the reduction.
    """
    _reject_argument("mean-score", argument)
    return lambda values: values.mean(axis=0)


@register_pooling("max", space=SCORE_SPACE)
def _max(argument: str | None) -> Reduce:
    """The single most confident crop decides. One edited region is enough."""
    _reject_argument("max", argument)
    return lambda values: values.max(axis=0)


@register_pooling("topk", space=SCORE_SPACE, usage="topk:k")
def _topk(argument: str | None) -> Reduce:
    """Average the ``k`` most confident crops -- a max that one crop cannot own.

    ``k`` larger than a bag uses the whole bag rather than failing: a
    ``center``-mode bag holds one crop, and asking for its top four is a
    well-defined question with the obvious answer.
    """
    if argument is None:
        raise ValueError("topk needs a k, as in 'topk:2'")
    try:
        k = int(argument)
    except ValueError:
        raise ValueError(f"topk needs an integer k, got {argument!r}") from None
    if k < 1:
        raise ValueError(f"topk needs k >= 1, got {k}")

    def reduce(values: np.ndarray) -> np.ndarray:
        if values.shape[0] <= k:
            return values.mean(axis=0)
        # Partition rather than sort: only membership of the top k matters.
        kept = np.partition(values, -k, axis=0)[-k:]
        return kept.mean(axis=0)

    return reduce


def _reject_argument(name: str, argument: str | None) -> None:
    if argument is not None:
        raise ValueError(f"pooling {name!r} takes no argument, got {argument!r}")


# -- ragged bags -----------------------------------------------------------
#
# Bags are not all the same size and never were: `center` and `resize` return
# one crop whatever --crops says, and an image smaller than the crop size can
# yield fewer. Everything here takes counts explicitly rather than assuming a
# rectangle, because the rectangle is an assumption that holds until it quietly
# does not.


def bag_offsets(counts: np.ndarray) -> np.ndarray:
    """The row at which each bag starts."""
    counts = np.asarray(counts, dtype=np.int64)
    return np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)


def _check_counts(values: np.ndarray, counts: np.ndarray) -> np.ndarray:
    counts = np.asarray(counts, dtype=np.int64)
    total = int(counts.sum())
    if total != values.shape[0]:
        raise ValueError(
            f"counts describe {total} rows but {values.shape[0]} were given; "
            "a cache whose bag counts disagree with its features is corrupt"
        )
    return counts


def segment_reduce(values: np.ndarray, counts: np.ndarray, reduce: Reduce) -> np.ndarray:
    """Apply ``reduce`` to each consecutive bag of rows, one result per bag."""
    values = np.asarray(values)
    counts = _check_counts(values, counts)
    if counts.size == 0:
        return values[:0]

    return np.stack(
        [reduce(chunk) for chunk in np.split(values, np.cumsum(counts)[:-1])]
    )


def truncate_bags(
    values: np.ndarray, counts: np.ndarray, limit: int | None
) -> tuple[np.ndarray, np.ndarray]:
    """Keep each bag's first ``limit`` rows. Bags already shorter are untouched.

    This is what makes a two-crop run reproducible from an eight-crop cache.
    ``texture`` ranks a fixed candidate set and returns its head, so the first
    two of eight crops *are* the two a ``--crops 2`` extraction would have
    chosen -- the same pixels, not a similar sample. ``tests/test_crops.py``
    pins that, and pins that ``random`` does not have the property.
    """
    values = np.asarray(values)
    counts = _check_counts(values, counts)
    if limit is None:
        return values, counts
    if limit < 1:
        raise ValueError(f"crop limit must be at least 1, got {limit}")

    kept = np.minimum(counts, limit)
    starts = bag_offsets(counts)
    rows = np.concatenate(
        [np.arange(start, start + n) for start, n in zip(starts, kept)]
    ).astype(np.int64) if counts.size else np.zeros(0, dtype=np.int64)
    return values[rows], kept


def take_bags(
    values: np.ndarray, counts: np.ndarray, index: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Select whole bags by *image* index, keeping their crops together.

    Train/calibration splits, leave-one-generator-out and per-rung masks all
    index images, not crop rows. Selecting rows directly is the bug this
    exists to make unavailable: it would split an image's crops across the
    train and calibration halves.
    """
    values = np.asarray(values)
    counts = _check_counts(values, counts)
    index = np.asarray(index)
    if index.dtype == bool:
        index = np.flatnonzero(index)

    starts = bag_offsets(counts)
    if index.size == 0:
        return values[:0], counts[:0]
    rows = np.concatenate(
        [np.arange(starts[i], starts[i] + counts[i]) for i in index]
    ).astype(np.int64)
    return values[rows], counts[index]


def repeat_labels(labels: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """One label per crop, inherited from the crop's image.

    This is the label noise instance-level training accepts: most crops of a
    tampered image are authentic content wearing a synthetic label.
    """
    labels = np.asarray(labels)
    counts = np.asarray(counts, dtype=np.int64)
    if labels.shape[0] != counts.shape[0]:
        raise ValueError(
            f"got {labels.shape[0]} labels for {counts.shape[0]} bags"
        )
    return np.repeat(labels, counts)
