"""Classification heads: the part of the model that is actually trained.

The backbone is frozen, so *this* is where the learning happens, and it is the
knob most worth turning. Each head is a scikit-learn estimator exposing
``fit`` and ``predict_proba``; swapping one swaps the training objective --
log loss, hinge loss, or a small non-linear net -- without touching extraction,
caching or evaluation.

    from byteprint.heads import register_head

    @register_head("my-head")
    def _build(config):
        return SomeSklearnClassifier(random_state=config.seed)

Then ``byteprint train --head my-head --plugin myteam.heads``.

The default stays logistic regression, deliberately: deterministic, no training
loop to debug, fits in under a second, and its outputs are probabilities that
can be calibrated. Anything more expressive has to earn its variance against
that baseline, which is exactly what the registry makes cheap to check.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC

from byteprint.registry import Registry

if TYPE_CHECKING:  # pragma: no cover -- avoids a probe <-> heads import cycle
    from byteprint.probe import ProbeConfig

HeadFactory = Callable[["ProbeConfig"], BaseEstimator]

HEADS: Registry[HeadFactory] = Registry("head")

DEFAULT_HEAD = "logreg"


def register_head(name: str, *, replace: bool = False):
    """Decorator: register a factory taking a :class:`ProbeConfig`, returning an estimator.

    The estimator must implement ``fit(X, y)`` and ``predict_proba(X)``; a head
    without calibrated probabilities cannot be thresholded at a false-positive
    budget, which is the only operating point this project reports.
    """

    def decorator(factory: HeadFactory) -> HeadFactory:
        HEADS.register(name, factory, replace=replace)
        return factory

    return decorator


@register_head("logreg")
def _logistic_regression(config: "ProbeConfig") -> BaseEstimator:
    """Log loss. The baseline every other head is measured against."""
    return LogisticRegression(
        C=config.C,
        max_iter=config.max_iter,
        class_weight=config.class_weight,
        random_state=config.seed,
    )


@register_head("linear-svm")
def _linear_svm(config: "ProbeConfig") -> BaseEstimator:
    """Hinge loss -- a max-margin boundary, wrapped so it still emits probabilities.

    Worth a look because it cares only about the points near the boundary,
    which is where laundered images end up.
    """
    return CalibratedClassifierCV(
        LinearSVC(
            C=config.C,
            max_iter=config.max_iter,
            class_weight=config.class_weight,
        ),
        method="sigmoid",
        cv=3,
    )


@register_head("mlp")
def _mlp(config: "ProbeConfig") -> BaseEstimator:
    """One hidden layer, cross-entropy. Non-linear, and correspondingly noisier.

    ``C`` is reused as an inverse-regularisation knob so one flag tunes every
    head; class weighting is not supported by sklearn's MLP, so a head that
    needs it should resample instead.
    """
    return MLPClassifier(
        hidden_layer_sizes=(256,),
        alpha=1.0 / config.C,
        max_iter=config.max_iter,
        early_stopping=True,
        random_state=config.seed,
    )


def build_head(config: "ProbeConfig") -> BaseEstimator:
    """Instantiate the head named by ``config.head``."""
    return HEADS.resolve(config.head)(config)
