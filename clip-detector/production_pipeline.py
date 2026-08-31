"""
The production detection pipeline, promoted after comparing several
alternatives (see README "Model iteration" section and
results_detector/experiments/): a cheap classical-feature domain
classifier routes each image to a domain-specialized logistic-regression
head trained on CLIP PRE-projection embeddings (768-dim, before CLIP's
image-text alignment layer), extended with a "reactivity" feature: the
CLIP-embedding shift caused by re-JPEG-compressing the image at quality
50 (see diagnostic_reactivity_delta.py / diagnostic_reactivity_
comprehensive.py / prelim_integration_test.py) - validated to add real,
non-spurious signal under cross-source, cross-generator, and
already-degraded-input testing (mean +0.028 AUROC in the pre-production
check), not just appearance-based classification.

This won the comparison: mean AUROC 0.9478 across the full 16-cell
robustness grid, vs. 0.9316 for the original single-generalist
post-projection baseline (+0.0162), and it beat every other variant
tried (balanced-augmentation generalist alone, post-projection
specialists, naive balanced+pre-projection stacking on one head) in 15
of 16 grid cells. The reactivity-delta extension (train_reactivity_
specialists.py) was added on top of that winning architecture after
separate validation - see README "Reactivity-delta feature" section.

Model files (in model/):
  - domain_classifier.pkl: RandomForestClassifier, 5-way
    (clean/jpeg/spatial/noise/colorjitter), trained on classical
    no-reference features (domain_classifier.py).
  - specialists.pkl: dict[group_name -> sklearn Pipeline
    (StandardScaler + LogisticRegression)], each trained on the extended
    1536-dim feature (CLIP pre-projection embedding + jpeg_q50-probe
    embedding delta) of images augmented within that domain group
    (train_reactivity_specialists.py).
"""

import os
import pickle

import numpy as np

from clip_features import embed_images_preproj
from domain_classifier import extract_color_iqa_features
from transforms import jpeg_compress

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
REACTIVITY_PROBE = lambda img: jpeg_compress(img, 50)

_domain_clf = None
_specialists = None


def load_production_models(model_dir=MODEL_DIR):
    global _domain_clf, _specialists
    if _domain_clf is None:
        with open(os.path.join(model_dir, "domain_classifier.pkl"), "rb") as f:
            _domain_clf = pickle.load(f)
        with open(os.path.join(model_dir, "specialists.pkl"), "rb") as f:
            _specialists = pickle.load(f)
    return _domain_clf, _specialists


def _extended_features(pil_images, batch_size):
    """CLIP pre-projection embedding concatenated with the embedding
    shift caused by a jpeg_q50 reactivity probe - see module docstring."""
    embs = embed_images_preproj(pil_images, batch_size=batch_size)
    probed_imgs = [REACTIVITY_PROBE(img) for img in pil_images]
    probed_embs = embed_images_preproj(probed_imgs, batch_size=batch_size)
    delta = probed_embs - embs
    return np.concatenate([embs, delta], axis=1)


def predict_proba(pil_images, batch_size=32, model_dir=MODEL_DIR, confidence_gated=True):
    """pil_images: list of PIL RGB images. Returns (probs, groups):
    probs is an (N,) array of P(AI-generated) per image; groups is the
    (N,) array of the domain classifier's TOP-1 guess per image (for
    diagnostics/logging - see below for how it's actually used).

    confidence_gated=True (default): soft mixture-of-experts. Rather than
    committing fully to the domain classifier's top-1 guess, every
    specialist's opinion is weighted by the domain classifier's own
    probability that the image belongs to that domain:
        P(AI) = sum_g  P(domain=g | image) * specialist_g.predict_proba(image)
    When the domain classifier is confident, this is ~equivalent to hard
    routing. When it's unsure (e.g. color-jitter vs. clean, its hardest
    call), it blends multiple specialists instead of betting everything
    on a single, possibly-wrong guess - no confidence threshold to tune.

    confidence_gated=False: the original hard top-1 routing (kept for
    comparison - see results_detector/robustness_table_hard_routing.json).
    """
    domain_clf, specialists = load_production_models(model_dir)

    color_feats = np.array([extract_color_iqa_features(img) for img in pil_images])
    feats = _extended_features(pil_images, batch_size)

    domain_probs = domain_clf.predict_proba(color_feats)  # (N, n_groups)
    group_order = domain_clf.classes_
    top1_groups = group_order[domain_probs.argmax(axis=1)]

    if not confidence_gated:
        probs = np.zeros(len(pil_images))
        for i, g in enumerate(top1_groups):
            probs[i] = specialists[g].predict_proba(feats[i:i + 1])[:, 1][0]
        return probs, top1_groups

    specialist_probs = np.column_stack(
        [specialists[g].predict_proba(feats)[:, 1] for g in group_order]
    )  # (N, n_groups)
    probs = (domain_probs * specialist_probs).sum(axis=1)
    return probs, top1_groups
