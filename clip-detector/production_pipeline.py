"""
The production detection pipeline, promoted after comparing several
alternatives (see README "Model iteration" section and
results_detector/experiments/): a cheap classical-feature domain
classifier routes each image to a domain-specialized logistic-regression
head trained on CLIP PRE-projection embeddings (768-dim, before CLIP's
image-text alignment layer), extended with a "reactivity" feature: the
CLIP-embedding shift caused by a probe transform (see
diagnostic_reactivity_delta.py / diagnostic_reactivity_comprehensive.py
/ prelim_integration_test.py) - validated to add real, non-spurious
signal under cross-source, cross-generator, and already-degraded-input
testing (mean +0.028 AUROC in the pre-production check), not just
appearance-based classification.

This won the comparison: mean AUROC 0.9478 across the full 16-cell
robustness grid, vs. 0.9316 for the original single-generalist
post-projection baseline (+0.0162), and it beat every other variant
tried (balanced-augmentation generalist alone, post-projection
specialists, naive balanced+pre-projection stacking on one head) in 15
of 16 grid cells. The reactivity-delta extension (train_reactivity_
specialists.py) was added on top of that winning architecture after
separate validation - see README "Reactivity-delta feature" section.

Domain-matched probes (retrain_jpeg_specialist_domain_matched.py): the
"jpeg" specialist uses a blur_s1.0 probe instead of the jpeg_q50 probe
the other 4 specialists use, since re-JPEG-probing an already-JPEG-
degraded image is redundant by construction - validated at full scale
(+0.0015 AUROC on the jpeg domain specifically, on the same held-out
test images). This costs a 3rd CLIP pass per image (base + jpeg_q50-
probed + blur_s1.0-probed), since confidence-gated routing means any
image can get nonzero weight on any specialist, so every specialist's
delta is computed for every image regardless of routing.

Model files (in model/):
  - domain_classifier.pkl: RandomForestClassifier, 5-way
    (clean/jpeg/spatial/noise/colorjitter), trained on classical
    no-reference features (domain_classifier.py).
  - specialists.pkl: dict[group_name -> sklearn Pipeline
    (StandardScaler + LogisticRegression)], each trained on the extended
    1536-dim feature (CLIP pre-projection embedding + a probe-embedding
    delta - jpeg_q50 for 4 domains, blur_s1.0 for "jpeg", see
    PROBE_FOR_GROUP below) of images augmented within that domain group.
  - calibration.json: a threshold calibrated to a 1% false-positive rate
    (calibrate_threshold.py), for callers that want a binary decision
    instead of a raw probability - see calibrated_predict() below. The
    required predict.py deliverable still reports raw probabilities per
    the brief's spec; this is an additional, optional convenience.
"""

import json
import os
import pickle

import numpy as np

from clip_features import embed_images_preproj
from domain_classifier import extract_color_iqa_features
from transforms import gaussian_blur, jpeg_compress

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

# Which reactivity probe each domain specialist's delta feature is built
# from. jpeg_q50 is shared by 4 domains (only 2 distinct probes are ever
# computed, regardless of how many specialists use each).
PROBES = {
    "jpeg_q50": lambda img: jpeg_compress(img, 50),
    "blur_s1.0": lambda img: gaussian_blur(img, 1.0),
}
PROBE_FOR_GROUP = {
    "clean": "jpeg_q50",
    "jpeg": "blur_s1.0",
    "spatial": "jpeg_q50",
    "noise": "jpeg_q50",
    "colorjitter": "jpeg_q50",
}

_domain_clf = None
_specialists = None
_calibration = None


def load_production_models(model_dir=MODEL_DIR):
    global _domain_clf, _specialists, _calibration
    if _domain_clf is None:
        with open(os.path.join(model_dir, "domain_classifier.pkl"), "rb") as f:
            _domain_clf = pickle.load(f)
        with open(os.path.join(model_dir, "specialists.pkl"), "rb") as f:
            _specialists = pickle.load(f)
        calib_path = os.path.join(model_dir, "calibration.json")
        if os.path.exists(calib_path):
            with open(calib_path) as f:
                _calibration = json.load(f)
    return _domain_clf, _specialists


def _extended_features_by_group(pil_images, batch_size):
    """CLIP pre-projection embedding, plus each distinct probe's
    embedding-shift delta computed once - returns (embs, {probe_name:
    delta}). Callers build a specialist's feature by concatenating embs
    with deltas[PROBE_FOR_GROUP[group]]."""
    embs = embed_images_preproj(pil_images, batch_size=batch_size)
    deltas = {}
    for probe_name, probe_fn in PROBES.items():
        probed_imgs = [probe_fn(img) for img in pil_images]
        probed_embs = embed_images_preproj(probed_imgs, batch_size=batch_size)
        deltas[probe_name] = probed_embs - embs
    return embs, deltas


def _feature_for_group(embs, deltas, group):
    return np.concatenate([embs, deltas[PROBE_FOR_GROUP[group]]], axis=1)


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
    embs, deltas = _extended_features_by_group(pil_images, batch_size)

    domain_probs = domain_clf.predict_proba(color_feats)  # (N, n_groups)
    group_order = domain_clf.classes_
    top1_groups = group_order[domain_probs.argmax(axis=1)]

    if not confidence_gated:
        probs = np.zeros(len(pil_images))
        for i, g in enumerate(top1_groups):
            feat = _feature_for_group(embs, deltas, g)[i:i + 1]
            probs[i] = specialists[g].predict_proba(feat)[:, 1][0]
        return probs, top1_groups

    specialist_probs = np.column_stack(
        [specialists[g].predict_proba(_feature_for_group(embs, deltas, g))[:, 1] for g in group_order]
    )  # (N, n_groups)
    probs = (domain_probs * specialist_probs).sum(axis=1)
    return probs, top1_groups


def calibrated_predict(pil_images, batch_size=32, model_dir=MODEL_DIR, confidence_gated=True):
    """Like predict_proba, but also returns a binary decision using the
    threshold calibrated to a 1% false-positive rate (model/
    calibration.json, calibrate_threshold.py) instead of a fixed 0.5 -
    the fixed-0.5 cutoff ignores that score distributions shift between
    domains/generators, and a real platform's authentic:synthetic ratio
    makes accuracy-at-0.5 a poor proxy for deployment usability.

    Returns (probs, groups, decisions) - decisions is an (N,) bool array,
    True = predicted AI-generated at the calibrated operating point.
    Falls back to 0.5 if model/calibration.json doesn't exist."""
    load_production_models(model_dir)
    probs, groups = predict_proba(pil_images, batch_size, model_dir, confidence_gated)
    threshold = _calibration["recommended_threshold"] if _calibration else 0.5
    decisions = probs >= threshold
    return probs, groups, decisions
