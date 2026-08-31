"""
Domain classifier: predicts which of the 5 coarse degradation groups
(transforms.GROUP_NAMES: clean, jpeg, spatial, noise, colorjitter) was
applied to an image, using cheap classical no-reference features (no
CLIP forward pass needed). Trained on synthetic degradations of a
generic image pool with known ground truth, same style as
degrade_model.py's blur/quality regressors.

This is the routing mechanism for the domain-specialist experiment
(train_domain_approaches.py / evaluate_domain_approaches.py): since the
organizers' robustness test images are each degraded in exactly ONE
domain at a time, detecting that domain and routing to a
domain-specialized classifier head is well-posed here.
"""

import random

import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

from degrade_model import blockiness_score, highfreq_ratio
from scipy.ndimage import laplace
from transforms import (
    GROUP_NAMES,
    center_crop,
    color_jitter,
    gaussian_blur,
    gaussian_noise,
    jpeg_compress,
    resize_roundtrip,
)

SEED = 1234


def extract_color_iqa_features(img_rgb):
    """img_rgb: PIL RGB image. Returns
    [laplacian_var, highfreq_ratio, blockiness, contrast, mean_saturation]
    - the last feature is what lets this distinguish colorjitter, which
    the grayscale-only features in degrade_model.py can't see."""
    arr = np.asarray(img_rgb, dtype=np.float64) / 255.0
    gray = arr.mean(axis=2)
    lap_var = float(laplace(gray).var())
    hf_ratio = float(highfreq_ratio(gray))
    block = blockiness_score(gray)
    contrast = float(gray.std())
    hsv = np.asarray(img_rgb.convert("HSV"), dtype=np.float64) / 255.0
    mean_sat = float(hsv[..., 1].mean())
    return np.array([lap_var, hf_ratio, block, contrast, mean_sat])


FEATURE_NAMES = ["laplacian_var", "highfreq_ratio", "blockiness", "contrast", "mean_saturation"]


def apply_group_transform(img_rgb, group, rng):
    """Apply one randomly-parameterized transform from `group` (a
    transforms.GROUP_NAMES value) to img_rgb, returning the degraded
    image. Mirrors transforms.py's specific grid values so the domain
    classifier's training distribution matches what it'll see at eval
    time."""
    if group == "clean":
        return img_rgb
    if group == "jpeg":
        return jpeg_compress(img_rgb, rng.choice([90, 70, 50, 30]))
    if group == "spatial":
        choice = rng.choice(["blur", "resize", "crop"])
        if choice == "blur":
            return gaussian_blur(img_rgb, rng.choice([0.5, 1.0, 2.0]))
        elif choice == "resize":
            return resize_roundtrip(img_rgb, rng.choice([0.5, 0.25]))
        else:
            return center_crop(img_rgb, 0.8)
    if group == "noise":
        return gaussian_noise(img_rgb, rng.choice([0.02, 0.05, 0.10]))
    if group == "colorjitter":
        return color_jitter(img_rgb, rng.choice([0.8, 1.2]))
    raise ValueError(group)


def build_training_data(base_images_rgb, n_per_image=3, seed=SEED):
    """base_images_rgb: list of PIL RGB images (content only, no class
    labels needed). For each image, samples n_per_image (group, transformed
    image) pairs across the 5 groups."""
    rng = random.Random(seed)
    X, y = [], []
    for img in base_images_rgb:
        for _ in range(n_per_image):
            group = rng.choice(GROUP_NAMES)
            degraded = apply_group_transform(img, group, rng)
            X.append(extract_color_iqa_features(degraded))
            y.append(group)
    return np.array(X), np.array(y)


def train_domain_classifier(X, y, seed=SEED):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=seed, stratify=y)
    clf = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    acc = float(accuracy_score(yte, pred))
    labels = sorted(set(y))
    cm = confusion_matrix(yte, pred, labels=labels).tolist()
    metrics = {"n_train": len(Xtr), "n_test": len(Xte), "accuracy": acc,
               "confusion_matrix_labels": labels, "confusion_matrix": cm}
    return clf, metrics
