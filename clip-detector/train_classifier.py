"""
Trains the real detector: CLIP embeddings + logistic-regression head,
with training-time augmentation drawn from the hackathon's robustness
transform grid (transforms.py), so the head learns to be invariant to
those transforms rather than just accurate on clean images.

The held-out test split is set aside BEFORE augmentation (so a held-out
image never contributes even an augmented copy to training) and its
path manifest is saved for robustness_eval.py to reuse untouched.

Usage:
    python train_classifier.py
"""

import json
import os
import pickle
import random
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from clip_features import DEVICE, MODEL_ID, embed_images, load_image_capped
from transforms import random_transform

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SEED = 1234
AUGMENT_FRACTION = 0.5

DATASETS = [
    ("data_fullres", "fullres"),
    ("data_sid_set", "sid_set"),
    ("data_defactify_train", "defactify"),
    # data_cifake_prod (dragonintelligence/CIFAKE-image-dataset, an HF
    # mirror of the organizer-recommended Kaggle CIFAKE dataset) was
    # TRIED here to properly use an organizer-listed resource in the
    # production model (previously only used in the abandoned Part-2
    # forensic-signal research). It was REMOVED after empirical testing
    # showed a clear, dual-confirmed regression: -0.020 mean AUROC on
    # our own 16-cell robustness grid (0.976 -> 0.956) AND -0.056 AUROC
    # on the independent generator-diagnostic benchmark (0.960 -> 0.904,
    # including real-image accuracy falling from 93% to 82%). The likely
    # cause: CIFAKE's native 32x32 resolution (heavily upsampled to
    # 512px) is fundamentally incompatible with a task that also applies
    # blur/resize/noise on top - already-blurry base images confuse the
    # spatial/noise specialists. The organizer's brief frames its
    # datasets as available resources, not mandatory inclusions, so a
    # measurable, dual-validated performance regression was not worth
    # accepting purely for checklist inclusion. Kept as
    # model/*_v4_with_cifake_WORSE.* for a fully transparent record of
    # this decision, including the before/after numbers.
    #
    # defactify (Rajarshi-Roy-research/Defactify_Image_Dataset, TRAIN
    # split only) adds generator diversity: 150 each from Stable
    # Diffusion 2.1, SDXL, SD3, DALL-E 3, Midjourney 6 (AI class ONLY -
    # see below). Added after a diagnostic eval
    # (fetch_generator_diagnostic.py / evaluate_generator_diagnostic.py,
    # using Defactify's disjoint TEST split) found the pre-existing
    # 2-source model generalized poorly to unseen generators (SD2.1 45%,
    # Midjourney6 48%, SD3 53% accuracy - near or below chance) despite
    # 0.98+ AUROC on its own held-out data.
    #
    # Defactify's REAL images (data_defactify_train/real_EXCLUDED_coco_
    # overlap_risk/) are deliberately NOT loaded here: they're sourced
    # from MS COCO, and the organizer's WildFake demonstration-only
    # benchmark uses COCO val2017 as its non-AIGC side. Defactify's ~16k
    # real images almost certainly come from the much larger train2017
    # split (COCO val2017 is only 5k images total), so systematic
    # overlap is unlikely - but since real-photo diversity is already
    # well covered by fullres/sid_set/cifake, there's no reason to carry
    # even a small, unconfirmed risk of training on images the organizer
    # explicitly reserved for demonstration. Only Defactify's AI-
    # generated images (no COCO/WildFake connection at all) are used.
    #
    # cifake (dragonintelligence/CIFAKE-image-dataset via fetch_data.py,
    # an HF mirror of the organizer-recommended Kaggle CIFAKE dataset -
    # birdy654/cifake-real-and-ai-generated-synthetic-images) was added
    # to actually use an organizer-listed dataset in the production
    # model; earlier in the project it was used only for the abandoned
    # hand-crafted forensic-signal research (README Part 2), never for
    # this CLIP-based detector, which was an oversight worth correcting.
    #
    # data_fullres2 (itsLeen/deepfake_vs_real_image) EXCLUDED: manual
    # inspection found its "Real_Art"/"AI_Art" classes are a "human art
    # vs AI art" dataset, not "real photo vs AI-generated image" - the
    # "real" class is dominated by paintings/illustrations, and the "AI"
    # class includes real photographs of media coverage about AI art
    # (e.g. a magazine graphic referencing the Kashtanova/Midjourney
    # copyright case). Not a reliable source for this task.
]


def collect_paths():
    records = []
    for data_dir, source in DATASETS:
        real_dir = os.path.join(BASE_DIR, data_dir, "real")
        ai_dir = os.path.join(BASE_DIR, data_dir, "ai")
        if os.path.isdir(real_dir):
            for f in sorted(os.listdir(real_dir)):
                if f.lower().endswith(".png"):
                    records.append((os.path.join(real_dir, f), 0, source))
        if os.path.isdir(ai_dir):
            for f in sorted(os.listdir(ai_dir)):
                if f.lower().endswith(".png"):
                    records.append((os.path.join(ai_dir, f), 1, source))
    return records


def compute_split(seed=SEED, test_size=0.15):
    """Returns (paths, labels, sources, train_idx, test_idx) - the exact
    same deterministic split used everywhere, so any script that needs
    "the same train/test partition as the real run" gets it by construction
    rather than by duplicating this logic."""
    records = collect_paths()
    paths = [r[0] for r in records]
    labels = [r[1] for r in records]
    sources = [r[2] for r in records]
    idx = list(range(len(records)))
    train_idx, test_idx = train_test_split(
        idx, test_size=test_size, random_state=seed, stratify=labels
    )
    return paths, labels, sources, train_idx, test_idx


def build_split(paths, labels, sources, indices, augment, rng, augment_fraction=AUGMENT_FRACTION):
    """Loads images for `indices`; if augment, a random subset (augment_fraction)
    also get one extra randomly-transformed copy appended. `rng` must be a
    seeded random.Random for reproducibility."""
    imgs, ys, metas = [], [], []
    for i in tqdm(indices, desc="load"):
        path, label, source = paths[i], labels[i], sources[i]
        img = load_image_capped(path)
        imgs.append(img)
        ys.append(label)
        metas.append({"path": path, "source": source, "transform": "clean"})
        if augment and rng.random() < augment_fraction:
            name, aug_img = random_transform(img, rng)
            imgs.append(aug_img)
            ys.append(label)
            metas.append({"path": path, "source": source, "transform": name})
    return imgs, np.array(ys), metas


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    paths, labels, sources, train_idx, test_idx = compute_split()
    sources_present = sorted(set(sources))
    print(f"Collected {len(paths)} images from {sources_present}")
    if not paths:
        print("No data found. Run fetch_data.py / sid_set_fetch.py first.")
        return

    rng = random.Random(SEED)

    print("Building train split (with robustness-grid augmentation)...")
    train_imgs, y_train, train_meta = build_split(paths, labels, sources, train_idx, True, rng)
    print(f"  {len(train_imgs)} images (incl. augmented copies)")

    print("Building held-out test split (clean only)...")
    test_imgs, y_test, test_meta = build_split(paths, labels, sources, test_idx, False, rng)
    print(f"  {len(test_imgs)} images")

    print("Extracting CLIP embeddings for train split...")
    t0 = time.time()
    X_train = embed_images(train_imgs, batch_size=32)
    print(f"  done in {time.time() - t0:.1f}s")

    print("Extracting CLIP embeddings for held-out test split...")
    t0 = time.time()
    X_test = embed_images(test_imgs, batch_size=32)
    print(f"  done in {time.time() - t0:.1f}s")

    print("Training logistic regression head...")
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    clf.fit(X_train, y_train)

    train_auc = float(roc_auc_score(y_train, clf.predict_proba(X_train)[:, 1]))
    test_auc = float(roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]))
    test_acc = float(accuracy_score(y_test, clf.predict(X_test)))
    print(f"Train AUROC: {train_auc:.4f}")
    print(f"Held-out test AUROC: {test_auc:.4f}  accuracy: {test_acc:.4f}")

    model_dir = os.path.join(BASE_DIR, "model")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "classifier_head.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(clf, f)

    n_params = 151_277_313  # CLIP ViT-B/32 param count, logged at load time too
    meta = {
        "backbone": MODEL_ID,
        "backbone_param_count_approx": n_params,
        "device_used_for_training": DEVICE,
        "n_train_images_incl_augmented": len(train_imgs),
        "n_test_images_heldout_clean": len(test_imgs),
        "train_sources": sources_present,
        "augment_fraction": AUGMENT_FRACTION,
        "train_auroc": train_auc,
        "test_auroc": test_auc,
        "test_accuracy": test_acc,
        "seed": SEED,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    meta_path = os.path.join(model_dir, "model_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    test_manifest_path = os.path.join(model_dir, "test_manifest.json")
    with open(test_manifest_path, "w") as f:
        json.dump(
            [{"path": paths[i], "label": labels[i], "source": sources[i]} for i in test_idx],
            f, indent=2,
        )

    print(f"Saved model to {model_path}")
    print(f"Saved meta to {meta_path}")
    print(f"Saved held-out test manifest ({len(test_idx)} images) to {test_manifest_path}")


if __name__ == "__main__":
    main()
