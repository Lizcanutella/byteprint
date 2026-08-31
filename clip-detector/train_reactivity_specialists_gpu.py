"""
GPU variant of train_reactivity_specialists.py: same architecture and
same validated jpeg_q50 reactivity-delta feature, but images are loaded
at NATIVE resolution (clip_features.load_image_native, no 512px cap)
and embedded via native-resolution texture crops (clip_features.
embed_images_preproj_crops_with_delta) instead of letting CLIPProcessor
squash the whole image down to 224x224.

Why this needed a GPU rather than the CPU sandbox the rest of this
project was built on: extracting `top_k` crops per image multiplies
backbone compute ~top_k-fold over the whole-image approach, which
already took ~2350s (~39min) for one pass over the balanced training
set on CPU - the crop version would take ~2-3x longer AGAIN per pass
(base+delta both need the crop-embedding path), pushing a single run
past 2 hours on this machine. On a free Colab/Kaggle T4 this should
take a few minutes.

Run this on a GPU runtime (Google Colab / Kaggle Notebooks are free):
    1. Upload/clone this project's code (not the data_*/ or model/
       directories - those get rebuilt by the fetch/train scripts).
    2. pip install -r requirements.txt, but swap the CPU torch line for
       a normal (CUDA) torch install - Colab/Kaggle already have a
       compatible CUDA torch preinstalled, so this is usually a no-op.
    3. Run the fetch scripts (fetch_data.py, sid_set_fetch.py,
       fetch_defactify_train.py) to repopulate data_fullres/,
       data_sid_set/, data_defactify_train/ - they stream from HF Hub,
       no local data upload needed.
    4. python train_reactivity_specialists_gpu.py
    5. Compare results_detector/experiments/reactivity_specialists_gpu_
       train_result.json's held-out AUROC against the CPU run's
       (model/model_meta.json's held_out_clean_test_auroc, 0.9989) - if
       it's not clearly better, the native-crop version isn't worth the
       extra inference cost (3x CLIP passes per image at predict time
       instead of 2), and the existing CPU-trained production model
       (model/specialists.pkl) should stay as the submission.

Usage:
    python train_reactivity_specialists_gpu.py [--top-k 3]
"""

import argparse
import json
import os
import pickle
import random
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from clip_features import DEVICE, embed_images_preproj_crops_with_delta, load_image_native
from domain_classifier import apply_group_transform
from train_classifier import SEED, compute_split
from transforms import GROUP_NAMES, jpeg_compress

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_DIR = os.path.join(BASE_DIR, "results_detector", "experiments")
PROBE_FN = lambda img: jpeg_compress(img, 50)
CHUNK_SIZE = 250


def embed_balanced_set_gpu(paths, labels, indices, rng, top_k, chunk_size=CHUNK_SIZE):
    """Same balanced per-domain augmentation as train_domain_approaches.
    embed_balanced_set, but native-resolution crop-based embedding with
    the reactivity-delta computed on the SAME crops (see clip_features.
    embed_images_preproj_crops_with_delta)."""
    all_base, all_delta, all_y, all_groups = [], [], [], []
    n_chunks = (len(indices) + chunk_size - 1) // chunk_size
    for start in range(0, len(indices), chunk_size):
        chunk = indices[start:start + chunk_size]
        imgs, ys, groups = [], [], []
        for i in chunk:
            try:
                img = load_image_native(paths[i])
            except (OSError, ValueError) as exc:
                print(f"  WARNING: skipping unreadable image {paths[i]}: {exc}")
                continue
            imgs.append(img)
            ys.append(labels[i])
            groups.append("clean")
            for group in GROUP_NAMES:
                if group == "clean":
                    continue
                degraded = apply_group_transform(img, group, rng)
                imgs.append(degraded)
                ys.append(labels[i])
                groups.append(group)

        base, delta = embed_images_preproj_crops_with_delta(imgs, PROBE_FN, top_k=top_k, batch_size=64)
        all_base.append(base)
        all_delta.append(delta)
        all_y.extend(ys)
        all_groups.extend(groups)
        print(f"  chunk {start // chunk_size + 1}/{n_chunks}: "
              f"{len(chunk)} originals -> {len(imgs)} augmented images embedded (base+delta, top_k={top_k})")
    return (np.concatenate(all_base, axis=0), np.concatenate(all_delta, axis=0),
            np.array(all_y), np.array(all_groups))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-k", type=int, default=3, help="native-resolution crops per image")
    args = parser.parse_args()

    if DEVICE != "cuda":
        print(f"WARNING: DEVICE={DEVICE!r}, not 'cuda'. This script is designed for a GPU "
              f"runtime (Colab/Kaggle) - it will run on CPU but very slowly. Continuing anyway.")

    random.seed(SEED)
    np.random.seed(SEED)
    os.makedirs(EXPERIMENT_DIR, exist_ok=True)

    paths, labels, sources, train_idx, test_idx = compute_split()
    print(f"Collected {len(paths)} images; train={len(train_idx)} test={len(test_idx)}; device={DEVICE}")

    rng = random.Random(SEED)
    print(f"Building + embedding BALANCED train set with native-crop base+delta features (top_k={args.top_k})...")
    t0 = time.time()
    X_base, X_delta, y_train, train_groups = embed_balanced_set_gpu(paths, labels, train_idx, rng, args.top_k)
    print(f"  done in {time.time() - t0:.1f}s  base_shape={X_base.shape} delta_shape={X_delta.shape}")
    X_train = np.concatenate([X_base, X_delta], axis=1)

    print("Building held-out test split (clean only) with native-crop base+delta features...")
    t0 = time.time()
    test_imgs, test_labels = [], []
    for i in tqdm(test_idx, desc="load-test"):
        try:
            test_imgs.append(load_image_native(paths[i]))
            test_labels.append(labels[i])
        except (OSError, ValueError) as exc:
            print(f"  WARNING: skipping unreadable image {paths[i]}: {exc}")
    y_test = np.array(test_labels)
    Xb_test, Xd_test = embed_images_preproj_crops_with_delta(test_imgs, PROBE_FN, top_k=args.top_k, batch_size=64)
    X_test = np.concatenate([Xb_test, Xd_test], axis=1)
    print(f"  done in {time.time() - t0:.1f}s")

    print("\nTraining per-domain specialists (extended 1536-dim feature, native crops)...")
    specialists = {}
    for group in GROUP_NAMES:
        mask = train_groups == group
        Xg, yg = X_train[mask], y_train[mask]
        clf_g = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
        clf_g.fit(Xg, yg)
        specialists[group] = clf_g
        print(f"  [{group}] trained on {int(mask.sum())} images")

    clean_auc = float(roc_auc_score(y_test, specialists["clean"].predict_proba(X_test)[:, 1]))
    clean_acc = float(accuracy_score(y_test, specialists["clean"].predict(X_test)))
    print(f"\nSanity check - 'clean' specialist on held-out clean test set: "
          f"AUROC={clean_auc:.4f} accuracy={clean_acc:.4f}")
    print(f"Compare against the CPU whole-image run: 0.9989 AUROC / 0.9887 accuracy "
          f"(model/model_meta.json). If this isn't clearly better, native crops aren't "
          f"worth the extra inference cost - keep the existing CPU-trained model/specialists.pkl.")

    out_path = os.path.join(EXPERIMENT_DIR, "specialists_preproj_gpu_native_crops.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(specialists, f)
    print(f"Saved specialists to {out_path}")

    result = {
        "variant": "native-resolution texture crops (GPU)",
        "top_k_crops": args.top_k,
        "probe": "jpeg_q50 (universal, all domains)",
        "feature": "concat[pooled CLIP preproj embedding over top_k native crops (768d), "
                   "same-crop jpeg_q50-probe embedding delta (768d)] = 1536d",
        "group_train_sizes": {g: int((train_groups == g).sum()) for g in GROUP_NAMES},
        "held_out_clean_test_auroc": clean_auc,
        "held_out_clean_test_accuracy": clean_acc,
        "cpu_whole_image_baseline_auroc": 0.9989,
        "cpu_whole_image_baseline_accuracy": 0.9887,
        "seed": SEED,
        "device": DEVICE,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    result_path = os.path.join(EXPERIMENT_DIR, "reactivity_specialists_gpu_train_result.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved result to {result_path}")


if __name__ == "__main__":
    main()
