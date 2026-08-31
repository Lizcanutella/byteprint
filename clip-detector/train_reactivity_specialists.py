"""
Full production integration of the validated jpeg_q50 reactivity-delta
finding (diagnostic_reactivity_delta.py -> diagnostic_reactivity_
comprehensive.py -> prelim_integration_test.py -> prelim_domain_matched_
probe.py): every domain specialist's feature is extended from the plain
768-dim CLIP pre-projection embedding to a 1536-dim
[embedding, embedding_of_jpeg_q50_probed_copy - embedding] vector.

Scoping decision (time-boxed, ~1 day to hackathon deadline): a SINGLE
universal probe (jpeg_q50) is used for all 5 domains, not the
domain-matched scheme (blur_s1.0 for the "jpeg" domain specifically)
found slightly better in prelim_domain_matched_probe.py. Domain-matching
only helps the "jpeg" domain, which already has the least headroom
(~0.99 baseline) and the smallest measured gain (+0.008 vs +0.027 to
+0.045 for the other domains) - and doing it properly at INFERENCE time
under confidence-gated soft routing would require a 3rd CLIP pass per
image (both probes, since any image can get nonzero weight on any
specialist), not worth the added latency/complexity this close to the
deadline. This is a disclosed trade-off, not an oversight.

Trains all 5 domain specialists on the full training set (same balanced
per-domain augmentation as train_domain_approaches_preproj.py: each
training original gets one clean copy + one sample from each of the 4
other domain groups), with the extended 1536-dim feature. Evaluates the
"clean" specialist against the held-out clean test set as a sanity
check before the full robustness_eval.py grid run confirms the
end-to-end production numbers.

Usage:
    python train_reactivity_specialists.py
"""

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

from clip_features import embed_images_preproj, load_image_capped
from domain_classifier import apply_group_transform
from train_classifier import SEED, compute_split
from transforms import GROUP_NAMES, jpeg_compress

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_DIR = os.path.join(BASE_DIR, "results_detector", "experiments")
PROBE_FN = lambda img: jpeg_compress(img, 50)
CHUNK_SIZE = 250


def embed_balanced_set_with_delta(paths, labels, indices, rng, chunk_size=CHUNK_SIZE):
    """Like train_domain_approaches.embed_balanced_set, but also computes
    the jpeg_q50-probe embedding delta for every (original, group) copy.
    Returns (X_base, X_delta, y, groups)."""
    all_base, all_delta, all_y, all_groups = [], [], [], []
    n_chunks = (len(indices) + chunk_size - 1) // chunk_size
    for start in range(0, len(indices), chunk_size):
        chunk = indices[start:start + chunk_size]
        imgs, ys, groups = [], [], []
        for i in chunk:
            path, label = paths[i], labels[i]
            img = load_image_capped(path)
            imgs.append(img)
            ys.append(label)
            groups.append("clean")
            for group in GROUP_NAMES:
                if group == "clean":
                    continue
                degraded = apply_group_transform(img, group, rng)
                imgs.append(degraded)
                ys.append(label)
                groups.append(group)

        probed_imgs = [PROBE_FN(img) for img in imgs]
        emb_base = embed_images_preproj(imgs, batch_size=32)
        emb_probed = embed_images_preproj(probed_imgs, batch_size=32)

        all_base.append(emb_base)
        all_delta.append(emb_probed - emb_base)
        all_y.extend(ys)
        all_groups.extend(groups)
        print(f"  chunk {start // chunk_size + 1}/{n_chunks}: "
              f"{len(chunk)} originals -> {len(imgs)} augmented images embedded (base+probe)")
    return (np.concatenate(all_base, axis=0), np.concatenate(all_delta, axis=0),
            np.array(all_y), np.array(all_groups))


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    os.makedirs(EXPERIMENT_DIR, exist_ok=True)

    paths, labels, sources, train_idx, test_idx = compute_split()
    print(f"Collected {len(paths)} images; train={len(train_idx)} test={len(test_idx)}")

    rng = random.Random(SEED)
    print("Building + embedding BALANCED train set with base+jpeg_q50-delta features...")
    t0 = time.time()
    X_base, X_delta, y_train, train_groups = embed_balanced_set_with_delta(paths, labels, train_idx, rng)
    print(f"  done in {time.time() - t0:.1f}s  base_shape={X_base.shape} delta_shape={X_delta.shape}")
    X_train = np.concatenate([X_base, X_delta], axis=1)

    print("Building held-out test split (clean only) with base+delta features...")
    t0 = time.time()
    test_imgs = [load_image_capped(paths[i]) for i in tqdm(test_idx, desc="load-test")]
    y_test = np.array([labels[i] for i in test_idx])
    Xb_test = embed_images_preproj(test_imgs, batch_size=32)
    probed_test_imgs = [PROBE_FN(img) for img in test_imgs]
    Xd_test = embed_images_preproj(probed_test_imgs, batch_size=32) - Xb_test
    X_test = np.concatenate([Xb_test, Xd_test], axis=1)
    print(f"  done in {time.time() - t0:.1f}s")

    print("\nTraining per-domain specialists (extended 1536-dim feature)...")
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

    out_path = os.path.join(EXPERIMENT_DIR, "specialists_preproj_v6_reactivity.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(specialists, f)
    print(f"Saved specialists to {out_path}")

    result = {
        "probe": "jpeg_q50 (universal, all domains)",
        "feature": "concat[CLIP preproj embedding (768d), jpeg_q50-probe embedding delta (768d)] = 1536d",
        "group_train_sizes": {g: int((train_groups == g).sum()) for g in GROUP_NAMES},
        "held_out_clean_test_auroc": clean_auc,
        "held_out_clean_test_accuracy": clean_acc,
        "seed": SEED,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    result_path = os.path.join(EXPERIMENT_DIR, "reactivity_specialists_train_result.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved result to {result_path}")


if __name__ == "__main__":
    main()
