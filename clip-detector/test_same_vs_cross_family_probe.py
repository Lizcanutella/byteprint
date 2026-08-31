"""
Tests the mechanism behind why only the "jpeg" domain needed a
domain-matched probe swap: does a probe from the SAME degradation
family as a domain redundantly add little information, while a
DIFFERENT-family probe (like the universal jpeg_q50 currently used)
adds real signal? Mirrors prelim_domain_matched_probe.py's methodology
(which tested this for the "jpeg" domain) - parameterized so any
domain can be targeted, with a same-family probe compared against the
current production cross-family probe (jpeg_q50) on the SAME degraded
images, so the comparison isolates the probe choice.

Usage:
    python test_same_vs_cross_family_probe.py --group spatial --same-family-probe blur_s1.0
    python test_same_vs_cross_family_probe.py --group noise --same-family-probe noise_s0.05
    python test_same_vs_cross_family_probe.py --group colorjitter --same-family-probe colorjitter_up20
"""

import argparse
import json
import random
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from clip_features import embed_images_preproj, load_image_capped
from domain_classifier import apply_group_transform
from train_classifier import SEED, compute_split
from transforms import color_jitter, gaussian_blur, gaussian_noise, jpeg_compress

N_SUBSAMPLE = 300  # smaller than before for speed, given time pressure
CHUNK_SIZE = 100

SAME_FAMILY_PROBES = {
    "blur_s1.0": lambda img: gaussian_blur(img, 1.0),
    "noise_s0.05": lambda img: gaussian_noise(img, 0.05),
    "colorjitter_up20": lambda img: color_jitter(img, 1.2),
}


def build_group_images(paths, labels, indices, group, rng, chunk_size=CHUNK_SIZE):
    all_imgs, all_base, all_y = [], [], []
    for start in range(0, len(indices), chunk_size):
        chunk = indices[start:start + chunk_size]
        degraded_imgs, ys = [], []
        for i in chunk:
            img = load_image_capped(paths[i])
            degraded_imgs.append(apply_group_transform(img, group, rng))
            ys.append(labels[i])
        emb_base = embed_images_preproj(degraded_imgs, batch_size=32)
        all_imgs.extend(degraded_imgs)
        all_base.append(emb_base)
        all_y.extend(ys)
    return all_imgs, np.concatenate(all_base, axis=0), np.array(all_y)


def embed_probe(imgs, probe_fn, chunk_size=CHUNK_SIZE):
    embs = []
    for start in range(0, len(imgs), chunk_size):
        chunk = [probe_fn(img) for img in imgs[start:start + chunk_size]]
        embs.append(embed_images_preproj(chunk, batch_size=32))
    return np.concatenate(embs, axis=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", required=True, choices=["clean", "jpeg", "spatial", "noise", "colorjitter"])
    parser.add_argument("--same-family-probe", required=True, choices=list(SAME_FAMILY_PROBES.keys()))
    args = parser.parse_args()

    target_group = args.group
    candidate_probes = {
        "jpeg_q50 (current production, cross-family)": lambda img: jpeg_compress(img, 50),
        f"{args.same_family_probe} (same-family as {target_group}, hypothesized redundant)":
            SAME_FAMILY_PROBES[args.same_family_probe],
    }

    random.seed(SEED)
    np.random.seed(SEED)

    paths, labels, sources, train_idx, test_idx = compute_split()
    labels_arr = np.array(labels)

    sub_idx, _ = train_test_split(
        train_idx, train_size=N_SUBSAMPLE, random_state=SEED, stratify=labels_arr[train_idx]
    )
    print(f"Training subsample: {len(sub_idx)} of {len(train_idx)} originals; group={target_group}")

    rng_train = random.Random(SEED + 200)
    rng_test = random.Random(SEED + 201)

    print(f"Building {target_group}-domain TRAIN images + base embeddings...")
    t0 = time.time()
    train_imgs, Xb_train, y_train = build_group_images(paths, labels, sub_idx, target_group, rng_train)
    print(f"  done in {time.time() - t0:.1f}s")

    print(f"Building {target_group}-domain TEST images + base embeddings...")
    t0 = time.time()
    test_imgs, Xb_test, y_test = build_group_images(paths, labels, test_idx, target_group, rng_test)
    print(f"  done in {time.time() - t0:.1f}s")

    baseline_clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    baseline_clf.fit(Xb_train, y_train)
    auc_baseline = float(roc_auc_score(y_test, baseline_clf.predict_proba(Xb_test)[:, 1]))
    print(f"\nBaseline (absolute embedding only, no delta): AUROC={auc_baseline:.4f}")

    results = {"baseline_no_delta": auc_baseline}
    for probe_name, probe_fn in candidate_probes.items():
        print(f"\n=== Probe: {probe_name} ===")
        t0 = time.time()
        Xp_train = embed_probe(train_imgs, probe_fn)
        Xp_test = embed_probe(test_imgs, probe_fn)
        print(f"  embedded probed copies in {time.time() - t0:.1f}s")

        Xd_train = Xp_train - Xb_train
        Xd_test = Xp_test - Xb_test

        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
        X_train_ext = np.concatenate([Xb_train, Xd_train], axis=1)
        X_test_ext = np.concatenate([Xb_test, Xd_test], axis=1)
        clf.fit(X_train_ext, y_train)
        auc = float(roc_auc_score(y_test, clf.predict_proba(X_test_ext)[:, 1]))
        results[probe_name] = auc
        print(f"  extended AUROC={auc:.4f}  delta vs baseline={auc - auc_baseline:+.4f}")

    out_path = f"results_detector/same_vs_cross_family_probe_{target_group}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")

    print("\n" + "=" * 70)
    print(f"SUMMARY ({target_group} domain, same-family vs cross-family probe)")
    print("=" * 70)
    print(f"baseline (no delta): {auc_baseline:.4f}")
    for probe_name in candidate_probes:
        print(f"  + {probe_name}: {results[probe_name]:.4f}  ({results[probe_name] - auc_baseline:+.4f})")


if __name__ == "__main__":
    main()
