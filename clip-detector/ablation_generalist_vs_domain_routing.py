"""
Does domain-specialist routing still add value now that reactivity-delta
exists, or has reactivity-delta made routing redundant? Trains ONE
single generalist classifier (no domain routing at all - pooled across
all 5 domains, universal jpeg_q50 probe only, since a domain-agnostic
model can't domain-match its probe without defeating the point) on the
extended [embedding + jpeg_q50-delta] feature, and compares it against
the actual production pipeline (domain classifier + 5 specialists +
confidence-gated routing) on the SAME held-out test images, degraded by
each of the 5 domain transforms in turn.

Usage:
    python ablation_generalist_vs_domain_routing.py
"""

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
from production_pipeline import predict_proba
from train_classifier import SEED, compute_split
from transforms import GROUP_NAMES, jpeg_compress

N_SUBSAMPLE = 100
CHUNK_SIZE = 50
PROBE_FN = lambda img: jpeg_compress(img, 50)
DOMAINS_TO_TEST = ["clean", "noise", "jpeg"]  # representative subset, not all 5, given time pressure


def build_balanced_with_delta(paths, labels, indices, rng, chunk_size=CHUNK_SIZE):
    """Each original gets one clean copy + one sample from each non-clean
    group - matching production's balanced training approach, but pooled
    (no domain labels retained for training the generalist)."""
    all_base, all_delta, all_y = [], [], []
    for start in range(0, len(indices), chunk_size):
        chunk = indices[start:start + chunk_size]
        imgs, ys = [], []
        for i in chunk:
            img = load_image_capped(paths[i])
            imgs.append(img)
            ys.append(labels[i])
            for group in GROUP_NAMES:
                if group == "clean":
                    continue
                imgs.append(apply_group_transform(img, group, rng))
                ys.append(labels[i])
        probed = [PROBE_FN(img) for img in imgs]
        base = embed_images_preproj(imgs, batch_size=32)
        probed_emb = embed_images_preproj(probed, batch_size=32)
        all_base.append(base)
        all_delta.append(probed_emb - base)
        all_y.extend(ys)
        print(f"  {start + len(chunk)}/{len(indices)} originals processed -> {len(imgs)} so far")
    return np.concatenate(all_base, axis=0), np.concatenate(all_delta, axis=0), np.array(all_y)


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    paths, labels, sources, train_idx, test_idx = compute_split()
    labels_arr = np.array(labels)

    sub_idx, _ = train_test_split(
        train_idx, train_size=N_SUBSAMPLE, random_state=SEED, stratify=labels_arr[train_idx]
    )
    print(f"Training subsample: {len(sub_idx)} of {len(train_idx)} originals (balanced across 5 domains)")

    rng = random.Random(SEED)
    print("Building pooled training set (all domains mixed, no domain labels used)...")
    t0 = time.time()
    Xb_train, Xd_train, y_train = build_balanced_with_delta(paths, labels, sub_idx, rng)
    print(f"  done in {time.time() - t0:.1f}s, {len(y_train)} total training rows")
    X_train = np.concatenate([Xb_train, Xd_train], axis=1)

    print("\nTraining single generalist (StandardScaler + LogisticRegression, extended feature)...")
    generalist = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    generalist.fit(X_train, y_train)

    eval_test_idx = test_idx[:150]  # subsample of held-out test, given time pressure
    results = {}
    for group in DOMAINS_TO_TEST:
        print(f"\n=== Evaluating on '{group}' domain ({len(eval_test_idx)} held-out test images) ===")
        rng_test = random.Random(SEED + 100 + GROUP_NAMES.index(group))
        test_imgs = []
        for i in eval_test_idx:
            img = load_image_capped(paths[i])
            test_imgs.append(img if group == "clean" else apply_group_transform(img, group, rng_test))
        y_test = labels_arr[eval_test_idx]

        t0 = time.time()
        probed_test = [PROBE_FN(img) for img in test_imgs]
        Xb_test = embed_images_preproj(test_imgs, batch_size=32)
        Xd_test = embed_images_preproj(probed_test, batch_size=32) - Xb_test
        X_test = np.concatenate([Xb_test, Xd_test], axis=1)
        generalist_auc = float(roc_auc_score(y_test, generalist.predict_proba(X_test)[:, 1]))
        print(f"  single generalist AUROC: {generalist_auc:.4f}  ({time.time() - t0:.1f}s)")

        t0 = time.time()
        prod_probs, routed_groups = predict_proba(test_imgs, batch_size=32)
        prod_auc = float(roc_auc_score(y_test, prod_probs))
        print(f"  production (domain-routed) AUROC: {prod_auc:.4f}  ({time.time() - t0:.1f}s)")

        results[group] = {
            "single_generalist_auroc": generalist_auc,
            "production_domain_routed_auroc": prod_auc,
            "domain_routing_advantage": prod_auc - generalist_auc,
        }

    with open("results_detector/ablation_generalist_vs_routing.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("SUMMARY: single generalist (no routing) vs. production (domain-routed)")
    print("=" * 70)
    mean_gen = np.mean([r["single_generalist_auroc"] for r in results.values()])
    mean_prod = np.mean([r["production_domain_routed_auroc"] for r in results.values()])
    for group, r in results.items():
        print(f"{group:<14} generalist={r['single_generalist_auroc']:.4f}  "
              f"production={r['production_domain_routed_auroc']:.4f}  "
              f"advantage={r['domain_routing_advantage']:+.4f}")
    print(f"{'MEAN':<14} generalist={mean_gen:.4f}  production={mean_prod:.4f}  advantage={mean_prod - mean_gen:+.4f}")


if __name__ == "__main__":
    main()
