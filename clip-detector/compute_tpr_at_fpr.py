"""
Computes TPR@1%FPR (and other target FPRs) from the saved per-image
robustness predictions, using the same threshold_at_fpr/tpr_at_fpr
methodology as BYTEPRINT's byteprint/metrics.py: the threshold is set
from the REAL-image score distribution only, then recall is measured
on the AI-image class at that threshold. Reported both pooled across
all 16 robustness-grid cells and on the clean cell alone.

Usage:
    python compute_tpr_at_fpr.py
"""

import json

import numpy as np


def threshold_at_fpr(y, s, target_fpr):
    negatives = np.sort(s[y == 0])
    allowed = int(np.floor(target_fpr * negatives.size + 1e-9))
    if allowed >= negatives.size:
        return -np.inf
    cutoff = negatives[negatives.size - allowed - 1]
    return np.nextafter(cutoff, np.inf)


def tpr_at_fpr(y, s, target_fpr):
    positives = s[y == 1]
    thresh = threshold_at_fpr(y, s, target_fpr)
    return float((positives >= thresh).mean())


def main():
    d = json.load(open("results_detector/robustness_per_image.json"))

    labels, scores = [], []
    for cell, records in d.items():
        for r in records:
            labels.append(r["label"])
            scores.append(r["pred_prob"])
    labels = np.array(labels)
    scores = np.array(scores)

    clean = d["clean"]
    cl_labels = np.array([r["label"] for r in clean])
    cl_scores = np.array([r["pred_prob"] for r in clean])

    result = {
        "pooled_16cell": {
            "n": int(len(labels)),
            "n_real": int((labels == 0).sum()),
            "n_ai": int((labels == 1).sum()),
            "tpr_at_1pct_fpr": tpr_at_fpr(labels, scores, 0.01),
            "tpr_at_0_1pct_fpr": tpr_at_fpr(labels, scores, 0.001),
        },
        "clean_only": {
            "n": int(len(cl_labels)),
            "tpr_at_1pct_fpr": tpr_at_fpr(cl_labels, cl_scores, 0.01),
            "tpr_at_0_1pct_fpr": tpr_at_fpr(cl_labels, cl_scores, 0.001),
        },
    }
    print(json.dumps(result, indent=2))
    with open("results_detector/tpr_at_fpr.json", "w") as f:
        json.dump(result, f, indent=2)
    print("Saved to results_detector/tpr_at_fpr.json")


if __name__ == "__main__":
    main()
