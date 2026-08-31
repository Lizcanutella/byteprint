# Summary table (BYTEPRINT-comparable metrics)

Computed for direct comparison against BYTEPRINT's own reporting conventions
(`byteprint/metrics.py`'s AUC / TPR@FPR / leave-one-generator-out).

| backbone | params | AUC | TPR@1%FPR | LOGO mean |
|---|---|---|---|---|
| CLIP ViT-B/32 (frozen, pre-projection + reactivity-delta) | 151.3M | 0.997 | 0.933 | 0.968 |

## What each column means here

- **params**: the frozen CLIP ViT-B/32 backbone (151,277,313 params). The
  actually-trained part on top is tiny by comparison: 5 domain specialists
  x 1,537 logistic-regression weights each = 7,685 trainable params total.
- **AUC**: mean AUROC across the official 16-cell robustness transform grid
  (`results_detector/robustness_table.json`). Two other AUC numbers exist
  depending on what's being asked: 0.999 on the clean-only held-out test,
  0.977 pooled on the fully-disjoint generator-diagnostic set
  (`results_detector/generator_diagnostic_results.json`).
- **TPR@1%FPR**: `compute_tpr_at_fpr.py` - same `threshold_at_fpr`/
  `tpr_at_fpr` methodology as `byteprint/metrics.py` (threshold set from the
  real-image score distribution only, then recall measured on the AI class).
  0.933 pooled across all 16 transform cells; 0.957 on clean images alone.
  Full numbers, including TPR@0.1%FPR: `results_detector/tpr_at_fpr.json`.
- **LOGO mean**: `logo_evaluation.py` - a true 5-fold leave-one-generator-out
  evaluation (SD2.1/SDXL/SD3/DALL-E3/Midjourney6), each fold training a
  classifier with that generator entirely excluded, using production's exact
  feature representation (CLIP pre-projection embedding + jpeg_q50
  reactivity-delta, 1536-dim). Per-generator breakdown:

  | held-out generator | AUROC |
  |---|---|
  | SD2.1 | 0.994 |
  | SDXL | 0.994 |
  | SD3 | 0.961 |
  | DALL-E3 | 0.969 |
  | Midjourney6 | 0.918 |

  Caveat: this trains a single classifier per fold, not a full retrain of
  the 5-domain routed specialist architecture (would have meant 5 full
  production retrains) - it uses production's exact feature representation
  on a true K-fold split, not the full routed architecture. Full result:
  `results_detector/logo_evaluation.json`.
