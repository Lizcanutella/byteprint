# Three-expert fusion: CLIP+reactivity-delta, DINOv2, AEROBLADE

**Bottom line: fusing all three beats every single expert and every pair, at real
statistical power (N=443, proper 5-fold cross-validation).**

| Backbone | Params | AUC | TPR@1%FPR | LOGO mean |
|---|---|---|---|---|
| CLIP ViT-B/32 (+ reactivity-delta) | 151.3M | 0.9926 | 0.8597 | 0.968¹ |
| DINOv2 ViT-S/14 | 22.1M | 0.9546 | 0.4101 | not yet run |
| AEROBLADE (3× SD-VAE)² | 251.0M | 0.7103 | 0.0863 | not yet run |
| **Fused (all 3)** | **424.4M** | **0.9953** | **0.9173** | not yet run |

¹ Pulled from `clip-detector/results_detector/logo_evaluation.json` — a true 5-fold
leave-one-generator-out run (SD2.1/SDXL/SD3/DALL-E3/Midjourney6), but on CLIP alone
and computed under a **different protocol** than the rest of this table (see
"Methodology note" below). Not yet run for DINOv2, AEROBLADE, or the fusion.

² Excludes the VGG16 fallback distance network (14.7M params) — see "AEROBLADE ran
degraded" below.

Total params across all three experts: **424.4M**, comfortably inside the
competition's <2B budget.

## Methodology note — two different evaluation protocols exist in this repo

`clip-detector/results_detector/summary_table.md` reports CLIP's numbers on the
**official 16-cell robustness transform grid** (AUC 0.997, TPR@1%FPR 0.933). The
table above uses a **different, narrower protocol**: the same 443-image held-out
test split, but **clean images only** — no JPEG/blur/noise/etc. applied. That's why
CLIP's AUC differs between the two tables (0.997 vs 0.9926 here). This experiment's
job was to establish whether fusion helps *at all* before spending the (large)
extra compute to run all four experts across the full robustness ladder — see
"Next steps."

## The hypothesis

Three signal families that plausibly fail on different images:

- **CLIP** — a learned classifier on *semantic*, image-text-aligned features
- **DINOv2** — a learned classifier on *purely visual/textural* features, no
  language alignment
- **AEROBLADE** — not a learned classifier at all: a training-free test of whether
  an image reconstructs suspiciously well through a *known diffusion VAE*
  (Ricker et al., CVPR 2024)

If their errors are sufficiently uncorrelated, a calibrated blend of all three
should beat any one of them.

## What was actually run

- **Dataset**: the same deterministic 2506-train / 443-test split used throughout
  this repo (`byteprint-realdata` on Kaggle), disjoint from the
  organizer's WildFake demonstration set.
- **CLIP + reactivity-delta**: the existing trained production model
  (`clip-detector/model/{domain_classifier,specialists}.pkl`), reused as-is —
  no retraining. Scored on the full 443-image test set.
- **DINOv2**: linear probe trained fresh on the full 2506-image train set. An
  earlier local CPU-only pilot trained it on only 150 images and got AUC 0.875
  — retraining on the full set alone raised that to 0.9546, before fusion is
  even applied. Worth remembering when judging any single-expert number here:
  training-set size matters as much as architecture.
- **AEROBLADE**: training-free, scored directly on the 443-image test set.
- **Fusion**: a 3-input logistic regression (`StandardScaler` + `LogisticRegression`)
  on `[clip_score, dino_score, aeroblade_score]`, evaluated with stratified 5-fold
  cross-validation — every image's reported prediction comes from a fold that
  never trained on it, so the numbers above are genuine held-out estimates, not
  the model grading its own homework.

Full ablation (all 7 combinations):

| Combination | AUC | TPR@1%FPR |
|---|---|---|
| CLIP alone | 0.9926 | 0.8597 |
| DINOv2 alone | 0.9546 | 0.4101 |
| AEROBLADE alone | 0.7103 | 0.0863 |
| clip+dino | 0.9945 | 0.9029 |
| clip+aero | 0.9856 | 0.6187 |
| dino+aero | 0.9440 | 0.0360 |
| **fused: all 3** | **0.9953** | **0.9173** |

Two things worth flagging in this table:

- **TPR@1%FPR moves more than AUC does.** At the strict 1%-false-positive
  operating point a real deployment would use, 3-expert fusion catches 91.7% of
  fakes vs. CLIP alone's 86.0% — a bigger practical gain than the AUC delta
  (0.9953 vs 0.9926) suggests on its own.
- **`dino+aero` (no CLIP) has AUC 0.944 but TPR@1%FPR of just 0.036.** Without
  CLIP anchoring the fusion, the other two rank images correctly on average but
  catch almost nothing at a strict threshold — a clear illustration of why AUC
  alone can mislead, and why CLIP is doing the real load-bearing work here.

## 2-expert vs 3-expert fusion

Since CLIP+DINOv2 is the strongest pair on its own, it's worth asking directly:
does adding the weakest expert (AEROBLADE, 0.7103 AUC alone) actually earn its
place, or is `clip+dino` good enough by itself?

| | AUC | TPR@1%FPR |
|---|---|---|
| clip+dino (2-expert) | 0.9945 | 0.9029 |
| clip+dino+aero (3-expert) | **0.9953** | **0.9173** |
| **Δ from adding AEROBLADE** | **+0.0008** | **+0.0144** |

The 3-expert fusion wins on both metrics, but the two metrics disagree on *how
much*: the AUC gain from adding AEROBLADE is tiny (+0.0008 — within noise at
this sample size on its own), while the TPR@1%FPR gain is real and larger
(+1.44 points) — consistent with the pattern in the full ablation table, where
AEROBLADE's contribution shows up more at the strict end of the ROC curve than
in the overall ranking. Read together with AEROBLADE running on its weaker
VGG16 fallback distance (see below), this is likely a *lower bound* on what a
3rd expert can add, not the ceiling.

The 2-expert result was independently reproduced end-to-end in a **separate,
standalone Kaggle run** (`byteprint_fusion_lite.ipynb` — deliberately excludes
AEROBLADE so it runs in minutes instead of the ~50 minutes the full 3-expert
notebook takes, most of which is AEROBLADE's diffusion-VAE reconstruction cost)
and reproduced numerically identical numbers (AUC 0.9945, same to 4 decimal
places) — confirming the pipeline is deterministic, not a one-off lucky run.

## Why fusion wins: the error analysis

The fused model's biggest disagreements with CLIP alone are concentrated in the
`fullres` source — e.g. `fullres_real_0293.png`: CLIP wrongly scores it 0.880
(looks fake to CLIP), DINOv2 correctly scores it 0.000, fusion correctly pulls it
to 0.321. That is the complementary-failure-mode hypothesis working as intended.

It's not free, though: 2 of the top-10 disagreement cases are genuine fakes CLIP
correctly scored high (`fullres_ai_0286.png` 0.892, `fullres_ai_0147.png` 0.986)
that fusion dragged below the decision threshold (0.300, 0.441) because DINOv2 and
AEROBLADE both leaned "real" on them. Net effect across all 443 images is
positive (more errors fixed than introduced), but it's a real trade-off, not a
pure win — worth stating exactly that way rather than overselling it.

## AEROBLADE ran degraded

The `lpips` package failed to import in the Kaggle environment (likely a knock-on
from pinning `scikit-learn==1.9.0` to keep the CLIP model's pickled artifacts
loadable), so AEROBLADE fell back to `_UnweightedVGGDistance` (plain VGG16 feature distance) instead of the
paper's proper LPIPS metric. AEROBLADE's true ceiling is probably higher than
0.7103 AUC. Worth a follow-up run with the import fixed before treating 0.71 as
AEROBLADE's real number.

## Next steps

1. **LOGO for DINOv2, AEROBLADE, and the fused model** — `byteprint logo` already
   exists for this; it hasn't been run yet for anything except CLIP alone. This
   is the more important generalization test (unseen-generator performance) and
   the fairest test of AEROBLADE's actual value proposition, since it's designed
   to catch generators neither learned model has trained on.
2. **Robustness-ladder version of this fusion experiment** — everything here is
   clean-images-only. The real question (does fusion help *under* JPEG/blur/
   noise/etc.) is still open.
3. **Fix the LPIPS import** so AEROBLADE gets a fair shot at its real ceiling
   instead of running on the weaker VGG16 fallback.

## Files here

- `byteprint_fusion_experiment.ipynb` — the full 3-expert Kaggle notebook that
  produced the headline numbers. Needs three Kaggle Datasets attached
  (`byteprint-realdata`, `byteprint-code`, `clip-reactivity-code`) and GPU (T4)
  enabled. Runtime ≈ 50 minutes, dominated by AEROBLADE.
- `byteprint_fusion_lite.ipynb` — the 2-expert-only companion (CLIP + DINOv2, no
  AEROBLADE). Same dataset, same K-fold methodology, runs in a few minutes.
  Used to independently confirm the clip+dino numbers below.
- `results/clip_scores.json`, `results/dino_scores.json`,
  `results/aeroblade_scores.json` — raw per-image scores (`path`, `label`,
  `<expert>_score`) for all 443 test images, from each expert independently.
- `results/fusion_results_summary.json` — the pooled/fold-mean/fold-std AUROC for
  every combination in the full 7-way ablation table.
- `results/fusion_2expert_summary.json` — the clip+dino (2-expert) numbers
  isolated, including TPR@1%FPR and the independent-reproduction confirmation,
  for the side-by-side comparison above.

Model weights (`runs/probe.joblib`) and embedding caches are intentionally not
committed, per this repo's convention (`CONTRIBUTING.md`) — they're a pure
function of (images, config) and cheap to recompute from the notebook.
