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
VGG16 fallback distance rather than proper LPIPS (the `lpips` package failed to
import in the Kaggle environment), this is likely a *lower bound* on what a
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

## Robustness ladder: does the clean-data fusion survive degradation?

Answers the open question above. Design: fit the fusion model **once**, on
clean-image scores only (the same 443-image `none` rung, same methodology as
the headline table) — then apply that *fixed, unmodified* model to every rung
of the official §5.2 ladder (`byteprint/launder.py`'s `OFFICIAL_LADDER`: JPEG
90/70/50/30, blur σ0.5/1.0/2.0, scale 0.5/0.25, noise σ0.02/0.05/0.10, jitter
±20%, crop 80%). CLIP has no native ladder concept (`production_pipeline.py`
just scores whatever images it's given), so each rung's transform is applied to
every test image first via `byteprint.launder.apply` — the same function
DINOv2's `--ladder official` uses internally, so both experts see identically
degraded pixels. AEROBLADE isn't in this run (see "2-expert vs 3-expert" above
for why the 2-expert pair is the one worth running standalone); adding it back
under the ladder is a next step, not done here.

| Rung | CLIP AUC | DINOv2 AUC | **Fused AUC** | CLIP TPR@1% | DINOv2 TPR@1% | **Fused TPR@1%** |
|---|---|---|---|---|---|---|
| none (clean) | 0.9926 | 0.9546 | 0.9950 | 0.8597 | 0.4101 | 0.8957 |
| jpeg:90 | 0.9926 | 0.9493 | 0.9944 | 0.9173 | 0.4101 | 0.8957 |
| jpeg:70 | 0.9892 | 0.9458 | 0.9931 | 0.8705 | 0.4065 | 0.8453 |
| jpeg:50 | 0.9919 | 0.9430 | 0.9948 | 0.8849 | 0.3381 | 0.8633 |
| jpeg:30 | 0.9870 | 0.9396 | 0.9930 | 0.8022 | 0.2734 | 0.8129 |
| blur:0.5 | 0.9928 | 0.9529 | 0.9941 | 0.9029 | 0.4388 | 0.8849 |
| blur:1.0 | 0.9922 | 0.9504 | 0.9935 | 0.8885 | 0.3849 | 0.8597 |
| blur:2.0 | 0.9885 | 0.9409 | 0.9906 | 0.7374 | 0.3489 | 0.7986 |
| scale:0.5 | 0.9924 | 0.9487 | 0.9922 | 0.8633 | 0.4065 | 0.8921 |
| scale:0.25 | 0.9910 | 0.9352 | 0.9916 | 0.8273 | 0.3525 | 0.7734 |
| noise:0.02 | 0.9884 | 0.9401 | 0.9937 | 0.8381 | 0.3741 | 0.8525 |
| noise:0.05 | 0.9875 | 0.9294 | 0.9934 | 0.7950 | 0.2698 | 0.8669 |
| noise:0.10 | 0.9767 | 0.9064 | 0.9824 | 0.7626 | 0.3129 | 0.8525 |
| jitter:0.2 | 0.9932 | 0.9540 | 0.9949 | 0.8237 | 0.4137 | 0.8094 |
| **crop:0.8** | **0.9509** | **0.9414** | **0.9746** | **0.3345** | **0.4245** | **0.6475** |
| **mean (15 rungs)** | **0.9871** | **0.9421** | **0.9914** | **0.8072** | **0.3710** | **0.8367** |

**Yes, the clean-data fusion survives the ladder** — it beats CLIP alone on AUC
in 14 of 15 rungs (essentially tied on the 15th, `scale:0.5`), and beats it on
mean TPR@1%FPR too (0.8367 vs 0.8072), despite never having seen a single
degraded image during fitting.

**The standout finding is `crop:0.8`.** Every other rung, CLIP alone is already
strong (TPR@1%FPR 0.74-0.92). Under crop, it collapses to **0.3345** — missing
two-thirds of fakes at the strict 1%-FPR threshold. Fusion recovers it to
**0.6475**, nearly double, because DINOv2's TPR barely moves on this rung
(0.4245, close to its usual range) — exactly the complementary-failure-mode
story the hypothesis predicted, now demonstrated under real degradation rather
than only on clean data.

**This also validates the transform-implementation concern flagged earlier.**
`clip-detector`'s own `crop80` result (which resizes the crop back up to full
size after cropping) is **0.9990 AUC** — near perfect. This run's `crop:0.8`
(byteprint's version, which leaves the image at its smaller, native-cropped
resolution) gives CLIP only **0.9509 AUC**. Same nominal "80% crop" spec, very
different outcome — concrete evidence the two transform implementations are not
interchangeable, and that byteprint's un-resized crop is the harder, more
realistic test of what a genuinely cropped/reframed profile picture looks like.

A parallel run swapping DINOv2 for SigLIP2-so400m (the backbone Mateo's own
sweep found beats it) is in progress — see "Next steps."

## Next steps

1. **LOGO for DINOv2, AEROBLADE, and the fused model** — `byteprint logo` already
   exists for this; it hasn't been run yet for anything except CLIP alone. This
   is the more important generalization test (unseen-generator performance) and
   the fairest test of AEROBLADE's actual value proposition, since it's designed
   to catch generators neither learned model has trained on.
2. **AEROBLADE under the ladder** — the robustness-ladder run above is CLIP+DINOv2
   only. Re-adding AEROBLADE (fixing the LPIPS import first — item 4 below) is
   the natural extension, though its diffusion-VAE reconstruction cost makes a
   full 15-rung run considerably more expensive than the clean-only run was.
3. **CLIP+SigLIP2 under the ladder** — swaps DINOv2 for the backbone Mateo's sweep
   found beats every alternative (`docs/results-backbone-sweep.md`). In progress.
4. **Fix the LPIPS import** so AEROBLADE gets a fair shot at its real ceiling
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
- `byteprint_fusion_ladder_lite.ipynb` — the CLIP+DINOv2 robustness-ladder
  notebook: fits fusion on the clean rung, applies it unmodified to all 15
  official-ladder rungs. Same three Kaggle Datasets, GPU (T4). Runtime ≈ 2 hours
  (CLIP's per-rung transform loop is the majority of it).
- `results/clip_ladder_scores.json`, `results/dino_ladder_scores.json` — raw
  per-image, per-rung scores (`path`, `label`/`labels`, spec, score) underlying
  the robustness-ladder table.
- `results/ladder_results_by_rung_2expert.json` — the per-rung and mean-across-
  rungs AUC/TPR@1%FPR for CLIP, DINOv2, and fused, as printed in the table above.

Model weights (`runs/probe.joblib`) and embedding caches are intentionally not
committed, per this repo's convention (`CONTRIBUTING.md`) — they're a pure
function of (images, config) and cheap to recompute from the notebook.
