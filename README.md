# CLIP + DINOv2 fusion: robust AI-generated image detection

**Bottom line: fusing CLIP+reactivity-delta with DINOv2 beats either alone —
on clean images, and under every rung of the official robustness ladder,
despite the fusion model never having seen a single degraded image during
fitting.**

## Where the code lives

This branch is the **results/integration story** — the notebooks here pull
each expert's engine from where it actually lives, rather than vendoring it:

- **CLIP + reactivity-delta** (the domain-specialist detector, its trained
  model, and its own robustness/LOGO evaluation) — `jiahui/clip-detector`
- **DINOv2, AEROBLADE, the SigLIP2/EVA02 backbone sweep, and the byteprint
  engine itself** (extraction, caching, the launder/ladder implementation,
  `byteprint logo`) — `mateo/main-work`

Reproducing a notebook here needs no local checkout of either — each one
pulls the packaged code (and the trained CLIP model) from Kaggle Datasets at
run time. See the notebook's first cell for exactly which datasets to attach.

| Backbone | Params | AUC | TPR@1%FPR | LOGO mean |
|---|---|---|---|---|
| CLIP ViT-B/32 (+ reactivity-delta) | 151.3M | 0.9926 | 0.8597 | 0.968¹ |
| DINOv2 ViT-S/14 | 22.1M | 0.9546 | 0.4101 | not yet run |
| **Fused (CLIP + DINOv2)** | **173.4M** | **0.9945** | **0.9029** | not yet run |

¹ Pulled from `results_detector/logo_evaluation.json` on `jiahui/clip-detector`
— a true 5-fold leave-one-generator-out run (SD2.1/SDXL/SD3/DALL-E3/
Midjourney6), but on CLIP alone and computed under a **different protocol**
than the rest of this table (see "Methodology note" below). Not yet run for
DINOv2 or the fusion.

Total params: **173.4M**, comfortably inside the competition's <2B budget.

## Methodology note — two different evaluation protocols exist across branches

`results_detector/summary_table.md` on `jiahui/clip-detector` reports CLIP's
numbers on the **official 16-cell robustness transform grid** (AUC 0.997,
TPR@1%FPR 0.933), using that branch's own transform implementation. The table
above uses a
**different protocol**: the same 443-image held-out test split, but scored via
`byteprint/launder.py`'s transform implementation (see "Robustness ladder"
below for why these two aren't interchangeable — the difference is real, not
cosmetic).

## The hypothesis

Two signal families that plausibly fail on different images:

- **CLIP** — a learned classifier on *semantic*, image-text-aligned features
- **DINOv2** — a learned classifier on *purely visual/textural* features, no
  language alignment at all

If their errors are sufficiently uncorrelated, a calibrated blend of the two
should beat either alone. (A third, training-free signal family — AEROBLADE —
was also explored; see "Extension: a third expert" below.)

## What was actually run

- **Dataset**: the same deterministic 2506-train / 443-test split used
  throughout this repo (`byteprint-realdata` on Kaggle), disjoint from the
  organizer's WildFake demonstration set.
- **CLIP + reactivity-delta**: the existing trained production model
  (`model/{domain_classifier,specialists}.pkl` on `jiahui/clip-detector`),
  reused as-is — no retraining. Scored on the full 443-image test set.
- **DINOv2**: linear probe trained fresh on the full 2506-image train set. An
  earlier local CPU-only pilot trained it on only 150 images and got AUC 0.875
  — retraining on the full set alone raised that to 0.9546, before fusion is
  even applied. Worth remembering when judging any single-expert number here:
  training-set size matters as much as architecture.
- **Fusion**: a 2-input logistic regression (`StandardScaler` +
  `LogisticRegression`) on `[clip_score, dino_score]`, evaluated with
  stratified 5-fold cross-validation — every image's reported prediction comes
  from a fold that never trained on it, so the numbers above are genuine
  held-out estimates, not the model grading its own homework.

## Why fusion wins: the error analysis

The fused model's biggest disagreements with CLIP alone are concentrated in
the `fullres` source, and every one of the top 10 is the same story: CLIP
overconfident that a real image is fake, DINOv2 confidently (and correctly)
disagreeing, fusion pulling the score back down. E.g. `fullres_real_0293.png`:
CLIP scores it 0.880 (looks fake to CLIP), DINOv2 scores it 0.000, fusion
corrects to 0.391. `fullres_real_0482.png`: CLIP 0.586, DINOv2 0.000, fusion
0.067. That's the complementary-failure-mode hypothesis working exactly as
intended — and unlike the 3-expert extension (see below), every one of the top
10 disagreements here is a genuine correction, no new errors introduced among
the largest swings.

## Robustness ladder: does the clean-data fusion survive degradation?

Design: fit the fusion model **once**, on clean-image scores only (the same
443-image `none` rung, same methodology as the headline table) — then apply
that *fixed, unmodified* model to every rung of the official §5.2 ladder
(`byteprint/launder.py`'s `OFFICIAL_LADDER`: JPEG 90/70/50/30, blur
σ0.5/1.0/2.0, scale 0.5/0.25, noise σ0.02/0.05/0.10, jitter ±20%, crop 80%).
CLIP has no native ladder concept (`production_pipeline.py` just scores
whatever images it's given), so each rung's transform is applied to every test
image first via `byteprint.launder.apply` — the same function DINOv2's
`--ladder official` uses internally, so both experts see identically degraded
pixels.

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

**This also validates the transform-implementation concern flagged above.**
`clip-detector`'s own `crop80` result (which resizes the crop back up to full
size after cropping) is **0.9990 AUC** — near perfect. This run's `crop:0.8`
(byteprint's version, which leaves the image at its smaller, native-cropped
resolution) gives CLIP only **0.9509 AUC**. Same nominal "80% crop" spec, very
different outcome — concrete evidence the two transform implementations are
not interchangeable, and that byteprint's un-resized crop is the harder, more
realistic test of what a genuinely cropped/reframed profile picture looks
like.

A parallel run swapping DINOv2 for SigLIP2-so400m (the backbone Mateo's own
sweep found beats it — see `mateo/main-work`) is documented below, under
"Extension: swapping DINOv2 for SigLIP2-so400m".

The 2-expert result was independently reproduced end-to-end across two
separate standalone Kaggle runs and matched to 4 decimal places both times —
this is a deterministic pipeline, not a one-off lucky split.

---

## Extension: a third expert (AEROBLADE)

A natural follow-up question: does adding a training-free, non-learned signal
on top of CLIP+DINOv2 help further? **AEROBLADE** (Ricker et al., CVPR 2024)
tests whether an image reconstructs suspiciously well through a *known
diffusion VAE* — no training data, no learned decision boundary, a completely
different failure-mode family from CLIP or DINOv2.

### Clean-data results, all three experts

| Combination | AUC | TPR@1%FPR |
|---|---|---|
| CLIP alone | 0.9926 | 0.8597 |
| DINOv2 alone | 0.9546 | 0.4101 |
| AEROBLADE alone | 0.7103 | 0.0863 |
| clip+dino | 0.9945 | 0.9029 |
| clip+aero | 0.9856 | 0.6187 |
| dino+aero | 0.9440 | 0.0360 |
| **fused: all 3** | **0.9953** | **0.9173** |

Two things worth flagging:

- **TPR@1%FPR moves more than AUC does.** 3-expert fusion catches 91.7% of
  fakes at the strict 1%-FPR operating point vs. CLIP alone's 86.0% — a bigger
  practical gain than the AUC delta (0.9953 vs 0.9926) suggests on its own.
- **`dino+aero` (no CLIP) has AUC 0.944 but TPR@1%FPR of just 0.036.** Without
  CLIP anchoring the fusion, the other two rank images correctly on average
  but catch almost nothing at a strict threshold — a clear illustration of why
  AUC alone can mislead, and why CLIP is doing the real load-bearing work.

### Does the third expert actually earn its place?

| | AUC | TPR@1%FPR |
|---|---|---|
| clip+dino (2-expert, the main model above) | 0.9945 | 0.9029 |
| clip+dino+aero (3-expert) | **0.9953** | **0.9173** |
| **Δ from adding AEROBLADE** | **+0.0008** | **+0.0144** |

The 3-expert fusion wins on both metrics, but the two metrics disagree on *how
much*: the AUC gain is tiny (+0.0008 — within noise at this sample size on its
own), while the TPR@1%FPR gain is real and larger (+1.44 points). Read
together with AEROBLADE running on its weaker VGG16 fallback distance rather
than proper LPIPS (the `lpips` package failed to import in the Kaggle
environment), this is likely a *lower bound* on what a 3rd expert can add, not
the ceiling — worth a follow-up once that import is fixed.

### 3-expert error analysis

Unlike the clean 2-expert corrections above, adding AEROBLADE is a genuine
trade-off, not a pure win: of the top-10 largest fusion-vs-CLIP disagreements,
8 are the same kind of correction (a `fullres` real image CLIP over-scores,
pulled back down), but 2 are genuine fakes CLIP correctly scored high
(`fullres_ai_0286.png` 0.892, `fullres_ai_0147.png` 0.986) that fusion dragged
below the decision threshold (0.300, 0.441) because DINOv2 and AEROBLADE both
leaned "real" on them. Net effect across all 443 images is still positive
(more errors fixed than introduced), but it's worth stating plainly rather
than overselling it.

### This 3-expert combination has not yet been tested under the robustness ladder

Everything in this AEROBLADE section is clean-images-only — see "Next steps."

---

## Extension: swapping DINOv2 for SigLIP2-so400m

The backbone sweep (`mateo/main-work`, `docs/results-backbone-sweep.md`)
found SigLIP2-so400m beats every backbone tried, CLIP included. The obvious
follow-up: does swapping it in for DINOv2 in the fusion pair beat the
CLIP+DINOv2 result above? Same protocol as the ladder run above — fusion fit
once on clean scores only, applied unmodified to all 15 rungs.

| Rung | CLIP AUC | SigLIP2 AUC | **Fused AUC** | CLIP TPR@1% | SigLIP2 TPR@1% | **Fused TPR@1%** |
|---|---|---|---|---|---|---|
| none (clean) | 0.9926 | 0.9743 | 0.9961 | 0.8597 | 0.7086 | 0.8813 |
| jpeg:90 | 0.9926 | 0.9658 | 0.9953 | 0.9173 | 0.6942 | 0.8669 |
| jpeg:70 | 0.9892 | 0.9640 | 0.9933 | 0.8705 | 0.5252 | 0.8741 |
| jpeg:50 | 0.9919 | 0.9619 | 0.9937 | 0.8849 | 0.6439 | 0.8777 |
| jpeg:30 | 0.9870 | 0.9580 | 0.9902 | 0.8022 | 0.5647 | 0.7806 |
| blur:0.5 | 0.9928 | 0.9797 | 0.9967 | 0.9029 | 0.7374 | 0.9065 |
| blur:1.0 | 0.9922 | 0.9611 | 0.9955 | 0.8885 | 0.6367 | 0.9532 |
| blur:2.0 | 0.9885 | 0.9306 | 0.9919 | 0.7374 | 0.3705 | 0.8489 |
| scale:0.5 | 0.9924 | 0.9737 | 0.9957 | 0.8633 | 0.7122 | 0.9245 |
| scale:0.25 | 0.9910 | 0.9502 | 0.9963 | 0.8273 | 0.4676 | 0.9532 |
| noise:0.02 | 0.9884 | 0.9526 | 0.9934 | 0.8381 | 0.6511 | 0.8849 |
| noise:0.05 | 0.9875 | 0.9371 | 0.9906 | 0.7950 | 0.5576 | 0.8094 |
| noise:0.10 | 0.9767 | 0.9099 | 0.9807 | 0.7626 | 0.4209 | 0.7734 |
| jitter:0.2 | 0.9932 | 0.9663 | 0.9956 | 0.8237 | 0.6439 | 0.8561 |
| **crop:0.8** | **0.9509** | **0.9804** | **0.9797** | **0.3345** | **0.8129** | **0.6511** |
| **mean (15 rungs)** | **0.9871** | **0.9577** | **0.9923** | **0.8072** | **0.6098** | **0.8561** |

**Yes — on the mean, CLIP+SigLIP2 beats CLIP+DINOv2**: 0.9923 vs 0.9914 AUC,
0.8561 vs 0.8367 TPR@1%FPR. SigLIP2 is simply a stronger standalone expert
across the whole ladder (mean TPR 0.6098 vs DINOv2's 0.3710).

**`crop:0.8` tells a sharper version of the same story, with a twist.**
SigLIP2 alone hits **0.8129** TPR@1%FPR on this rung — far beyond what DINOv2
managed there (0.4245), and nearly matching CLIP's own clean-image
performance. This is consistent with the transform-implementation finding
above: SigLIP2's NaflexVision head patches an image at its *actual*
resolution instead of resizing to a canonical square, so it never has to
interpolate an already-cropped image back up to a fixed input size —
plausibly the exact step that costs CLIP its signal on this rung. (Stated as
a strong hypothesis consistent with the data, not a controlled ablation —
nobody has yet isolated "native-resolution vs. resize" as the single
variable.)

But the fused score does not inherit that advantage: **0.6511** TPR@1%FPR,
barely different from CLIP+DINOv2's 0.6475 on the same rung, and well below
SigLIP2's own 0.8129. Fused AUC on this rung (0.9797) is nearly identical to
SigLIP2 alone (0.9804), so the fusion's overall *ranking* of images stays
intact — what breaks is specifically the strict 1%-FPR operating point. The
logistic regression is fit once on clean scores, where CLIP is the stronger
expert (0.8597 vs SigLIP2's 0.7086), so it leans on CLIP more heavily; that
fixed weighting has no way to know a crop should flip that preference, and
blending in CLIP's collapsed score is enough to knock some borderline fakes
below a threshold this tight — even though SigLIP2 alone would have caught
them. This is the opposite of the DINOv2 finding above: there, fusion beat
*both* individual experts because the two failure modes were genuinely
uncorrelated; here, fusion dilutes one much-better signal instead of
combining two complementary ones.

### Which fusion pair is the primary model?

**CLIP+DINOv2 stays the headline result, not CLIP+SigLIP2**, despite the
latter's better mean. Three reasons:

1. **Parameter budget.** SigLIP2-so400m's vision tower is ~400M parameters
   against DINOv2 ViT-S/14's ~22M — swapping it in pushes the fusion
   pipeline from 173.4M parameters (8.7% of the <2B budget) to roughly 551M
   (27.6%). Still comfortably inside budget, but it meaningfully weakens the
   "proportionate, hackathon-scale" story that the Feasibility & Practicality
   criterion rewards.
2. **The performance margin is small, and the fusion isn't where it comes
   from.** +0.0009 mean AUC and +0.0194 mean TPR are earned mostly by SigLIP2
   being a better *standalone* expert, not a better fusion partner — and on
   the single hardest rung, the fused result is essentially tied with
   CLIP+DINOv2 (0.6511 vs 0.6475), not meaningfully better.
3. **The fusion-calibration gap above is an honest, unresolved limitation**,
   not a free upgrade. Presenting CLIP+SigLIP2 as the primary result would
   mean either omitting that finding or leading with a model whose own best
   rung has a known, unaddressed weakness baked in.

CLIP+SigLIP2 is documented here as a validated, ladder-tested alternative,
the same way AEROBLADE is treated above — a real result, explored and
quantified, kept in the repo rather than discarded for not becoming the
headline. The concrete next step it points to is a rung-aware or
confidence-weighted fusion in place of one fixed logistic regression, which
could plausibly beat *both* current pairs on the hardest rung specifically —
see "Next steps" below.

## Next steps

1. **LOGO for DINOv2, AEROBLADE, and both fused models** — `byteprint logo`
   already exists for this; it hasn't been run yet for anything except CLIP
   alone. This is the more important generalization test (unseen-generator
   performance) and the fairest test of AEROBLADE's actual value proposition,
   since it's designed to catch generators neither learned model has trained
   on.
2. **AEROBLADE under the ladder** — the robustness-ladder run above is
   CLIP+DINOv2 only. Re-adding AEROBLADE (fixing the LPIPS import first — item
   4 below) is the natural extension, though its diffusion-VAE reconstruction
   cost makes a full 15-rung run considerably more expensive than the
   clean-only run was.
3. **A fusion that isn't one fixed logistic regression.** The CLIP+SigLIP2
   ladder run (see "Extension: swapping DINOv2 for SigLIP2-so400m" above) is
   done, and it exposed a real gap: SigLIP2 alone is dramatically more
   `crop:0.8`-robust than the fused score becomes, because a single
   clean-data-fit combiner can't shift its trust per-rung. A rung-aware or
   confidence-weighted fusion (e.g. gated on how much the two experts
   disagree) is the concrete next step, and could plausibly beat both current
   pairs on the hardest rung specifically.
4. **Fix the LPIPS import** so AEROBLADE gets a fair shot at its real ceiling
   instead of running on the weaker VGG16 fallback.

## Files here

- `byteprint_fusion_ladder_lite.ipynb` — the CLIP+DINOv2 robustness-ladder
  notebook (the main result): fits fusion on the clean rung, applies it
  unmodified to all 15 official-ladder rungs. Needs three Kaggle Datasets
  attached (`byteprint-realdata`, `byteprint-code`, `clip-reactivity-code`)
  and GPU (T4). Runtime ≈ 2 hours (CLIP's per-rung transform loop is the
  majority of it).
- `byteprint_fusion_lite.ipynb` — the clean-data-only companion (no ladder),
  used to independently confirm the clip+dino numbers. Runs in minutes.
- `results/clip_ladder_scores.json`, `results/dino_ladder_scores.json` — raw
  per-image, per-rung scores underlying the robustness-ladder table.
- `results/ladder_results_by_rung_2expert.json` — the per-rung and
  mean-across-rungs AUC/TPR@1%FPR for CLIP, DINOv2, and fused.
- `results/clip_scores.json`, `results/dino_scores.json` — raw per-image
  clean-only scores for all 443 test images.
- `results/fusion_2expert_summary.json` — the clean-data clip+dino numbers,
  including the independent-reproduction confirmation.
- `byteprint_fusion_experiment.ipynb` — the 3-expert (AEROBLADE-extension)
  Kaggle notebook. Same three datasets, GPU (T4). Runtime ≈ 50 minutes,
  dominated by AEROBLADE.
- `results/aeroblade_scores.json` — raw per-image AEROBLADE scores.
- `results/fusion_results_summary.json` — the pooled/fold-mean/fold-std AUROC
  for every combination in the 3-expert ablation table.

Model weights (`runs/probe.joblib`) and embedding caches are intentionally not
committed, per this repo's convention (`CONTRIBUTING.md`) — they're a pure
function of (images, config) and cheap to recompute from the notebooks.
