# Robust AI-Image Detector (hackathon submission)

Detects AI-generated vs. authentic images, robust to real-world
post-processing (compression, blur, resizing, noise, color adjustment,
cropping). Built for Hackathon Challenge 5, "Robust Detection of
AI-Generated Images Under Real-World Transformations."

## Project overview

**Approach (production, `production_pipeline.py`)**: a cheap classical
no-reference-feature domain classifier (`domain_classifier.py` -
RandomForest over Laplacian variance / high-frequency ratio / JPEG
blockiness / contrast / mean saturation) first detects which of 5
degradation groups an image looks like it's in (clean / JPEG / spatial
[blur+resize+crop] / noise / color-jitter) - the organizers confirmed
robustness test images are each degraded in exactly ONE domain at a
time, never stacked, which is what makes this routing approach
well-posed. The image is then classified by a domain-SPECIALIZED
logistic-regression head trained on that domain's augmented images,
using CLIP's PRE-projection vision features (768-dim, `clip-vit-base-
patch32`, ~151M params - see "Model iteration" below for why
pre-projection beats the standard post-projection embedding here). This
builds on the published "Universal Fake Image Detection" result (Ojha
et al., CVPR 2023): frozen CLIP features + a linear probe generalize
well across generators.

**Why this approach, not hand-crafted forensics**: before landing on
CLIP embeddings, this project spent significant effort on 7 hand-crafted
statistical/forensic signals (noise-residual reactivity, spectral
analysis, DCT statistics, cross-channel correlation - see "Part 2"
below), each rigorously validated for content leakage, dataset-specific
shortcuts, and cross-dataset consistency. **None reached usable
accuracy** (best effective AUROC ~0.62 on the weaker of two datasets).
That negative result is the direct motivation for the learned approach.

## Model iteration (how we got from baseline to production)

Initially trained on 3,025 images pooled from three single-source
datasets (`saberzl/SID_Set` [organizer-recommended, 600+600 sampled],
`itsLeen/deepfake_vs_real_image_detection` [999], and
`itsLeen/deepfake_vs_real_image` [826]), 15% held-out test split (454
images). Mean effective AUROC across the full 16-cell transform grid,
for each architecture variant tried, in order:

| variant | mean AUROC (16 cells) | notes |
|---|---:|---|
| Baseline: single generalist head, post-projection CLIP, one random augmentation/image | 0.9316 | |
| + balanced augmentation (every image gets 1 sample from every domain, not 1 random pick) | 0.9408 | single generalist, still post-projection |
| + domain-specialist routing instead of one generalist (post-projection) | 0.9362 | worse than balanced-alone - routing hurt on ambiguous/mild cases (see below) |
| Pre-projection CLIP features alone (no balanced aug, no routing) | n/a (clean-test only: 0.9506) | single biggest individual lever found |
| Balanced augmentation + pre-projection, naive single generalist | clean-test: 0.9433 | **stacking FAILED** - worse than either individual improvement |
| Balanced augmentation + pre-projection + domain-specialist routing | **0.9477** | best architecture found - but see the data-quality fix below |

Two findings drove this architecture:
1. **Domain-specialist routing only helps when routing is confident.**
   With post-projection features, specialists *lost* to a plain
   generalist exactly on cells where the domain classifier's routing
   accuracy was low (e.g. `clean`: 34% routing accuracy, specialist
   AUROC dropped from 0.946 to 0.931) - a misrouted image gets a
   confidently-wrong specialist.
2. **Naive feature+augmentation stacking overfits; structured stacking
   (routing) is what unlocks it.** Pre-projection features and balanced
   augmentation each help alone, but training ONE generalist head on
   both combined made things worse (train AUROC hit 0.999, a classic
   overfitting signature). Routing to per-domain specialists *in the
   pre-projection space* resolved this: specialists there beat the
   generalist in 15/16 cells - including cells with LOW routing
   accuracy, because the richer pre-projection space makes each
   specialist a more broadly-capable classifier on its own, so a wrong
   routing decision costs much less than in the post-projection space.

## Data-quality fix: one training source was mislabeled/off-topic

Manual inspection of `itsLeen/deepfake_vs_real_image` (the "fullres2"
source above) found it is **not** a real-photo-vs-AI-image dataset: its
raw class names are `Real_Art` / `AI_Art`. Visual audit of sampled
images confirmed:
- **`Real_Art`** is dominated by paintings and digital illustrations
  (an oil-painted violinist in a gilt frame, a flower still-life
  painting, watercolor abstracts) - not authentic photographs.
- **`AI_Art`** is also contaminated, in a worse way: it contains real
  *photographs of media coverage about AI art* - a magazine graphic
  captioned "KASHTANOVA MIDJOURNEY" (referencing the real Kris
  Kashtanova AI-copyright case) and a promotional TV-show cast collage -
  i.e. genuinely real photographs mislabeled as AI-generated.

This dataset was **removed from training entirely** (not filtered - the
contamination pattern wasn't cheaply separable from the rest).
Retraining on the remaining 2,199 images (`fullres` + `sid_set` only,
1,869 train / 330 held-out test, same architecture/procedure as the
best row above) gave a large, uniform improvement:

| | mean AUROC (16 cells) | mean accuracy |
|---|---:|---:|
| Best architecture, contaminated 3-source data | 0.9477 | 0.8771 |
| **Same architecture, clean 2-source data (final production)** | **0.9821** | **0.9366** |

This +0.034 AUROC jump from removing one bad data source is larger than
the gain from *any* single architecture change tried above (balanced
augmentation: +0.009; pre-projection: comparable order; specialist
routing: comparable order) - a concrete illustration that a proper
data-source audit had more leverage than further model iteration would
have. The final `predict.py`/`model/` artifacts are all from this
clean-data run; the contaminated-data models are kept for comparison
(`model/*_CONTAMINATED.*`, `results_detector/experiments/`).

## Confidence-gated routing (final refinement)

The domain classifier's hard top-1 routing decision (used above) throws
away information: when it's genuinely unsure (e.g. mild JPEG vs. clean,
or light noise vs. clean - its hardest calls), committing fully to one
specialist bets everything on a possibly-wrong guess. `production_pipeline.py`
now defaults to a **soft mixture-of-experts** instead: every specialist's
prediction is weighted by the domain classifier's own probability that
the image is in that domain -
`P(AI) = sum_g P(domain=g) * specialist_g.predict_proba(image)` -
rather than picking one winner. No confidence threshold to tune; it
naturally reduces to hard routing when the domain classifier is
confident, and blends multiple opinions when it isn't.

| | mean AUROC (16 cells) | mean accuracy |
|---|---:|---:|
| Hard top-1 routing | 0.9821 | 0.9366 |
| **Confidence-gated (soft mixture) - final** | **0.9840** | **0.9379** |

A modest, real gain (+0.0019 AUROC), concentrated exactly where
expected: the biggest wins are on the most ambiguous cells (`clean`:
+0.0086, `jpeg_q90`: +0.0084, `noise_s0.02`: +0.0109 - all conditions
easily confused with "clean"), with small losses (<0.005) on a few
already-confident cells where blending in other specialists adds a
touch of dilution. Net: 9/16 cells improved. Both modes are available
via `predict_proba(..., confidence_gated=True/False)`; the hard-routing
numbers are preserved in `results_detector/robustness_table_hard_routing.json`.

## Generator-diversity gap: discovery and fix

After the confidence-gated model above (0.984 mean grid AUROC) was
promoted, real-world testing by the user on a ChatGPT/DALL-E 3 image
produced a confident, wrong prediction (`pred=0.008`, i.e. "confidently
real" on an actually-AI image). Rather than treat this as a one-off,
we built a proper diagnostic: `fetch_generator_diagnostic.py` /
`evaluate_generator_diagnostic.py` pull a **held-out, generator-labeled
sample from `Rajarshi-Roy-research/Defactify_Image_Dataset`'s TEST
split** (real/MS-COCO + Stable Diffusion 2.1/SDXL/SD3 + DALL-E 3 +
Midjourney 6, 60 images each, never used in training) and measure
accuracy broken down by generator.

**Result: the DALL-E 3 miss was a symptom, not an isolated bug.** The
pre-existing model generalized poorly to almost every generator it
hadn't seen in training:

| generator | accuracy (before) |
|---|---:|
| real (control) | 98.3% |
| Stable Diffusion 2.1 | 45.0% (worse than chance) |
| Stable Diffusion XL | 75.0% |
| Stable Diffusion 3 | 53.3% (near chance) |
| DALL-E 3 | 68.3% |
| Midjourney 6 | 48.3% (worse than chance) |

0.984 mean AUROC on our own robustness grid was real, but it was
measuring in-distribution generalization (held-out images from the
*same* training sources), not the broader generator landscape that
actually matters in deployment.

**Fix**: `fetch_defactify_train.py` pulls 150 real + 150-each-generator
images from Defactify's **TRAIN** split (strictly disjoint from the
TEST-split diagnostic above) and adds them as a third training source
(`train_classifier.DATASETS`). Retraining the full pipeline (baseline,
domain classifier, specialists) on this generator-diverse pool and
re-running the *same, untouched* diagnostic set:

| generator | accuracy (before) | accuracy (after) | Δ |
|---|---:|---:|---:|
| real (control) | 98.3% | 93.3% | -5.0pp |
| Stable Diffusion 2.1 | 45.0% | 73.3% | **+28.3pp** |
| Stable Diffusion XL | 75.0% | 91.7% | +16.7pp |
| Stable Diffusion 3 | 53.3% | 90.0% | **+36.7pp** |
| **DALL-E 3** | 68.3% | **95.0%** | **+26.7pp** |
| Midjourney 6 | 48.3% | 86.7% | **+38.4pp** |
| **pooled AUROC** | 0.9185 | **0.9597** | +0.041 |

Every previously near-or-below-chance generator is now solidly
detected, at the cost of a modest 5-point dip on real images (a
reasonable trade-off, not a regression - the model is now less
overfit to the specific training generators' fingerprints). The
original user-reported image moved from `pred=0.008` to `pred=0.092` -
a real, ~11x shift toward correct, though this specific hard example
still doesn't cross the 0.5 threshold (see `results_detector/
known_failure_examples/` for the image and both predictions,
documented transparently rather than cherry-picked).

This is the single largest improvement found in this project on a
*generalization* metric - larger than any architecture change - and it
came from the same lesson as the earlier data-quality fix: **audit
training data against the actual problem, not just internal metrics.**
Our own robustness grid couldn't have caught this gap, because it only
measures held-out images from the same sources used for training.

## Dataset compliance: final composition and why

After the generator-diversity fix above, we audited the training data
against the organizer's three listed datasets and their explicit
"do not train on this" instruction, and made two further corrections:

| organizer dataset | status |
|---|---|
| **SID_Set** (recommended) | ✅ Used - train split, labels 0/1 only |
| **CIFAKE** (Kaggle, recommended) | ⚠️ **Tried, then removed** - see below |
| **WildFake** (modelscope demo benchmark) | ✅ Never fetched (access friction + explicitly reserved for demonstration, not training) |

**CIFAKE was tried and reverted.** It had been used only in the
abandoned Part-2 forensic-signal research, never in this CLIP-based
detector - an oversight. Adding it (500 real + 500 AI, via the same HF
mirror used earlier) produced a clear, **dual-confirmed regression**:
mean robustness-grid AUROC fell 0.976→0.956, and the generator
diagnostic fell 0.960→0.904 (real-image accuracy specifically dropping
93%→82%). The likely cause: CIFAKE's native 32×32 resolution (upsampled
16x to 512px) is fundamentally incompatible with a task that also
stress-tests blur/resize/noise robustness - an already-blurry base
image confuses those specialists. Given both independent benchmarks
agreed, we excluded CIFAKE from the final model rather than keep a
measurably worse detector for checklist inclusion; the brief frames its
datasets as available resources, not mandatory ingredients, and the
comparison is fully preserved (`model/*_v4_with_cifake_WORSE.*`) for
transparency.

**Defactify's real (MS COCO) images were also removed as a precaution.**
WildFake's non-AIGC side is specifically COCO val2017 (~5k images);
Defactify's ~16k real images almost certainly come from the much larger
train2017 split, making systematic overlap unlikely - but since
real-photo diversity was already well covered by `fullres`/`sid_set`,
there was no reason to carry even a small, unconfirmed risk of training
on images the organizer reserved for demonstration. Only Defactify's 5
generator classes (no COCO/WildFake connection) remain in training.
This change was essentially free: the robustness-grid mean actually
*improved slightly* (0.976→0.981) without it, though the generator
diagnostic's real-image accuracy dropped (93%→82%) for an identifiable
reason - the prior version had training images from the exact same
COCO distribution as this diagnostic's real-image side, an advantage
specific to that one benchmark rather than a sign of better real-photo
detection generally (our own robustness grid's real-photo accuracy,
using fullres/SID_Set-sourced photos, is unaffected or better).

**Final training composition**: `fullres` (999) + `sid_set` (1,200) +
`defactify` AI-only, 5 generators (750) = 2,949 images, 2,506 train /
443 held-out test.

## Reactivity-delta feature (final architecture extension)

A user question - "instead of learning what real/AI images look like,
can the model learn how different degradations *change* a real image
vs. an AI image?" - led to a validated addition on top of the
architecture above. This is conceptually related to the classical
"noise-residual reactivity" signals from Part 2 below (measuring a
signal's *response* to a probe, not its absolute value), which were
rigorously falsified on hand-crafted pixel statistics (best ~0.62
AUROC). The new question: does the same idea work in **CLIP embedding
space** instead?

**Feature**: for every image, compute the CLIP pre-projection embedding
twice - once as-is, once after re-JPEG-compressing it at quality 50 -
and use the *difference* between the two (768-dim) concatenated with
the original embedding (768-dim) as a 1536-dim feature, instead of the
embedding alone.

**Validation sequence** (each step run before committing more time to
the next, same discipline as every other change in this project):

| check | absolute embedding alone | + reactivity delta |
|---|---:|---:|
| Cross-source generalization (leave-one-source-out CV) | 0.72 | **0.92** |
| Cross-generator generalization (leave-one-generator-out, Defactify TEST split) | 0.91 | **0.97** |
| Survives an already-degraded (blurred) input, not just clean | 0.82 | **0.91** |
| Full production integration, real training/eval pipeline, all 5 domains | see below | see below |

A domain-matched probe (using `blur_s1.0` instead of `jpeg_q50`
specifically for the "jpeg" domain, since re-JPEG-probing an
already-JPEG-degraded image is redundant by construction) measured
slightly better for that one domain (+0.008 vs +0.0004) - but was **not
adopted in production**: under confidence-gated soft routing, every
image needs a delta computed for every specialist it might get weighted
toward, so domain-matching would cost a 3rd CLIP pass per image (two
different probes) instead of 2, for a gain confined to the domain with
the least headroom (~0.99 baseline already). A single universal
`jpeg_q50` probe was used instead - a disclosed, time-boxed scoping
decision, not an oversight (`prelim_domain_matched_probe.py` keeps the
domain-matched comparison for the record).

**Production result**: retraining all 5 domain specialists on the
extended 1536-dim feature and re-running the full validation suite:

| | mean AUROC (16-cell grid) | mean accuracy | generator-diagnostic AUROC |
|---|---:|---:|---:|
| v5 (embedding alone) | 0.981 | 93.0% | 0.940 |
| **v6 (+ reactivity delta, final)** | **0.997** | **97.6%** | **0.977** |

Every one of the 16 robustness-grid cells now scores above 0.995 AUROC,
including the noise cells that were the one persistent weak spot in
every earlier version (0.96-0.97 -> 0.997-0.999). The gain is not
COCO-distribution-specific or a training-data artifact: it holds up on
the fully-disjoint generator-diagnostic set too, improving every single
generator class (real 81.7%->88.3%, SD2.1 86.7%->96.7%, SDXL
98.3%->100%, SD3 91.7%->98.3%, DALL-E3 95.0%->98.3%, Midjourney6
81.7%->96.7%). Cost: inference now requires 2 CLIP forward passes per
image instead of 1 (the original + the jpeg_q50-probed copy).

One honest limitation: the two hardest real-world misses found during
manual testing (a ChatGPT-generated NTU convocation photo, and a
deliberately-blurred ChatGPT dog image - see `known_failure_examples/`)
are **still misses** after this fix (0.030 and 0.008 respectively,
essentially unchanged). This feature targets generator-identity and
degradation-robustness signal; those two images represent a different
problem - a content/style distribution gap (unusual real-world prompts
vs. the simple, COCO-caption-style training examples) - which this
change was never expected to fix.

## GPU experiment: native-resolution texture crops (tried, not adopted)

After comparing this project's architecture against a teammate's
separate submission (BYTEPRINT: DINOv2 + a training-free autoencoder-
reconstruction expert), one of their design choices looked directly
applicable here: they extract several native-resolution texture-rich
crops per image rather than resizing the whole image down to fit the
backbone's input, on the reasoning that resizing is a low-pass filter
applied straight to the forensic evidence. Since CLIP's own processor
squashes any input to 224x224 internally regardless of size, the same
idea was ported over: `crops.py` selects the `top_k` 224x224 crops
richest in high-frequency detail (Laplacian-response variance) from the
*native*-resolution image, embedded via a free GPU (Google Colab, then
Kaggle after hitting Colab's usage limit) since extracting multiple
crops per image multiplies backbone compute several-fold - impractical
on this project's CPU-only sandbox.

Tested with `top_k=3` crops, on an enlarged training pool (6,006 images
across the same three sources, vs. 2,949 in production) fetched fresh
via GPU-side streaming:

| | held-out clean-test AUROC | accuracy |
|---|---|---|
| **CPU production (whole-image, 2,949 images)** | **0.9989** | **98.87%** |
| GPU native-crop (top_k=3, 6,006 images) | 0.9910 | 95.23% |

**Result: a regression, not an improvement - not adopted.** Two
confounded variables changed at once (more data AND crops), so this
doesn't cleanly isolate which one hurt, but the combined result is
clearly worse either way, which is enough to decide against promoting
it. The likely reason the insight didn't transfer: BYTEPRINT's crop
strategy was designed around **DINOv2**, a self-supervised backbone
tuned to represent local visual/texture structure. **CLIP** (this
project's backbone) is trained on image-text alignment, which pushes it
toward *holistic, whole-scene* semantic understanding - averaging three
small local patches likely throws away exactly the global context CLIP
relies on, while a resize (which DINOv2 would also suffer from) doesn't
cost CLIP as much since it wasn't leaning on high-frequency local detail
in the first place. A second plausible factor: applying a degradation
(blur/noise/jpeg) to the full native image before cropping changes its
*effective* visual severity compared to applying it then resizing to
224 (what production's whole-image pipeline actually does) - the same
nominal parameter value looks much milder at native scale than after a
resize concentrates it.

This is a disclosed negative result, not a bug: `crops.py`,
`clip_features.embed_images_preproj_crops[_with_delta]`, and
`train_reactivity_specialists_gpu.py` are kept in the repo for the
record, but `model/specialists.pkl` (the CPU-trained whole-image
version) remains the production model. A follow-up experiment
(training on the larger data pool *without* crops, to isolate whether
more data alone would have helped) was left incomplete due to time
constraints - see git history / conversation log for the in-progress
attempt.

## Results (final production pipeline, v6 with reactivity-delta)

**Held-out test set (clean images): AUROC 0.999, accuracy 98.9%.**

**Robustness table** (443 held-out test images, every cell from the
hackathon's transform grid, full data in
`results_detector/robustness_table.json`):

| transform | AUROC | accuracy | FPR | FNR |
|---|---|---|---|---|
| clean (baseline) | 0.999 | 0.989 | 0.030 | 0.000 |
| JPEG q90/70/50/30 | 0.999 / 0.995 / 0.998 / 0.996 | 0.957-0.977 | 0.036-0.073 | 0.014-0.025 |
| Gaussian blur σ 0.5/1.0/2.0 | 0.998 / 0.998 / 0.995 | 0.968-0.977 | 0.036-0.048 | 0.011-0.029 |
| Resize 0.5x/0.25x roundtrip | 0.998 / 0.997 | 0.973-0.980 | 0.018-0.042 | 0.018-0.022 |
| Gaussian noise σ 0.02/0.05/0.10 | 0.997 / 0.998 / 0.997 | 0.973-0.975 | 0.012-0.018 | 0.029-0.032 |
| Color jitter ±20% | 0.999 / 0.995 | 0.975-0.980 | 0.024-0.042 | 0.007-0.025 |
| Center crop 80% | 0.999 | 0.989 | 0.024 | 0.004 |

**Mean across all 16 cells: AUROC 0.997, accuracy 97.6%** - the best
result of every version tried in this project. Every cell now sits at
0.995-0.999 AUROC; the Gaussian-noise cells that were the one
consistent weak spot in every earlier version (0.96-0.97 AUROC) are now
solved (0.997-0.998), the direct result of adding the reactivity-delta
feature above.

**Generator diagnostic** (held-out Defactify TEST split, never
trained on): pooled AUROC 0.977 (real 88.3%, SD2.1 96.7%, SDXL 100%,
SD3 98.3%, DALL-E3 98.3%, Midjourney6 96.7%) - every class improved
over the pre-reactivity-delta version (see "Reactivity-delta feature"
above for the full before/after). Real-image accuracy is still the
softest spot, consistent with the dataset-compliance trade-off
discussed above (no COCO-distribution-matched real images in training,
by design, to avoid the organizer's reserved validation data).

## Error analysis (see `results_detector/error_analysis_summary.json`, `error_fp_*.png`/`error_fn_*.png`, and `results_detector/known_failure_examples/`)

- **False positives** (real images flagged as AI) are genuine,
  unremarkable photographs - well-composed real photography that
  shares visual statistics with the diverse set of generators now in
  training.
- **False negatives** (AI images flagged as real) cluster on genuinely
  photorealistic generations - the kind that are hard to distinguish
  from real photos on visual inspection even for a human, plus the
  documented DALL-E 3 case in `known_failure_examples/` (a complex,
  text-and-crowd event-photography composition unlike anything in the
  simple, COCO-caption-style DALL-E 3 training examples - a content/
  style gap distinct from the generator-identity gap that was fixed).
- **Trade-off discussion**: false-negative rate is generally higher
  than false-positive rate under noise degradation (FNR 0.029-0.032 vs.
  FPR 0.012-0.018), while most other cells show the opposite (FPR >
  FNR, most visibly on jpeg_q70: FPR 0.073 vs FNR 0.025). In a content-moderation deployment, false
  positives (flagging real user photos as AI) and false negatives
  (missing real fakes) have different costs depending on context - the
  fixed 0.5 threshold used throughout this project is a reasonable
  default, not a tuned choice, and a deployment-specific threshold
  (or per-domain thresholds, since the specialists already produce
  independent calibrations) would be a natural next step.

## Setup & installation

```bash
python3 -m virtualenv venv   # this session's environment used virtualenv
                              # since python3-venv wasn't installable without sudo;
                              # a plain `python3 -m venv venv` works fine if available
source venv/bin/activate
pip install -r requirements.txt
```

## Reproduce

```bash
# 1. Fetch training data (each is cached; re-runs skip already-downloaded data).
#    NOTE: fetch_data.FULLRES2_CANDIDATES (itsLeen/deepfake_vs_real_image)
#    is intentionally NOT fetched here - see "Data-quality fix" above.
python fetch_data.py                 # itsLeen/deepfake_vs_real_image_detection -> data_fullres/
python sid_set_fetch.py              # saberzl/SID_Set -> data_sid_set/
python fetch_defactify_train.py      # Defactify TRAIN split, 5 generators, AI-only -> data_defactify_train/ai/
#    (fetch_defactify_train.py deliberately skips this dataset's real/
#    MS-COCO-sourced images entirely - see "Dataset compliance" above
#    for why; only the AI-generated side is fetched.)
#
#    CIFAKE is intentionally NOT fetched for the production model - see
#    "Dataset compliance" above for the measured regression that led to
#    excluding it. fetch_data.CIFAKE_CANDIDATES + acquire_data() are
#    still there if you want to reproduce that comparison yourself.

# 2. (Optional but recommended) fetch the held-out generator diagnostic
#    set BEFORE training, so you can compare before/after like we did.
#    This pulls from Defactify's TEST split - strictly disjoint from
#    step 1's TRAIN-split fetch, so it stays a valid held-out check.
python fetch_generator_diagnostic.py  # -> data_generator_diagnostic/

# 3. Train the production pipeline
python train_classifier.py                  # -> model/classifier_head.pkl (simple baseline + fresh test_manifest.json)
python train_domain_classifier.py           # -> results_detector/experiments/domain_classifier.pkl
python train_reactivity_specialists.py      # -> results_detector/experiments/specialists_preproj_v6_reactivity.pkl
#    (extends train_domain_approaches_preproj.py's specialists with the
#    jpeg_q50 reactivity-delta feature - see "Reactivity-delta feature"
#    above. The pre-reactivity-delta specialists are still available via
#    train_domain_approaches_preproj.py if you want the v5 comparison.)
#    then promote into model/ (what predict.py and robustness_eval.py load):
cp results_detector/experiments/domain_classifier.pkl model/
cp results_detector/experiments/specialists_preproj_v6_reactivity.pkl model/specialists.pkl

# 4. Evaluate robustness across the full transform grid (production pipeline)
python robustness_eval.py            # -> results_detector/robustness_table.json + plot

# 5. Error analysis
python error_analysis.py             # -> results_detector/error_analysis_summary.json + example grids

# 6. Re-check the generator diagnostic (now that training includes Defactify)
python evaluate_generator_diagnostic.py  # compare against your step 2 baseline numbers

# 7. Run the required deliverable script on any directory of images
python predict.py --input_dir <DIR> --output out.json
```

See "Model iteration", "Dataset compliance", and "Reactivity-delta
feature" above for the earlier, simpler baselines and every
intermediate data/architecture variant - all kept in
`results_detector/experiments/` and `model/*_v2_2source.*` /
`model/*_v3_INCLUDES_COCO_REAL*` / `model/*_v4_with_cifake_WORSE.*` /
`model/*_v5_no_reactivity.*` / `model/*_CONTAMINATED.*` (never deleted;
filenames track which data-quality/diversity/compliance/architecture
stage each artifact came from).

## Limitations & what we'd improve with more time

- **The two hardest real-world misses found by manual testing remain
  unfixed.** A ChatGPT-generated NTU convocation photo and a
  deliberately-blurred ChatGPT dog image both still score confidently
  "real" (0.030, 0.008) even after the reactivity-delta fix closed the
  generator-diversity and noise-robustness gaps. These represent a
  content/style distribution gap - unusual real-world prompts and
  compositions vs. the simple, COCO-caption-style training examples -
  a different problem than what any fix so far has targeted. Closing it
  would need training data that includes more varied, "in-the-wild"
  AI-generated content (event photography, casual snapshots), not more
  generator diversity or more robustness augmentation.
- **Generator coverage is still not exhaustive.** The fixes above closed
  the gap for 5 major generators (SD2.1/SDXL/SD3/DALL-E3/Midjourney6),
  but newer or less common generators (Flux, Imagen, Firefly, and
  whatever ships next) are untested and may reveal the same kind of gap
  - this class of problem (train/test both drawn from a finite generator
  list) can't be fully solved, only continuously monitored and patched,
  which is a real operational commitment for any deployed version of
  this detector, not a one-time fix.
- **The reactivity-delta feature uses a single universal probe
  (jpeg_q50) rather than a domain-matched one**, a disclosed time-boxed
  scoping decision (see "Reactivity-delta feature" above) - a
  domain-matched probe measured better specifically for the "jpeg"
  domain (+0.008 vs +0.0004) but would cost a 3rd CLIP pass per image
  under soft routing. With more time, a cheaper way to get domain-
  matched probing without the extra pass (e.g. computing all candidate
  probe deltas once and letting a single specialist select among them)
  would be worth exploring.
- **Inference now costs 2 CLIP forward passes per image** (original +
  jpeg_q50-probed copy) instead of 1, to compute the reactivity-delta
  feature - a real latency/throughput trade-off for the accuracy gain,
  not yet benchmarked end-to-end.
- **Domain-classifier routing confidence gating uses the router's
  probabilities as-is**, not a separately calibrated confidence measure
  - a proper calibration step (e.g. Platt scaling on the router's
  outputs) might make the soft-mixture blending even more effective.
- **Backbone frozen, only the linear heads are trained.** With GPU access
  (per the plan, this session was CPU-only proof-of-concept; the "real"
  run moves to GPU), fine-tuning the last few CLIP layers, or trying
  CLIP:ViT-L/14 (still well under 2B params) instead of ViT-B/32, would
  likely improve accuracy further.
- **Decision threshold is a fixed 0.5**, not tuned for the FP/FN
  cost trade-off discussed above - a deployment-specific threshold
  (or per-domain thresholds, since the specialists already produce
  independent calibrations) would be a quick, meaningful improvement.
- **No adversarial robustness testing** - the transform grid covers
  realistic incidental degradation, not an adversary deliberately
  optimizing to evade the detector, which is a materially different
  (and harder) threat model.
- **SID_Set's "locally edited/inpainted" class (label 2) was not used**
  - it's excluded from binary training but could support a 3-way
    detector or a segmentation-style "which region is fake" extension,
    which is closer to real moderation needs than a single whole-image
    label.

## Team member contributions

_(fill in if working as a team; solo otherwise)_

---

# Part 2: exploratory forensic-signal search (prior work / appendix)

The sections below document the classical, hand-crafted signal search
that preceded the CLIP-based detector above. Kept as supporting material
for the "Innovation & Problem Insight" narrative - it's the evidence for
*why* a learned approach was chosen, not itself the submitted detector.

**Data-quality note**: this section's `fullres2` profile uses
`itsLeen/deepfake_vs_real_image`, later found (see "Data-quality fix"
above) to be a human-art-vs-AI-art dataset, not real-photo-vs-AI-image,
and excluded from the production detector's training data. It was used
here only as a *second, differently-sourced* dataset for cross-dataset
consistency checks on the classical signals (i.e. "does this signal's
effect direction hold on a dataset built differently") - a role where
the mismatch matters less, but the numbers below should still be read
with that caveat rather than as clean real-photo-vs-AI-photo results.

Tests whether real and AI-generated images react differently, in their
**noise residual** (image minus its denoised version), to a controlled
perturbation ("probe"). If the *change* in the residual under the probe
(Δ) separates real from AI images, that's a candidate signal for an
AI-image detector.

## What this experiment does

1. **Data**: downloads ~500 real + ~500 AI images from a single
   Hugging Face dataset (CIFAKE-family: CIFAR-10 reals vs.
   Stable-Diffusion-generated fakes) so both classes share identical
   origin/processing — this rules out accidentally detecting
   file-format/source differences instead of AI-vs-real.
2. **Provenance control**: every image, both classes, is re-encoded to
   JPEG quality 95 and reloaded before any measurement, so real and AI
   images share identical final compression provenance.
3. **Pipeline** (per image): resize/crop to 512x512 grayscale ->
   `R0 = I - denoise(I)` -> probe `I -> I'` (mild Gaussian blur) ->
   `R1 = I' - denoise(I')` -> `Δ_energy = mean((R1-R0)^2)` and
   `Δ_spectral = |high-freq energy(R1) - high-freq energy(R0)|`.
4. **Two tests**: Test A on the clean (q95) images, Test B on images
   additionally degraded to JPEG q50 first (simulates images that have
   been re-compressed/laundered by the internet — the realistic case).
5. **Content-leakage check**: verifies the residual looks like noise,
   not leaked scene content, by checking the correlation between
   Δ_energy and image "busyness" (variance of the Laplacian).
6. **Metrics**: AUROC (sklearn, positive class = AI) for Δ_energy and
   Δ_spectral, in both tests, plus histograms and a verdict.

## Running it

```bash
python3 -m venv venv   # or: python3 -m virtualenv venv, if venv is unavailable
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

Downloaded images are cached in `./data/real/` and `./data/ai/` — a
second run skips the download if >50 images of each class are already
present.

Outputs land in `./results/`:
- `results.json` — all AUROCs, counts, leakage correlation, data
  source, verdict.
- `hist_test{A,B}_{energy,spectral}.png` — overlaid real-vs-ai
  histograms.
- `leakage_example_{real,ai}_*.png` — side-by-side [original | R0]
  visualizations for the leakage sanity check.

## Swapping the denoiser or probe

Both are plain functions in `pipeline.py`:

```python
def denoiser_wavelet(img_float):
    return denoise_wavelet(img_float, rescale_sigma=True)

def probe_gaussian_blur(img_float, sigma=0.8):
    return gaussian_filter(img_float, sigma=sigma)

DEFAULT_DENOISER = denoiser_wavelet
DEFAULT_PROBE = probe_gaussian_blur
```

Write a new function with the same signature (`float [0,1] grayscale
array in -> float array out`) and point `DEFAULT_DENOISER` /
`DEFAULT_PROBE` at it (or pass `denoiser=` / `probe=` explicitly to
`compute_deltas(...)` in `main.py`'s `process_image`). E.g. to try a
stronger probe:

```python
DEFAULT_PROBE = lambda img: probe_gaussian_blur(img, sigma=1.5)
```

`bm3d` can be used as an alternative denoiser if installed
(`pip install bm3d`) — wrap `bm3d.bm3d(img_float, sigma_psd=...)` in a
function with the same signature.

## Swapping the data source

`fetch_data.py` tries a list of Hugging Face CIFAKE-family dataset
repos in `CIFAKE_CANDIDATES`, in order, and uses the first one whose
schema it can confidently parse (an `Image` column + a `ClassLabel`
column whose class names it can map to "real" vs. "ai"/"fake"). To
point it at a different single dataset (must contain **both** classes
with a clear label, from the same origin/processing):

1. Add/replace the repo id in `CIFAKE_CANDIDATES`.
2. If the automatic label-mapping heuristic (`find_label_mapping` in
   `fetch_data.py`) can't confidently detect real vs. fake for your
   dataset's class names, it will print the schema and skip rather
   than guess — extend the `real_tokens` / `fake_tokens` sets if your
   dataset uses different class-name vocabulary.

If no Hugging Face source is reachable, `fetch_data.py` prints
copy-pasteable manual instructions (including a Kaggle CLI command for
CIFAKE) and exits without fabricating data — it will **not** assemble
the two classes from two different sources.

## Result: CIFAKE (32x32-native) vs. full-resolution — the effect does NOT generalize

Two profiles were run (`python main.py --profile cifake` and
`python main.py --profile fullres`), same pipeline code, two different
single-source datasets:

| profile | dataset | native res | Test A Δ_energy AUROC | Test B Δ_energy AUROC | leakage corr (A / B) | verdict |
|---|---|---|---|---|---|---|
| `cifake` | dragonintelligence/CIFAKE-image-dataset | 32x32 (16x upsampled) | 0.661 | 0.698 | -0.43 / -0.41 (ok) | **UPSIDE** |
| `fullres` | itsLeen/deepfake_vs_real_image_detection | native (varies, no upsampling) | 0.598 | 0.650 | 0.62 / 0.58 (**FLAGGED**) | **DEAD** |

**The CIFAKE "UPSIDE" result does not replicate at full resolution, and
the full-resolution run's own leakage check flags why.** CIFAKE's images
are CIFAR-10-derived, natively 32x32 pixels; this pipeline upscales them
16x to 512x512 (identically for both classes) before measuring
residuals. Visual inspection of the CIFAKE leakage-example PNGs showed
the residual dominated by a grid-like pattern consistent with JPEG block
boundaries interacting with the upsampling. On full-resolution images
(no upsampling), the residual visibly traces scene edges (e.g. an
aircraft's silhouette, a mountain ridge line) rather than looking like
flat noise, and the quantitative check confirms it: correlation between
Δ_energy and scene busyness is 0.62 (Test A) / 0.58 (Test B), both over
the 0.5 flag threshold. So on real, full-resolution images, Δ_energy is
substantially explained by *how busy the scene is*, not by real-vs-AI
"reactivity" — and once that's accounted for, Test A's effective AUROC
(0.598) doesn't even clear the DEAD/PREDICTED threshold (0.65).

**Conclusion: the CIFAKE result was very likely a resolution/upsampling
artifact, not a genuine generative-model fingerprint.** This is exactly
why the leakage check and the two-dataset comparison matter — a single
promising AUROC number on one (low-resolution) dataset would have been
misleading on its own.

One data-quality note from the full-resolution run: one downloaded AI
image (`ai_0251.png`) turned out to be a fully-black/degenerate 512x512
frame (either a corrupted source record or a truncated download), which
produced NaN residual statistics and crashed the first attempt. Both
`fetch_data.py` (skip near-constant images during acquisition) and
`main.py` (skip any record with a non-finite result, with a warning)
were hardened against this; the final full-resolution run used n_real=500,
n_ai=499.

## Follow-up: making the signal content-invariant BY CONSTRUCTION

Since the leakage check showed Δ_energy was substantially explained by
scene busyness, `pipeline.py` adds a second statistic,
**`delta_energy_norm`**, instead of just correcting Δ_energy after the
fact: it divides the pixel-wise squared residual-difference `(R1-R0)^2`
by a local scene-activity map (box-filter local variance, window=15,
`LOCAL_ACTIVITY_WINDOW`/`LOCAL_ACTIVITY_EPS` in `pipeline.py`) *before*
averaging, so busy/edge pixels can't dominate the pooled statistic
regardless of which class the image is from. This is the standard trick
from PRNU/sensor-noise forensics (dividing out local scene variance
before comparing noise statistics across images of different content).
It's visible directly in the leakage-example PNGs: the third panel,
`(R1-R0)^2 / local activity`, clearly suppresses the strong edge
reactivity along object silhouettes that dominated the raw residual.

**Result: the fix overturns both original findings.**

| profile | raw Δ_energy verdict | Δ_energy_norm verdict | what changed |
|---|---|---|---|
| `cifake` | UPSIDE (AUROC 0.661) | **DEAD** (AUROC 0.624) | Test A AUROC drops below 0.65; Test A leakage corr gets *worse* (-0.43 → -0.55, now flagged) |
| `fullres` | DEAD (AUROC 0.599) | **DEAD** (AUROC 0.581) | stays DEAD; Test A leakage corr improves (0.62 → 0.38, now ok), Test B corr worsens (0.58 → 0.66) |

Both datasets converge on **DEAD** once the statistic is made
content-invariant. The CIFAKE "UPSIDE" result specifically does not
survive this correction — its apparent signal was, like the
full-resolution case, substantially a function of scene content rather
than real-vs-AI reactivity. Note the normalization isn't a clean win
across the board (it improves the leakage correlation in three of the
four test/profile combinations but worsens it in one, Test B fullres),
which is itself informative: a single global local-variance window
isn't a complete fix for content leakage, but it's enough to show the
original positive result doesn't hold up.

**Conclusion:** across two independent single-source datasets and two
ways of measuring reactivity (raw and content-normalized), this
particular denoiser+probe+statistic combination shows no robust
real-vs-AI signal. Promising next directions, roughly in order of how
likely they are to change this conclusion: (1) a genuinely
content-invariant denoiser (e.g. BM3D) instead of post-hoc
normalization of a content-adaptive one (wavelet thresholding is itself
scene-dependent, which may be part of why normalization doesn't fully
clean it up); (2) a probe targeted at generator-specific artifacts
(repeated JPEG re-compression at multiple qualities, or a
resampling/upsampling probe) rather than generic Gaussian blur; (3) a
larger, more diverse full-resolution dataset spanning multiple
generators, since `itsLeen/deepfake_vs_real_image_detection` alone may
not represent all generation styles.

## Follow-up: degradation-awareness (predict native blur/JPEG-quality, then stratify)

`degrade_model.py` + `stratify_experiment.py` implement a different
idea: train a small regressor to predict how blurred/JPEG-compressed an
image already was ("native degradation") *before* our own
provenance-control step touches it, using classical no-reference
features (Laplacian variance, high-frequency FFT power ratio, an 8x8
blockiness score, contrast) and a RandomForest trained on synthetic
degradations with known ground truth (held-out validation: JPEG-quality
R²=0.88/MAE=4.7, blur-sigma R²=0.71/MAE=0.29 - the regressor works).
Run with:

```bash
python stratify_experiment.py --profile fullres   # or --profile cifake
```

**What it found, and why the headline number is misleading if taken at
face value:**

1. **Confound check**: native predicted JPEG quality differs
   significantly by class (real mean 96.0 vs. AI mean 94.5,
   Mann-Whitney p=2.2e-7) - real photos in this dataset arrived very
   slightly higher-quality/less-compressed than the AI images, on
   average. Native blur did not differ significantly (p=0.098).
2. **Stratifying Δ_energy by native-quality tercile**: AUROC is higher
   in the most-already-degraded tercile (0.70) than in the
   cleanest-looking tercile (0.57) - consistent with the earlier
   Test A vs. Test B finding that further JPEG compression didn't kill
   the (weak) signal, it slightly increased it.
3. **Combined classifier** (`Δ_energy + Δ_energy_norm + native_quality +
   native_blur + busyness`, 5-fold CV logistic regression) reaches
   0.723 effective AUROC, vs. 0.599 for Δ_energy alone - a seemingly
   big win.
4. **The ablation that matters**: a classifier using ONLY
   `native_quality + native_blur + busyness` - **zero residual/reactivity
   statistics at all** - already scores 0.7245, matching or exceeding
   the "combined" score. And `Δ_energy + Δ_energy_norm` combined
   (no degradation features) scores only 0.591, no better than
   Δ_energy alone.

**Conclusion: the apparent improvement is not degradation-awareness
rescuing the noise-residual signal - it's native JPEG quality and scene
busyness acting as a standalone shortcut**, unrelated to the reactivity
hypothesis this whole experiment was built to test. That shortcut is
almost certainly dataset-specific: it likely reflects this particular
AI image source's save/export pipeline producing slightly different
JPEG quality/compression conventions than this particular set of real
photos, not a property of "AI-generated-ness" in general. It's the same
category of problem as the original CIFAKE resolution artifact - a
plausible-looking number that traces back to incidental dataset
metadata rather than the thing being tested - just discovered here via
an ablation instead of a correlation-with-busyness check. **The
noise-residual reactivity signal itself remains at ~0.59 effective
AUROC even in its best combined form, still below the 0.65 "not dead"
threshold.**

Output artifacts per profile: `results{_fullres}/stratify_results.json`
(regressor validation, confound stats, stratified table, all ablation
AUROCs) and three plots -
`stratify_hist_native_quality.png` (native-quality distribution by
class), `stratify_bar_auroc_by_bin.png` (Δ_energy AUROC per tercile),
`stratify_scatter_quality_vs_delta.png` (Δ_energy vs. native quality,
colored by class).

## Multi-signal search framework (`harness.py`)

Rather than keep hand-rolling the same validation logic for every new
idea, `harness.py` factors it into one reusable entry point,
`run_signal_experiment(name, feature_fn, profile_name, needs_color=...)`,
that runs ANY candidate signal through the full gauntlet every prior
signal in this project had to survive:

- Test A (clean, JPEG q95) and Test B (pre-degraded, JPEG q50) AUROC.
- Leakage check: correlation vs. `pipeline.busyness()`.
- **Shortcut-ablation check**: does a classifier using ONLY
  `[native_quality, native_blur, busyness]` (the regressor from
  `degrade_model.py`, no real signal at all) already explain the
  feature's apparent AUROC? This is what caught the degradation-features
  false win in the section above; every new signal gets it automatically.
- Every run appends to `results/leaderboard.json` (shared across all
  profiles/signals) and saves per-image scores to a `.npz` so a later
  run on a second profile can be cross-checked for direction consistency.

A new signal is just a function `feature_fn(img_float) -> dict[str, float]`
- see `signal_spectral_slope.py`, `signal_dct_stats.py`, and
`signal_channel_corr.py` for templates (the last one needs
`needs_color=True` since it looks at cross-channel structure).

**A second full-resolution dataset was added for real cross-dataset
validation**: `fullres2` (`itsLeen/deepfake_vs_real_image`, a different
content domain - stylized/art images - from the same curator as
`fullres`'s dataset, registered in `fetch_data.FULLRES2_CANDIDATES` /
`main.PROFILES["fullres2"]`). Without a second dataset, nothing here
could distinguish "genuine effect" from "artifact of this one dataset" -
exactly the mistake CIFAKE's result made.

`leaderboard_report.py` prints the full comparison table plus, for any
signal run on 2+ profiles, whether the effect's **direction** is
consistent across datasets (a flipped sign means the "effect" is
dataset-specific noise, not real) - and a SURVIVORS section listing only
signals that clear AUROC ≥ 0.65, aren't leakage-flagged, aren't
shortcut-dominated, AND are direction-consistent across profiles.

### Results of the first breadth pass (Tier 1 signals)

Ran 3 new signal families (radial spectral-slope/peak anomaly, per-block
DCT statistics, cross-channel residual correlation) through the harness
on both `fullres` and `fullres2`, alongside the two signals from earlier
sections. **Current leaderboard: 7 signal families tested, zero
survivors.** The closest thing to a promising lead is
`channel_corr`'s cross-channel correlation features: direction-consistent
across both datasets and NOT shortcut-dominated on `fullres2`, but still
under the 0.65 bar on the weaker profile (`fullres`, ~0.55-0.57) - a
candidate worth keeping for a future combined-classifier attempt, not a
usable signal on its own. Everything else (DCT stats, spectral radial
slope) is either shortcut-dominated, leakage-flagged, or simply doesn't
separate the classes. Run `python leaderboard_report.py` for the current
full table.

Not yet attempted (see the original plan): local self-similarity/
non-local redundancy (Tier 1 #5), and the Tier 2 stretch signals
(resampling-periodicity detection, double-JPEG/first-quantization-table
detection, a pretrained-CNN-embedding ceiling baseline).
