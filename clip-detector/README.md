# Robust AI-Image Detector

Tells real photos apart from AI-generated images, and keeps working after
the image has been compressed, blurred, resized, noised, recolored, or
cropped — the things that actually happen to an image between being made
and being seen. Built for Hackathon Challenge 5, *"Robust Detection of
AI-Generated Images Under Real-World Transformations."*

## The result, up front

| | value |
|---|---:|
| Mean AUROC, official 16-cell robustness grid | **0.997** |
| Mean accuracy, same grid | 97.6% |
| AUROC on a fully unseen generator-diagnostic set | 0.977 |
| TPR @ 1% false-positive rate | 0.933 |
| Leave-one-generator-out mean AUROC | 0.968 |
| Backbone size | 151.3M params (frozen) |
| Trainable parameters | 7,685 |

That last row is not a typo. The frozen CLIP backbone does the heavy
lifting; everything this project actually *trained* is five small
logistic-regression heads, together smaller than a single attention
layer of the backbone that feeds them.

For context, here's the same table computed identically for a
teammate's separate submission to this challenge (BYTEPRINT: SigLIP2 +
a training-free reconstruction expert) — see "Comparing against
BYTEPRINT" below for the full methodology and an important caveat about
what this comparison does and doesn't prove:

| | backbone | params | AUC | TPR@1%FPR | LOGO mean |
|---|---|---:|---:|---:|---:|
| This project | CLIP ViT-B/32 | 151.3M | 0.997 | 0.933 | 0.968 |
| BYTEPRINT | SigLIP2-so400m | 430M | 0.950 | 0.585 | 0.721 |

## How it works

Six ideas stacked on top of a frozen backbone, in the order a request
actually flows through them:

**1. A frozen CLIP backbone, read at the right layer.** Every image goes
through `openai/clip-vit-base-patch32` (151M params, never fine-tuned),
but not through the layer most people would reach for. CLIP's public
embedding is the *post-projection* output — optimized to align images
with text captions. This project uses the *pre-projection* output
instead (768-dim, straight out of the vision encoder, before that
projection layer) — the projection strips out visual detail that isn't
needed for caption-matching, and it turns out some of that stripped
detail is exactly what a forensic classifier needs. This one choice was
the single biggest lever found in the entire project (see "The
insights behind the design" below).

**2. A domain classifier that guesses what kind of damage the image has taken.**
Before deciding real-or-AI, a cheap RandomForest (`domain_classifier.py`)
looks at five classical, no-reference image statistics — Laplacian
variance, high-frequency FFT ratio, JPEG blockiness, contrast, mean
saturation — and estimates which of five "domains" the image is
currently in: clean, JPEG-compressed, spatially distorted
(blur/resize/crop), noisy, or color-jittered. This works because the
competition's own robustness grid degrades every test image in exactly
one way at a time, never stacked — so "which single thing happened to
this image" is a well-posed question. 5-way accuracy: 75%.

**3. Five domain specialists instead of one generalist.** Rather than
one classifier trying to handle every kind of damage, there are five —
each a small logistic regression trained only on images degraded the
same way it specializes in. A noise specialist never has to also be
good at judging blurry images. This beats a single generalist by a
wide, consistent margin (details below), because CLIP's embedding
space itself shifts depending on what happened to the image, and one
decision boundary can't cover all of that shifted territory well.

**4. Confidence-gated blending, not a hard decision.** The router
doesn't just pick its top guess and commit — every specialist's opinion
gets weighted by how likely the router thinks that domain actually is:

```
P(AI-generated) = Σ over domains g:  P(domain = g | image) × specialist_g(image)
```

When the router is confident, this collapses to picking one specialist.
When it isn't (its hardest calls: mild JPEG vs. clean, light noise vs.
clean), the image gets a blended opinion instead of a coin-flip bet on
a single, possibly-wrong guess.

**5. Reactivity-delta: don't just look at the image, watch how it reacts.**
On top of the plain embedding, every specialist also gets a second
feature: the *shift* in CLIP's embedding when the image is
re-compressed (or, for the "jpeg" specialist specifically, re-blurred —
see below) and re-embedded. It's not "what does this image look like,"
it's "how does this image's representation move when I nudge it" — a
signal about the image's origin, not just its current appearance. This
is the single largest architectural improvement found after the initial
pipeline was working (full validation story below).

**6. A threshold calibrated for the real world, not 0.5.** A flat 0.5
cutoff ignores that a real deployment sees vastly more authentic images
than fake ones, so accuracy-at-0.5 flatters a detector in a way that
doesn't hold up in production. `calibrate_threshold.py` sets a
threshold at a 1% false-positive budget instead, using the same method
BYTEPRINT's own metrics module uses — so the two projects' numbers are
directly comparable.

## The insights behind the design

The architecture above didn't arrive fully formed — five findings, each
backed by a controlled comparison, shaped it. They're presented here as
lessons, not a lab diary, in the order they matter most for
understanding *why* the system looks the way it does.

### 1. Frozen CLIP, but the pre-projection layer, is worth more than any architecture trick

| variant | mean AUROC (16 cells) |
|---|---:|
| Post-projection CLIP, single generalist, one random augmentation | 0.9316 |
| + balanced augmentation (every image sees every domain, not one at random) | 0.9408 |
| + domain-specialist routing (still post-projection) | 0.9362 — *worse than balanced-alone* |
| **Pre-projection CLIP alone** (no balanced aug, no routing) | clean-test only: 0.9506 |
| Balanced augmentation + pre-projection, one generalist | clean-test: 0.9433 — **stacking failed** |
| Balanced augmentation + pre-projection + domain routing | **0.9477** — best architecture found |

Two things stand out. First, domain-specialist routing actively *hurt*
in the post-projection space — a misrouted image got a confidently
wrong specialist, and the domain classifier's accuracy on "clean"
images was only 34% at the time, so misrouting was common. Second,
just stacking two good ideas (balanced augmentation + richer features)
onto one generalist head made things *worse*, not better — a classic
overfitting signature (training AUROC hit 0.999). Routing only started
paying off once it had the richer pre-projection space to route
*within* — specialists there beat the generalist in 15 of 16 cells,
including cells where the router itself was still often wrong, because
a wrong routing decision costs much less when every specialist is
individually more capable.

### 2. A data-quality audit beat every architecture change combined

One of the three original training sources (`itsLeen/deepfake_vs_real_image`)
turned out not to be a real-vs-AI dataset at all — its actual classes
were `Real_Art` (paintings and illustrations, not photographs) and
`AI_Art` (which included real photographs of magazine coverage about AI
art, mislabeled as AI-generated). Removing it entirely:

| | mean AUROC (16 cells) | mean accuracy |
|---|---:|---:|
| Best architecture, contaminated 3-source data | 0.9477 | 87.7% |
| **Same architecture, clean 2-source data** | **0.9821** | **93.7%** |

+0.034 AUROC from deleting one bad data source — bigger than the gain
from any single architecture change tried. The lesson generalized: a
data audit is often higher-leverage than another round of model
tuning, and it's exactly what caught the next, bigger problem.

### 3. Held-out accuracy from the same sources isn't the same as generalization

After the fixes above, a real-world test — a ChatGPT/DALL-E 3 image the
user actually tried — came back confidently *wrong* (predicted 0.008,
i.e. "definitely real," on an actually-AI image), despite the model
scoring 0.984 mean AUROC on its own held-out test set. Rather than
shrug it off, a proper diagnostic was built: 60 held-out images each
from real photos, SD2.1, SDXL, SD3, DALL-E 3, and Midjourney 6, pulled
from a source never touched during training. Score by generator:

| generator | accuracy before |
|---|---:|
| real (control) | 98.3% |
| Stable Diffusion 2.1 | 45.0% — worse than chance |
| Stable Diffusion XL | 75.0% |
| Stable Diffusion 3 | 53.3% — near chance |
| DALL-E 3 | 68.3% |
| Midjourney 6 | 48.3% — worse than chance |

The DALL-E 3 miss wasn't a fluke — it was a symptom of the model having
only ever learned two generators' fingerprints. **0.984 AUROC on our
own test set was real, but it was measuring in-distribution
performance, not the generator landscape that actually matters.** The
fix — adding 750 images spanning all five generators from a strictly
train-side split (never touching the diagnostic split above) — closed
almost every gap:

| generator | before | after | change |
|---|---:|---:|---:|
| real (control) | 98.3% | 93.3% | −5.0pp |
| Stable Diffusion 2.1 | 45.0% | 73.3% | **+28.3pp** |
| Stable Diffusion XL | 75.0% | 91.7% | +16.7pp |
| Stable Diffusion 3 | 53.3% | 90.0% | **+36.7pp** |
| **DALL-E 3** | 68.3% | **95.0%** | **+26.7pp** |
| Midjourney 6 | 48.3% | 86.7% | **+38.4pp** |
| pooled AUROC | 0.919 | **0.960** | +0.041 |

The largest generalization gain in the whole project, and it's a
finding no amount of tuning against the existing robustness grid could
have surfaced — that grid only ever tested held-out images from the
*same* training sources. (The original real-world image that started
this thread moved from 0.008 to 0.092 — a real ~11x shift, though this
specific hard example still doesn't cross 0.5. It's kept, unmodified,
in `results_detector/known_failure_examples/` rather than quietly
dropped.)

### 4. A probe reveals more than a look — and it's a rediscovery of a classic idea

A user question mid-project — "can the model learn how a degradation
*changes* an image, not just what the image looks like?" — led to the
reactivity-delta feature described above. It's conceptually related to
classic **Error Level Analysis** (re-compress an image, compare it to
the original) — a well-known forensic technique, but normally used to
find *which region* of a real photo was locally edited, working
directly on pixels. This project's version works at the *whole-image*
level, in *CLIP's embedding space*, for classifying origin rather than
localizing edits. That relocation is what made it work: **this project
already tried the literal pixel-level version of this idea (Part 2,
below) and it failed** (~0.62 AUROC, best case). The embedding-space
version is a different implementation of the same underlying question,
and it validated through four independent, increasingly hostile checks
before being adopted:

| check | absolute embedding alone | + reactivity delta |
|---|---:|---:|
| Cross-source generalization (leave-one-source-out) | 0.72 | 0.92 |
| Cross-generator generalization (leave-one-generator-out) | 0.91 | 0.97 |
| Survives being applied to an already-degraded input | 0.82 | 0.91 |
| Full production integration, real pipeline | 0.981 | **0.997** |

Every one of the 16 robustness-grid cells moved above 0.995 AUROC,
including the Gaussian-noise cells that had been the one persistent
weak spot through every earlier version. The gain held up on the fully
disjoint generator-diagnostic set too — every generator class improved,
not just the ones already strong.

### 5. Cross-family probes beat same-family probes, everywhere it was tested

The reactivity-delta feature above uses one probe (re-compress at JPEG
quality 50) for every domain — except one. Applying a JPEG probe to an
image whose *own* damage is already JPEG compression is redundant by
construction; there's little new information in compressing an
already-compressed image again. Testing this directly, across every
non-clean domain, by swapping in a probe from the *same* family as each
domain's own degradation and comparing it against the universal
cross-family probe:

| domain | same-family probe | same-family gain | cross-family (jpeg_q50) gain | gap |
|---|---|---:|---:|---:|
| jpeg | jpeg_q50 | +0.0004 | +0.0078 (blur probe) | 19x |
| spatial | blur | +0.0285 | +0.0492 | 1.7x |
| noise | noise | +0.0235 | +0.0704 | 3.0x |
| colorjitter | colorjitter | **+0.0024** | +0.0391 | **16x** |

Cross-family wins in every domain tested, no exceptions. Two of the
four (jpeg and colorjitter) show *near-total* redundancy for a
same-family probe; the other two show it's still real, just muted. The
one domain where this mattered enough to change ("jpeg," whose
universal probe used to *be* same-family) got fixed: its specialist now
uses a blur probe instead, validated at full scale (2,506 training
images, 443 held-out test): **0.9953 AUROC, up from 0.9938** on the
identical test images. Cost: a third CLIP pass per image, since
confidence-gated blending means every specialist's own probe delta
needs computing for every image, regardless of which domain it actually
routes to.

*Honest gap*: the full 16-cell grid hasn't been re-run end-to-end on
this specific change — an attempt hit severe, unexplained slowdowns on
this project's compute environment and was killed rather than risk the
submission deadline. The improvement is validated in isolation (full
scale, above) and on a 3-domain subset (below), not yet reconfirmed
holistically across all 16 cells.

### 6. Domain routing and reactivity-delta are complementary, not redundant

A fair question once reactivity-delta exists: does it make the domain
routing pointless? Tested directly — a single pooled classifier (no
routing, universal probe only) against the live routed pipeline, same
held-out images:

| domain | single generalist | routed pipeline | routing's advantage |
|---|---:|---:|---:|
| clean | 0.9837 | 0.9982 | +0.0146 |
| noise | 0.9769 | 0.9942 | +0.0173 |
| jpeg | 0.9588 | 0.9949 | +0.0362 |
| **mean** | 0.9731 | 0.9958 | **+0.0227** |

Routing still adds a real, consistent improvement in every domain
tested — the two mechanisms answer different questions. Routing
calibrates *where* the decision boundary sits, since CLIP's raw
embedding shifts depending on what happened to the image (an
appearance-space problem). Reactivity-delta adds a signal about the
image's *origin* that generalizes across that shift (a different axis
of information entirely). The biggest routing advantage lands on
"jpeg" — a pooled generalist has no domain to match a probe to, so it's
stuck with exactly the redundant-probe problem the fix above solved.

## What we tried and didn't keep

Not every idea worked, and the negative results are kept rather than
quietly dropped — they're evidence the final architecture was actually
tested against alternatives, not just the first thing that worked.

- **Native-resolution texture crops** (borrowed from BYTEPRINT's own
  design: sample several native-resolution patches instead of resizing
  the whole image, since resizing is a low-pass filter on forensic
  evidence). Tested at three different scales on this project's CLIP
  backbone — it **regressed performance every time** (clean domain
  −0.008, spatial −0.025, jpeg −0.039 AUROC). The likely reason: CLIP's
  strength is holistic, whole-scene semantic understanding, not local
  texture (that's DINOv2's strength, which is why it works for
  BYTEPRINT's design and not this one) — averaging a few small patches
  throws away exactly the context CLIP relies on.
- **More training data alone**, isolated from the crop change: doubling
  the dataset (2,949 → 6,050 images, same architecture) moved AUROC by
  +0.0003 — statistically noise. The model was already close to its
  ceiling on the original data; volume alone had nowhere useful to go.
- **AEROBLADE-style reconstruction error** (BYTEPRINT's training-free
  second signal: measure how well a latent-diffusion VAE decoder
  reconstructs an image). A genuinely promising idea to try — unlike
  crops, it's an orthogonal signal source, not a different way of
  extracting the same one — but the attempt to validate it hit a
  stalled multi-hundred-megabyte weight download and was abandoned
  given the deadline. Flagged as real, untested future work, not a
  dead end.

## Dataset & organizer compliance

| organizer dataset | status |
|---|---|
| **SID_Set** (recommended) | ✅ Used — train split, labels 0/1 only |
| **CIFAKE** (Kaggle, recommended) | ⚠️ Tried, then reverted |
| **WildFake** (modelscope demo benchmark) | ✅ Never touched — explicitly reserved for demonstration, not training |

CIFAKE was added properly (500 real + 500 AI) to make sure an
organizer-recommended resource was actually used, not just referenced
in an earlier abandoned experiment. It caused a clear, dual-confirmed
regression: robustness-grid AUROC fell 0.976→0.956, generator
diagnostic fell 0.960→0.904. The likely cause: CIFAKE's native 32×32
resolution, upsampled 16x, is fundamentally incompatible with a task
that also stress-tests blur/resize/noise robustness. Both benchmarks
agreed, so it was excluded rather than kept for checklist credit — the
brief frames its datasets as available resources, not mandatory
ingredients, and the comparison is preserved for transparency
(`model/*_v4_with_cifake_WORSE.*`).

Defactify's real (MS-COCO-sourced) images were also excluded as a
precaution, even though the overlap risk with the organizer's reserved
WildFake/COCO-val2017 validation set was assessed as unlikely
(Defactify draws from the much larger train2017 pool). Real-photo
diversity was already covered by the other two sources, so there was
no reason to carry even a small, unconfirmed risk — and the change was
essentially free (robustness grid improved slightly without those
images).

**Final training composition**: `fullres` (999) + `sid_set` (1,200) +
`defactify` AI-only across 5 generators (750) = 2,949 images, 2,506
train / 443 held-out test.

## Full results

**Held-out clean test**: AUROC 0.999, accuracy 98.9%.

**Robustness grid** (443 held-out images, every official transform,
full data in `results_detector/robustness_table.json`):

| transform | AUROC | accuracy | FPR | FNR |
|---|---|---|---|---|
| clean (baseline) | 0.999 | 0.989 | 0.030 | 0.000 |
| JPEG q90/70/50/30 | 0.999 / 0.995 / 0.998 / 0.996 | 0.957–0.977 | 0.036–0.073 | 0.014–0.025 |
| Gaussian blur σ 0.5/1.0/2.0 | 0.998 / 0.998 / 0.995 | 0.968–0.977 | 0.036–0.048 | 0.011–0.029 |
| Resize 0.5x/0.25x roundtrip | 0.998 / 0.997 | 0.973–0.980 | 0.018–0.042 | 0.018–0.022 |
| Gaussian noise σ 0.02/0.05/0.10 | 0.997 / 0.998 / 0.997 | 0.973–0.975 | 0.012–0.018 | 0.029–0.032 |
| Color jitter ±20% | 0.999 / 0.995 | 0.975–0.980 | 0.024–0.042 | 0.007–0.025 |
| Center crop 80% | 0.999 | 0.989 | 0.024 | 0.004 |

**Mean across all 16 cells: AUROC 0.997, accuracy 97.6%.** Every cell
sits at 0.995–0.999 — the Gaussian-noise cells that were the one
consistent weak spot in every earlier version are now solved. (This
table is the last version with a full end-to-end re-validation — see
the honest gap noted in "insight 5" above for what's changed since.)

**Generator diagnostic** (held-out, never trained on): pooled AUROC
0.977 — real 88.3%, SD2.1 96.7%, SDXL 100%, SD3 98.3%, DALL-E3 98.3%,
Midjourney6 96.7%. Real-image accuracy is the softest spot, a direct
consequence of the compliance decision above (no COCO-distribution
real images in training, by design).

**Threshold calibration**: a flat 0.5 cutoff ignores that score
distributions shift between domains and generators, and that a real
platform sees far more authentic images than fake ones. Calibrated to a
1% false-positive budget instead (same method as BYTEPRINT's own
`byteprint/metrics.py`):

| | fixed 0.5 | calibrated to 1% FPR |
|---|---:|---:|
| mean accuracy | 97.6% | 95.4% |
| mean FPR | 1.2%–7.3% (by cell) | 0.98% (on target) |
| mean FNR | 0%–3.2% (by cell) | 6.7% |

The expected trade-off of enforcing a strict false-positive budget —
some recall is sacrificed to keep false positives where a real
deployment needs them. `production_pipeline.calibrated_predict()`
exposes this for callers that want a binary decision; the required
`predict.py` deliverable still reports raw probabilities per the
brief's exact spec.

### Comparing against BYTEPRINT

Computed with BYTEPRINT's own metric definitions (`byteprint/metrics.py`),
for a direct comparison against a teammate's separate submission to
this same challenge:

| backbone | params | AUC | TPR@1%FPR | LOGO mean |
|---|---|---:|---:|---:|
| CLIP ViT-B/32 (this project) | 151.3M | 0.997 | 0.933 | 0.968 |
| SigLIP2-so400m (BYTEPRINT) | 430M | 0.950 | 0.585 | 0.721 |

LOGO (leave-one-generator-out) breakdown for this project:

| held-out generator | AUROC |
|---|---:|
| SD2.1 | 0.994 |
| SDXL | 0.994 |
| SD3 | 0.961 |
| DALL-E3 | 0.969 |
| Midjourney6 | 0.918 |

**Caveat worth stating plainly**: these are each team's own
self-reported numbers on each team's own evaluation setup, not a
controlled head-to-head on identical held-out data — different training
data and different test images could contribute to part of the gap.
Read it as "each team's best result under the same metric definitions,"
not "identical benchmark, decisive win." Full methodology notes in
`results_detector/summary_table.md`.

## Error analysis & known limitations

**What the errors look like:**
- **False positives** (real flagged as AI) are genuine, unremarkable
  photographs that happen to share visual statistics with the diverse
  generator set now in training.
- **False negatives** (AI flagged as real) cluster on the most
  photorealistic generations — hard to tell apart even by eye — plus
  the documented DALL-E 3 case below.
- Under noise degradation specifically, false negatives outnumber false
  positives (FNR 2.9–3.2% vs. FPR 1.2–1.8%); most other cells show the
  reverse. Different deployment contexts weigh these differently, which
  is exactly why the calibrated threshold above exists.

**Open limitations, honestly stated:**

- **Two real-world misses remain unfixed.** A ChatGPT-generated NTU
  convocation photo and a deliberately-blurred ChatGPT dog image both
  still score confidently "real" (0.030, 0.008), even after every fix
  above. This is a content/style distribution gap — unusual real-world
  prompts and compositions vs. the simple, COCO-caption-style training
  examples — not something generator diversity or robustness
  augmentation was ever going to fix. Closing it needs training data
  that looks like how people actually use these tools, not more
  generators or more transforms.
- **Generator coverage isn't exhaustive.** Five major generators are
  covered; newer or less common ones (Flux, Imagen, Firefly, whatever
  ships next) are untested. This class of problem can only be
  continuously monitored and patched, never fully solved.
- **Inference costs 3 CLIP forward passes per image** now (original +
  two probed copies), not yet benchmarked for latency end-to-end.
- **The domain router's confidence isn't separately calibrated** — a
  Platt-scaling step on its raw probabilities might make the soft
  blending even more effective.
- **The backbone is frozen** — with GPU access, fine-tuning the last
  few CLIP layers, or trying ViT-L/14 (still well under the 2B-param
  budget), would likely help further.
- **No adversarial robustness testing** — the transform grid covers
  realistic incidental degradation, not an adversary deliberately
  trying to evade the detector, a materially harder threat model.
- **SID_Set's tampered/locally-edited class was never used** — excluded
  from binary training, but could support a 3-way or segmentation-style
  detector, arguably closer to real moderation needs.

## Setup & installation

```bash
python3 -m virtualenv venv   # or plain `python3 -m venv venv` if available
source venv/bin/activate
pip install -r requirements.txt
```

## Reproduce

```bash
# 1. Fetch training data (cached; re-runs skip already-downloaded data).
#    itsLeen/deepfake_vs_real_image is intentionally NOT fetched here -
#    see "Dataset & organizer compliance" above.
python fetch_data.py                 # itsLeen/deepfake_vs_real_image_detection -> data_fullres/
python sid_set_fetch.py              # saberzl/SID_Set -> data_sid_set/
python fetch_defactify_train.py      # Defactify TRAIN split, 5 generators, AI-only -> data_defactify_train/ai/
#    (this deliberately skips Defactify's real/MS-COCO-sourced images
#    entirely - only the AI-generated side is fetched.)
#    CIFAKE is intentionally NOT fetched for the production model - see
#    "Dataset & organizer compliance" for the measured regression.
#    fetch_data.CIFAKE_CANDIDATES + acquire_data() are still there if
#    you want to reproduce that comparison yourself.

# 2. (Optional but recommended) fetch the held-out generator diagnostic
#    set BEFORE training, so you can compare before/after like we did.
#    Strictly disjoint from step 1's TRAIN-split fetch.
python fetch_generator_diagnostic.py  # -> data_generator_diagnostic/

# 3. Train the production pipeline
python train_classifier.py                  # -> model/classifier_head.pkl (simple baseline + fresh test_manifest.json)
python train_domain_classifier.py           # -> results_detector/experiments/domain_classifier.pkl
python train_reactivity_specialists.py      # -> results_detector/experiments/specialists_preproj_v6_reactivity.pkl
cp results_detector/experiments/domain_classifier.pkl model/
cp results_detector/experiments/specialists_preproj_v6_reactivity.pkl model/specialists.pkl

# 4. Evaluate robustness across the full transform grid
python robustness_eval.py            # -> results_detector/robustness_table.json + plot

# 5. Error analysis
python error_analysis.py             # -> results_detector/error_analysis_summary.json + example grids

# 6. Re-check the generator diagnostic
python evaluate_generator_diagnostic.py

# 7. Domain-matched jpeg specialist (v7) - retrain just this one specialist
python retrain_jpeg_specialist_domain_matched.py  # -> model/experiments_jpeg_specialist_domain_matched.pkl
#    (swap the "jpeg" entry into model/specialists.pkl - check the
#    script's printed comparison AUROC before promoting)

# 8. Calibrate a deployment threshold at a target false-positive rate
python calibrate_threshold.py        # -> model/calibration.json

# 9. Run the required deliverable script on any directory of images
python predict.py --input_dir <DIR> --output out.json
```

Every intermediate data/architecture variant is kept, never deleted —
`results_detector/experiments/` and `model/*_v2_2source.*` /
`*_v3_INCLUDES_COCO_REAL*` / `*_v4_with_cifake_WORSE.*` /
`*_v5_no_reactivity.*` / `*_v6_universal_probe.*` / `*_CONTAMINATED.*`
track which data-quality/compliance/architecture stage each artifact
came from.

## Team member contributions

_(fill in if working as a team; solo otherwise)_

---

# Part 2: exploratory forensic-signal search (prior work / appendix)

The sections below document the classical, hand-crafted signal search
that preceded the CLIP-based detector above. Kept as supporting material
for the "Innovation & Problem Insight" narrative - it's the evidence for
*why* a learned approach was chosen, not itself the submitted detector.

**Data-quality note**: this section's `fullres2` profile uses
`itsLeen/deepfake_vs_real_image`, later found (see "Dataset &
organizer compliance" above) to be a human-art-vs-AI-art dataset, not
real-photo-vs-AI-image, and excluded from the production detector's
training data. It was used here only as a *second, differently-sourced*
dataset for cross-dataset consistency checks on the classical signals
(i.e. "does this signal's effect direction hold on a dataset built
differently") - a role where the mismatch matters less, but the numbers
below should still be read with that caveat rather than as clean
real-photo-vs-AI-photo results.

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
