# BYTEPRINT

**Robust detection of AI-generated images under real-world transformations.**

Every generator leaves a byteprint. A camera sensor imprints grain, optics and
a compression history; a latent-diffusion decoder imprints its own manifold
instead. BYTEPRINT looks for both traces at once — and, because an image is
compressed, cropped and reposted before anyone sees it, it is graded on what
survives the wash rather than on clean accuracy.

By team ByteSized. The models are byte-sized too: the default backbone is 0.43B
parameters, well inside the competition's <2B budget, and the smallest
registered backbone runs on a CPU with no staged weights at all.

Two independent detectors behind one interface, fused at the score level:

1. **Frozen backbone + linear probe** — SigLIP2-so400m features,
   native-resolution texture crops, a calibrated logistic-regression head.
   **This is the detector that works.**
2. **AEROBLADE** — training-free reconstruction error through a bank of latent
   diffusion autoencoders.

The argument for two experts was that they fail on different inputs. That has
now been measured, and on SID_Set it does not hold: **the reconstruction expert
scores AUC 0.5822 alone, and fusing it moves the pooled AUC by +0.0001** — it is
chance-level (0.4975) on tampered images, because a local edit in a real
photograph is not what a whole-image reconstruction detector was built to catch.
It stays in the repo as a measured negative result rather than a load-bearing
component; the mechanism, and the three ways this test was hostile to the
method, are in
[`docs/results-recon-fusion.md`](docs/results-recon-fusion.md).

So read BYTEPRINT as a strong single-expert detector with a second expert that
has been honestly evaluated and reported. Around both sits an evaluation harness
built on the numbers that actually predict deployment behaviour.

## Competition context

Built for the challenge in **[`docs/competition-brief.md`](docs/competition-brief.md)**
(§5, "Robust Detection of AI-Generated Images Under Real-World Transformations").
The requirements that bind the design:

| Requirement | Consequence here |
|---|---|
| **Binary** image-level AIGC vs authentic | One confidence score per image — not 3-class, not localization, whatever SID_Set additionally supports |
| **Models <2B parameters** | Rules out the NTIRE-2026-winning DINOv3-7B recipe. Budget: DINOv2-giant ≈1.1B, SigLIP2-so400m ≈400M, EVA02-L ≈300M |
| **Robustness is the graded axis** | The laundering ladder is the headline result, not clean AUC — which is exactly what `eval --by-spec` was built to report |
| **Fixed transform list** (§5.2) | JPEG 90/70/50/30 · blur σ 0.5/1.0/2.0 · resize 0.5×/0.25× · noise σ 0.02/0.05/0.10 · color jitter ±20% · center crop 80% — transcribed as `OFFICIAL_LADDER` and pinned by a test |
| **Demo validation set is off-limits for training** | COCO val2017 (4,998) + DALL·E Advanced (8,843), a WildFake subset |
| **Required output interface** | A script: image directory → JSON of `{image_path, pred}` — `byteprint score`, also as `scripts/score_directory.py` |

Judged on technical execution (35%), innovation & insight (20%), impact (20%),
feasibility (15%), presentation (10%). Feasibility explicitly rewards
*proportionate* resource usage, so the compute budget is stated rather than
maximised.

### Known gaps against the brief

One decision is still open:

- **How much tampered images cost.** SID_Set's class 2 (a real photograph with
  an AI-edited region) counts as AIGC here, and rather than assert that, the
  materialiser files those images under their own generator directory so the
  choice is measurable. It is now measured: they score AUC 0.8513 against 0.9537
  for fully synthetic images, and a probe trained on one type transfers to the
  other at only 0.6457. Both numbers argued for a crop strategy with some notion
  of *where* to look. Two were built — `anomaly` and `ela` — and **the answer is
  no**: on SID_Set they move tampered AUC from 0.8513 to 0.8317 and 0.8318, the
  wrong way, from two independent cues that agree to four decimals. Crop
  placement is not what limits tampered detection
  ([`docs/results-crop-localisation.md`](docs/results-crop-localisation.md)).

  That pointed at mean-pooling over crops as the real constraint — a localised
  edit averaged against authentic content before the head ever sees it — so
  pooling was moved out of the cache and max, top-k and mean-score reductions
  were tested. **That answer is also no.** At matched crop count every
  score-space arm loses to plain mean pooling on both backbones, and they
  measurably damage the low-FPR operating point, because a max selects each
  image's *noisiest* crop
  ([`docs/results-crop-pooling.md`](docs/results-crop-pooling.md)).

  Two rounds of "look at the right part of the image, or combine the parts more
  cleverly" have now come back negative. What has actually moved tampered
  detection twice is seeing **more** of the image — 2 → 8 crops takes tampered
  AUC 0.9176 → 0.9494 — which is a coverage story, not a localisation one, and
  the opposite of the intuition both rounds were built on. The gap itself is
  still open: tampered images remain the harder class.

Closed since the first draft:

- The laundering ladder now matches §5.2 exactly — colour jitter and centre crop
  are implemented, noise σ is normalised to the brief's 0–1 convention (values
  above 1.0 are rejected rather than reinterpreted), and the rung list is pinned
  by a test so it cannot drift.
- The directory → JSON entry point exists: `byteprint score`, below.

## Expert 1 — frozen backbone + linear probe

The backbone is never fine-tuned. That is deliberate: a general-purpose
pretrained representation transfers across unseen generators far better than a
network trained end-to-end on one of them, which reliably descends onto
generator-specific shortcuts. It also means embeddings are a pure function of
(image, laundering spec, extraction config), so they are computed **once** and
cached — after which retraining the probe, recalibrating a threshold, or running
leave-one-generator-out costs seconds.

```
                 ┌─ backbone (frozen) ───► probe cache ─► probe ─┐
images ─► crops ─┤                                               ├─► fused score
                 └─ VAE bank + LPIPS ────► recon cache ─► -min ──┘
```

Both experts implement the same `embed(crops) -> (n_crops, dim)` interface, so
extraction, caching, resume and the laundering ladder apply to each unchanged.

## Expert 2 — AEROBLADE reconstruction error

A latent diffusion model's images sit close to the output manifold of its VAE
decoder, so that decoder round-trips them with much less perceptual damage than
it does a real photograph. Reconstruct, measure LPIPS distance, and a *small*
distance is itself evidence of synthesis — no training data, no training run.

Robustness comes from reconstructing through several autoencoders and taking the
minimum: an image only has to sit near one generator's manifold. That minimum is
applied at scoring time, so which autoencoders count can be ablated without
recomputing anything.

Verified on genuine SD-1.5 decoder output (real photos passed through
encode/decode), 20 vs 20 at 112px:

| | AUC | TPR@1%FPR |
|---|---|---|
| AEROBLADE, min over 3 VAEs | **0.9650** | **0.9500** |

with the per-autoencoder means showing the mechanism directly — the decoder that
*made* the image separates, the other two barely move:

| autoencoder | real | SD-1.5 output |
|---|---|---|
| `sd15` | 0.1287 | **0.0869** |
| `vae-mse` | 0.1190 | 0.1119 |
| `sdxl` | 0.1212 | 0.1179 |

Ricker et al., CVPR 2024 ([arXiv:2401.17879](https://arxiv.org/abs/2401.17879)).

**At scale on SID_Set, this expert does not carry weight.** The 20 vs 20 table
above is a mechanism check on genuine SD-1.5 decoder output, not a result. Run
over the full corpus it scores **AUC 0.5822**, and the multi-decoder bank turns
out to be redundant here — min, mean and max aggregation all give 0.582, so it
cost 3× the compute for no coverage. The distances are healthy and point the
predicted direction; the effect is simply small.
See [`docs/results-recon-fusion.md`](docs/results-recon-fusion.md).

> **Cost.** ~1.2GB of VAE weights, and on CPU roughly **2.9 s/crop** at 112px or
> **13.7 s/crop** at 224px with three autoencoders. Budget accordingly, or cut
> `--aes` down to one. This expert wants a GPU.

## Fusion

`fuse` joins the two caches **on key** — they are never zipped, because
extraction runs happen at different times, images can fail in one pass and not
the other, and augmentation draws different specs. A two-feature logistic
regression over `[probe_score, recon_score]` is then fitted and calibrated, and
the `probe only / recon only / fused` ablation falls out for free.

**Measured outcome on SID_Set: +0.0001.** Fusing the reconstruction expert with
the DINOv2-L probe moves pooled AUC from 0.9025 to 0.9026; against the stronger
SigLIP2 probe it is very slightly negative, 0.9497 → 0.9493. The fusion
machinery itself behaves correctly — it recovers the better expert rather than
being dragged down by the weaker one, which is exactly what a sound score-level
fusion should do with a weak input. On this corpus the two-expert architecture
is a one-expert architecture with overhead, and that is reported rather than
quietly dropped:
[`docs/results-recon-fusion.md`](docs/results-recon-fusion.md).

### A second fusion result, on a different corpus and protocol

[`experiments/fusion/`](experiments/fusion/README.md) reports a *three*-expert
fusion — CLIP + reactivity-delta, DINOv2, AEROBLADE — reaching **AUC 0.9953 and
TPR@1%FPR 0.9173**, with 5-fold cross-validation and a full 7-way ablation.
That is a real result and it is not in conflict with the +0.0001 above, because
the two were not measured on the same thing. **Do not read the two side by
side without this table:**

| | this README | `experiments/fusion/` |
|---|---|---|
| corpus | SID_Set, 16,000 train / 1,600 test | a 2,506 / 443 split of `byteprint-realdata` |
| evaluation | all 15 §5.2 rungs — **24,000 laundered views** | **clean images only**, 443 |
| what fusion is measured against | a 0.9497 SigLIP2 probe | a 0.9926 CLIP + reactivity-delta model |

The decisive difference is the second row. Robustness under the §5.2 ladder is
the graded axis of this competition, and the three-expert numbers are clean-only
— their own README names this as open work. A weak expert also has much more
room to help a fusion when the ladder is not there to separate the experts'
failure modes for it. Treat 0.9953 as a clean-image result on a different
corpus, not as this project's headline, until the ladder version of it runs.

## The deliverable interface

An image directory in, a JSON file out, with `pred` the likelihood the image is
AI-generated:

```bash
byteprint score IMAGE_DIR --probe runs/probe.joblib --out predictions.json
# or, as a standalone script:
python scripts/score_directory.py IMAGE_DIR --probe runs/probe.joblib --out predictions.json
```

```json
[
  { "image_path": "/data/inbox/img_000.png", "pred": 0.9376 },
  { "image_path": "/data/inbox/img_001.png", "pred": 0.0002 }
]
```

Nothing but the probe file and the directory is required — the extraction
settings (backbone, crop size, crop count, crop mode) travel *inside* the saved
probe, because scoring at a different crop size than you trained at degrades
quietly rather than loudly. `--relative` reports bare filenames instead of
absolute paths, for a harness keyed on those, and `--workers 4` roughly halves
the wall clock on a large directory without changing a number in the output.

**Every discovered image gets exactly one entry.** A directory of ten thousand
images will contain a truncated download or an HTML page named `.png`, and a
scorer that dies on the first one has scored nothing. Unreadable files are
reported with `pred: 0.5` — maximum uncertainty, the honest answer for an image
we never saw — plus an `error` field naming the cause, so the output stays
complete and the failure stays visible:

```json
{ "image_path": "/data/inbox/corrupt.png", "pred": 0.5,
  "error": "cannot identify image file '/data/inbox/corrupt.png'" }
```

`--strict` turns the first failure back into a crash.

## Install

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e '.[dev]'
uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cpu torch torchvision
```

## Quick start

```bash
byteprint list                                                  # every swappable part
byteprint fixture --out data --per-class 40 --size 256          # synthetic smoke-test data
byteprint extract --data data/train --cache runs/cache/train --crops 2 --augment 4
byteprint train   --cache runs/cache/train --out runs/probe.joblib --target-fpr 0.01
byteprint eval    --cache runs/cache/test  --probe runs/probe.joblib --by-spec
byteprint logo    --cache runs/cache/train                      # leave-one-generator-out
byteprint predict --probe runs/probe.joblib IMAGE...          # ad-hoc, a few files
byteprint score   IMAGE_DIR --probe runs/probe.joblib --out predictions.json
```

Scoring the official robustness ladder means extracting every rung of it:

```bash
byteprint extract --data data/test --cache runs/cache/ladder --ladder official
byteprint eval    --cache runs/cache/ladder --probe runs/probe.joblib --by-spec
```

Adding the second expert:

```bash
byteprint extract --data data/train --cache runs/cache/recon_train \
               --expert recon --crop-size 112 --crops 1
byteprint fuse    --dino-cache runs/cache/train --recon-cache runs/cache/recon_train \
               --probe runs/probe.joblib --out runs/fused.joblib
byteprint eval    --cache runs/cache/test --recon-cache runs/cache/recon_test \
               --fused runs/fused.joblib --by-spec
```

Autoencoders available to `--aes`: `sd15`, `sd14`, `vae-mse`, `vae-ema`, `sdxl`
(default `sd15,vae-mse,sdxl` — three genuinely different decoders; the others are
near duplicates that add cost without coverage). The `stabilityai/stable-diffusion-2-*`
repos are no longer resolvable on the Hub and are deliberately absent rather than
listed and broken.

## Your own data

```
data/
  train/
    real/                 any nesting; every image is label 0
    fake/
      sdxl/               one directory per generator
      flux/
  test/
    real/
    fake/
      midjourney/         held-out generator
```

Fakes placed directly under `fake/` are kept as generator `unknown` — they train
fine, they just cannot be broken out in per-generator reporting.

## Swappable parts

Six people on six branches should not be editing the same `if` ladder. The
backbone, the head (and so the training objective), the crop strategy and the
autoencoder bank are **registry entries**: you add one by writing a function in
your own module and decorating it, then point any command at that module.

```python
# myteam/heads.py
from byteprint.heads import register_head
from sklearn.ensemble import HistGradientBoostingClassifier

@register_head("gbt")
def _build(config):
    return HistGradientBoostingClassifier(random_state=config.seed)
```

```bash
byteprint list                                            # what is registered
byteprint train --plugin myteam.heads --head gbt ...      # or BYTEPRINT_PLUGINS=myteam.heads
```

| Extension point | Decorator | Shipped |
|---|---|---|
| backbone — the model | `@register_backbone` | `dinov2_vits14/vitb14/vitl14/vitg14`, `dinov2_{large,giant}_hf`, `eva02_large_timm`, `siglip2_so400m_hf` |
| head — the training objective | `@register_head` | `logreg` (log loss), `linear-svm` (hinge), `mlp` |
| crop strategy — where you look | `@register_crop_mode` | `texture`, `anomaly`, `ela`, `random`, `center`, `resize` |
| autoencoder bank | `AUTOENCODERS` dict | `sd15`, `sd14`, `vae-mse`, `vae-ema`, `sdxl` |

Names are validated when they are *used*, not when arguments are parsed, so a
`--plugin` entry is as first-class as a built-in one. Registering a name twice
is an error rather than a silent override — two branches both claiming `"probe"`
and letting import order decide is a day-long bug.

One thing is deliberately **not** pluggable: `OFFICIAL_LADDER`. It is §5.2 of
the brief transcribed verbatim and pinned by a test, because changing it
invalidates every robustness number already quoted. Exploratory chains go in
`STRESS_LADDER` and are reported separately.

Full contracts and worked examples: [`docs/extending.md`](docs/extending.md).

## The four design decisions

**1. Crop at native resolution; never resize.** Resizing to fit a backbone's
input is a low-pass filter applied directly to the evidence. The default mode
samples candidate crops, scores each by Laplacian-response variance, and keeps
the most texture-rich. `--crop-mode resize` reproduces the naive baseline so you
can measure the difference rather than take it on faith.

**1b. Looking where the edit is, tested and rejected.** *(A negative result, kept
because it cost a day and rules something out.)* The reasoning was that ranking
windows by high-frequency energy is right when the whole image is synthetic and
wrong when only a region of it is: a region produced by a decoder is usually
*cleaner* than the captured frame around it, so `texture` would not merely miss
the edit but sort it last. On a fixture that plants a locally denoised patch —
structure intact, sensor grain removed — `texture` does land on the planted
region **0 times out of 40**.

`--crop-mode anomaly` ranks each candidate window by how far its high-frequency
statistics sit from *the rest of the same image*, in MAD units, using the median
so that windows inside the edit cannot define the baseline they are measured
against. It finds the planted region **40 times out of 40**.

The separation is not delicate: images carrying an edit score a maximum
deviation of 66+, uniform images at most 1.19, and the hand-over threshold sits
at 6.0 — in the empty gap between them rather than on a slope. That matters
because the mode must not buy back the 0.851 tampered number by spending the
0.954 full-synthetic one: a uniformly generated image has no odd region, so when
nothing stands out `anomaly` defers to `texture` and reproduces its ranking
exactly, which is pinned by a test rather than assumed.

Across the §5.2 ladder it holds 24/24 on twelve of the fifteen rungs and goes
blind on `blur:2.0` and `scale:0.25` — which destroy the grain everywhere, and
so destroy the contrast the cue reads. It fails *safely* there: the fallback
fires on every image, so the mode degrades to today's behaviour rather than
returning windows ranked by noise.

`ela` (classic error-level analysis) is registered as the absolute-cue control.
The expectation was that it would collapse under recompression while a
within-image contrast survived. **It does not, and the control is what caught
it**: ranked as an outlier rather than by raw energy, ELA holds across the ladder
and beats `anomaly` outright on exactly the two rungs where the grain is gone.
The two cues fail on different rungs — ELA has its own blind spot at `jpeg:90`,
where re-encoding at the image's own quality leaves almost no residual — which is
the same complementary-failure argument this project already makes for fusing two
experts, and the reason both stay registered.

Cost is ~1.8× `texture` per image, not the 4× a naive implementation gives: a
window's texture score is the variance of its luma Laplacian, which is the same
band the fingerprint needs, so both come out of one pass and the fallback path
costs nothing extra.

**And on SID_Set it does not work.** All of the above is a planted proxy, and it
did not transfer:

| | `texture` | `anomaly` | `ela` |
|---|---|---|---|
| **tampered** | **0.8513** | 0.8317 | 0.8318 |
| full synthetic | **0.9537** | 0.9511 | 0.9505 |
| pooled AUC | **0.9025** | 0.8914 | 0.8912 |
| TPR @ 1% FPR | 0.3362 | **0.3693** | 0.3504 |
| TPR @ 0.1% FPR | 0.1325 | **0.1683** | 0.1357 |

Tampered AUC went the wrong way, and the two cues — grain statistics and
compression response, physically unrelated — agree within 0.0001. Two independent
instruments converging on the same wrong answer is what makes this conclusive:
crop placement is not the constraint. `anomaly` does lift the operating point
(+27% relative at 0.1% FPR), which was not predicted, has no mechanism yet, and
is not shared by `ela` — so it is a lead, not a finding.

Both modes stay registered as measured negative results. The default is still
`texture`. What the run appeared to indict was **mean-pooling**: crop embeddings
were averaged into one row per image, so aiming crops at a region whose evidence
is then averaged away costs more than it gains.

Full tables, the cost breakdown, why the fixture was confidently wrong, and an
exact reproduction of the published baseline that fell out of the control arm:
**[`docs/results-crop-localisation.md`](docs/results-crop-localisation.md)**.

**That follow-up has since run, and pooling was refuted too.** The cache now
stores one row per crop and pooling is a train-time flag, so mean, max and top-k
are a sweep over one extraction. At matched crop count, max and top-k lose to
mean on both backbones and badly damage the low-FPR operating point — a maximum
selects each image's *noisiest* crop, which is how an authentic photograph
becomes a confident false positive. What the run did find, through a control that
cost nothing, is that **crop count** is the largest lever measured so far:
2 → 8 crops takes SigLIP2 from 0.9497 to **0.9688** pooled and from 0.5854 to
**0.6995** TPR@1%FPR. Two rounds of "look at the right part of the image" have
now come back negative, while "look at more of it" worked twice.
**[`docs/results-crop-pooling.md`](docs/results-crop-pooling.md)**.

**2. Train on the damage.** `--augment N` draws N distinct random laundering
chains per image (JPEG, downscale, blur, noise, and combinations). Because the
cache is keyed by spec, augmented views cost backbone time once, not once per
epoch.

**2b. Overlap the decode with the backbone.** Both pipelines are CPU-bound, not
GPU-bound: decoding a full-size photograph and scoring 32 candidate crop windows
costs far more than pushing two crops through a frozen ViT. Choosing crops is
the larger half — 52 ms/image against 31 ms to decode, at 1024px — so both
stages move onto a thread pool behind `--workers N`:

| | `extract` | `score` |
|---|---|---|
| `--workers 1` | 12.7 img/s | 12.8 img/s |
| `--workers 4` | **27.8 img/s (2.2×)** | **29.1 img/s (2.3×)** |
| `--workers 8` | 25.4 img/s | 21.5 img/s |

The pool is deliberately invisible to the result. In `extract`, the laundering
draw and the cache-skip check stay sequential and rows are appended in
submission order; in `score`, chunks are cut at the same paths whatever the
worker count, because an image's crops are seeded by its position within its
chunk. The backbone is only ever called from the calling thread — a torch module
is not safe to run forward on concurrently. Caches built at `--workers 1`, `4`
and `12` are byte-identical, as is the predictions file, and that is pinned by
tests rather than asserted here.

The gain flattens past four threads as the GIL takes it back, and on small
images the handover costs more than the work — which is why the default is `1`
and the flag is opt-in.

**3. Calibrate, don't threshold at 0.5.** Probe scores shift systematically
between generators and laundering paths. `train` holds out a calibration split
and fits the threshold at `--target-fpr`.

**4. Report TPR at a fixed low FPR.** On a real platform authentic images
outnumber synthetic ones by orders of magnitude, so accuracy and even AUC
flatter a detector that would be unusable. Every report leads with AUC but
carries TPR@1%FPR and TPR@0.1%FPR, plus a per-generator breakdown — an average
over generators hides the one you cannot detect at all.

## Results on SID_Set

`eval --by-spec` scores each rung of the §5.2 ladder separately. The headline
configuration is the default one: SigLIP2-so400m, 2×224px texture crops, 16,000
SID_Set training images with `--augment 3`, scored on 1,600 held-out images
across all fifteen rungs — 24,000 views.

> **This configuration has since been beaten on two independent axes.**
> Everything in this section describes it and remains accurate for it; it is
> kept as the reference point every later result is measured against.
>
> - **More crops.** The same pipeline at `--crops 8` scores **0.9688** pooled,
>   **0.6995** TPR@1%FPR, **0.4626** TPR@0.1%FPR, 0.9494 tampered and 0.7642
>   LOGO mean, at 4× the backbone forward when scoring.
>   [`docs/results-crop-pooling.md`](docs/results-crop-pooling.md)
> - **A shallower read.** Reading layer 12 of 27 alongside the pooled output,
>   at 2 crops, scores **0.9717** pooled, **0.6922** TPR@1%FPR and **0.4409**
>   TPR@0.1%FPR. Reading layer 9 *alone* beats the baseline at **34% of the
>   parameters and 2.9× the throughput**.
>   [`docs/results-depth-frontier.md`](docs/results-depth-frontier.md)
>
> **These two gains are not additive, and we measured that rather than assuming
> it.** Crossing them in one run gives an interaction of **−0.0088 AUC**: each is
> worth ~+0.019 alone and ~+0.010 once the other is present, and at 1% FPR depth
> keeps only 22% of its value once you have 8 crops. The arithmetic that gets you
> to 0.99 is wrong.
> [`docs/results-depth-crops.md`](docs/results-depth-crops.md)

| | AUC |
|---|---|
| **pooled over the ladder** | **0.9497** |
| best rung | 0.9805 |
| worst rung (`noise:0.10`) | 0.8624 |
| full synthetic | 0.9817 |
| tampered | 0.9176 |
| unseen manipulation type (LOGO mean) | 0.7208 |

At the operating point: **TPR@1%FPR 0.5854**, TPR@0.1%FPR 0.2554. That is the
number to read rather than the AUC — at a threshold strict enough to wrongly
flag one authentic image in a hundred, the detector catches 59% of
AI-generated images.

### The backbone is what decided it

Six frozen extractors through one identical pipeline, changing nothing else.
**Ranking them by parameter count nearly inverts ranking them by AUC** — on this
task the pretraining objective is the lever, not scale:

| backbone | pretraining | params | AUC | TPR@1%FPR | LOGO mean | ladder span |
|---|---|---|---|---|---|---|
| `dinov2_large_hf` (the original baseline) | self-distillation | 0.30B | 0.9025 | 0.3362 | 0.6457 | 0.064 |
| `dinov2_giant_hf` | self-distillation | 1.14B | 0.9261 | 0.4170 | 0.6388 | **0.052** |
| `eva02_large_timm` | MIM from CLIP | 0.30B | 0.9182 | 0.4036 | 0.5871 | 0.120 |
| `clip_b32_proj_hf` | language-supervised | **0.09B** | 0.9227 | 0.4457 | 0.7032 | 0.096 |
| `clip_b32_hf` | language-supervised | **0.09B** | 0.9319 | 0.4575 | 0.7025 | 0.095 |
| **`siglip2_so400m_hf`** | language-supervised | 0.43B | **0.9497** | **0.5854** | **0.7208** | 0.118 |

**The top two are both language-supervised, from different model families.** The
smallest entry in the table beats the largest — CLIP ViT-B/32 clears DINOv2-giant
on AUC, on the operating point and on transfer at 7.6% of its parameters. Ranked
by pretraining objective the table sorts cleanly; ranked by parameter count it
does not sort at all. CLIP's own numbers, and what they do *not* say about the
detector on `jiahui/clip-detector`, are in
**[`docs/results-clip-backbone.md`](docs/results-clip-backbone.md)**.

**The one honest argument against the default is in the last column.**
Robustness is the graded axis, and DINOv2-giant keeps the flattest ladder (span
0.052 against 0.118) and the highest floor (0.8906 against 0.8624). Head to head
per rung SigLIP2 takes 13 of 15 and giant takes the two heaviest-noise rungs.
Every backbone's worst rung is `noise:0.10`, without exception. EVA02 ran at
224px against a native 448, so it is under-tested here rather than beaten.

Full table, per-rung numbers and cost:
**[`docs/results-backbone-sweep.md`](docs/results-backbone-sweep.md)**.

### The depth frontier

Every number above reads one thing: the tower's final pooled output. Not
because it was chosen, but because it is all the shipped adapters return.
Tapping eleven depths of SigLIP2 plus that pooled output — all from the *same*
forward pass, so the whole curve costs one extraction — says the final layer is
the wrong layer to read, and not by a little.

| read | AUC | TPR@1%FPR | TPR@0.1%FPR | carried params | throughput |
|---|---|---|---|---|---|
| layer 5 (19% of depth) | 0.9513 | 0.5635 | 0.3334 | 77M (0.19×) | **4.98×** |
| layer 9 (33%) | 0.9612 | 0.6101 | 0.2308 | 138M (0.34×) | **2.90×** |
| layer 12 (44%) | **0.9617** | 0.6072 | 0.3815 | 184M (0.45×) | 2.20× |
| layer 27, mean-pooled | 0.9429 | 0.5490 | 0.1969 | 413M | 1.00× |
| pooler *(what we shipped)* | 0.9497 | 0.5854 | 0.2554 | 413M | 1.00× |
| **layer 12 + pooler** | **0.9717** | **0.6922** | **0.4409** | 413M | 1.00× |

Read it as two different detectors, not one:

- **Cheaper *and* better.** Layer 9 beats the shipped baseline on AUC and on the
  operating point while carrying a third of the weights and running three times
  faster. Layer 5 matches the shipped AUC at 19% of the parameters. This
  continues the arc the backbone sweep started — 1.14B, then 0.43B, now an
  effective 0.14B — three steps down the parameter axis, none costing accuracy.
- **Best accuracy.** Layer 12 concatenated with the pooled output. It needs the
  full tower, so it buys accuracy and no speed. **Do not quote 0.9717 and "0.34×
  parameters" in the same sentence** — they are different rows.

The mechanism is the part worth keeping. **The worst rung changes with depth:**
every tap up to layer 9 fails worst on `blur:2.0`, every tap from layer 12 on
fails worst on `noise:0.10`, and the crossover sits exactly where the curve
peaks. Shallow features are high-frequency, so blur destroys them; deep features
are semantic and blur-tolerant, but noise perturbs them. Layer 12 is where
neither failure has taken hold — an optimum made of two competing failure modes,
which aggregate AUC hides completely.

Two honest weaknesses. A control holding depth fixed (`layer 27 + pooler`, the
same 2,304 columns) gains only +0.0056 AUC against the two-depth version's
+0.0220, so about a third of the fusion gain is capacity and two thirds is
genuine depth diversity — both real, only the second a finding. And **transfer
does not improve**: the pooler's LOGO mean of 0.7208 is still the best in that
column, with layer 12 at 0.6398. Unseen-type transfer remains this project's
weakest axis, and depth does not move it.

Predictions were registered before the run in
[`docs/depth-frontier-prediction.md`](docs/depth-frontier-prediction.md); one of
the three was refuted, and it is the most interesting thing in
**[`docs/results-depth-frontier.md`](docs/results-depth-frontier.md)**.

### Robustness summary — clean versus transformed

The graded axis, on one page. Every rung of the §5.2 ladder, SigLIP2 at 8 crops,
1,600 held-out images per rung. `none` is the clean control; everything else is
the same images after one transform.

| rung | pooler (shipped read) | layer 12 (best read) | Δ |
|---|---|---|---|
| `none` *(clean)* | 0.9816 | **0.9869** | +0.0053 |
| `jpeg:90` | 0.9781 | 0.9852 | +0.0071 |
| `jpeg:70` | 0.9721 | 0.9844 | +0.0123 |
| `jpeg:50` | 0.9676 | 0.9809 | +0.0133 |
| `jpeg:30` | 0.9539 | 0.9703 | +0.0164 |
| `blur:0.5` | 0.9850 | 0.9862 | +0.0012 |
| `blur:1.0` | 0.9715 | 0.9773 | +0.0058 |
| `blur:2.0` | 0.9321 | 0.9307 | −0.0014 |
| `scale:0.5` | 0.9694 | 0.9795 | +0.0101 |
| `scale:0.25` | 0.9249 | 0.9405 | +0.0156 |
| `noise:0.02` | 0.9565 | 0.9778 | +0.0213 |
| `noise:0.05` | 0.9286 | 0.9613 | +0.0327 |
| `noise:0.10` | 0.8897 | 0.9215 | +0.0318 |
| `jitter:0.2` | 0.9749 | 0.9856 | +0.0107 |
| `crop:0.8` | 0.9807 | 0.9860 | +0.0053 |
| **pooled** | **0.9608** | **0.9712** | **+0.0104** |
| **spread, clean → worst** | 0.0919 | **0.0654** | |

Three things to read off it. **Degradation is graceful** — the worst rung of the
best read is 0.9215, so nothing collapses. **Heavy noise is the binding
constraint** — `noise:0.10` is the worst rung for both reads, with `blur:2.0`
and `scale:0.25` next; JPEG at any quality in the list is close to free. And
**the mid-depth read's advantage is largest exactly
where the ladder bites hardest** — +0.0318 on `noise:0.10` against +0.0053 on
clean — which is the opposite of what we predicted before the run, and the
reason the depth result matters for a robustness-graded task rather than being a
clean-accuracy curiosity.

These numbers come from the half-scale-train run in
[`docs/results-depth-crops.md`](docs/results-depth-crops.md), so read them
against each other and not against the tables above; the ladder itself is full
scale.

### How the baseline got here, and the control that cleared it

The first run used a DINOv2-L probe — one 48 GB GPU, 4h 40m — and scored pooled
AUC 0.9025, TPR@1%FPR 0.3362, with a ladder spanning **0.064 AUC** from 0.8553
(`noise:0.10`) to 0.9191 (`blur:1.0`). Clean was not even its best rung:
`--augment 3` *replaces* the spec list rather than adding to it, so the probe
trained almost entirely on laundered views, which is where deployed images live.

The three weaknesses it exposed, and where each one now stands under the default
backbone:

- **The operating point.** 0.3362 → **0.5854** at 1% FPR. Much improved, and
  still the number that decides whether this is deployable.
- **Tampered images are harder than fully synthetic ones.** 0.8513 vs 0.9537 →
  **0.9176 vs 0.9817**. The gap narrowed but did not close, and crop placement
  turned out not to be its cause — see the known gaps above.
- **Transfer to an unseen manipulation type.** 0.6457 → **0.7208**. Still the
  weakest axis, and still far below the ~0.95 in-distribution figure.

One caveat used to bound all of it: SID_Set's reals are 100% JPEG while its
fully-synthetic images are 100% PNG. Every class is re-encoded to PNG so the
container cannot be the classifier, but JPEG history survives in the pixels —
which would make 0.9025 a measure of the dataset rather than of the detector.

**That control has now run, and it clears the number.** Re-encoding *both*
classes through JPEG-95 and rerunning the identical pipeline gives **0.9022**,
against the baseline's 0.9025. No rung moves by more than 0.0016; the
per-generator split is unchanged to three decimal places. Compression history
was not what the probe was reading.

| | baseline (PNG) | control (JPEG-95) |
|---|---|---|
| pooled over the ladder | 0.9025 | **0.9022** |
| full synthetic | 0.9537 | 0.9535 |
| tampered | 0.8513 | 0.8510 |
| LOGO mean (unseen type) | 0.6457 | 0.6486 |

What the control does not settle: the reals are now double-JPEG and the
synthetics single, which is itself detectable. But if compression were the
feature, changing it from "JPEG vs none" to "double vs single" should have
perturbed *something*, and nothing moved.

Full tables, per-rung numbers and stage timings:
**[`docs/results-sid-set-first-run.md`](docs/results-sid-set-first-run.md)** and
**[`docs/results-jpeg95-control.md`](docs/results-jpeg95-control.md)**.

### The same ladder on the bundled fixture

For contrast — DINOv2-S, 40 images per class, `--augment 4`. The fixture's fakes
are a planted periodic grid, so its numbers mean *the wiring works*, nothing
more, and its failure modes are artifacts of the plant:

| rung | AUC | TPR@1%FPR | |
|------|-----|-----------|---|
| `none` | 0.9400 | 0.6500 | |
| `jpeg:90` | 0.9175 | 0.6500 | |
| `jpeg:70` | 0.9300 | 0.8500 | |
| `jpeg:50` | 0.8275 | 0.7000 | |
| `jpeg:30` | 0.7600 | 0.5000 | |
| `blur:0.5` | 0.8900 | 0.5000 | |
| `blur:1.0` | 0.8625 | 0.6000 | |
| `blur:2.0` | **0.3550** | 0.0500 | ← below chance |
| `scale:0.5` | 0.8950 | 0.5500 | |
| `scale:0.25` | **0.4100** | 0.1000 | ← below chance |
| `noise:0.02` | 0.9050 | 0.5000 | |
| `noise:0.05` | 0.8475 | 0.1000 | |
| `noise:0.10` | 0.7325 | 0.1000 | |
| `jitter:0.2` | 0.9250 | 0.6500 | |
| `crop:0.8` | 0.9275 | 0.6000 | |

A single pooled number over that same set reads **0.8349**, which sounds
respectable and tells you nothing about the two rungs where the detector is
*worse than a coin*. Both destroy high-frequency detail, which on this fixture
is the entire signal — the failure mode is legible only because the rungs are
reported apart. On real data those same two rungs hold 0.889 and 0.890, which
is the point: per-rung reporting is what tells you whether a collapse is real.

Note also what the pooled number does to the operating point: TPR@1%FPR over
everything is 0.3833, but the per-rung values range from 0.85 to 0.05. A single
threshold fitted on the pooled distribution is the wrong threshold for every
rung individually.

> The bundled fixture plants an explicit periodic grid artifact in the "fake"
> class, so its absolute numbers mean *the wiring works*, nothing more. Both
> classes are written as PNG at identical size on purpose: reals-as-JPEG against
> fakes-as-PNG is the most common way this benchmark gets accidentally faked,
> and yields a 99% container-format classifier.

## Backbones

Two families. The `torch.hub` entries download on demand; the `_hf` and `_timm`
entries read from a local HuggingFace cache and **will not download**, which is
what an offline compute node needs — stage their weights first.

| name | dim | params | notes |
|------|-----|--------|-------|
| `siglip2_so400m_hf` | 1152 | 0.43B | **default** — best on every axis, see [the sweep](docs/results-backbone-sweep.md) |
| `clip_b32_hf` | 768 | 0.09B | second of six at a twelfth of giant's size; CLIP's pre-projection feature |
| `clip_b32_proj_hf` | 512 | 0.09B | the same tower through `visual_projection` — slightly worse, except on transfer |
| `dinov2_giant_hf` | 1536 | 1.14B | flattest robustness ladder, but 13× CLIP's parameters for less accuracy |
| `eva02_large_timm` | 1024 | 0.30B | natively a 448 model; under-tested at our 224 crops |
| `dinov2_large_hf` | 1024 | 0.30B | the original published baseline |
| `dinov2_vits14` | 384 | 0.02B | no staging needed; runs on CPU |
| `dinov2_vitb14` | 768 | 0.09B | |
| `dinov2_vitl14` | 1024 | 0.30B | |
| `dinov2_vitg14` | 1536 | 1.14B | over half the <2B budget on its own |

That list is a registry, not a fixed menu: `@register_backbone` adds one from
your own module. See [Swappable parts](#swappable-parts).

Device is auto-detected (`cuda` → `mps` → `cpu`). `--crop-size` must be a
multiple of the backbone's patch size — 14 for the DINOv2 and EVA02 entries,
16 for SigLIP2, 32 for CLIP. 224 satisfies all three.

## Tests

```bash
.venv/bin/python -m pytest
```

## A note on the fixture and the second expert

On the bundled fixture the reconstruction expert scores **below chance**
(AUC 0.185) and fusion correctly ignores it, staying at the probe's 1.0000. That
is the right answer, not a bug: the fixture's fakes are a planted sinusoid, not
latent-diffusion output, so AEROBLADE has nothing to detect — and the grid makes
those images marginally *harder* to reconstruct, hence the anti-correlation. It
is also a genuine check on the fusion: a useless expert did not drag the
ensemble down.

The SD-1.5 table above shows the mechanism working on the output it was designed
for. What neither the fixture nor that table can answer — and what SID_Set now
does — is whether the mechanism is worth anything on the competition corpus. It
is not: AUC 0.5822 overall, and chance-level (0.4975) on tampered images. That
is a property of this corpus and these crops rather than a verdict on AEROBLADE;
[`docs/results-recon-fusion.md`](docs/results-recon-fusion.md) sets out the
three ways the test was hostile to the method.

## Limitations, and what we would do with more time

The honest list, in the order we would attack it.

**One corpus.** Every headline number is SID_Set. The two classes it offers are
*fully synthetic* and *tampered*, which are two kinds of editing, not two
generators — so what we report as transfer (LOGO) is transfer to an unseen
*manipulation type*, and it should never be quoted as generator
generalisation. Nothing here has been tested against a diffusion model the
probe has not seen. That is the single largest gap between this and a
deployable detector.

**Transfer is the weakest axis and nothing has moved it.** In-distribution
pooled AUC is ~0.97; unseen-manipulation-type transfer is 0.72–0.76. The
backbone sweep moved it, crop count moved it a little, and depth did not move
it at all — the shipped attention pooler still has the best LOGO of any read we
have tried, which is unexplained and worth understanding rather than
papering over.

**Tampered images remain the harder class**, and two rounds of the obvious fix
came back negative. Crop *placement* (`anomaly`, `ela`) made it worse; crop
*pooling* (max, top-k, mean-score) made it worse. What helped twice was seeing
more of the image. We would take that seriously next and test crop counts
beyond 8 — the curve between 2 and 8 is unmeasured, and one metric already
falls at 8 on DINOv2, so it is not monotone.

**One seed, one split, no error bars.** The large effects here (+0.02 AUC,
+0.11 TPR@1%FPR) are far outside plausible seed variance, but adjacent taps —
layer 9 at 0.9612 against layer 12 at 0.9617 — are not, and we treat them as
tied. Repeat seeds are the cheapest missing experiment in the project.

**The second expert does not earn its place on this corpus.** AEROBLADE scores
0.5822 alone and fusion moves the pooled AUC by +0.0001. We kept it as a
measured negative rather than deleting the evidence, but BYTEPRINT should be
described as a strong single-expert detector, not a working ensemble. The
three-expert result in [`experiments/fusion/`](experiments/fusion/README.md)
suggests fusion *can* pay with a stronger second expert — on clean images, on
another corpus. Running that over the §5.2 ladder is the experiment we would
run next with a spare GPU.

**Error analysis is thin.** We report distributional metrics (AUC, TPR at fixed
FPR, per-rung, per-class) but have not yet characterised *which images* fail.
`experiments/fusion/` contains a per-image disagreement analysis for its own
corpus; the equivalent for the SigLIP2 probe on SID_Set is not done.

**No adversarial evaluation.** The §5.2 ladder is non-adversarial degradation.
An attacker optimising against the detector is a different threat model and we
have not touched it.

## Team

| | |
|---|---|
| **Mateo** | Detection pipeline and CLI, extraction/caching layer, laundering ladder (`OFFICIAL_LADDER`), backbone sweep, crop-mode and crop-pooling studies, depth-frontier study, SLURM job scripts, test suite |
| **jiahui** | CLIP + reactivity-delta detector and domain routing, three-expert fusion experiment and its ablations ([`experiments/fusion/`](experiments/fusion/README.md)) |
| **Fan Kuan** | **Video Direction & Editing:** Translated complex ML concepts (like the depth-frontier crossover) into a polished, high-impact demo video. |
## Not included

No DIRE (full diffusion inversion), no adversarial evaluation, no localization,
no MLLM rationales.
