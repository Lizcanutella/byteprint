# BYTEPRINT

**Robust detection of AI-generated images under real-world transformations.**

Every generator leaves a byteprint. A camera sensor imprints grain, optics and
a compression history; a latent-diffusion decoder imprints its own manifold
instead. BYTEPRINT looks for both traces at once — and, because an image is
compressed, cropped and reposted before anyone sees it, it is graded on what
survives the wash rather than on clean accuracy.

By team ByteSized. The models are byte-sized too: the whole thing fits well
inside the competition's <2B parameter budget, and the default configuration
runs on a CPU.

Two independent detectors fused at the score level:

1. **DINOv2 + linear probe** — frozen self-supervised features, native-resolution
   texture crops, a calibrated logistic-regression head.
2. **AEROBLADE** — training-free reconstruction error through a bank of latent
   diffusion autoencoders.

They fail on different inputs, so fusing them buys coverage rather than decimal
places. Around both sits an evaluation harness built on the numbers that
actually predict deployment behaviour.

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
  other at only 0.6457. Both numbers argue for a crop strategy with some notion
  of *where* to look, which does not exist yet.

Closed since the first draft:

- The laundering ladder now matches §5.2 exactly — colour jitter and centre crop
  are implemented, noise σ is normalised to the brief's 0–1 convention (values
  above 1.0 are rejected rather than reinterpreted), and the rung list is pinned
  by a test so it cannot drift.
- The directory → JSON entry point exists: `byteprint score`, below.

## Expert 1 — DINOv2 probe

The backbone is never fine-tuned. That is deliberate: a general-purpose
self-supervised representation transfers across unseen generators far better
than a network trained end-to-end on one of them, which reliably descends onto
generator-specific shortcuts. It also means embeddings are a pure function of
(image, laundering spec, extraction config), so they are computed **once** and
cached — after which retraining the probe, recalibrating a threshold, or running
leave-one-generator-out costs seconds.

```
                 ┌─ DINOv2 (frozen) ─────► dino cache ─► probe ──┐
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

> **Cost.** ~1.2GB of VAE weights, and on CPU roughly **2.9 s/crop** at 112px or
> **13.7 s/crop** at 224px with three autoencoders. Budget accordingly, or cut
> `--aes` down to one. This expert wants a GPU.

## Fusion

`fuse` joins the two caches **on key** — they are never zipped, because
extraction runs happen at different times, images can fail in one pass and not
the other, and augmentation draws different specs. A two-feature logistic
regression over `[probe_score, recon_score]` is then fitted and calibrated, and
the `probe only / recon only / fused` ablation falls out for free.

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
| backbone — the model | `@register_backbone` | `dinov2_vits14/vitb14/vitl14/vitg14` |
| head — the training objective | `@register_head` | `logreg` (log loss), `linear-svm` (hinge), `mlp` |
| crop strategy — where you look | `@register_crop_mode` | `texture`, `random`, `center`, `resize` |
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

`eval --by-spec` scores each rung of the §5.2 ladder separately. DINOv2-L probe,
2×224px texture crops, 16,000 SID_Set training images with `--augment 3`, scored
on 1,600 held-out images across all fifteen rungs — one 48 GB GPU, 4h 40m.

| | AUC | TPR@1%FPR |
|---|---|---|
| pooled over the ladder | 0.9025 | 0.3362 |
| clean (`none`) | 0.9112 | 0.4100 |
| best rung (`blur:1.0`) | 0.9191 | 0.3750 |
| worst rung (`noise:0.10`) | 0.8553 | 0.2550 |

**The whole ladder spans 0.064 AUC.** Robustness is the graded axis, and this is
the number to point at: the worst of fifteen real-world transformations costs
six points. Clean is not even the best rung — `--augment 3` *replaces* the spec
list rather than adding to it, so the probe trained almost entirely on laundered
views, which is where deployed images live.

Three weaknesses the same run exposes, stated plainly:

- **The operating point is mediocre.** At a threshold strict enough to wrongly
  flag 1 authentic image in 100, two thirds of AI-generated images still get
  through. AUC 0.90 flatters it.
- **Tampered images are much harder** (AUC 0.8513) than fully synthetic ones
  (0.9537) — only a *region* is generated, and texture-ranked crops have no
  reason to land on it.
- **Transfer to an unseen manipulation type is weak**: mean leave-one-out AUC
  0.6457, down from ~0.90 in-distribution.

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

| name | dim | notes |
|------|-----|-------|
| `dinov2_vits14` | 384 | default; runs on CPU |
| `dinov2_vitb14` | 768 | |
| `dinov2_vitl14` | 1024 | the published strong baseline; wants a GPU |
| `dinov2_vitg14` | 1536 | ~1.1B params — over half the <2B budget on its own |

That list is a registry, not a fixed menu: `@register_backbone` adds one from
your own module. See [Swappable parts](#swappable-parts).

Device is auto-detected (`cuda` → `mps` → `cpu`). `--crop-size` must be a
multiple of 14.

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

The expert's real validation is the SD-1.5 table above, which the fixture cannot
provide.

## Not included

No DIRE (full diffusion inversion), no adversarial evaluation, no localization,
no MLLM rationales.
