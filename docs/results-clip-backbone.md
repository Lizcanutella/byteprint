# CLIP ViT-B/32 in the sweep — the smallest backbone finishes second

`jiahui/clip-detector` reports strong numbers for a CLIP-based detector, but on
its own test set and its own 16-cell transform grid, so they cannot be read
against the sweep table. This run puts that branch's *backbone* through the
identical pipeline the other four went through and changes exactly one thing.

**CLIP ViT-B/32 places second of six on pooled AUC, beating DINOv2-giant at 7.6%
of its parameters.** 0.9319 against giant's 0.9261, and it wins the operating
point (TPR@1%FPR 0.4575 against 0.4170) and unseen-type transfer (0.7025 against
0.6388) by much wider margins than it wins the AUC.

## Read this first: what was and was not measured

This is a **backbone** comparison. The clip-detector branch's actual
contribution is a jpeg-reactivity-delta feature — the embedding shift caused by
re-compressing the image at quality 50 — concatenated onto the CLIP feature, and
a RandomForest domain router feeding five per-domain specialists under
confidence-gated soft mixing. **None of that is exercised here.** The numbers
below say what CLIP features are worth inside BYTEPRINT's pipeline. They say
nothing about whether that branch's method works, and they should not be quoted
as though they do.

Nor is the reverse comparison available. That branch's headline — mean AUROC
0.9974 over a 16-cell grid, TPR@1%FPR 0.9325 — is measured on a different corpus
(its own mix of `fullres`, SID_Set and Defactify-AI-only), a different and
smaller transform grid, 443 held-out clean images against our 1,600, and without
SID_Set's tampered class. Two detectors scored on two different test sets are
not ranked by comparing the numbers.

## What was held fixed

Everything except the backbone — the same constants as the original sweep:

| | |
|---|---|
| Crops | 2 per image, **224px**, `texture` mode, native resolution |
| Head | logistic regression, calibrated at 1% FPR |
| Train | 16,000 SID_Set images (8,000 real / 4,000 full-synthetic / 4,000 tampered), `--augment 3` |
| Test | 1,600 held-out images × 15 §5.2 rungs = 24,000 views |
| Compute | 2× 48 GB GPUs, both arms concurrent: **1h35m GPU time / 47m wall**, 0 failures |

224px is not a handicap here, and that is worth stating because it was for
EVA02. ViT-B/32 is natively a 224 model, so these crops are its design
resolution — CLIP is the one entry in the table being measured at exactly the
input size it was trained for.

## Two arms, because the choice of feature is a real one

CLIP publishes two image features and they differ by one matrix. The
pre-projection 768-d pooled vision output is what the clip-detector branch's
specialists are fitted on; the 512-d post-projection embedding is the shared
image/text space CLIP is more usually read through. Both were registered and
both were run, so the question is settled rather than assumed.

| | `clip_b32_hf` (pre-proj, 768-d) | `clip_b32_proj_hf` (post-proj, 512-d) | Δ |
|---|---|---|---|
| pooled AUC | **0.9319** | 0.9227 | **+0.0092** |
| TPR@1%FPR | **0.4575** | 0.4457 | +0.0118 |
| TPR@0.1%FPR | **0.2014** | 0.1876 | +0.0138 |
| full synthetic | **0.9694** | 0.9653 | +0.0041 |
| tampered | **0.8944** | 0.8801 | +0.0143 |
| clean rung | **0.9633** | 0.9566 | +0.0067 |
| LOGO mean | 0.7025 | **0.7032** | −0.0007 |

**The branch's choice is vindicated, with one exception that is more interesting
than the rule.** Pre-projection wins every in-distribution measure, by most on
the tampered class. But on transfer to an unseen manipulation type the two are a
dead heat — 0.7025 against 0.7032, a gap of 0.0007 that is noise.

A mechanism that fits: the projection was fitted to align images with captions,
so it keeps what a caption can describe and is free to discard what one cannot.
A generator fingerprint is exactly the sort of thing no caption mentions, which
is why dropping the projection helps in-distribution. What survives the
projection is semantic, and semantics is what generalises — so the projected
feature loses nothing on the transfer axis. That is a story consistent with the
numbers, not a tested claim.

## The sweep table, now six entries

Pooled over all fifteen rungs (24,000 views, 12,000 real / 12,000 fake):

| backbone | pretraining | params | AUC | TPR@1%FPR | TPR@0.1%FPR | LOGO mean | ladder span |
|---|---|---|---|---|---|---|---|
| `dinov2_large_hf` | self-distillation | 0.30B | 0.9025 | 0.3362 | 0.1325 | 0.6457 | 0.064 |
| `dinov2_giant_hf` | self-distillation | 1.14B | 0.9261 | 0.4170 | 0.1416 | 0.6388 | **0.052** |
| `eva02_large_timm` | MIM from CLIP | 0.30B | 0.9182 | 0.4036 | 0.2437 | 0.5871 | 0.120 |
| `clip_b32_proj_hf` | language-supervised | **0.09B** | 0.9227 | 0.4457 | 0.1876 | 0.7032 | 0.096 |
| **`clip_b32_hf`** | language-supervised | **0.09B** | **0.9319** | **0.4575** | 0.2014 | **0.7025** | 0.095 |
| **`siglip2_so400m_hf`** | language-supervised | 0.43B | **0.9497** | **0.5854** | **0.2554** | **0.7208** | 0.118 |

## What the sixth entry changes

**The sweep's central finding gets much stronger.** It used to rest on one
model: SigLIP2 at 38% of DINOv2-giant's parameters beating it. Now the *smallest
model in the table* — 0.09B, a twelfth of giant — also beats giant, on every
axis, and the parameter ranking is close to the inverse of the AUC ranking
across a 13× span. The largest model in the table is third.

**And the finding is now about the objective, not one checkpoint.** The top two
entries are both language-supervised contrastive models, from different
families, three years and 5× parameters apart. Third and last are the two
self-distillation entries; fifth is the masked-image-modelling one. Ranked by
pretraining objective the table sorts cleanly; ranked by parameter count it does
not sort at all. That was a one-model observation before this run and it is a
pattern now.

**Transfer is where the gap is widest, and it is the axis we care most about.**
LOGO mean was the project's weakest number. Both language-supervised backbones
clear 0.70; none of the other three reaches 0.65, and EVA02 sits at 0.5871.
CLIP-B/32 buys +0.0637 of transfer over DINOv2-giant while costing a
thirteenth of the parameters.

**Tampered images.** 0.8944, second only to SigLIP2's 0.9176 and ahead of
giant's 0.8852. The class that the crop-localisation work failed to move keeps
responding to the backbone instead.

## The honest weaknesses

**It is not the best backbone, and the gap to SigLIP2 is not small.** −0.0178
pooled AUC, and −0.1279 at TPR@1%FPR — that last is the number a deployment
feels, and it is a 22% relative loss. SigLIP2 stays the default. CLIP-B/32's
argument is efficiency: 21% of SigLIP2's parameters for 78% of its operating
point.

**Downscaling hurts it more than the others.** `scale:0.25` is 0.8780, its
second-worst rung, and its ladder is visibly steeper on the resize and blur
rungs than on the JPEG ones. A 32-pixel patch is coarse, and a quarter-scale
image leaves very little inside one — this is the cost of the patch size that
makes it cheap.

**A claim from the original sweep needs amending.** That doc says every
backbone's worst rung is `noise:0.10` "without exception". For
`clip_b32_hf` it is a *tie*: `noise:0.10` and `blur:2.0` both land on 0.8712, to
four decimals. For the projected arm `noise:0.10` still takes it, by 0.0001
(0.8608 against 0.8609). The claim survives as "worst or tied-worst", which is
weaker than how it was written.

**One seed, one split, as everywhere else in this sweep.** The CLIP-vs-giant gap
on AUC is +0.0058 and is *not* obviously outside seed variance. The gaps that
carry the argument are the operating point (+0.0405) and transfer (+0.0637),
which are far larger.

**A fair test of the clip-detector branch's method is still outstanding.** The
right experiment is its reactivity-delta feature and its routing, on this split
and this ladder — not its backbone. Its cross-*generator* LOGO number (0.9675 on
Defactify) is measuring something our LOGO does not, since SID_Set gives two
manipulation types rather than two generators, so neither number bounds the
other.

## Cost

| arm | train extraction | probe + ladder | total |
|---|---|---|---|
| `clip_b32_hf` | 33m05s | 14m24s | 47m29s |
| `clip_b32_proj_hf` | 33m05s | 14m26s | 47m31s |

Both ran concurrently on the two 48 GB cards, so the wall clock is 47m and the
GPU time 1h35m. That is the cheapest arm in the sweep — SigLIP2 cost 1h03m
alone — which is the practical form of the finding above.

## Reproducing

```bash
export BYTEPRINT_ROOT=<your compute directory>
cd "$BYTEPRINT_ROOT"
for backbone in clip_b32_hf clip_b32_proj_hf; do
  sbatch --nodelist="$BYTEPRINT_GPU_NODE" \
      --export=ALL,BYTEPRINT_BACKBONE=$backbone src/scripts/run_sid_set.sbatch
done
```

`run_sid_set.sbatch` is already parameterised by `BYTEPRINT_BACKBONE` and writes
per-backbone caches, probes and reports, so this needed no new script — only the
two registrations in `byteprint/backbone_hf.py`.

Stage `openai/clip-vit-base-patch32` into `HF_HOME` first. It publishes no
`safetensors`, only `pytorch_model.bin`, so an `allow_patterns` that asks for
safetensors silently fetches the configs and none of the weights, and the
failure surfaces much later as a `local_files_only` load error.
