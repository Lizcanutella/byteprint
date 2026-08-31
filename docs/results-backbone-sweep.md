# The backbone sweep — pretraining objective beats parameter count

Three staged backbones had never been run. This sweep puts all four through the
identical pipeline and changes exactly one thing: the frozen feature extractor.

**The headline: SigLIP2-so400m wins on every axis that matters, at 38% of
DINOv2-giant's parameters.** Pooled AUC 0.9025 → **0.9497**, and TPR@1%FPR
0.3362 → **0.5854** — the operating point, which is what a deployment actually
feels, improves by 74% relative.

## What was held fixed

Everything except the backbone. Same 16,000 train / 1,600 test images, same
crops, same head, same ladder, same seed:

| | |
|---|---|
| Crops | 2 per image, **224px**, `texture` mode, native resolution |
| Head | logistic regression, calibrated at 1% FPR |
| Train | 16,000 SID_Set images (8,000 real / 4,000 full-synthetic / 4,000 tampered), `--augment 3` |
| Test | 1,600 held-out images × 15 §5.2 rungs = 24,000 views |
| Compute | 2× 48 GB GPUs on a SLURM cluster, **9h01m GPU time / 4h37m wall clock**, 0 failures |

224px for every backbone is deliberate and is *not* each model's best
configuration — EVA02 is natively a 448 model, and SigLIP2-naflex is designed
for variable resolution. A 448 crop covers four times the image area, which
makes it a different detector rather than the same detector with a different
backbone. One variable at a time. Native-resolution runs are the obvious
follow-up, and EVA02 in particular is being judged below its weight here.

## The result

Pooled over all fifteen rungs (24,000 views, 12,000 real / 12,000 fake):

| backbone | params | AUC | TPR@1%FPR | TPR@0.1%FPR | LOGO mean |
|---|---|---|---|---|---|
| `dinov2_large_hf` (baseline) | 0.30B | 0.9025 | 0.3362 | 0.1325 | 0.6457 |
| `dinov2_giant_hf` | 1.14B | 0.9261 | 0.4170 | 0.1416 | 0.6388 |
| `eva02_large_timm` | 0.30B | 0.9182 | 0.4036 | 0.2437 | 0.5871 |
| **`siglip2_so400m_hf`** | **0.43B** | **0.9497** | **0.5854** | **0.2554** | **0.7208** |

Per generator, and the shape of each ladder:

| backbone | full synthetic | tampered | best rung | worst rung | span |
|---|---|---|---|---|---|
| `dinov2_large_hf` | 0.9537 | 0.8513 | 0.9191 | 0.8553 | 0.064 |
| `dinov2_giant_hf` | 0.9670 | 0.8852 | 0.9425 | **0.8906** | **0.052** |
| `eva02_large_timm` | 0.9589 | 0.8775 | 0.9533 | 0.8333 | 0.120 |
| `siglip2_so400m_hf` | **0.9817** | **0.9176** | **0.9805** | 0.8624 | 0.118 |

Every backbone's worst rung is `noise:0.10`, without exception. Heavy Gaussian
noise is the ladder's hardest rung and no choice of features changes that.

> **Amended.** A later arm weakened this to *worst or tied-worst*: CLIP ViT-B/32
> lands `noise:0.10` and `blur:2.0` on the same 0.8712, to four decimals. See
> [the CLIP run](results-clip-backbone.md).

## What the sweep actually shows

**Scale is not the lever; the pretraining objective is.** DINOv2-giant costs
3.8× DINOv2-large's parameters and buys +0.024 AUC. SigLIP2-so400m, at 38% of
giant's parameters, buys another +0.024 on top of giant. Ranking the four by
parameter count gives almost the opposite of ranking them by AUC. What separates
them is how they were pretrained: SigLIP2 is language-supervised contrastive,
DINOv2 is self-distillation, EVA02 is masked image modelling distilled from
CLIP. The language-supervised features are the ones that carry the signal here.

**The operating point moved, not just the ranking metric.** AUC improving 0.90 →
0.95 is easy to under-rate. TPR@1%FPR going 0.3362 → 0.5854 means that at a
threshold strict enough to wrongly flag one authentic image in a hundred, the
detector catches 59% of AI-generated images instead of 34%. At 0.1% FPR it
nearly doubles, 0.1325 → 0.2554.

**Transfer to an unseen manipulation type improves too, and by more than the
in-distribution number.** LOGO mean 0.6457 → 0.7208. This is the metric that has
been the project's weakest, and it is the one where the gap between backbones is
widest — EVA02 collapses to 0.5871 while SigLIP2 reaches 0.7208, a spread of
0.13 against an in-distribution spread of 0.03.

**In-distribution accuracy and robustness are not the same ranking.** This is the
one place SigLIP2 does not win outright. DINOv2-giant has the flattest ladder
(span 0.052 against SigLIP2's 0.118) and the highest floor (worst rung 0.8906
against 0.8624). Head to head per rung, **SigLIP2 wins 13 of 15; DINOv2-giant
wins the two heaviest-noise rungs**, `noise:0.05` (0.9201 vs 0.9125) and
`noise:0.10` (0.8906 vs 0.8624). SigLIP2 is better everywhere else, often by
0.03–0.04. Since robustness is the graded axis, this deserves stating plainly
rather than burying: if the grading weights the worst case rather than the
average, giant's flatter curve is a real argument, and it is the only argument
against SigLIP2 in this table.

## The honest weaknesses

**Tampered images are still the hard class**, though much less so: 0.8513 →
0.9176. The texture-ranked crop strategy still has no notion of *where* an edit
is, and that remains the most promising unexplored direction.

**Transfer is still the real gap.** 0.7208 is a large improvement on 0.6457 and
still far below the ~0.95 in-distribution number. And the caveat from the first
run stands unchanged: SID_Set gives two *manipulation types*, not two
generators, so this measures transfer to an unseen kind of editing rather than
to an unseen diffusion model. It should not be quoted as generator
generalisation.

**EVA02 is under-tested, not beaten.** At 224px it is running at a quarter of its
native input area. Its strong clean rung (0.9487, second only to SigLIP2) next to
its poor `blur:2.0` (0.8660) and `scale:0.25` (0.8678) is the signature of a
model whose features live at a higher resolution than we gave it. Concluding
"EVA02 is worse" from this table would be wrong; the supported claim is "EVA02 at
224px is worse".

**One seed, one split.** No error bars. The differences between SigLIP2 and the
rest (+0.024 AUC, +0.17 TPR) are far larger than any plausible seed variance, but
the gap between DINOv2-giant and EVA02 (0.9261 vs 0.9182) is not obviously
outside it.

## Cost

| backbone | train extraction | ladder | total |
|---|---|---|---|
| `dinov2_giant_hf` | 1h31m | 45m | 2h16m |
| `eva02_large_timm` | 44m | 21m | 1h05m |
| `siglip2_so400m_hf` | 42m | 21m | 1h03m |

The two 0.3–0.43B models cost less than half of giant and one beats it
decisively — which is the practical form of the finding above. All runs used
`--workers 4`; the published DINOv2-large baseline predates the prefetch pool,
so its 4h40m is not comparable as a timing.

## Since this ran: two more arms, and a stronger version of the finding

CLIP ViT-B/32 — the backbone under `jiahui/clip-detector` — went through this
same pipeline afterwards, in both of its feature widths. It is the smallest
model in the table and it finishes **second of six**:

| backbone | pretraining | params | AUC | TPR@1%FPR | LOGO mean |
|---|---|---|---|---|---|
| `clip_b32_hf` (pre-projection) | language-supervised | **0.09B** | 0.9319 | 0.4575 | 0.7025 |
| `clip_b32_proj_hf` (post-projection) | language-supervised | 0.09B | 0.9227 | 0.4457 | 0.7032 |

That sharpens the finding above rather than complicating it. The argument used
to rest on one model beating one larger model; now the smallest entry beats
DINOv2-giant at **7.6% of its parameters**, on AUC, on the operating point and
on transfer, and the parameter ranking is close to the inverse of the AUC
ranking across a 13× span. The top two entries are both language-supervised
contrastive models from different families — so this is a property of the
pretraining objective, not of one checkpoint.

Full write-up, including why the un-projected feature wins in-distribution while
the two tie on transfer: **[`results-clip-backbone.md`](results-clip-backbone.md)**.

## Reproducing

```bash
export BYTEPRINT_ROOT=<your compute directory>
cd "$BYTEPRINT_ROOT"
for backbone in dinov2_giant_hf eva02_large_timm siglip2_so400m_hf; do
  sbatch --nodelist="$BYTEPRINT_GPU_NODE" \
      --export=ALL,BYTEPRINT_BACKBONE=$backbone src/scripts/run_sid_set.sbatch
done
```

`run_sid_set.sbatch` is already parameterised by `BYTEPRINT_BACKBONE` and writes
per-backbone caches, probes and reports, so the sweep needs no new script.

Both new backbones read from the staged HuggingFace cache and are registered in
`byteprint/backbone_hf.py`. SigLIP2-naflex does not accept an image tensor — it
takes a sequence of flattened patches plus the grid they came from — so
`NaflexVision` adapts it. Its patch flattening is pixel-major and channel-minor,
matching transformers' own `convert_image_to_patches`, and is pinned by a test
against a transcription of that function: the other ordering yields embeddings of
exactly the right shape and no meaning at all.
