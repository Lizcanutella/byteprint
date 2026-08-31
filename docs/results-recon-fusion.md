# The reconstruction expert and the fusion — a negative result

The second expert existed in code, with tests, and had never been run on more
than 20 real images. Fusion was the argument for a two-expert architecture, and
it needed a number. It now has one.

**The number is that it does not work on SID_Set.** The AEROBLADE reconstruction
expert scores **AUC 0.5822** alone, and fusing it with the DINOv2-large probe
moves the pooled AUC from 0.9025 to **0.9026**. Against the stronger SigLIP2
probe it is very slightly *negative*: 0.9497 → 0.9493.

This is reported as a result rather than quietly dropped, because the reason it
fails is specific, measurable, and says something about the corpus.

## Setup

| | |
|---|---|
| Expert | AEROBLADE: reconstruct each crop through a bank of VAEs, measure LPIPS distance |
| Autoencoders | `sd15`, `vae-mse`, `sdxl` — the three mutually distinct decoders |
| Perceptual distance | **real LPIPS** (VGG backbone with fitted linear weights), verified, not the unweighted fallback |
| Aggregation | minimum distance across decoders, negated, per the paper |
| Fusion | logistic regression over the two experts' scalar scores |
| Images | the *same* splits as the DINOv2 runs, joined on cache key |
| Compute | one 48 GB GPU, **4h37m** (train 3h05m, ladder 1h32m), 0 failures |

The two caches were joined on key, never zipped by row order. Because the
train-side laundering specs are drawn from a seeded RNG walked over the sample
list, the recon run had to mirror the DINOv2 run's `--augment 3 --seed 0`
exactly. That was verified *before* the job ran, by simulating the draw against
the existing cache: **48,000 of 48,000 keys matched**, so the fusion is fitted on
the full training set rather than on a fragment.

## The result

| | probe only | recon only | fused | Δ |
|---|---|---|---|---|
| **with `dinov2_large_hf`** | | | | |
| pooled AUC | 0.9025 | 0.5822 | **0.9026** | **+0.0001** |
| TPR@1%FPR | 0.3362 | 0.0267 | 0.3417 | +0.0055 |
| **with `siglip2_so400m_hf`** | | | | |
| pooled AUC | 0.9497 | 0.5822 | **0.9493** | **−0.0004** |
| TPR@1%FPR | 0.5854 | 0.0267 | 0.5827 | −0.0027 |

Fusing a 0.58 expert with a 0.90 one gains a ten-thousandth of a point. Fusing it
with a 0.95 one costs slightly more than it gains. The two-expert architecture,
on this corpus, is a one-expert architecture with overhead.

## Why — the expert is uninformative, not broken

That distinction matters, so it was checked rather than assumed.

**The distances are healthy and point the right way.** Finite, no NaNs, and
well-spread, with synthetic images reconstructing *better* than real ones in
every decoder — exactly AEROBLADE's predicted direction:

| decoder | real mean | fake mean | real sd | fake sd | AUC alone |
|---|---|---|---|---|---|
| `sd15` | 0.1520 | 0.1334 | 0.0695 | 0.0694 | 0.5824 |
| `vae-mse` | 0.1336 | 0.1147 | 0.0713 | 0.0705 | 0.5842 |
| `sdxl` | 0.1327 | 0.1147 | 0.0729 | 0.0729 | 0.5814 |

The effect is real. It is just small — a gap of ~0.019 in means against a spread
of ~0.070 — so the distributions overlap almost completely.

**The three-decoder bank buys nothing here.** This is the more interesting
finding. AEROBLADE's robustness comes from taking the minimum across several
autoencoders, on the theory that an image need only sit near *one* generator's
manifold. On SID_Set the three decoders are nearly perfectly redundant:

| aggregation | AUC |
|---|---|
| minimum (the paper's) | 0.5822 |
| mean | 0.5827 |
| maximum | 0.5822 |

When min, mean and max all agree to three decimal places, the columns are
carrying the same information. The bank cost 3× the compute for no coverage.
Because the minimum is applied at *scoring* time rather than extraction, this was
ablated with no recompute.

**Half the AIGC class is invisible to it by construction.** This is the
mechanism:

| class | recon-only AUC | n |
|---|---|---|
| full synthetic | 0.6668 | 6,000 |
| tampered | **0.4975** | 6,000 |

0.4975 is chance. A tampered SID_Set image is a real photograph with a locally
generated region, so most of its pixels genuinely *are* a real photograph and
reconstruct like one. Worse, `texture` crop selection picks the highest-frequency
224px windows in the frame, which have no particular reason to be the edited
region. The expert is asked whether a crop came from a VAE, and is usually shown
a crop that did not.

The whole of the expert's 0.58 comes from the fully-synthetic half, where it
reaches a modest 0.667.

**Per rung**, it degrades under compression, best on `crop:0.8` (0.6733) and
clean (0.6621), worst on `jpeg:50` (0.5711) and `noise:0.10` (0.5777) — a VAE
fingerprint is a high-frequency property and JPEG is designed to remove exactly
that.

## What this does and does not establish

**Established.** On SID_Set, with 224px texture-ranked crops and this bank of
three decoders, reconstruction error adds nothing to a frozen-feature probe. The
score-level fusion is sound — it recovers the better expert rather than being
dragged down, which is what a well-behaved fusion should do with a weak input.

**Not established: that AEROBLADE does not work.** Three of this setup's choices
are hostile to it, and none of them is the method's fault:

1. **Crops, not images.** AEROBLADE reconstructs whole generated images at
   generation resolution. We hand it 224px windows cut from full-size
   photographs, which for the real class have been through a camera pipeline and
   resizing that a VAE never saw.
2. **The bank may not contain the generator.** The premise is proximity to a
   decoder we own. SID_Set's synthetic images are not documented as SD1.5/SDXL
   outputs, and if their generator's VAE is absent from the bank the signal is
   weak by construction — which the near-identical per-decoder AUCs are
   consistent with.
3. **Tampering is out of scope for it.** A local edit in a real photograph is not
   what a whole-image reconstruction detector was designed to catch.

A fair test would reconstruct whole images, on a corpus of fully-synthetic images
from known latent-diffusion generators. That is a different experiment, not a
tweak to this one.

## The cost, stated plainly

4h37m on one 48 GB GPU for a result of +0.0001 AUC. Two things made it more
expensive than it needed to be, both worth recording:

- **`--batch-size` is inert for this expert.** `embed()` is called once per view
  with `crops_per_image=2`, so `ReconExpert`'s internal loop never sees more than
  2 crops and each VAE runs on a batch of 2 regardless of the flag. The GPU sat
  mostly idle at 4.0 views/s. Batching across views is a change to the extraction
  pipeline rather than to the expert, and would likely be worth 3–5×.
- **The three-decoder bank was 3× the cost of one** for information one decoder
  already had, as the aggregation table above shows.

## Reproducing

```bash
export BYTEPRINT_ROOT=<your compute directory>
cd "$BYTEPRINT_ROOT"
sbatch --nodelist="$BYTEPRINT_GPU_NODE" src/scripts/run_recon_fusion.sbatch
```

The job extracts reconstruction distances over both splits, mirroring the DINOv2
run's augmentation and seed so the caches join, then fits the fusion and prints
the probe / recon / fused ablation per rung. `BYTEPRINT_BACKBONE` selects which
DINOv2-side probe to fuse against; it defaults to the published baseline.

Weights must be staged from a machine with internet first — the three
autoencoders into `HF_HOME`, **and** torchvision's VGG16 checkpoint into
`TORCH_HOME`, which `lpips` fetches through `torch.hub` rather than HuggingFace.
Missing the second is silent: `PerceptualDistance` catches the failure and
substitutes an unweighted feature distance, which is a different and weaker
expert than the one reported here. This run was verified to be using real LPIPS
before it started.
