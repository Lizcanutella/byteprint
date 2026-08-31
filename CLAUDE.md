# techjam — AIGC image detection

Entry for the competition described in `docs/competition-brief.md` (transcribed
from `competition_info.md`). Read the brief before design work; the constraints
below are the ones that silently invalidate a submission if forgotten.

## Hard constraints

- **Models must be <2B parameters.** This rules out the NTIRE-2026-winning
  DINOv3-7B recipe entirely. Budget: DINOv2-giant ≈1.1B, SigLIP2-so400m ≈400M,
  EVA02-L ≈300M. *Open question: whether <2B is per-model or summed across an
  ensemble — assume summed until clarified, it is the safe reading.*
- **The task is binary.** Image-level AI-generated vs authentic, one confidence
  score per image. Not 3-class, not localization — regardless of what SID_Set
  supports.
- **Never train on the demo validation set**: COCO val2017 (4,998 non-AIGC) +
  DALL·E Advanced (8,843 AIGC), a subset of WildFake. It is explicitly excluded
  and scores nothing.
- **Robustness is the graded axis**, not clean accuracy. The official transform
  list (§5.2) is fixed and specific — match it exactly, do not improvise a
  different ladder.
- Datasets must be public or properly licensed. SID_Set is CC-BY-4.0. Check any
  gated weights' licence (DINOv3's is custom) before relying on them.

## Required deliverable

A script taking an **image directory** and writing a **JSON file** with
`image_path` and `pred` per image, `pred` = likelihood the image is AIGC.
This is a hard interface requirement, not a suggestion.

Built: `byteprint score IMAGE_DIR --probe P --out predictions.json`, also as
`scripts/score_directory.py`. The extraction config travels inside the saved
probe, so scoring needs nothing but the probe and the directory. Every
discovered image gets exactly one entry; unreadable files score 0.5 with an
`error` field rather than aborting the run.

## Where the numbers stand

`siglip2_so400m_hf` is the default backbone and the best result: pooled **AUC
0.9497**, **TPR@1%FPR 0.5854**, unseen-type transfer 0.7208, over the full §5.2
ladder on SID_Set. It beat DINOv2-giant at 38% of its parameters — on this task
the pretraining objective matters more than scale, which is the finding worth
repeating. Full table in `docs/results-backbone-sweep.md`.

The sweep is now six backbones, and the finding held up when it was tested
again: CLIP ViT-B/32 (0.09B) places **second**, beating DINOv2-giant (1.14B) on
AUC, operating point and transfer. The top two are both language-supervised
contrastive models from different families, so this is a property of the
pretraining objective rather than of one checkpoint. That run measured
`jiahui/clip-detector`'s *backbone* only — not its reactivity-delta feature or
its domain routing, which are that branch's actual contribution, and which
remain untested on our split and ladder. Do not quote it as a verdict on that
detector. See `docs/results-clip-backbone.md`.

The second expert is currently **not** carrying weight: AEROBLADE reconstruction
error scores 0.5822 alone and fusion moves the pooled AUC by +0.0001. It is
chance-level (0.4975) on tampered images, because a local edit in a real
photograph is not what a whole-image reconstruction detector was built to catch.
Do not describe BYTEPRINT as a working two-expert detector without saying this;
`docs/results-recon-fusion.md` has the mechanism and the three reasons the test
was hostile to the method.

## Judging weights

Technical execution 35% · Innovation & insight 20% · Impact & relevance 20% ·
Feasibility & practicality 15% · Presentation 10%.

Note the tension: Feasibility rewards **proportionate resource usage** and a
"hackathon-scale prototype". A multi-day multi-GPU campaign can read as
disproportionate. Prefer a defensible, reproducible result over a maximal one,
and state the compute budget explicitly.

## Compute

Cluster specifics — host, node inventory, paths, logins — live in
`CLUSTER.local.md`, which is **gitignored and must stay that way**. It is not in
a fresh clone; ask the team for it. `tests/test_public_repo_hygiene.py` fails the
build if any identifying detail leaks into a tracked file.

What may be said in public documentation: that we used a SLURM cluster, the
number and class of GPUs, and wall-clock cost. Stating the compute budget is
rewarded by the Feasibility criterion; naming the machine is not.

- Compute nodes have **no internet** — stage datasets and weights first.
- Datasets already staged: SID_Set (131 GB), CIFAKE (49 MB), and cached
  backbones (eva02-large, siglip2-so400m, dinov2-giant, dinov2-large).

## Repo conventions

- TDD throughout; `pytest` must stay green (`.venv/bin/python -m pytest`).
- Two experts behind one interface — `embed(crops) -> (n_crops, dim)` — so
  extraction, caching, resume and the laundering ladder work on any of them.
- Embeddings are cached and joined **on key**, never zipped by row order.
- Report TPR at a fixed low FPR alongside AUC; accuracy at threshold 0.5 is
  meaningless when score distributions shift between generators.
- The package is `byteprint` (project **BYTEPRINT**, team ByteSized). Swappable parts
  — backbone, head/loss, crop strategy — are **registries**: add one with
  `@register_backbone` / `@register_head` / `@register_crop_mode` in your own
  module and load it with `--plugin`, never by editing a shared file. See
  `docs/extending.md`; `byteprint list` shows what is registered.
- `OFFICIAL_LADDER` in `byteprint/launder.py` is §5.2 transcribed and pinned by a
  test. Do not add rungs to it — exploratory chains go in `STRESS_LADDER`. Noise
  σ is normalised (0–1), not 0–255.
