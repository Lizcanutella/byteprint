# techjam — AIGC image detection

Entry for the competition described in `docs/competition-brief.md` (transcribed
from `competition_info.md`). Read the brief before design work; the constraints
below are the ones that silently invalidate a submission if forgotten.

## This branch is the results/integration story, not engine development

`main` holds the CLIP+DINOv2 fusion writeup, notebooks, and results (see
`README.md`). The engines it fuses live on their own branches:

- **CLIP + reactivity-delta** (domain-specialist detector, trained model, its
  own robustness/LOGO evaluation) — `jiahui/clip-detector`
- **DINOv2, AEROBLADE, the SigLIP2/EVA02 backbone sweep, the byteprint engine**
  (extraction, caching, `byteprint/launder.py`, `byteprint logo`) —
  `mateo/main-work`

If the task is engine work — a new backbone, a new head, extending the
laundering ladder, fixing something in `byteprint/` — switch to
`mateo/main-work` first; this branch doesn't have that code checked out
locally, only notebooks that pull it from packaged Kaggle Datasets at run
time. If the task is the CLIP detector specifically, switch to
`jiahui/clip-detector`.

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
This is a hard interface requirement, not a suggestion. The actual scorer
(`byteprint score` / `scripts/score_directory.py`) lives on `mateo/main-work`.

## Judging weights

Technical execution 35% · Innovation & insight 20% · Impact & relevance 20% ·
Feasibility & practicality 15% · Presentation 10%.

Note the tension: Feasibility rewards **proportionate resource usage** and a
"hackathon-scale prototype". A multi-day multi-GPU campaign can read as
disproportionate. Prefer a defensible, reproducible result over a maximal one,
and state the compute budget explicitly.
