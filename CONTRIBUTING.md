# Working on BYTEPRINT

Six people, six branches, one engine. This file is the short version of how to
not stand on each other's feet.

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e '.[dev]'
uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cpu torch torchvision
.venv/bin/python -m pytest          # must be green before you push
```

`byteprint list` prints every backbone, head, crop mode, autoencoder and ladder
the engine currently knows about. Start there.

## Branching

```bash
git switch -c <yourname>/<what>     # e.g. ana/siglip2-backbone
```

One branch per experiment. `main` stays green: if `pytest` fails on `main`,
everyone else's branch is now also broken and nobody can tell whose change did
it.

## Add your idea as a plugin, not a patch

The parts most likely to be swapped are **registries**, not `if` ladders. Adding
a backbone, a training objective or a crop strategy means writing a function in
**your own module** and decorating it — no edit to a shared file, so no merge
conflict with the five other branches. See [`docs/extending.md`](docs/extending.md).

The one shared file you should not casually edit is
`byteprint/launder.py`'s `OFFICIAL_LADDER`: it is the competition brief's fixed
transform list, it is pinned by a test, and changing it silently invalidates
every robustness number anyone has quoted. Exploratory chains go in
`STRESS_LADDER`.

## Tests

TDD, as the rest of the repo does it: write the failing test, then the code.
Tests are named as sentences describing the behaviour, not `test_foo_works`.
`pytest` runs in ten seconds and uses stub backbones — if your test needs to
download weights, it is an experiment, not a test.

## What goes in git, and what does not

| In | Why |
|---|---|
| `byteprint/`, `tests/` | the engine |
| `docs/`, `README.md`, `CONTRIBUTING.md` | how to use and extend it |
| `pyproject.toml` | the environment, pinned |
| `scripts/`, job scripts | reproducing a run is part of the deliverable — but see below |
| result tables, small JSON predictions | evidence; a reviewer should not have to rerun to see them |

| Out | Why |
|---|---|
| `data/` | SID_Set is 131 GB. Stage it on the cluster, reference it by path |
| `runs/`, `**/features.npy`, `**/records.jsonl` | embedding caches; a pure function of (images, config), so recompute |
| `*.joblib`, `*.pt`, `*.safetensors` | model weights — attach to a release, do not commit |
| `.claude/`, `*.local.md` | environment-identifying notes; this repo is public |
| `.venv/`, `__pycache__/` | machine-local |

**The repo is a public deliverable.** Before you commit, assume a stranger will
read it: no hostnames, no logins, no absolute paths under someone's home
directory, no API keys, no dataset copies whose licence you have not checked.
`pytest` enforces this — see `tests/test_public_repo_hygiene.py`.

## Compute

Long runs go to a batch cluster. The setup notes — host, nodes, paths — are in
`CLUSTER.local.md`, which is deliberately **not** in this repo; ask the team.

Two things bite everyone at least once, and neither is a secret:

- Compute nodes have **no internet**. Stage datasets and pretrained weights
  before you submit a job.
- Never put the repo, the venv or a cache in `$HOME`. Use the compute
  directory the local notes name.

If you commit a job script, parameterise every path through an environment
variable so the committed version names no login, host or absolute path.
