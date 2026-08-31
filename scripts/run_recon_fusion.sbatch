#!/bin/bash
# The reconstruction expert on real data, and the fusion that is the whole
# argument for a two-expert architecture.
#
# Submit it the same way as the other jobs -- every path is an environment
# variable, so this file names no host, login or absolute directory:
#
#     export BYTEPRINT_ROOT=<your compute directory>
#     cd "$BYTEPRINT_ROOT"
#     sbatch --nodelist="$BYTEPRINT_GPU_NODE" src/scripts/run_recon_fusion.sbatch
#
# WHAT MAKES THIS JOB DIFFERENT from run_sid_set.sbatch: it does not train a
# probe and it does not materialise anything. It extracts a *second* view of
# splits that already exist, then joins that view to the DINOv2 caches those
# same splits already produced.
#
# The join is on cache key, and the key is (path, size, mtime, spec). The specs
# on the train side are drawn from a seeded RNG walked over the sample list, so
# this job MUST mirror the DINOv2 run's `--augment` and `--seed` exactly or the
# two caches share no keys at all and `fuse` exits with "no shared images".
# That is a loud failure rather than a wrong number, which is the point -- but
# it is also why none of the three numbers below is a free parameter.
#SBATCH --job-name=byteprint_recon
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/%x_%j.out

set -euo pipefail

BYTEPRINT_ROOT="${BYTEPRINT_ROOT:?set this to your compute directory}"
cd "$BYTEPRINT_ROOT"

PY="$BYTEPRINT_ROOT/.venv/bin/python"
SRC="${BYTEPRINT_SRC:-$BYTEPRINT_ROOT/src}"
DATA="$BYTEPRINT_ROOT/data"
RUNS="$BYTEPRINT_ROOT/runs"

# Compute nodes have no internet. Two separate caches matter here: the
# autoencoders come out of HF_HOME, and the VGG16 backbone that LPIPS is built
# on comes out of TORCH_HOME via torch.hub. Miss the second and
# PerceptualDistance catches the failure and silently substitutes an
# unweighted feature distance -- a weaker expert, reported nowhere.
export HF_HOME="$DATA/hf_home"
export TORCH_HOME="$DATA/torch_home"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$SRC"
export TOKENIZERS_PARALLELISM=false

# The probe side this fuses against. Defaults to the project default backbone;
# set it to dinov2_large_hf to reproduce the ablation against the original
# published baseline instead.
BACKBONE="${BYTEPRINT_BACKBONE:-siglip2_so400m_hf}"
AES="${BYTEPRINT_AES:-sd15,vae-mse,sdxl}"

# Pinned to the baseline run, not chosen. See the header.
CROPS=2
CROP_SIZE=224
AUGMENT=3
SEED=0

BATCH="${BYTEPRINT_BATCH:-32}"
WORKERS="${BYTEPRINT_WORKERS:-4}"

TRAIN_DIR="$DATA/sid_train"
TEST_DIR="$DATA/sid_test"

DINO_TRAIN_CACHE="$RUNS/cache/sid_train_$BACKBONE"
DINO_LADDER_CACHE="$RUNS/cache/sid_ladder_$BACKBONE"
PROBE="$RUNS/probe_sid_$BACKBONE.joblib"

RECON_TRAIN_CACHE="$RUNS/cache/sid_train_recon"
RECON_LADDER_CACHE="$RUNS/cache/sid_ladder_recon"
FUSED="$RUNS/fused_sid_$BACKBONE.joblib"
REPORT="$RUNS/report_fused_$BACKBONE.txt"

mkdir -p "$RUNS/cache" slurm_logs

echo "=== BYTEPRINT recon expert + fusion ==="
echo "autoencoders=$AES  fusing against=$BACKBONE"
echo "mirroring the DINOv2 run: crops=$CROPS@$CROP_SIZE augment=$AUGMENT seed=$SEED"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
date

# The DINOv2 side has to exist already -- this job extracts the second expert,
# not the first. Checking now costs a second and saves discovering it after the
# extraction stages have run.
for required in "$DINO_TRAIN_CACHE" "$DINO_LADDER_CACHE" "$PROBE"; do
  if [ ! -e "$required" ]; then
    echo "missing $required -- run run_sid_set.sbatch for $BACKBONE first" >&2
    exit 1
  fi
done

# -- 1. recon over the train split ---------------------------------------
# Same directory, same augmentation draw, same seed as the DINOv2 run, so the
# keys line up and the join is dense rather than empty.
echo "--- extracting recon distances over train ($AUGMENT laundered views per image)"
date
"$PY" -m byteprint.cli -v extract \
    --data "$TRAIN_DIR" --cache "$RECON_TRAIN_CACHE" \
    --expert recon --aes "$AES" \
    --crops "$CROPS" --crop-size "$CROP_SIZE" \
    --augment "$AUGMENT" --seed "$SEED" \
    --batch-size "$BATCH" --device cuda --workers "$WORKERS"

# -- 2. recon over the official ladder -----------------------------------
echo "--- extracting recon distances over the official ladder"
date
"$PY" -m byteprint.cli -v extract \
    --data "$TEST_DIR" --cache "$RECON_LADDER_CACHE" \
    --expert recon --aes "$AES" \
    --crops "$CROPS" --crop-size "$CROP_SIZE" \
    --ladder official --seed "$SEED" \
    --batch-size "$BATCH" --device cuda --workers "$WORKERS"

# -- 3. fit the fusion, and report the ablation --------------------------
# `fuse` fits on the train join and prints the probe/recon/fused ablation on
# its held-out calibration split; `eval --fused --by-spec` then reports the
# same three, per rung, over the ladder. Recon alone is the number that says
# whether the second expert carries any signal at all on this corpus.
{
  echo "### BYTEPRINT — recon expert + fusion, SID_Set"
  echo "### autoencoders=$AES  dino=$BACKBONE  crops=$CROPS@$CROP_SIZE augment=$AUGMENT"
  echo
  echo "## Fitting the fusion (ablation on the held-out calibration split)"
  "$PY" -m byteprint.cli fuse \
      --dino-cache "$DINO_TRAIN_CACHE" --recon-cache "$RECON_TRAIN_CACHE" \
      --probe "$PROBE" --out "$FUSED" --target-fpr 0.01 --seed "$SEED"
  echo
  echo "## Per-rung ablation over the official §5.2 ladder"
  echo "## probe alone / recon alone / fused -- the argument for two experts"
  "$PY" -m byteprint.cli eval \
      --cache "$DINO_LADDER_CACHE" --recon-cache "$RECON_LADDER_CACHE" \
      --fused "$FUSED" --by-spec
} | tee "$REPORT"

echo "=== done ==="
date
echo "report: $REPORT"
