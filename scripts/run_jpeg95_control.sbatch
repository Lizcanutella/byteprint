#!/bin/bash
# The JPEG-95 control for the SID_Set run.
#
# The baseline scored AUC 0.9025 on a corpus whose reals are 100% JPEG-family
# and whose fully-synthetic images are 100% PNG. Materialisation equalises the
# container by writing everything as PNG, but JPEG quantisation artifacts
# survive into the pixels of the reals and are absent from the synthetics --
# so the baseline is an upper bound until both classes carry the same kind of
# compression damage. This job runs exactly that comparison.
#
# Only one thing changes from run_sid_set.sbatch: both splits are re-encoded
# through JPEG-95 before extraction. Same images, same seeds, same backbone,
# same crops, same augmentation, same ladder -- otherwise the difference in the
# result would not be attributable to the compression.
#
#     export BYTEPRINT_ROOT=<your compute directory>
#     cd "$BYTEPRINT_ROOT"
#     sbatch --nodelist="$BYTEPRINT_GPU_NODE" src/scripts/run_jpeg95_control.sbatch
#
# The recompressed splits are *derived from the baseline splits*, not
# re-materialised from parquet: that guarantees the two runs see the identical
# image set, and costs minutes instead of the materialiser's hour and a half.
#SBATCH --job-name=byteprint_jpeg95
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --output=slurm_logs/%x_%j.out

set -euo pipefail

BYTEPRINT_ROOT="${BYTEPRINT_ROOT:?set this to your compute directory}"
cd "$BYTEPRINT_ROOT"

PY="$BYTEPRINT_ROOT/.venv/bin/python"
SRC="${BYTEPRINT_SRC:-$BYTEPRINT_ROOT/src}"
DATA="$BYTEPRINT_ROOT/data"
RUNS="$BYTEPRINT_ROOT/runs"

# Compute nodes have no internet. Weights come from the staged cache.
export HF_HOME="$DATA/hf_home"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$SRC"
export TOKENIZERS_PARALLELISM=false

# Identical to the baseline run. Changing any of these makes the control
# uninterpretable, so they are not parameterised down to taste.
BACKBONE="${BYTEPRINT_BACKBONE:-dinov2_large_hf}"
CROPS="${BYTEPRINT_CROPS:-2}"
CROP_SIZE="${BYTEPRINT_CROP_SIZE:-224}"
AUGMENT="${BYTEPRINT_AUGMENT:-3}"
BATCH="${BYTEPRINT_BATCH:-32}"
WORKERS="${BYTEPRINT_WORKERS:-4}"
ENCODING="${BYTEPRINT_ENCODING:-jpeg:95}"
# Encoding is PIL and releases the GIL, so this can use more threads than the
# extraction pool, which is bounded by the backbone handover.
ENCODE_WORKERS="${BYTEPRINT_ENCODE_WORKERS:-12}"

BASE_TRAIN="$DATA/sid_train"
BASE_TEST="$DATA/sid_test"
TRAIN_DIR="$DATA/sid_train_jpeg95"
TEST_DIR="$DATA/sid_test_jpeg95"
TRAIN_CACHE="$RUNS/cache/sid_train_jpeg95_$BACKBONE"
LADDER_CACHE="$RUNS/cache/sid_ladder_jpeg95_$BACKBONE"
PROBE="$RUNS/probe_sid_jpeg95_$BACKBONE.joblib"
REPORT="$RUNS/report_sid_jpeg95_$BACKBONE.txt"

mkdir -p "$RUNS/cache" slurm_logs

echo "=== BYTEPRINT JPEG-95 control ==="
echo "encoding=$ENCODING backbone=$BACKBONE crops=$CROPS size=$CROP_SIZE augment=$AUGMENT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
date

for split in "$BASE_TRAIN:$TRAIN_DIR" "$BASE_TEST:$TEST_DIR"; do
  src="${split%%:*}"; dst="${split##*:}"
  [ -d "$src" ] || { echo "missing baseline split $src -- run run_sid_set.sbatch first" >&2; exit 1; }
  # Guarded on a marker rather than on the directory existing: a pass killed
  # mid-encode leaves a partial tree, and extracting from that would silently
  # train on a fraction of the corpus. Re-running is safe -- already-written
  # files are skipped.
  if [ ! -f "$dst/.recompressed" ]; then
    echo "--- re-encoding $(basename "$src") as $ENCODING"
    "$PY" "$SRC/scripts/recompress_split.py" "$src" "$dst" \
        --encoding "$ENCODING" --workers "$ENCODE_WORKERS"
    touch "$dst/.recompressed"
  fi
done

echo "--- extracting train ($AUGMENT laundered views per image)"
date
"$PY" -m byteprint.cli -v extract \
    --data "$TRAIN_DIR" --cache "$TRAIN_CACHE" \
    --backbone "$BACKBONE" --crops "$CROPS" --crop-size "$CROP_SIZE" \
    --augment "$AUGMENT" --batch-size "$BATCH" --device cuda --workers "$WORKERS"

echo "--- training probe"
"$PY" -m byteprint.cli train \
    --cache "$TRAIN_CACHE" --out "$PROBE" --target-fpr 0.01

echo "--- extracting the official ladder over the test split"
date
"$PY" -m byteprint.cli -v extract \
    --data "$TEST_DIR" --cache "$LADDER_CACHE" \
    --backbone "$BACKBONE" --crops "$CROPS" --crop-size "$CROP_SIZE" \
    --ladder official --batch-size "$BATCH" --device cuda --workers "$WORKERS"

{
  echo "### BYTEPRINT — JPEG-95 CONTROL — encoding=$ENCODING"
  echo "### backbone=$BACKBONE, crops=$CROPS@$CROP_SIZE, augment=$AUGMENT"
  echo "### Compare against report_sid_$BACKBONE.txt, which differs only in"
  echo "### that its splits were PNG (reals JPEG-derived, synthetics not)."
  echo
  echo "## Per-rung robustness (the §5.2 ladder)"
  "$PY" -m byteprint.cli eval --cache "$LADDER_CACHE" --probe "$PROBE" --by-spec
  echo
  echo "## Leave-one-generator-out"
  "$PY" -m byteprint.cli logo --cache "$TRAIN_CACHE"
} | tee "$REPORT"

echo "=== done ==="
date
echo "report: $REPORT"
