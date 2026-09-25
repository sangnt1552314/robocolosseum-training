#!/bin/bash
set -euo pipefail

ROBO_ROOT=/scratch/e1583535/projects/robocolosseum-training
G05_ROOT=/scratch/e1583535/projects/GalaxeaVLA

export HF_HOME=/scratch/e1583535/cache
export G05_OUTPUT_DIR="$ROBO_ROOT/outputs/g05"

export TMPDIR=/scratch/e1583535/tmp
export TMP="$TMPDIR"
export TEMP="$TMPDIR"

export WANDB_PROJECT="RoboColosseum"
export WANDB_ENTITY="tsangb34-national-university-of-singapore-students-union"

mkdir -p "$TMPDIR"
mkdir -p "$G05_OUTPUT_DIR"

# The g05 virtualenv is activated by the PBS job before this script runs.

# Copy RoboColosseum configs into GalaxeaVLA/Hydra.
# finetune.sh resolves task configs with realpath and rejects symlinks that
# resolve outside GalaxeaVLA/configs/task, so copy real files instead.
mkdir -p "$G05_ROOT/configs/task" "$G05_ROOT/configs/data/parts_meta"

# Drop any stale symlinks left by earlier runs so cp writes real files.
rm -f \
  "$G05_ROOT/configs/task/so101_stack_white_bowls.yaml" \
  "$G05_ROOT/configs/data/so101_stack_white_bowls.yaml" \
  "$G05_ROOT/configs/data/parts_meta/so101.yaml"

cp -f \
  "$ROBO_ROOT/scripts/g05/configs/task/so101_stack_white_bowls.yaml" \
  "$G05_ROOT/configs/task/so101_stack_white_bowls.yaml"

cp -f \
  "$ROBO_ROOT/scripts/g05/configs/data/so101_stack_white_bowls.yaml" \
  "$G05_ROOT/configs/data/so101_stack_white_bowls.yaml"

cp -f \
  "$ROBO_ROOT/scripts/g05/configs/data/parts_meta/so101.yaml" \
  "$G05_ROOT/configs/data/parts_meta/so101.yaml"

cd "$G05_ROOT"

echo "G0.5 full fine-tuning"
echo "GPU:             1"
echo "Micro batch:     1"
echo "Grad accumulation: 32"
echo "Effective batch: 32"
echo "Steps:           4690"

nvidia-smi -L

bash scripts/run/finetune.sh \
  1 \
  so101_stack_white_bowls

# Export the final checkpoint and upload it to the HuggingFace Hub.
# G0.5 has no built-in push_to_hub, so this runs the standalone upload script.
# A failure here must not mark a successful training run as failed.
# HF_TOKEN comes from the PBS env or the repo .env file (loaded by the upload script).
if [ -n "${HF_TOKEN:-}" ] || [ -f "$ROBO_ROOT/.env" ]; then
  echo "Uploading fine-tuned checkpoint to HuggingFace Hub..."
  bash "$ROBO_ROOT/scripts/g05/upload_so101_stack_white_bowls_100episodes.sh" \
    || echo "Upload failed; run scripts/g05/upload_so101_stack_white_bowls_100episodes.sh manually."
else
  echo "HF_TOKEN not set and no .env found; skipping automatic upload."
  echo "Run scripts/g05/upload_so101_stack_white_bowls_100episodes.sh manually to upload."
fi
