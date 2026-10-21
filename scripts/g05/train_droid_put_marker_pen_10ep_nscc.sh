#!/bin/bash
set -euo pipefail

# G0.5 (GalaxeaVLA) fine-tuning on the DROID (Franka) put-marker-pen dataset on NSCC ASPIRE 2B,
# 8 x H200. Same native recipe as train_droid_put_marker_pen_10ep.sh (Hopper, 1 GPU): GalaxeaVLA's
# configs/task/so100.yaml settings (g05-base init, global batch 16, default LR 8e-5, warmup_ratio
# 0.05, warmup_constant_cosine schedule, action_dim 20, relative-joint arm actions with an absolute
# gripper, q01/q99 norm, same tokenizer settings), with our Franka task/data configs added through
# hydra.searchpath (scripts/g05/configs; nothing is copied into or edited in GalaxeaVLA).
#
# Multi-GPU change: finetune.sh launches torchrun on 8 GPUs (DDP) and model.batch_size is per GPU,
# so model.batch_size=2 x 8 GPUs keeps the global batch at 16. G0.5 derives the step count from
# max_epochs (len(train_dataloader) per rank), so 10 epochs, the warmup (5% of the run) and the
# cosine schedule all stay the same as on 1 GPU.
# Overridden here via Hydra:
#   - dataset path            -> our local LeRobot v3 dataset
#   - model.batch_size=2      -> per GPU (global 16)
#   - max_epochs=10
#   - checkpointing_steps=1000000 -> no periodic checkpoints; finetune.py always saves the final step.
# After training the final checkpoint is exported, uploaded to the Hub repo root (no
# checkpoints/step_<N>/ copies) and cleaned up locally by upload_droid_put_marker_pen_10ep_nscc.sh.

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../nscc" && pwd)/env.sh"
source "$G05_VENV/bin/activate"

NUM_GPUS=8
PER_GPU_BATCH=2  # x 8 GPUs = global 16 (the native recipe's batch)

DATASET="$DATASETS_ROOT/droid-put-marker-pen"

JOB_NAME="g05-droid-put-marker-pen-10ep-nscc"
OUTPUT_DIR="$MODELS_ROOT/$JOB_NAME"

# Required by configs/train.yaml; the run dir itself is pinned to $OUTPUT_DIR below.
export G05_OUTPUT_DIR="$MODELS_ROOT"
# Used by train.yaml for the run name.
export EXP_NAME="$JOB_NAME"
export OMP_NUM_THREADS=4

# train.yaml reads the wandb project from WANDB_PROJECT and the entity from WANDB_ENTITY (its
# fallback is "Galaxea-AI"), so without an entity, null it to use the logged-in account's default.
WANDB_ARGS=()
if [ -z "${WANDB_ENTITY:-}" ]; then
    WANDB_ARGS+=(logger.workspace=null)
fi

test -d "$DATASET"
test -f "$G05_ROOT/checkpoints/g05-base/checkpoints/model_state_dict.pt"
test -f "$G05_ROOT/checkpoints/action_tokenizer.pt"

cd "$G05_ROOT"

echo "G0.5 fine-tuning (native so100 recipe, DROID Franka data config)"
echo "Dataset:    $DATASET"
echo "Output dir: $OUTPUT_DIR"
echo "GPUs:       $NUM_GPUS"
echo "Batch:      $PER_GPU_BATCH per GPU (global $((PER_GPU_BATCH * NUM_GPUS)))"
echo "Epochs:     10 (final checkpoint only)"

nvidia-smi -L

bash scripts/run/finetune.sh \
  "$NUM_GPUS" \
  robocolosseum_droid_franka \
  "hydra.searchpath=[file://$ROBO_ROOT/scripts/g05/configs]" \
  "data.embodiment_datasets.Droid_Franka.dataset_groups=[{weight: 1.0, dataset_dirs: ['$DATASET']}]" \
  model.batch_size="$PER_GPU_BATCH" \
  model.max_epochs=10 \
  checkpointing_steps=1000000 \
  "${WANDB_ARGS[@]}" \
  "hydra.run.dir=$OUTPUT_DIR"

# Export + upload the final checkpoint to the Hub root, then clean up the local checkpoint files.
# A failure here must not mark a successful training run as failed; cleanup only happens after
# every upload is verified on the Hub, so rerunning the upload script is always safe.
bash "$ROBO_ROOT/scripts/g05/upload_droid_put_marker_pen_10ep_nscc.sh" \
  || echo "Upload failed; rerun scripts/g05/upload_droid_put_marker_pen_10ep_nscc.sh manually."
