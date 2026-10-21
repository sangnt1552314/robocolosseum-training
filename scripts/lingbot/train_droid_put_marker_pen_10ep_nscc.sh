#!/bin/bash
set -euo pipefail

# LingBot-VLA v2 6B full fine-tuning on the DROID (Franka) put-marker-pen dataset on NSCC ASPIRE 2B,
# 8 x H200. Same recipe as train_droid_put_marker_pen_10ep.sh (Hopper, 1 GPU), following LingBot's
# real-world recipe (configs/vla/real_robot/real_robot.yaml + Training_Config.md): per-joint meanstd
# norm, state-relative arm actions / absolute gripper, Muon, LR 5e-5 constant, VLM + vision + action
# expert all trained, torch.compile off. Cameras: left, right, wrist; 7 arm joints + gripper.
#
# Multi-GPU change (configs/droid_put_marker_pen_10ep_nscc.yaml): global batch stays 32, now split
# as micro batch 4 x 8 GPUs (DDP) with no gradient accumulation instead of 8 x 1 GPU x 4 accum, so
#   10 epochs = 16,150 steps (1615 steps/epoch = floor(51699 / 32)), unchanged.
# Dataloader workers per rank drop from 24 to 8 (8 ranks share the node's 96 cores).
# No intermediate checkpoints: only the final step is saved, and only that step is uploaded to the
# Hub afterwards by upload_droid_put_marker_pen_10ep_nscc.sh.
#
# Norm stats are computed automatically on the first run if the JSON does not exist yet.

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../nscc" && pwd)/env.sh"
source "$LINGBOT_VENV/bin/activate"

NUM_GPUS=8

CONFIG="$ROBO_ROOT/scripts/lingbot/configs/droid_put_marker_pen_10ep_nscc.yaml"
DATASET="$DATASETS_ROOT/droid-put-marker-pen"
NORM_STATS="$ROBO_ROOT/scripts/lingbot/norm_stats/droid_put_marker_pen_nscc.json"

JOB_NAME="lingbot-vla-v2-6b-droid-put-marker-pen-10ep-nscc"
STEPS=16150  # = max_steps / save_steps in $CONFIG (10 epochs)
OUTPUT_DIR="$MODELS_ROOT/$JOB_NAME"
export WANDB_NAME="$JOB_NAME"
export WANDB_RUN_ID="$JOB_NAME"
export WANDB_RESUME=allow

# Keep HF models local/offline during training (train.sh does the same).
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4

# Sanity checks
test -f "$CONFIG"
test -d "$DATASET"
test -d "$MODELS_DIR/lingbot-vla-v2-6b"
test -d "$MODELS_DIR/Qwen3-VL-4B-Instruct"
test -f "$MODELS_DIR/moge-2-vitb-normal/model.pt"

# Norm stats need a GPU, so compute them here on the first run instead of on the login node.
# Later runs/resubmissions reuse the file.
if [ ! -f "$NORM_STATS" ]; then
    echo "Norm stats not found; computing $NORM_STATS"
    bash "$ROBO_ROOT/scripts/lingbot/compute_norm_stats_droid_put_marker_pen_nscc.sh"
fi
test -f "$NORM_STATS"

mkdir -p "$OUTPUT_DIR"

cd "$LINGBOT_ROOT"

echo "========================================"
echo "LingBot-VLA v2 full fine-tuning"
echo "Dataset: DROID put marker pen (100 episodes, 51,699 frames)"
echo "Steps: $STEPS = 10 epochs (final checkpoint only)"
echo "GPUs: $NUM_GPUS"
echo "Micro batch: 4 per GPU"
echo "Gradient accumulation: 1"
echo "Global batch: 32"
echo "========================================"

# Equivalent to LingBot's train.sh (torchrun over all local GPUs), without its `tee log.txt` into
# the framework checkout; the PBS job captures stdout/stderr instead.
torchrun --standalone --nnodes=1 --nproc-per-node="$NUM_GPUS" \
  tasks/vla/train_lingbotvla.py \
  "$CONFIG" \
  --train.use_wandb true \
  --train.wandb_project "$WANDB_PROJECT" \
  --train.wandb_name "$JOB_NAME" \
  --train.output_dir "$OUTPUT_DIR"

# Push the final checkpoint to the Hub (the trainer has no push_to_hub).
bash "$ROBO_ROOT/scripts/lingbot/upload_droid_put_marker_pen_10ep_nscc.sh" "$STEPS"
