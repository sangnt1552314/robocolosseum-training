#!/bin/bash
set -euo pipefail

# pi05 fine-tuning on the DROID (Franka) pour-water dataset, with the same recipe as our SO-101
# runs (see train_so101_cube_drawer_40k.sh): base lerobot/pi05_base, full fine-tuning (vision
# encoder trainable), quantile normalization, default chunk 50 / 50 action steps, default pi05 LR
# schedule (2.5e-5 peak, cosine to 2.5e-6; lerobot shrinks the 1K warmup / 30K decay to the run
# length when --steps < 30K).
# Cameras: left, wrist (+ one masked empty slot, as in pi05_base's 3-camera layout). Trains on
# droid-pour-water-left-wrist (built by tools/make_droid_left_wrist_dataset.py): the right
# camera is removed and info.json lists left_image before wrist, which sets the camera order.
# Batch 32 on 1 GPU (gradient checkpointing on to fit batch 32 in memory).
# 10 epochs = 28,720 steps (2,872 steps/epoch = ceil(91891 / 32)). No intermediate checkpoints: only
# the final step is saved locally, and only the final model is pushed to the Hub repo root
# (save_checkpoint_to_hub=false, so no checkpoints/<step>/ on the Hub).

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export HF_HOME=/scratch/e1583535/cache
export HF_DATASETS_CACHE=/scratch/e1583535/cache/datasets

# Keep wandb staging/cache off the 40 GB home quota.
export WANDB_DATA_DIR=/scratch/e1583535/cache/wandb/data
export WANDB_CACHE_DIR=/scratch/e1583535/cache/wandb/cache
export WANDB_ARTIFACT_DIR=/scratch/e1583535/cache/wandb/artifacts

export FFMPEG_ROOT=/scratch/e1583535/opt/ffmpeg-8.0.3
export PATH="$FFMPEG_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$FFMPEG_ROOT/lib:${LD_LIBRARY_PATH:-}"

# Local-only dataset; repo_id is just a label when --dataset.root points at it.
DATASET="local/droid-pour-water-left-wrist"
DATASET_ROOT="/scratch/e1583535/datasets/droid-pour-water-left-wrist"
SOURCE_DATASET_ROOT="/scratch/e1583535/datasets/droid-pour-water"

JOB_NAME="pi05-droid-pour-water-10ep"
MODELS_ROOT="/scratch/Projects/CFP-05/CFP05-CF-002/robocolosseum-finetuned-models"
OUTPUT_DIR="$MODELS_ROOT/$JOB_NAME"

HUB_REPO_ID="tsangb34/$JOB_NAME"

BATCH_SIZE=32
STEPS=28720
SAVE_FREQ="$STEPS"  # only the final checkpoint

WANDB_PROJECT="RoboColosseum"
WANDB_ENTITY="tsangb34-national-university-of-singapore-students-union"

# Build the left + wrist dataset if it does not exist yet (a few minutes; videos are copied,
# not re-encoded).
if [ ! -f "$DATASET_ROOT/meta/info.json" ]; then
    echo "Left + wrist dataset not found; building $DATASET_ROOT"
    python "$REPO_ROOT/tools/make_droid_left_wrist_dataset.py" "$SOURCE_DATASET_ROOT" "$DATASET_ROOT"
fi
test -f "$DATASET_ROOT/meta/info.json"

lerobot-train \
    --dataset.repo_id="$DATASET" \
    --dataset.root="$DATASET_ROOT" \
    --dataset.video_backend=pyav \
    --policy.type=pi05 \
    --policy.pretrained_path=lerobot/pi05_base \
    --policy.device=cuda \
    --policy.dtype=bfloat16 \
    --policy.normalization_mapping='{"ACTION":"QUANTILES","STATE":"QUANTILES","VISUAL":"IDENTITY"}' \
    --policy.empty_cameras=1 \
    --policy.freeze_vision_encoder=false \
    --policy.train_expert_only=false \
    --policy.gradient_checkpointing=true \
    --policy.compile_model=false \
    --policy.push_to_hub=true \
    --policy.repo_id="$HUB_REPO_ID" \
    --policy.private=false \
    --output_dir="$OUTPUT_DIR" \
    --job_name="$JOB_NAME" \
    --batch_size="$BATCH_SIZE" \
    --num_workers=8 \
    --steps="$STEPS" \
    --log_freq=20 \
    --save_checkpoint=true \
    --save_freq="$SAVE_FREQ" \
    --save_checkpoint_to_hub=false \
    --wandb.enable=true \
    --wandb.project="$WANDB_PROJECT" \
    --wandb.entity="$WANDB_ENTITY" \
    --wandb.run_id="$JOB_NAME" \
    --wandb.disable_artifact=true
