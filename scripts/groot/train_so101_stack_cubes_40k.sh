#!/bin/bash
set -euo pipefail

export HF_HOME=/scratch/e1583535/cache
export HF_DATASETS_CACHE=/scratch/e1583535/cache/datasets

# Keep wandb staging/cache off the 40 GB home quota.
export WANDB_DATA_DIR=/scratch/e1583535/cache/wandb/data
export WANDB_CACHE_DIR=/scratch/e1583535/cache/wandb/cache
export WANDB_ARTIFACT_DIR=/scratch/e1583535/cache/wandb/artifacts

export FFMPEG_ROOT=/scratch/e1583535/opt/ffmpeg-8.0.3
export PATH="$FFMPEG_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$FFMPEG_ROOT/lib:${LD_LIBRARY_PATH:-}"

DATASET="Jiamo0912/so101-stack-cubes"
DATASET_ROOT="/scratch/e1583535/datasets/so101-stack-cubes"

JOB_NAME="groot-n17-so101-stack-cubes-40k"
MODELS_ROOT="/scratch/Projects/CFP-05/CFP05-CF-002/robocolosseum-finetuned-models"
OUTPUT_DIR="$MODELS_ROOT/$JOB_NAME"

HUB_REPO_ID="tsangb34/$JOB_NAME"

WANDB_PROJECT="RoboColosseum"
WANDB_ENTITY="tsangb34-national-university-of-singapore-students-union"

lerobot-train \
    --dataset.repo_id="$DATASET" \
    --dataset.root="$DATASET_ROOT" \
    --dataset.image_transforms.enable=true \
    --dataset.video_backend=pyav \
    --policy.type=groot \
    --policy.device=cuda \
    --policy.base_model_path=nvidia/GR00T-N1.7-3B \
    --policy.embodiment_tag=new_embodiment \
    --policy.chunk_size=32 \
    --policy.n_action_steps=32 \
    --policy.use_relative_actions=true \
    --policy.relative_exclude_joints='["gripper"]' \
    --policy.use_bf16=true \
    --policy.max_steps=40000 \
    --policy.push_to_hub=true \
    --policy.repo_id="$HUB_REPO_ID" \
    --policy.private=false \
    --batch_size=32 \
    --steps=40000 \
    --persistent_workers=true \
    --save_checkpoint=true \
    --save_freq=10000 \
    --save_checkpoint_to_hub=true \
    --use_policy_training_preset=true \
    --env_eval_freq=0 \
    --eval_steps=0 \
    --log_freq=20 \
    --output_dir="$OUTPUT_DIR" \
    --job_name="$JOB_NAME" \
    --wandb.enable=true \
    --wandb.project="$WANDB_PROJECT" \
    --wandb.entity="$WANDB_ENTITY" \
    --wandb.run_id="$JOB_NAME" \
    --wandb.disable_artifact=true
