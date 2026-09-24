#!/bin/bash
set -euo pipefail

export HF_HOME=/scratch/e1583535/cache
export HF_DATASETS_CACHE=/scratch/e1583535/cache/datasets

DATASET="tsangb34/robocolosseum-so101-soccer-red_bowl-40episodes"
OUTPUT_DIR="outputs/pi05-so101-soccer-red_bowl-40episodes"

HUB_REPO_ID="tsangb34/pi05-so101-soccer-red_bowl-40episodes"

JOB_NAME="pi05-so101-soccer-red_bowl-40episodes"
WANDB_PROJECT="RoboColosseum"
WANDB_ENTITY="tsangb34-national-university-of-singapore-students-union"

lerobot-train \
    --dataset.repo_id="$DATASET" \
    --dataset.video_backend=pyav \
    --policy.type=pi05 \
    --policy.pretrained_path=lerobot/pi05_base \
    --policy.device=cuda \
    --policy.dtype=bfloat16 \
    --policy.normalization_mapping='{"ACTION":"MEAN_STD","STATE":"MEAN_STD","VISUAL":"IDENTITY"}' \
    --policy.chunk_size=50 \
    --policy.n_action_steps=50 \
    --policy.empty_cameras=1 \
    --policy.freeze_vision_encoder=false \
    --policy.train_expert_only=false \
    --policy.gradient_checkpointing=true \
    --policy.optimizer_lr=2.5e-5 \
    --policy.scheduler_warmup_steps=31 \
    --policy.scheduler_decay_steps=921 \
    --policy.scheduler_decay_lr=2.5e-6 \
    --policy.push_to_hub=true \
    --policy.repo_id="$HUB_REPO_ID" \
    --policy.private=false \
    --output_dir="$OUTPUT_DIR" \
    --job_name="$JOB_NAME" \
    --batch_size=16 \
    --num_workers=4 \
    --steps=921 \
    --log_freq=20 \
    --save_checkpoint=true \
    --save_freq=921 \
    --save_checkpoint_to_hub=false \
    --wandb.enable=true \
    --wandb.project="$WANDB_PROJECT" \
    --wandb.entity="$WANDB_ENTITY" \
    --wandb.run_id="$JOB_NAME"