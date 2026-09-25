#!/bin/bash
set -euo pipefail

# pi05 recipe from the VLA-REPLICA paper (https://irvlutd.github.io/VLAReplica/):
# quantile normalization, batch 16, 1 GPU, 40K steps, chunk 32 / 32 action steps,
# default vision encoder (trainable) and default LR schedule.

export HF_HOME=/scratch/e1583535/cache
export HF_DATASETS_CACHE=/scratch/e1583535/cache/datasets

DATASET="Jiamo0912/robocolosseum-so101-stack-white_bowls-100episodes"
DATASET_ROOT="/scratch/e1583535/datasets/robocolosseum-so101-stack-white_bowls-100episodes"
OUTPUT_DIR="outputs/pi05-so101-stack-white_bowls-100episodes-vlareplica-40k"

HUB_REPO_ID="tsangb34/pi05-so101-stack-white_bowls-100episodes-vlareplica-40k"

# Number of finished checkpoints kept on local disk (all of them stay on the Hub).
export KEEP_LOCAL_CHECKPOINTS=2

JOB_NAME="pi05-so101-stack-white_bowls-100episodes-vlareplica-40k"
WANDB_PROJECT="RoboColosseum"
WANDB_ENTITY="tsangb34-national-university-of-singapore-students-union"

source "$(dirname "$0")/../utils/resumable_lerobot_train.sh"

# Checkpoints every 5K steps are pushed to the Hub and pruned locally; re-submitting
# the job resumes from the latest local/Hub checkpoint after a crash or walltime kill.
run_resumable_train "$OUTPUT_DIR" "$HUB_REPO_ID" \
    --dataset.repo_id="$DATASET" \
    --dataset.root="$DATASET_ROOT" \
    --dataset.video_backend=pyav \
    --policy.type=pi05 \
    --policy.pretrained_path=lerobot/pi05_base \
    --policy.device=cuda \
    --policy.dtype=bfloat16 \
    --policy.normalization_mapping='{"ACTION":"QUANTILES","STATE":"QUANTILES","VISUAL":"IDENTITY"}' \
    --policy.chunk_size=32 \
    --policy.n_action_steps=32 \
    --policy.empty_cameras=1 \
    --policy.freeze_vision_encoder=false \
    --policy.train_expert_only=false \
    --policy.gradient_checkpointing=false \
    --policy.compile_model=false \
    --policy.push_to_hub=true \
    --policy.repo_id="$HUB_REPO_ID" \
    --policy.private=false \
    --output_dir="$OUTPUT_DIR" \
    --job_name="$JOB_NAME" \
    --batch_size=16 \
    --num_workers=4 \
    --steps=40000 \
    --log_freq=20 \
    --save_checkpoint=true \
    --save_freq=5000 \
    --save_checkpoint_to_hub=true \
    --wandb.enable=true \
    --wandb.project="$WANDB_PROJECT" \
    --wandb.entity="$WANDB_ENTITY" \
    --wandb.run_id="$JOB_NAME"
