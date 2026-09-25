#!/bin/bash
set -euo pipefail

# GR00T N1.7 recipe from the VLA-REPLICA paper (https://irvlutd.github.io/VLAReplica/):
# batch 32, 40K steps, chunk 32 / 32 action steps, default vision encoder (frozen:
# tune_visual=false) and default LR (1e-4 cosine, 5% warmup).

export HF_HOME=/scratch/e1583535/cache
export HF_DATASETS_CACHE=/scratch/e1583535/cache/datasets

export FFMPEG_ROOT=/scratch/e1583535/opt/ffmpeg-8.0.3
export PATH="$FFMPEG_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$FFMPEG_ROOT/lib:${LD_LIBRARY_PATH:-}"

DATASET="Jiamo0912/robocolosseum-so101-stack-white_bowls-100episodes"
DATASET_ROOT="/scratch/e1583535/datasets/robocolosseum-so101-stack-white_bowls-100episodes"
OUTPUT_DIR="outputs/groot-n17-so101-stack-white_bowls-100episodes-vlareplica-40k"

HUB_REPO_ID="tsangb34/groot-n17-so101-stack-white_bowls-100episodes-vlareplica-40k"

# Number of finished checkpoints kept on local disk (all of them stay on the Hub).
export KEEP_LOCAL_CHECKPOINTS=2

BATCH_SIZE=32
STEPS=40000

JOB_NAME="groot-n17-so101-stack-white_bowls-100episodes-vlareplica-40k"
WANDB_PROJECT="RoboColosseum"
WANDB_ENTITY="tsangb34-national-university-of-singapore-students-union"

source "$(dirname "$0")/../utils/resumable_lerobot_train.sh"

# Checkpoints every 5K steps are pushed to the Hub and pruned locally; re-submitting
# the job resumes from the latest local/Hub checkpoint after a crash or walltime kill.
run_resumable_train "$OUTPUT_DIR" "$HUB_REPO_ID" \
    --dataset.repo_id="$DATASET" \
    --dataset.root="$DATASET_ROOT" \
    --dataset.video_backend=pyav \
    --dataset.image_transforms.enable=true \
    --policy.type=groot \
    --policy.device=cuda \
    --policy.base_model_path=nvidia/GR00T-N1.7-3B \
    --policy.embodiment_tag=new_embodiment \
    --policy.chunk_size=32 \
    --policy.n_action_steps=32 \
    --policy.use_relative_actions=true \
    --policy.relative_exclude_joints='["gripper"]' \
    --policy.use_bf16=true \
    --policy.model_params_fp32=true \
    --policy.tune_llm=true \
    --policy.tune_projector=true \
    --policy.tune_diffusion_model=true \
    --policy.tune_vlln=true \
    --policy.tune_top_llm_layers=0 \
    --policy.max_steps="$STEPS" \
    --policy.push_to_hub=true \
    --policy.repo_id="$HUB_REPO_ID" \
    --policy.private=false \
    --batch_size="$BATCH_SIZE" \
    --steps="$STEPS" \
    --num_workers=8 \
    --prefetch_factor=2 \
    --persistent_workers=true \
    --seed=42 \
    --save_checkpoint=true \
    --save_freq=5000 \
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
