#!/bin/bash
set -euo pipefail

# GR00T N1.7 recipe from the VLA-REPLICA paper (https://irvlutd.github.io/VLAReplica/):
# batch 32, 40K steps, chunk 32 / 32 action steps, default vision encoder (frozen:
# tune_visual=false) and default LR (1e-4 cosine, 5% warmup).
# Checkpoints are saved at 20K and 40K and each is pushed to the Hub under checkpoints/<step>/.

export HF_HOME=/scratch/e1583535/cache
export HF_DATASETS_CACHE=/scratch/e1583535/cache/datasets

# Keep wandb staging/cache off the 40 GB home quota.
export WANDB_DATA_DIR=/scratch/e1583535/cache/wandb/data
export WANDB_CACHE_DIR=/scratch/e1583535/cache/wandb/cache
export WANDB_ARTIFACT_DIR=/scratch/e1583535/cache/wandb/artifacts

export FFMPEG_ROOT=/scratch/e1583535/opt/ffmpeg-8.0.3
export PATH="$FFMPEG_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$FFMPEG_ROOT/lib:${LD_LIBRARY_PATH:-}"

DATASET="Jiamo0912/so101-cube-drawer"
DATASET_ROOT="/scratch/e1583535/datasets/so101-cube-drawer"

JOB_NAME="groot-n17-so101-cube-drawer-vlareplica-40k"
MODELS_ROOT="/scratch/Projects/CFP-05/CFP05-CF-002/robocolosseum-finetuned-models"
OUTPUT_DIR="$MODELS_ROOT/$JOB_NAME"

HUB_REPO_ID="tsangb34/$JOB_NAME"

BATCH_SIZE=32
STEPS=40000

WANDB_PROJECT="RoboColosseum"
WANDB_ENTITY="tsangb34-national-university-of-singapore-students-union"

# save_freq=20000 writes checkpoints at 20K and 40K (the final step); save_checkpoint_to_hub
# pushes each one to <repo>/checkpoints/<step>/, and push_to_hub uploads the final model.
lerobot-train \
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
    --save_freq=20000 \
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
