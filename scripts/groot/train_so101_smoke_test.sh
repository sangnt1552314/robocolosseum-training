#!/bin/bash
set -euo pipefail

export HF_HOME=/scratch/e1583535/cache
export HF_DATASETS_CACHE=/scratch/e1583535/cache/datasets

export FFMPEG_ROOT=/scratch/e1583535/opt/ffmpeg-8.0.3
export PATH="$FFMPEG_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$FFMPEG_ROOT/lib:${LD_LIBRARY_PATH:-}"

DATASET="liu-tao/so101_dataset"
OUTPUT_DIR="outputs/groot_so101_smoke_test"

lerobot-train \
    --dataset.repo_id="$DATASET" \
    --dataset.video_backend=pyav \
    --dataset.image_transforms.enable=false \
    --policy.type=groot \
    --policy.device=cuda \
    --policy.base_model_path=nvidia/GR00T-N1.7-3B \
    --policy.embodiment_tag=new_embodiment \
    --policy.chunk_size=16 \
    --policy.n_action_steps=16 \
    --policy.use_relative_actions=true \
    --policy.relative_exclude_joints='["gripper"]' \
    --policy.use_bf16=true \
    --policy.tune_llm=true \
    --policy.tune_visual=true \
    --policy.tune_projector=true \
    --policy.tune_diffusion_model=true \
    --policy.tune_vlln=true \
    --policy.push_to_hub=false \
    --batch_size=1 \
    --steps=10 \
    --num_workers=0 \
    --save_checkpoint=true \
    --save_freq=10 \
    --use_policy_training_preset=true \
    --env_eval_freq=0 \
    --eval_steps=0 \
    --log_freq=1 \
    --output_dir="$OUTPUT_DIR" \
    --job_name=groot_so101_smoke_test \
    --wandb.enable=false