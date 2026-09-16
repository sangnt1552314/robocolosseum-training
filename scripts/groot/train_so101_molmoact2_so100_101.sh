#!/bin/bash
set -euo pipefail

export HF_HOME=/scratch/e1583535/cache
export HF_DATASETS_CACHE=/scratch/e1583535/cache/datasets

DATASET="allenai/MolmoAct2-SO100_101-Dataset"
OUTPUT_DIR="outputs/groot_n17_molmoact2_so100_101"

# Multi-GPU: effective batch size = BATCH_SIZE x NUM_GPUS (DDP replicates the full model per GPU).
# 2x H200 141GB: ~50GB/GPU for fp32 weights+grads+Adam, leaving ~90GB for activations.
NUM_GPUS=2
BATCH_SIZE=64        # per-GPU -> global batch 128 on 2 H200

# Full fine-tune schedule. Cosine w/ 5% warmup (GR00T N1.7 native recipe).
# Dataset: 19,227,195 frames -> 1 epoch = 19,227,195 / (BATCH_SIZE * NUM_GPUS) = ~150,212 steps.
# STEPS=300000 ~= 2 epochs (matches pi05's ~2-epoch / 38.4M-sample reference).
STEPS=300000
SAVE_FREQ=25000

JOB_NAME="groot_n17_molmoact2_so100_101"
HUB_REPO_ID="tsangb34/groot_n17_molmoact2_so100_101"
WANDB_PROJECT="RoboColosseum"
WANDB_ENTITY="tsangb34-national-university-of-singapore-students-union"

accelerate launch \
    --multi_gpu \
    --num_processes="$NUM_GPUS" \
    "$(which lerobot-train)" \
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
    --policy.model_params_fp32=true \
    --policy.tune_llm=true \
    --policy.tune_visual=true \
    --policy.tune_projector=true \
    --policy.tune_diffusion_model=true \
    --policy.tune_vlln=true \
    --policy.optimizer_lr=1e-4 \
    --policy.optimizer_weight_decay=1e-5 \
    --policy.warmup_ratio=0.05 \
    --policy.gradient_checkpointing=true \
    --policy.push_to_hub=true \
    --policy.repo_id="$HUB_REPO_ID" \
    --policy.private=false \
    --batch_size="$BATCH_SIZE" \
    --steps="$STEPS" \
    --num_workers=4 \
    --save_checkpoint=true \
    --save_freq="$SAVE_FREQ" \
    --save_checkpoint_to_hub=false \
    --use_policy_training_preset=true \
    --env_eval_freq=0 \
    --eval_steps=0 \
    --log_freq=20 \
    --resume=true \
    --output_dir="$OUTPUT_DIR" \
    --job_name="$JOB_NAME" \
    --wandb.enable=true \
    --wandb.project="$WANDB_PROJECT" \
    --wandb.entity="$WANDB_ENTITY" \
    --wandb.run_id="$JOB_NAME"