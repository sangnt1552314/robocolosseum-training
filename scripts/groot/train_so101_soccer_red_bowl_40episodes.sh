#!/bin/bash
set -euo pipefail

export HF_HOME=/scratch/e1583535/cache
export HF_DATASETS_CACHE=/scratch/e1583535/cache/datasets

export FFMPEG_ROOT=/scratch/e1583535/opt/ffmpeg-8.0.3
export PATH="$FFMPEG_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$FFMPEG_ROOT/lib:${LD_LIBRARY_PATH:-}"

DATASET="tsangb34/robocolosseum-so101-soccer-red_bowl-40episodes"
OUTPUT_DIR="outputs/groot-n17-so101-soccer-red_bowl-40episodes"

NUM_GPUS=1

# Match the pi05/molmoact2 soccer runs on this dataset for a fair comparison.
BATCH_SIZE=16

STEPS=921

# GR00T N1.7 native fine-tuning LR (cosine w/ 5% warmup).
LR=1e-4

SAVE_FREQ=921

JOB_NAME="groot-n17-so101-soccer-red_bowl-40episodes"
HUB_REPO_ID="tsangb34/groot-n17-so101-soccer-red_bowl-40episodes"

WANDB_PROJECT="RoboColosseum"
WANDB_ENTITY="tsangb34-national-university-of-singapore-students-union"
WANDB_RUN_ID="$JOB_NAME"

LAST_CONFIG="$OUTPUT_DIR/checkpoints/last/pretrained_model/train_config.json"

COMMON_ARGS=(
    --dataset.repo_id="$DATASET"
    --dataset.video_backend=pyav
    --dataset.image_transforms.enable=true
    --policy.type=groot
    --policy.device=cuda
    --policy.base_model_path=nvidia/GR00T-N1.7-3B
    --policy.embodiment_tag=new_embodiment
    --policy.chunk_size=16
    --policy.n_action_steps=16
    --policy.use_relative_actions=true
    --policy.relative_exclude_joints='["gripper"]'
    --policy.use_bf16=true
    --policy.model_params_fp32=true
    # FULL FINE-TUNING
    --policy.tune_llm=true
    --policy.tune_visual=true
    --policy.tune_projector=true
    --policy.tune_diffusion_model=true
    --policy.tune_vlln=true
    --policy.tune_top_llm_layers=0
    --policy.optimizer_lr="$LR"
    --policy.optimizer_weight_decay=1e-5
    --policy.warmup_ratio=0.05
    --policy.push_to_hub=true
    --policy.repo_id="$HUB_REPO_ID"
    --policy.private=false
    --batch_size="$BATCH_SIZE"
    --steps="$STEPS"
    --num_workers=8
    --prefetch_factor=2
    --persistent_workers=true
    --seed=42
    --save_checkpoint=true
    --save_freq="$SAVE_FREQ"
    --save_checkpoint_to_hub=false
    --use_policy_training_preset=true
    --env_eval_freq=0
    --eval_steps=0
    --log_freq=20
    --output_dir="$OUTPUT_DIR"
    --job_name="$JOB_NAME"
    --wandb.enable=true
    --wandb.project="$WANDB_PROJECT"
    --wandb.entity="$WANDB_ENTITY"
    --wandb.run_id="$WANDB_RUN_ID"
)

if [ -f "$LAST_CONFIG" ]; then
    echo "Resuming from:"
    echo "$LAST_CONFIG"

    accelerate launch \
        --num_processes="$NUM_GPUS" \
        "$(which lerobot-train)" \
        --resume=true \
        --config_path="$LAST_CONFIG" \
        "${COMMON_ARGS[@]}"
else
    echo "Starting new GR00T N1.7 full fine-tuning run"

    accelerate launch \
        --num_processes="$NUM_GPUS" \
        "$(which lerobot-train)" \
        "${COMMON_ARGS[@]}"
fi