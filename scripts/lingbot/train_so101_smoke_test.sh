#!/bin/bash
set -euo pipefail

export HF_HOME=/scratch/e1583535/cache
export HF_DATASETS_CACHE=/scratch/e1583535/cache/datasets

export FFMPEG_ROOT=/scratch/e1583535/opt/ffmpeg-8.0.3
export PATH="$FFMPEG_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$FFMPEG_ROOT/lib:${LD_LIBRARY_PATH:-}"

LINGBOT_ROOT=/scratch/e1583535/projects/lingbot-vla-v2
LINGBOT_ENV=/scratch/e1583535/virtualenvs/lingbot-vla-v2

source "$LINGBOT_ENV/bin/activate"

OUTPUT=/scratch/e1583535/projects/robocolosseum-training/outputs/lingbot_so101_smoke_test
CONFIG_PATH=/scratch/e1583535/projects/robocolosseum-training/scripts/lingbot/so101_smoke.yaml
DATASET=/scratch/e1583535/cache/lerobot/liu-tao/so101_dataset

cd "$LINGBOT_ROOT"

CUDA_VISIBLE_DEVICES=0 \
bash train.sh \
    tasks/vla/train_lingbotvla.py \
    $CONFIG_PATH \
    --data.data_name so101 \
    --data.train_path "$DATASET" \
    --data.robot_config_root ./configs/robot_configs \
    --data.norm_stats_file assets/norm_stats/so101.json \
    --train.output_dir "$OUTPUT" \
    --train.micro_batch_size 1 \
    --train.gradient_accumulation_steps 1 \
    --train.max_steps 10 \
    --train.save_steps 10 \
    --train.enable_gradient_checkpointing true \
    --train.freeze_vit false \
    --train.freeze_vision_encoder false \
    --train.train_expert_only false \
    --train.use_wandb false