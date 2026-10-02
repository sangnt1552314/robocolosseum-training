#!/bin/bash
set -euo pipefail

LINGBOT_ROOT=/scratch/e1583535/projects/lingbot-vla-v2
ROBO_ROOT=/scratch/e1583535/projects/robocolosseum-training
DATASET=/scratch/e1583535/datasets/so101-stack-bowls
NORM_STATS=$ROBO_ROOT/scripts/lingbot/norm_stats/so101_stack_bowls.json

source /scratch/e1583535/virtualenvs/lingbot-vla-v2/bin/activate

mkdir -p "$ROBO_ROOT/scripts/lingbot/norm_stats"

cd "$LINGBOT_ROOT"

bash train.sh \
  scripts/compute_norm_stats.py \
  ./configs/vla/norm_compute/post_data.yaml \
  --data.data_name so101_stack_bowls \
  --data.train_path "$DATASET" \
  --data.robot_config_root "$ROBO_ROOT/scripts/lingbot/configs/robot_configs" \
  --data.norm_path "$NORM_STATS"