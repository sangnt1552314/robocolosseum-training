#!/bin/bash
set -euo pipefail

LINGBOT_ROOT=/scratch/e1583535/projects/lingbot-vla-v2
ROBO_ROOT=/scratch/e1583535/projects/robocolosseum-training
DATASET=/scratch/e1583535/datasets/droid-put-marker-pen
NORM_STATS=$ROBO_ROOT/scripts/lingbot/norm_stats/droid_put_marker_pen.json

source /scratch/e1583535/virtualenvs/lingbot-vla-v2/bin/activate

mkdir -p "$ROBO_ROOT/scripts/lingbot/norm_stats"

cd "$LINGBOT_ROOT"

bash train.sh \
  scripts/compute_norm_stats.py \
  ./configs/vla/norm_compute/post_data.yaml \
  --data.data_name droid_put_marker_pen \
  --data.train_path "$DATASET" \
  --data.robot_config_root "$ROBO_ROOT/scripts/lingbot/configs/robot_configs" \
  --data.norm_path "$NORM_STATS"