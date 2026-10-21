#!/bin/bash
set -euo pipefail

# LingBot norm stats for droid-put-marker-pen on NSCC ASPIRE 2B (same as
# compute_norm_stats_droid_put_marker_pen.sh, with the NSCC paths). Needs a GPU
# (compute_norm_stats.py initialises NCCL), so it runs inside the training job on one GPU.

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../nscc" && pwd)/env.sh"
source "$LINGBOT_VENV/bin/activate"

DATASET="$DATASETS_ROOT/droid-put-marker-pen"
NORM_STATS="$ROBO_ROOT/scripts/lingbot/norm_stats/droid_put_marker_pen_nscc.json"

export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1

mkdir -p "$(dirname "$NORM_STATS")"

cd "$LINGBOT_ROOT"

CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nnodes=1 --nproc-per-node=1 \
  scripts/compute_norm_stats.py \
  ./configs/vla/norm_compute/post_data.yaml \
  --data.data_name droid_put_marker_pen \
  --data.train_path "$DATASET" \
  --data.robot_config_root "$ROBO_ROOT/scripts/lingbot/configs/robot_configs_nscc" \
  --data.norm_path "$NORM_STATS"

test -f "$NORM_STATS"
echo "Wrote $NORM_STATS"
