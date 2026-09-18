#!/bin/bash
set -euo pipefail

ROBO_ROOT=/scratch/e1583535/projects/robocolosseum-training
G05_ROOT=/scratch/e1583535/projects/GalaxeaVLA
G05_ENV=/scratch/e1583535/virtualenvs/g05

DATASET=/scratch/e1583535/datasets/robocolosseum-so101-soccer-red_bowl-40episodes

export HF_HOME=/scratch/e1583535/cache
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"

export G05_OUTPUT_DIR="$ROBO_ROOT/outputs/g05"

export WANDB_PROJECT="RoboColosseum"
export WANDB_ENTITY="tsangb34-national-university-of-singapore-students-union"

export TMPDIR=/scratch/e1583535/tmp
export TEMP="$TMPDIR"
export TMP="$TMPDIR"

mkdir -p "$TMPDIR"
mkdir -p "$G05_OUTPUT_DIR"


source "$G05_ENV/bin/activate"


# Dataset sanity check
test -f "$DATASET/meta/info.json" || {
    echo "Dataset not found: $DATASET"
    exit 1
}


# Put RoboColosseum configs where G0.5 Hydra expects them
mkdir -p "$G05_ROOT/configs/task"
mkdir -p "$G05_ROOT/configs/data/parts_meta"

ln -sfn \
    "$ROBO_ROOT/scripts/g05/configs/task/so101_soccer_red_bowl.yaml" \
    "$G05_ROOT/configs/task/so101_soccer_red_bowl.yaml"

ln -sfn \
    "$ROBO_ROOT/scripts/g05/configs/data/so101_soccer_red_bowl.yaml" \
    "$G05_ROOT/configs/data/so101_soccer_red_bowl.yaml"

ln -sfn \
    "$ROBO_ROOT/scripts/g05/configs/data/parts_meta/so101.yaml" \
    "$G05_ROOT/configs/data/parts_meta/so101.yaml"


cd "$G05_ROOT"

echo "=========================================="
echo "G0.5 SO101 Soccer -> Red Bowl"
echo "=========================================="
echo "Dataset:  $DATASET"
echo "Base:     g05-base"
echo "GPUs:     2"
echo "Batch:    8/GPU -> global 16"
echo "Steps:    921"
echo "=========================================="

nvidia-smi -L


bash scripts/run/finetune.sh \
    2 \
    so101_soccer_red_bowl