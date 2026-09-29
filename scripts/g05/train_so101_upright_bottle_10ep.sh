#!/bin/bash
set -euo pipefail

# G0.5 (GalaxeaVLA) fine-tuning on the SO-101 upright-bottle dataset using the native SO-100/101
# recipe as-is: configs/task/so100.yaml + configs/data/so100.yaml (the same so100 processor the
# released g05-so101 checkpoint uses), initialised from g05-base (configs/model/g05.yaml default).
# Native settings kept: batch 16, LR 8e-5, warmup_ratio 0.05 (overrides warmup_steps in
# finetune.py, so ~700 steps here), warmup_constant_cosine schedule,
# action_dim 20, relative-joint arm actions, q01/q99 norm, camera canonicalisation/random drop.
# Only these are overridden via Hydra (nothing is copied into or edited in GalaxeaVLA):
#   - dataset path            -> our local LeRobot v3 dataset
#   - datastatics_path=null   -> norm stats computed fresh from this dataset (native so100 points
#                                at the pretraining mixture's data/stats/so100_stats.json)
#   - max_epochs=10           -> GalaxeaVLA's own real-robot fine-tuning length (r1lite/r1pro
#                                task configs); native so100 uses 4 epochs, sized for the large
#                                SO-100 pretraining mixture. ~21.3K steps at batch 16 on this dataset.
#   - checkpointing_steps=5500 -> 4 checkpoints: 5500, 11000, 16500 + the final step (~21.3K;
#                                native saves every 2000 steps, ~8 checkpoints).
# After training every checkpoint is exported, uploaded to the Hub and cleaned up locally by
# upload_so101_upright_bottle_10ep.sh.

ROBO_ROOT=/scratch/e1583535/projects/robocolosseum-training
G05_ROOT=/scratch/e1583535/projects/GalaxeaVLA

DATASET=/scratch/e1583535/datasets/so101-upright-bottle

JOB_NAME="g05-so101-upright-bottle-10ep"
MODELS_ROOT="/scratch/Projects/CFP-05/CFP05-CF-002/robocolosseum-finetuned-models"
OUTPUT_DIR="$MODELS_ROOT/$JOB_NAME"

export HF_HOME=/scratch/e1583535/cache
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"

# Required by configs/train.yaml; the run dir itself is pinned to $OUTPUT_DIR below.
export G05_OUTPUT_DIR="$MODELS_ROOT"
# Used by train.yaml for the run name (wandb experiment name = so100_$EXP_NAME).
export EXP_NAME="$JOB_NAME"

export TMPDIR=/scratch/e1583535/tmp
export TMP="$TMPDIR"
export TEMP="$TMPDIR"

# train.yaml reads the wandb project/entity from these.
export WANDB_PROJECT="RoboColosseum"
export WANDB_ENTITY="tsangb34-national-university-of-singapore-students-union"

# Keep wandb staging/cache off the 40 GB home quota.
export WANDB_DATA_DIR=/scratch/e1583535/cache/wandb/data
export WANDB_CACHE_DIR=/scratch/e1583535/cache/wandb/cache
export WANDB_ARTIFACT_DIR=/scratch/e1583535/cache/wandb/artifacts

mkdir -p "$TMPDIR" "$MODELS_ROOT"

# The g05 virtualenv is activated by the PBS job before this script runs.

test -d "$DATASET"
test -f "$G05_ROOT/checkpoints/g05-base/checkpoints/model_state_dict.pt"

cd "$G05_ROOT"

echo "G0.5 fine-tuning (native so100 recipe)"
echo "Dataset:    $DATASET"
echo "Output dir: $OUTPUT_DIR"
echo "GPU:        1"
echo "Batch:      16"
echo "Epochs:     10 (checkpoints at 5500, 11000, 16500 + final)"

nvidia-smi -L

bash scripts/run/finetune.sh \
  1 \
  so100 \
  "data.embodiment_datasets.so100.dataset_groups=[{weight: 1.0, dataset_dirs: ['$DATASET']}]" \
  datastatics_path=null \
  model.max_epochs=10 \
  checkpointing_steps=5500 \
  "hydra.run.dir=$OUTPUT_DIR"

# Export + upload every checkpoint to the Hub, then clean up the local checkpoint files.
# A failure here must not mark a successful training run as failed; cleanup only happens after
# every upload is verified on the Hub, so rerunning the upload script is always safe.
bash "$ROBO_ROOT/scripts/g05/upload_so101_upright_bottle_10ep.sh" \
  || echo "Upload failed; rerun scripts/g05/upload_so101_upright_bottle_10ep.sh manually."
