#!/bin/bash
set -euo pipefail

# G0.5 (GalaxeaVLA) fine-tuning on the DROID (Franka) towel-in-basket dataset with the same native
# recipe as our SO-101 runs: GalaxeaVLA's configs/task/so100.yaml settings (g05-base init, batch 16,
# default LR 8e-5, warmup_ratio 0.05, warmup_constant_cosine schedule, action_dim 20,
# relative-joint arm actions with an absolute gripper, q01/q99 norm, same tokenizer settings).
# Nothing is copied into or edited in GalaxeaVLA: hydra.searchpath adds our scripts/g05/configs, which
# hold the Franka task/data configs:
#   - task/robocolosseum_droid_franka.yaml -> configs/task/so100.yaml with the data config below
#                                              and datastatics_path=null (stats from this dataset)
#   - data/robocolosseum_droid_franka.yaml -> 7 arm joints + gripper sliced from the 8-D
#                                              state/action, cameras left (exterior) + wrist
#                                              (wrist_left) + a dummy wrist_right slot; the right
#                                              camera is not used. Embodiment Droid_Franka.
# Overridden here via Hydra:
#   - dataset path            -> our local LeRobot v3 dataset
#   - max_epochs=10 (twice the 5-epoch runs)
#   - checkpointing_steps=1000000 -> no periodic checkpoints; finetune.py always saves the final step.
# After training the final checkpoint is exported, uploaded to the Hub repo root (no
# checkpoints/step_<N>/ copies) and cleaned up locally by upload_droid_towel_in_basket_10ep.sh.

ROBO_ROOT=/scratch/e1583535/projects/robocolosseum-training
G05_ROOT=/scratch/e1583535/projects/GalaxeaVLA

DATASET=/scratch/e1583535/datasets/droid-towel-in-basket

JOB_NAME="g05-droid-towel-in-basket-10ep"
MODELS_ROOT="/scratch/Projects/CFP-05/CFP05-CF-002/robocolosseum-finetuned-models"
OUTPUT_DIR="$MODELS_ROOT/$JOB_NAME"

export HF_HOME=/scratch/e1583535/cache
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"

# Required by configs/train.yaml; the run dir itself is pinned to $OUTPUT_DIR below.
export G05_OUTPUT_DIR="$MODELS_ROOT"
# Used by train.yaml for the run name.
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

echo "G0.5 fine-tuning (native so100 recipe, DROID Franka data config)"
echo "Dataset:    $DATASET"
echo "Output dir: $OUTPUT_DIR"
echo "GPU:        1"
echo "Batch:      16"
echo "Epochs:     10 (final checkpoint only)"

nvidia-smi -L

bash scripts/run/finetune.sh \
  1 \
  robocolosseum_droid_franka \
  "hydra.searchpath=[file://$ROBO_ROOT/scripts/g05/configs]" \
  "data.embodiment_datasets.Droid_Franka.dataset_groups=[{weight: 1.0, dataset_dirs: ['$DATASET']}]" \
  model.max_epochs=10 \
  checkpointing_steps=1000000 \
  "hydra.run.dir=$OUTPUT_DIR"

# Export + upload the final checkpoint to the Hub root, then clean up the local checkpoint files.
# A failure here must not mark a successful training run as failed; cleanup only happens after
# every upload is verified on the Hub, so rerunning the upload script is always safe.
bash "$ROBO_ROOT/scripts/g05/upload_droid_towel_in_basket_10ep.sh" \
  || echo "Upload failed; rerun scripts/g05/upload_droid_towel_in_basket_10ep.sh manually."
