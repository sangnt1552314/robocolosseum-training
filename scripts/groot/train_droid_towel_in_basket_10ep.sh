#!/bin/bash
set -euo pipefail

# GR00T N1.7 FULL fine-tuning on the DROID (Franka) towel-in-basket dataset: System 2 (Eagle VLM:
# LLM + vision encoder) and System 1 (projector, DiT action head, VLLN) are all trained
# (tune_llm / tune_visual / tune_projector / tune_diffusion_model / tune_vlln = true).
# Otherwise the same recipe as our SO-101 runs: base nvidia/GR00T-N1.7-3B, new_embodiment,
# default chunk 40 / 40 action steps, state-relative arm actions with an absolute gripper, default LR
# (1e-4, cosine, 5% warmup; policy.max_steps sets the warmup length, so it equals --steps).
# Cameras: left, wrist. Trains on droid-towel-in-basket-left-wrist (built by
# tools/make_droid_left_wrist_dataset.py): LeRobot's GR00T processor uses every video key of the
# dataset in info.json order and has no CLI camera selection, so the right camera is removed and
# left_image is listed before wrist.
# Batch 32 on 1 GPU.
# 10 epochs = 8,570 steps (857 steps/epoch = ceil(27395 / 32)). No intermediate checkpoints: only
# the final step is saved locally, and only the final model is pushed to the Hub repo root
# (save_checkpoint_to_hub=false, so no checkpoints/<step>/ on the Hub).

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export HF_HOME=/scratch/e1583535/cache
export HF_DATASETS_CACHE=/scratch/e1583535/cache/datasets

# Keep wandb staging/cache off the 40 GB home quota.
export WANDB_DATA_DIR=/scratch/e1583535/cache/wandb/data
export WANDB_CACHE_DIR=/scratch/e1583535/cache/wandb/cache
export WANDB_ARTIFACT_DIR=/scratch/e1583535/cache/wandb/artifacts

export FFMPEG_ROOT=/scratch/e1583535/opt/ffmpeg-8.0.3
export PATH="$FFMPEG_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$FFMPEG_ROOT/lib:${LD_LIBRARY_PATH:-}"

# Local-only dataset; repo_id is just a label when --dataset.root points at it.
DATASET="local/droid-towel-in-basket-left-wrist"
DATASET_ROOT="/scratch/e1583535/datasets/droid-towel-in-basket-left-wrist"
SOURCE_DATASET_ROOT="/scratch/e1583535/datasets/droid-towel-in-basket"

JOB_NAME="groot-n17-droid-towel-in-basket-10ep"
MODELS_ROOT="/scratch/Projects/CFP-05/CFP05-CF-002/robocolosseum-finetuned-models"
OUTPUT_DIR="$MODELS_ROOT/$JOB_NAME"

HUB_REPO_ID="tsangb34/$JOB_NAME"

BATCH_SIZE=32
STEPS=8570
SAVE_FREQ="$STEPS"  # only the final checkpoint

WANDB_PROJECT="RoboColosseum"
WANDB_ENTITY="tsangb34-national-university-of-singapore-students-union"

# Build the left + wrist dataset if it does not exist yet (a few minutes; videos are copied,
# not re-encoded).
if [ ! -f "$DATASET_ROOT/meta/info.json" ]; then
    echo "Left + wrist dataset not found; building $DATASET_ROOT"
    python "$REPO_ROOT/tools/make_droid_left_wrist_dataset.py" "$SOURCE_DATASET_ROOT" "$DATASET_ROOT"
fi
test -f "$DATASET_ROOT/meta/info.json"

# relative_exclude_joints matches action names by substring: "gripper" keeps gripper_target
# absolute while joint_target_0..6 become state-relative.
lerobot-train \
    --dataset.repo_id="$DATASET" \
    --dataset.root="$DATASET_ROOT" \
    --dataset.video_backend=pyav \
    --dataset.image_transforms.enable=true \
    --policy.type=groot \
    --policy.device=cuda \
    --policy.base_model_path=nvidia/GR00T-N1.7-3B \
    --policy.embodiment_tag=new_embodiment \
    --policy.use_relative_actions=true \
    --policy.relative_exclude_joints='["gripper"]' \
    --policy.use_bf16=true \
    --policy.model_params_fp32=true \
    --policy.tune_llm=true \
    --policy.tune_visual=true \
    --policy.tune_projector=true \
    --policy.tune_diffusion_model=true \
    --policy.tune_vlln=true \
    --policy.tune_top_llm_layers=0 \
    --policy.max_steps="$STEPS" \
    --policy.push_to_hub=true \
    --policy.repo_id="$HUB_REPO_ID" \
    --policy.private=false \
    --batch_size="$BATCH_SIZE" \
    --steps="$STEPS" \
    --num_workers=8 \
    --prefetch_factor=2 \
    --persistent_workers=true \
    --save_checkpoint=true \
    --save_freq="$SAVE_FREQ" \
    --save_checkpoint_to_hub=false \
    --use_policy_training_preset=true \
    --env_eval_freq=0 \
    --eval_steps=0 \
    --log_freq=20 \
    --output_dir="$OUTPUT_DIR" \
    --job_name="$JOB_NAME" \
    --wandb.enable=true \
    --wandb.project="$WANDB_PROJECT" \
    --wandb.entity="$WANDB_ENTITY" \
    --wandb.run_id="$JOB_NAME" \
    --wandb.disable_artifact=true
