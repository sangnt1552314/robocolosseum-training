#!/bin/bash
set -euo pipefail

# GR00T N1.7 on the Unitree G1 carry-box dataset with the default GR00T fine-tuning recipe
# (frozen LLM and vision encoder; projector, DiT and VLLN tuned): batch 32, 40K steps,
# chunk 32 / 32 action steps.
#
# G1-specific notes (this is NOT the SO101 single-arm layout):
# - observation.state is 29 measured joint positions, while action is 23-D
#   (pivot 7 + commanded L/R end-effector poses 12 + L/R trigger/squeeze 4). The dimensions do
#   not line up, so state-relative actions (action - state) are meaningless here and are disabled.
# - observation.state.wbc is not used as the policy state: its pivot and gripper dims are copies
#   of the action at the same frame, which would leak the target.
# - Cameras: head, left_wrist, right_wrist, all 480x640 H.264. Trains on g1-carry-box-left-eye
#   (built by tools/make_g1_left_eye_dataset.py): the original head camera is a 480x1280
#   side-by-side stereo frame, and LeRobot's GR00T processor stacks cameras before resizing, so
#   all cameras must share one shape. Only the left eye is kept (as in OpenHLM). At inference,
#   crop the robot's head frame with tools/g1_head_left_eye.py:crop_head_left_eye.
# - Launched through tools/lerobot_train_pyav_single_thread.py instead of `lerobot-train`: with
#   PyAV's default threading, FFmpeg starts a decoder thread per core on every per-frame
#   container open, which slows data loading. Same CLI args as lerobot-train.

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
DATASET="local/g1-carry-box-left-eye"
DATASET_ROOT="/scratch/e1583535/datasets/g1-carry-box-left-eye"

JOB_NAME="groot-n17-g1-carry-box-40k"
MODELS_ROOT="/scratch/Projects/CFP-05/CFP05-CF-002/robocolosseum-finetuned-models"
OUTPUT_DIR="$MODELS_ROOT/$JOB_NAME"

HUB_REPO_ID="tsangb34/$JOB_NAME"

BATCH_SIZE=32
STEPS=40000

WANDB_PROJECT="RoboColosseum"
WANDB_ENTITY="tsangb34-national-university-of-singapore-students-union"

python "$REPO_ROOT/tools/lerobot_train_pyav_single_thread.py" \
    --dataset.repo_id="$DATASET" \
    --dataset.root="$DATASET_ROOT" \
    --dataset.video_backend=pyav \
    --dataset.image_transforms.enable=true \
    --policy.type=groot \
    --policy.device=cuda \
    --policy.base_model_path=nvidia/GR00T-N1.7-3B \
    --policy.embodiment_tag=new_embodiment \
    --policy.chunk_size=32 \
    --policy.n_action_steps=32 \
    --policy.use_relative_actions=false \
    --policy.use_bf16=true \
    --policy.max_steps="$STEPS" \
    --policy.push_to_hub=true \
    --policy.repo_id="$HUB_REPO_ID" \
    --policy.private=false \
    --batch_size="$BATCH_SIZE" \
    --steps="$STEPS" \
    --num_workers=10 \
    --persistent_workers=true \
    --save_checkpoint=true \
    --save_freq=10000 \
    --save_checkpoint_to_hub=true \
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
