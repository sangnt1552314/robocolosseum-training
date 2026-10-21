#!/bin/bash
set -euo pipefail

# pi05 fine-tuning on the DROID (Franka) put-marker-pen dataset on NSCC ASPIRE 2B, 8 x H200.
# Same recipe as train_droid_put_marker_pen_10ep.sh (Hopper, 1 GPU): base lerobot/pi05_base, full
# fine-tuning (vision encoder trainable), quantile normalization, default chunk 50 / 50 action steps,
# default pi05 LR schedule (2.5e-5 peak, cosine to 2.5e-6; lerobot scales the 1K warmup / 30K decay
# by --steps / 30K when --steps < 30K, so warmup 538 and decay 16,160 steps).
# Cameras: left, wrist (+ one masked empty slot, as in pi05_base's 3-camera layout). Trains on
# droid-put-marker-pen-left-wrist (built by tools/make_droid_left_wrist_dataset.py): the right
# camera is removed and info.json lists left_image before wrist, which sets the camera order.
#
# Multi-GPU change: lerobot-train is launched with `accelerate launch` on 8 GPUs (DDP). lerobot's
# --batch_size is per process, so batch 4 x 8 GPUs keeps the effective batch at 32. The step count
# and the LR schedule (built from --steps, not from the number of processes) are unchanged:
#   10 epochs = 16,160 steps (1616 steps/epoch = ceil(51699 / 32)).
# Gradient checkpointing stays on as on Hopper (it does not change the math; pi05 checkpoints with
# use_reentrant=False, which works with DDP's find_unused_parameters=True).
# No intermediate checkpoints: only the final step is saved locally, and only the final model is
# pushed to the Hub repo root (save_checkpoint_to_hub=false, so no checkpoints/<step>/ on the Hub).
# Framework: huggingface/lerobot v0.6.1 (native pi05 policy), venv $LEROBOT_VENV. The PaliGemma
# tokenizer (google/paligemma-3b-pt-224, gated) is pre-fetched into $HF_HOME by setup_envs.sh.

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../nscc" && pwd)/env.sh"
source "$LEROBOT_VENV/bin/activate"

NUM_GPUS=8

# Local-only dataset; repo_id is just a label when --dataset.root points at it.
DATASET="local/droid-put-marker-pen-left-wrist"
DATASET_ROOT="$DATASETS_ROOT/droid-put-marker-pen-left-wrist"
SOURCE_DATASET_ROOT="$DATASETS_ROOT/droid-put-marker-pen"

JOB_NAME="pi05-droid-put-marker-pen-10ep-nscc"
OUTPUT_DIR="$MODELS_ROOT/$JOB_NAME"

HUB_REPO_ID="$HF_HUB_ORG/$JOB_NAME"  # RoboColosseum/<job> (scripts/nscc/env.sh)

BATCH_SIZE=4  # per GPU -> effective 32
STEPS=16160
SAVE_FREQ="$STEPS"  # only the final checkpoint

# Build the left + wrist dataset if it does not exist yet (a few minutes; videos are copied,
# not re-encoded).
if [ ! -f "$DATASET_ROOT/meta/info.json" ]; then
    echo "Left + wrist dataset not found; building $DATASET_ROOT"
    python "$ROBO_ROOT/tools/make_droid_left_wrist_dataset.py" "$SOURCE_DATASET_ROOT" "$DATASET_ROOT"
fi
test -f "$DATASET_ROOT/meta/info.json"
: "${HF_TOKEN:?HF_TOKEN not set; add it to $ROBO_ROOT/.env}"

# A rerun cannot reuse the old output dir: lerobot refuses an existing --output_dir unless
# --resume=true, and only the final step is checkpointed, so there is nothing to resume from. Fail here,
# before 8 ranks start. A rerun also needs a fresh W&B run id: qsub -v WANDB_RUN_ID=<new id> <pbs>.
WANDB_RUN_ID="${WANDB_RUN_ID:-$JOB_NAME}"
if [ -e "$OUTPUT_DIR" ]; then
    echo "ERROR: $OUTPUT_DIR exists from an earlier run. Move it aside first, e.g." >&2
    echo "  mv $OUTPUT_DIR $OUTPUT_DIR.old && qsub -v WANDB_RUN_ID=$JOB_NAME-r2 <this job's .pbs>" >&2
    exit 1
fi

WANDB_ARGS=(
    --wandb.enable=true
    --wandb.project="$WANDB_PROJECT"
    --wandb.run_id="$WANDB_RUN_ID"
    --wandb.disable_artifact=true
)
if [ -n "${WANDB_ENTITY:-}" ]; then
    WANDB_ARGS+=(--wandb.entity="$WANDB_ENTITY")
fi

export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
# Store-based barrier right after init_process_group, so no rank creates the output dir (rank 0's
# wandb.init does) before every rank has passed lerobot's "output dir must not exist" check.
export TORCH_DIST_INIT_BARRIER=1

echo "pi05 | $JOB_NAME | GPUs: $NUM_GPUS | batch $BATCH_SIZE/GPU (effective $((BATCH_SIZE * NUM_GPUS))) | steps $STEPS"

# pi05 has a dtype field, so lerobot passes mixed_precision=bf16 to Accelerator itself (as on the
# 1-GPU Hopper run); the launch flag below only covers the fallback.
set +e
accelerate launch \
    --multi_gpu \
    --num_machines=1 \
    --num_processes="$NUM_GPUS" \
    --mixed_precision=bf16 \
    --dynamo_backend=no \
    --main_process_port="$(nscc_free_port)" \
    "$(command -v lerobot-train)" \
    --dataset.repo_id="$DATASET" \
    --dataset.root="$DATASET_ROOT" \
    --dataset.video_backend=pyav \
    --policy.type=pi05 \
    --policy.pretrained_path=lerobot/pi05_base \
    --policy.device=cuda \
    --policy.dtype=bfloat16 \
    --policy.normalization_mapping='{"ACTION":"QUANTILES","STATE":"QUANTILES","VISUAL":"IDENTITY"}' \
    --policy.empty_cameras=1 \
    --policy.freeze_vision_encoder=false \
    --policy.train_expert_only=false \
    --policy.gradient_checkpointing=true \
    --policy.compile_model=false \
    --policy.push_to_hub=true \
    --policy.repo_id="$HUB_REPO_ID" \
    --policy.private=true \
    --output_dir="$OUTPUT_DIR" \
    --job_name="$JOB_NAME" \
    --batch_size="$BATCH_SIZE" \
    --num_workers=8 \
    --steps="$STEPS" \
    --log_freq=20 \
    --save_checkpoint=true \
    --save_freq="$SAVE_FREQ" \
    --save_checkpoint_to_hub=false \
    "${WANDB_ARGS[@]}"
rc=$?
set -e

# lerobot pushes the final model from rank 0 while ranks 1-7 wait in an NCCL barrier with torch's
# 10-minute default timeout, so a slow upload can abort the job after the final checkpoint is saved.
# The checkpoint only exists once the last step finished; push it from a single process if anything
# is missing on the Hub (a no-op when lerobot's own push completed).
if [ -f "$OUTPUT_DIR/checkpoints/last/pretrained_model/model.safetensors" ]; then
    bash "$ROBO_ROOT/scripts/nscc/push_lerobot_final.sh" "$JOB_NAME" && rc=0
fi
exit "$rc"
