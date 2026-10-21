#!/bin/bash
set -euo pipefail

# GR00T N1.7 FULL fine-tuning on the DROID (Franka) put-marker-pen dataset on NSCC ASPIRE 2B, 8 x H200.
# Same recipe as train_droid_put_marker_pen_10ep.sh (Hopper, 1 GPU): System 2 (Cosmos-Reason2-2B VLM,
# a Qwen3-VL layout: LLM + vision encoder) and System 1 (projector, DiT action head, VLLN) are all
# trained (tune_llm / tune_visual / tune_projector / tune_diffusion_model / tune_vlln = true,
# tune_top_llm_layers=0). Base nvidia/GR00T-N1.7-3B, new_embodiment, default chunk 40 / 40 action
# steps, state-relative arm actions with an absolute gripper, bf16 autocast on fp32 parameters
# (use_bf16 / model_params_fp32), default LR (1e-4 AdamW, cosine, 5% warmup; policy.max_steps sets
# the warmup length, so it equals --steps), the dataset's default image transforms.
# Cameras: left, wrist. Trains on droid-put-marker-pen-left-wrist (built by
# tools/make_droid_left_wrist_dataset.py): LeRobot's GR00T processor uses every video key of the
# dataset in info.json order and has no CLI camera selection, so the right camera is removed and
# left_image is listed before wrist.
#
# Multi-GPU change: lerobot-train is launched with `accelerate launch` on 8 GPUs (DDP). lerobot's
# --batch_size is per process, so batch 4 x 8 GPUs keeps the effective batch at 32. The step count
# and policy.max_steps are unchanged, and so is the LR schedule (warmup ceil(0.05 x 16,160) = 808
# steps, then cosine decay to 0 at step 16,160; accelerate does not rescale it because lerobot sets
# step_scheduler_with_optimizer=False):
#   10 epochs = 16,160 steps (1616 steps/epoch = ceil(51699 / 32)).
# --num_workers / --prefetch_factor are per process, as on Hopper: 8 ranks x 8 workers = 64 workers
# (PBS gives the job the whole node, 192 cores).
# "10 epochs" counts frames, as lerobot logs it and as on Hopper: GR00T's sampler skips the last 39
# frames of each episode (drop_n_last_frames), so 16,160 x 32 samples are ~10.8 passes over the
# trainable windows. The step count is the Hopper one.
# No intermediate checkpoints: only the final step is saved locally, and only the final model is
# pushed to the Hub repo root (save_checkpoint_to_hub=false, so no checkpoints/<step>/ on the Hub).
# Framework: huggingface/lerobot v0.6.1 (native groot policy, GR00T N1.7), venv $LEROBOT_VENV. Every
# flag of the Hopper run (LeRobot fork) exists upstream with the same meaning. At startup each rank
# also reads the gated nvidia/Cosmos-Reason2-2B tokenizer/processor (pre-fetched into $HF_HOME;
# HF_TOKEN has access) and computes the per-timestep relative-action stats from the dataset in memory.

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../nscc" && pwd)/env.sh"
source "$LEROBOT_VENV/bin/activate"

NUM_GPUS=8

# Local-only dataset; repo_id is just a label when --dataset.root points at it.
DATASET="local/droid-put-marker-pen-left-wrist"
DATASET_ROOT="$DATASETS_ROOT/droid-put-marker-pen-left-wrist"
SOURCE_DATASET_ROOT="$DATASETS_ROOT/droid-put-marker-pen"

JOB_NAME="groot-n17-droid-put-marker-pen-10ep-nscc"
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

echo "GR00T N1.7 | $JOB_NAME | GPUs: $NUM_GPUS | batch $BATCH_SIZE/GPU (effective $((BATCH_SIZE * NUM_GPUS))) | steps $STEPS"

# mixed_precision=no matches the single-GPU Hopper run: GrootConfig has no `dtype` field and
# use_amp=false, so lerobot-train passes mixed_precision=None and Accelerate falls back to the
# launcher's value ("no", the default plain `lerobot-train` got on Hopper). The policy runs the GR00T
# forward under bf16 autocast itself (use_bf16=true) on fp32 parameters (model_params_fp32=true), so
# gradients, the DDP all-reduce and AdamW stay fp32.
# relative_exclude_joints matches action names by substring: "gripper" keeps gripper_target
# absolute while joint_target_0..6 become state-relative.
set +e
accelerate launch \
    --multi_gpu \
    --num_machines=1 \
    --num_processes="$NUM_GPUS" \
    --mixed_precision=no \
    --dynamo_backend=no \
    --main_process_port="$(nscc_free_port)" \
    "$(command -v lerobot-train)" \
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
    --policy.private=true \
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
    --job_name="$JOB_NAME" \
    "${WANDB_ARGS[@]}" \
    --output_dir="$OUTPUT_DIR"
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
