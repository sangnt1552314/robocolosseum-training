#!/bin/bash
set -euo pipefail

# MolmoAct2 fine-tuning on the DROID (Franka) put-marker-pen dataset on NSCC ASPIRE 2B, 8 x H200.
# Same recipe as train_droid_put_marker_pen_10ep.sh (Hopper, 1 GPU): base allenai/MolmoAct2, full
# fine-tuning (VLM, vision tower, connector and action expert; LoRA off), continuous action mode,
# 8 flow timesteps, default chunk 30 / 30 action steps, quantile normalization, franka_droid
# setup_type / control mode with the gripper un-normalized, cameras left/right/wrist, the guide's
# augmentation (95% crop at 16:9, +-5 deg rotation, color jitter) and default LRs (VLM 1e-5 /
# ViT 5e-6 / connector 5e-6 / action expert 5e-5, 200 warmup steps, cosine decay to 1e-6).
#
# Multi-GPU change: lerobot-train is launched with `accelerate launch` on 8 GPUs (DDP). lerobot's
# --batch_size is per process, so batch 8 x 8 GPUs keeps the effective batch at 64. The step count
# (and therefore the LR schedule) is unchanged:
#   10 epochs = 8,080 steps (808 steps/epoch = ceil(51699 / 64)).
# No intermediate checkpoints: only the final step is saved locally, and only the final model is
# pushed to the Hub repo root (save_checkpoint_to_hub=false, so no checkpoints/<step>/ on the Hub).
# Framework: huggingface/lerobot v0.6.1 (native molmoact2 policy), venv $LEROBOT_VENV.

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../nscc" && pwd)/env.sh"
source "$LEROBOT_VENV/bin/activate"

NUM_GPUS=8

# Local copy of RoboColosseum/droid-put-marker-pen; repo_id is just a label when --dataset.root
# points at it.
DATASET="local/droid-put-marker-pen"
DATASET_ROOT="$DATASETS_ROOT/droid-put-marker-pen"

JOB_NAME="molmoact2-droid-put-marker-pen-10ep-nscc"
OUTPUT_DIR="$MODELS_ROOT/$JOB_NAME"

HUB_REPO_ID="$HF_HUB_ORG/$JOB_NAME"  # RoboColosseum/<job> (scripts/nscc/env.sh)

BATCH_SIZE=8  # per GPU -> effective 64
STEPS=8080
SAVE_FREQ="$STEPS"  # only the final checkpoint

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

echo "MolmoAct2 | $JOB_NAME | GPUs: $NUM_GPUS | batch $BATCH_SIZE/GPU (effective $((BATCH_SIZE * NUM_GPUS))) | steps $STEPS"

# mixed_precision=no matches the single-GPU Hopper runs (the model itself runs in bfloat16 via
# --policy.model_dtype).
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
    --dataset.image_transforms.max_num_transforms=3 \
    --dataset.image_transforms.random_order=false \
    --dataset.image_transforms.tfs='{"crop":{"weight":1.0,"type":"RandomResizedCrop","kwargs":{"size":[360,640],"scale":[0.9025,0.9025],"ratio":[1.7777778,1.7777778]}},"rotation":{"weight":1.0,"type":"RandomRotation","kwargs":{"degrees":[-5.0,5.0],"interpolation":2}},"color":{"weight":1.0,"type":"ColorJitter","kwargs":{"brightness":0.2,"contrast":[0.8,1.2],"saturation":[0.8,1.2],"hue":0.05}}}' \
    --policy.type=molmoact2 \
    --policy.checkpoint_path=allenai/MolmoAct2 \
    --policy.device=cuda \
    --policy.action_mode=continuous \
    --policy.inference_action_mode=continuous \
    --policy.setup_type="single franka robotic arm in droid" \
    --policy.control_mode="absolute joint pose" \
    --policy.image_keys='["observation.images.left_image","observation.images.right_image","observation.images.wrist"]' \
    --policy.model_dtype=bfloat16 \
    --policy.normalize_gripper=false \
    --policy.normalization_mapping='{"ACTION":"QUANTILES","STATE":"QUANTILES","VISUAL":"IDENTITY"}' \
    --policy.gradient_checkpointing=true \
    --policy.num_flow_timesteps=8 \
    --policy.freeze_embedding=true \
    --policy.enable_lora_vlm=false \
    --policy.enable_lora_action_expert=false \
    --policy.train_action_expert_only=false \
    --policy.enable_knowledge_insulation=false \
    --policy.optimizer_lr=1e-5 \
    --policy.optimizer_vit_lr=5e-6 \
    --policy.optimizer_connector_lr=5e-6 \
    --policy.optimizer_action_expert_lr=5e-5 \
    --policy.scheduler_warmup_steps=200 \
    --policy.scheduler_decay_steps="$STEPS" \
    --policy.scheduler_decay_lr=1e-6 \
    --policy.push_to_hub=true \
    --policy.repo_id="$HUB_REPO_ID" \
    --policy.private=false \
    --policy.tags='["molmoact2","droid","franka","robocolosseum","put-marker-pen"]' \
    --save_checkpoint=true \
    --save_freq="$SAVE_FREQ" \
    --save_checkpoint_to_hub=false \
    --steps="$STEPS" \
    --batch_size="$BATCH_SIZE" \
    --num_workers=8 \
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
