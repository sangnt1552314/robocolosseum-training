#!/bin/bash
set -euo pipefail

# MolmoAct2 fine-tuning on the DROID (Franka) towel-in-basket dataset via lerobot-train, with the
# same recipe as our SO-101 runs (see train_so101_cube_drawer_40k.sh): base allenai/MolmoAct2,
# full fine-tuning (VLM, vision tower, connector and action expert; LoRA off), continuous action
# mode, 8 flow timesteps, default chunk 30 / 30 action steps, quantile normalization, the guide's
# augmentation and default LRs (VLM 1e-5 / ViT 5e-6 / connector 5e-6 / action expert 5e-5,
# 200 warmup steps, cosine decay to 1e-6 over the whole run).
# DROID-specific settings:
#   - setup_type / control_mode / normalize_gripper=false match MolmoAct2's franka_droid
#     pretraining tag (norm_stats.json metadata_by_tag) and its 8-D joint_0..6 + gripper layout;
#     the gripper (already in [0, 1]) passes through un-normalized as in pretraining.
#   - Cameras: left, right, wrist (the order of franka_droid's exterior_1_left, exterior_2_left,
#     wrist_left).
#   - Crop: 95% per side at the native 360x640 (16:9) ratio.
# Batch 64 on 1 H200 GPU (lerobot-train has no gradient accumulation, so the whole batch is one
# forward/backward pass; gradient checkpointing is on to fit it). 5 epochs = 2,145 steps (429
# steps/epoch = ceil(27395 / 64)); only the final checkpoint is saved and pushed to the Hub under
# checkpoints/<step>/.

export HF_HOME=/scratch/e1583535/cache
export HF_DATASETS_CACHE=/scratch/e1583535/cache/datasets

# Keep wandb staging/cache off the 40 GB home quota.
export WANDB_DATA_DIR=/scratch/e1583535/cache/wandb/data
export WANDB_CACHE_DIR=/scratch/e1583535/cache/wandb/cache
export WANDB_ARTIFACT_DIR=/scratch/e1583535/cache/wandb/artifacts
# A W&B run with this run id was already created (empty, no logged steps) by an aborted launch on
# 2026-10-11; "allow" makes wandb.init attach to it instead of failing on the existing id.
export WANDB_RESUME=allow

export FFMPEG_ROOT=/scratch/e1583535/opt/ffmpeg-8.0.3
export PATH="$FFMPEG_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$FFMPEG_ROOT/lib:${LD_LIBRARY_PATH:-}"

# Local-only dataset; repo_id is just a label when --dataset.root points at it.
DATASET="local/droid-towel-in-basket"
DATASET_ROOT="/scratch/e1583535/datasets/droid-towel-in-basket"

JOB_NAME="molmoact2-droid-towel-in-basket-5ep"
MODELS_ROOT="/scratch/Projects/CFP-05/CFP05-CF-002/robocolosseum-finetuned-models"
OUTPUT_DIR="$MODELS_ROOT/$JOB_NAME"

HUB_REPO_ID="tsangb34/$JOB_NAME"

BATCH_SIZE=64
STEPS=2145
SAVE_FREQ="$STEPS"  # only the final checkpoint

WANDB_PROJECT="RoboColosseum"
WANDB_ENTITY="tsangb34-national-university-of-singapore-students-union"

test -f "$DATASET_ROOT/meta/info.json"

lerobot-train \
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
    --policy.tags='["molmoact2","droid","franka","robocolosseum","towel-in-basket"]' \
    --save_checkpoint=true \
    --save_freq="$SAVE_FREQ" \
    --save_checkpoint_to_hub=true \
    --steps="$STEPS" \
    --batch_size="$BATCH_SIZE" \
    --num_workers=8 \
    --log_freq=20 \
    --job_name="$JOB_NAME" \
    --wandb.enable=true \
    --wandb.project="$WANDB_PROJECT" \
    --wandb.entity="$WANDB_ENTITY" \
    --wandb.run_id="$JOB_NAME" \
    --wandb.disable_artifact=true \
    --output_dir="$OUTPUT_DIR"
