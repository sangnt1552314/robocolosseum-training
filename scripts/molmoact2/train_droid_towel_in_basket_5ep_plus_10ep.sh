#!/bin/bash
set -euo pipefail

# Continued fine-tuning ("5 + 10 epochs") of tsangb34/molmoact2-droid-towel-in-basket-5ep for 10 more
# epochs on the same DROID towel-in-basket dataset, to compare against the single-run 10- and 15-epoch
# models (train_droid_towel_in_basket_{10,15}ep.sh).
# Same recipe as train_droid_towel_in_basket_5ep.sh, but the weights are initialized from the
# 5-epoch model via --policy.path (lerobot loads its saved config and model.safetensors; the
# --policy.* flags below are CLI overrides on that config and match it). This is a new run, not a
# --resume: optimizer state is fresh and the LR schedule restarts (200 warmup steps, cosine decay to
# 1e-6 over these 4,290 steps). A true --resume would be useless here, because the first run's
# cosine schedule was baked to decay_steps=2145, so the LR would stay at 1e-6 for all 10 extra
# epochs. Total: warmup + cosine over 5 epochs, then warmup + cosine over 10 epochs, versus one
# cosine over the whole run for the 10ep/15ep runs.
# Normalization stats are recomputed from the dataset (not --resume), identical to the 5ep run's.
# Batch 64 on 1 H200 GPU, gradient checkpointing on. 10 epochs = 4,290 steps (429 steps/epoch =
# ceil(27395 / 64)). A checkpoint every 5 epochs (save_freq=2145: steps 2145 = 5+5, 4290 = 5+10);
# save_checkpoint_to_hub pushes each one to <repo>/checkpoints/<step>/ (tagged <step>) as it is
# written, and push_to_hub uploads the final model to the repo root.

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
DATASET="local/droid-towel-in-basket"
DATASET_ROOT="/scratch/e1583535/datasets/droid-towel-in-basket"

# Starting weights: the 5-epoch run (final model at the Hub repo root).
INIT_POLICY="tsangb34/molmoact2-droid-towel-in-basket-5ep"

JOB_NAME="molmoact2-droid-towel-in-basket-5ep-plus-10ep"
MODELS_ROOT="/scratch/Projects/CFP-05/CFP05-CF-002/robocolosseum-finetuned-models"
OUTPUT_DIR="$MODELS_ROOT/$JOB_NAME"

HUB_REPO_ID="tsangb34/$JOB_NAME"

BATCH_SIZE=64
STEPS=4290
SAVE_FREQ=2145  # every 5 epochs: 2145 (5+5), 4290 (5+10)

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
    --policy.path="$INIT_POLICY" \
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
    --policy.tags='["molmoact2","droid","franka","robocolosseum","towel-in-basket","continued"]' \
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
