#!/bin/bash
set -euo pipefail

# MolmoAct2 fine-tuning via lerobot-train, following the LeRobot MolmoAct2 guide
# (https://github.com/huggingface/lerobot/blob/main/docs/source/molmoact2.mdx), mapped onto the
# option names of our lerobot fork (/scratch/e1583535/projects/lerobot):
#   train_mode_vlm=fft  ->  enable_lora_vlm=false + train_action_expert_only=false
#   dtype=bfloat16      ->  model_dtype=bfloat16
# Starts from base allenai/MolmoAct2 with full fine-tuning (VLM, vision tower, connector and
# action expert trained; LoRA off), continuous action mode, 8 flow timesteps, the guide's image
# augmentation and LRs (VLM 1e-5 / ViT 5e-6 / connector 5e-6 / action expert 5e-5),
# batch 16 on 1 GPU, 40K steps, chunk 32 / 32 action steps, quantile normalization.
# Image crop: the guide's 256x256 / ratio-1 crop is sized for LIBERO's square frames; on our
# 480x640 frames it always falls back to a fixed center 480x480 crop (drops 80 px per side,
# absent at inference), so we crop 95% per side at the native 4:3 ratio instead.
# No joint_signs / joint_offsets: in lerobot they only transform the state (before
# normalization) and the predicted action (after unnormalization), which is correct for
# zero-shot with the checkpoint's norm_stats, but when fine-tuning with dataset stats the
# state gets clamped and the action targets stay in our frame, so inference would be wrong.
# Checkpoints are saved every 10K steps and each is pushed to the Hub under checkpoints/<step>/.

export HF_HOME=/scratch/e1583535/cache
export HF_DATASETS_CACHE=/scratch/e1583535/cache/datasets

# Keep wandb staging/cache off the 40 GB home quota.
export WANDB_DATA_DIR=/scratch/e1583535/cache/wandb/data
export WANDB_CACHE_DIR=/scratch/e1583535/cache/wandb/cache
export WANDB_ARTIFACT_DIR=/scratch/e1583535/cache/wandb/artifacts

export FFMPEG_ROOT=/scratch/e1583535/opt/ffmpeg-8.0.3
export PATH="$FFMPEG_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$FFMPEG_ROOT/lib:${LD_LIBRARY_PATH:-}"

DATASET="Jiamo0912/so101-cube-drawer"
DATASET_ROOT="/scratch/e1583535/datasets/so101-cube-drawer"

JOB_NAME="molmoact2-so101-cube-drawer-40k"
MODELS_ROOT="/scratch/Projects/CFP-05/CFP05-CF-002/robocolosseum-finetuned-models"
OUTPUT_DIR="$MODELS_ROOT/$JOB_NAME"

HUB_REPO_ID="tsangb34/$JOB_NAME"

WANDB_PROJECT="RoboColosseum"
WANDB_ENTITY="tsangb34-national-university-of-singapore-students-union"

# save_freq=10000 writes checkpoints at 10K, 20K, 30K and 40K; save_checkpoint_to_hub
# pushes each one to <repo>/checkpoints/<step>/, and push_to_hub uploads the final model.
lerobot-train \
    --dataset.repo_id="$DATASET" \
    --dataset.root="$DATASET_ROOT" \
    --dataset.video_backend=pyav \
    --dataset.image_transforms.enable=true \
    --dataset.image_transforms.max_num_transforms=3 \
    --dataset.image_transforms.random_order=false \
    --dataset.image_transforms.tfs='{"crop":{"weight":1.0,"type":"RandomResizedCrop","kwargs":{"size":[480,640],"scale":[0.9025,0.9025],"ratio":[1.3333333,1.3333333]}},"rotation":{"weight":1.0,"type":"RandomRotation","kwargs":{"degrees":[-5.0,5.0],"interpolation":2}},"color":{"weight":1.0,"type":"ColorJitter","kwargs":{"brightness":0.2,"contrast":[0.8,1.2],"saturation":[0.8,1.2],"hue":0.05}}}' \
    --policy.type=molmoact2 \
    --policy.checkpoint_path=allenai/MolmoAct2 \
    --policy.device=cuda \
    --policy.action_mode=continuous \
    --policy.inference_action_mode=continuous \
    --policy.chunk_size=32 \
    --policy.n_action_steps=32 \
    --policy.setup_type="single so100/so101 robotic arm in molmoact2" \
    --policy.control_mode="absolute joint pose" \
    --policy.image_keys='["observation.images.front","observation.images.wrist"]' \
    --policy.model_dtype=bfloat16 \
    --policy.normalize_gripper=true \
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
    --policy.scheduler_decay_steps=40000 \
    --policy.scheduler_decay_lr=1e-6 \
    --policy.push_to_hub=true \
    --policy.repo_id="$HUB_REPO_ID" \
    --policy.private=false \
    --policy.tags='["molmoact2","so101","robocolosseum","cube-drawer"]' \
    --save_checkpoint=true \
    --save_freq=20000 \
    --save_checkpoint_to_hub=true \
    --steps=40000 \
    --batch_size=16 \
    --num_workers=4 \
    --log_freq=20 \
    --job_name="$JOB_NAME" \
    --wandb.enable=true \
    --wandb.project="$WANDB_PROJECT" \
    --wandb.entity="$WANDB_ENTITY" \
    --wandb.run_id="$JOB_NAME" \
    --wandb.disable_artifact=true \
    --output_dir="$OUTPUT_DIR"
