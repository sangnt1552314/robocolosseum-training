#!/bin/bash
set -euo pipefail

export HF_HOME=/scratch/e1583535/cache
export HF_DATASETS_CACHE=/scratch/e1583535/cache/datasets

DATASET="tsangb34/robocolosseum-so101-soccer-red_bowl-40episodes"
OUTPUT_DIR="outputs/molmoact2-so101-soccer-red_bowl-40episodes"

HUB_REPO_ID="tsangb34/molmoact2-so101-soccer-red_bowl-40episodes"

JOB_NAME="molmoact2-so101-soccer-red_bowl-40episodes"
WANDB_PROJECT="RoboColosseum"
WANDB_ENTITY="tsangb34-national-university-of-singapore-students-union"

lerobot-train \
    --dataset.repo_id="$DATASET" \
    --dataset.video_backend=pyav \
    --dataset.image_transforms.enable=true \
    --dataset.image_transforms.max_num_transforms=3 \
    --dataset.image_transforms.random_order=false \
    --dataset.image_transforms.tfs='{"crop":{"weight":1.0,"type":"RandomResizedCrop","kwargs":{"size":[256,256],"scale":[0.9025,0.9025],"ratio":[1.0,1.0]}},"rotation":{"weight":1.0,"type":"RandomRotation","kwargs":{"degrees":[-5.0,5.0],"interpolation":2}},"color":{"weight":1.0,"type":"ColorJitter","kwargs":{"brightness":0.2,"contrast":[0.8,1.2],"saturation":[0.8,1.2],"hue":0.05}}}' \
    --policy.type=molmoact2 \
    --policy.checkpoint_path=allenai/MolmoAct2 \
    --policy.device=cuda \
    --policy.action_mode=both \
    --policy.inference_action_mode=continuous \
    --policy.discrete_action_tokenizer=allenai/MolmoAct2-FAST-Tokenizer \
    --policy.chunk_size=30 \
    --policy.n_action_steps=30 \
    --policy.setup_type="single so100/so101 robotic arm in molmoact2" \
    --policy.control_mode="absolute joint pose" \
    --policy.image_keys='["observation.images.front","observation.images.wrist"]' \
    --policy.joint_signs='[1,-1,1,1,1,1]' \
    --policy.joint_offsets='[0,90,90,0,0,0]' \
    --policy.model_dtype=bfloat16 \
    --policy.normalize_gripper=true \
    --policy.normalization_mapping='{"ACTION":"MEAN_STD","STATE":"MEAN_STD","VISUAL":"IDENTITY"}' \
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
    --policy.scheduler_warmup_steps=40 \
    --policy.scheduler_decay_steps=921 \
    --policy.scheduler_decay_lr=1e-6 \
    --policy.push_to_hub=true \
    --policy.repo_id="$HUB_REPO_ID" \
    --policy.private=false \
    --policy.tags='["molmoact2","so101","robocolosseum","soccer-red_bowl"]' \
    --save_checkpoint_to_hub=false \
    --steps=921 \
    --batch_size=16 \
    --num_workers=4 \
    --log_freq=20 \
    --save_checkpoint=true \
    --save_freq=921 \
    --job_name="$JOB_NAME" \
    --wandb.enable=true \
    --wandb.project="$WANDB_PROJECT" \
    --wandb.entity="$WANDB_ENTITY" \
    --output_dir="$OUTPUT_DIR"