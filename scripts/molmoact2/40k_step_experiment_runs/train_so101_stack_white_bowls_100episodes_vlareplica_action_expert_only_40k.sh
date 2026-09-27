#!/bin/bash
set -euo pipefail

# Ablation of train_so101_stack_white_bowls_100episodes_vlareplica_40k.sh: same VLA-REPLICA recipe,
# but only the action expert is trained (train_action_expert_only=true). Every parameter whose name
# does not contain "action_expert" (ViT, connector, LLM, embeddings, lm_head) is frozen and the VLM is
# kept in eval mode. train_action_expert_only requires action_mode=continuous, so the discrete
# FAST-token loss is dropped; with the VLM frozen it would have no trainable parameters anyway.
# Checkpoints are saved every 10K steps and each is pushed to the Hub under checkpoints/<step>/.

export HF_HOME=/scratch/e1583535/cache
export HF_DATASETS_CACHE=/scratch/e1583535/cache/datasets

export FFMPEG_ROOT=/scratch/e1583535/opt/ffmpeg-8.0.3
export PATH="$FFMPEG_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$FFMPEG_ROOT/lib:${LD_LIBRARY_PATH:-}"

# Keep wandb staging/cache off the 40 GB home quota.
export WANDB_DATA_DIR=/scratch/e1583535/cache/wandb/data
export WANDB_CACHE_DIR=/scratch/e1583535/cache/wandb/cache
export WANDB_ARTIFACT_DIR=/scratch/e1583535/cache/wandb/artifacts

DATASET="Jiamo0912/robocolosseum-so101-stack-white_bowls-100episodes"
DATASET_ROOT="/scratch/e1583535/datasets/robocolosseum-so101-stack-white_bowls-100episodes"

JOB_NAME="molmoact2-so101-stack-white_bowls-100episodes-vlareplica-action-expert-only-40k"
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
    --dataset.image_transforms.tfs='{"crop":{"weight":1.0,"type":"RandomResizedCrop","kwargs":{"size":[256,256],"scale":[0.9025,0.9025],"ratio":[1.0,1.0]}},"rotation":{"weight":1.0,"type":"RandomRotation","kwargs":{"degrees":[-5.0,5.0],"interpolation":2}},"color":{"weight":1.0,"type":"ColorJitter","kwargs":{"brightness":0.2,"contrast":[0.8,1.2],"saturation":[0.8,1.2],"hue":0.05}}}' \
    --policy.type=molmoact2 \
    --policy.checkpoint_path=allenai/MolmoAct2-SO100_101 \
    --policy.device=cuda \
    --policy.action_mode=continuous \
    --policy.inference_action_mode=continuous \
    --policy.discrete_action_tokenizer=allenai/MolmoAct2-FAST-Tokenizer \
    --policy.chunk_size=32 \
    --policy.n_action_steps=32 \
    --policy.setup_type="single so100/so101 robotic arm in molmoact2" \
    --policy.control_mode="absolute joint pose" \
    --policy.image_keys='["observation.images.front","observation.images.wrist"]' \
    --policy.joint_signs='[1,-1,1,1,1,1]' \
    --policy.joint_offsets='[0,90,90,0,0,0]' \
    --policy.model_dtype=bfloat16 \
    --policy.normalize_gripper=true \
    --policy.normalization_mapping='{"ACTION":"QUANTILES","STATE":"QUANTILES","VISUAL":"IDENTITY"}' \
    --policy.gradient_checkpointing=true \
    --policy.num_flow_timesteps=8 \
    --policy.freeze_embedding=true \
    --policy.enable_lora_vlm=false \
    --policy.enable_lora_action_expert=false \
    --policy.train_action_expert_only=true \
    --policy.enable_knowledge_insulation=false \
    --policy.push_to_hub=true \
    --policy.repo_id="$HUB_REPO_ID" \
    --policy.private=false \
    --policy.tags='["molmoact2","so101","robocolosseum","stack-white_bowls"]' \
    --save_checkpoint=true \
    --save_freq=10000 \
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
