#!/bin/bash
set -euo pipefail

# MolmoAct2 recipe from the VLA-REPLICA paper (https://irvlutd.github.io/VLAReplica/):
# start from MolmoAct2-SO100_101, quantile normalization, batch 16, 1 GPU, 40K steps,
# chunk 32 / 32 action steps, default vision encoder (trainable) and default LRs.
# Checkpoints every 5K steps are pushed to the Hub and pruned locally, so re-submitting
# the job resumes after a crash or walltime kill.

export HF_HOME=/scratch/e1583535/cache
export HF_DATASETS_CACHE=/scratch/e1583535/cache/datasets

DATASET="Jiamo0912/robocolosseum-so101-stack-white_bowls-100episodes"
DATASET_ROOT="/scratch/e1583535/datasets/robocolosseum-so101-stack-white_bowls-100episodes"
OUTPUT_DIR="outputs/molmoact2-so101-stack-white_bowls-100episodes-vlareplica-40k"

HUB_REPO_ID="tsangb34/molmoact2-so101-stack-white_bowls-100episodes-vlareplica-40k"

# Number of finished checkpoints kept on local disk (all of them stay on the Hub).
export KEEP_LOCAL_CHECKPOINTS=2

JOB_NAME="molmoact2-so101-stack-white_bowls-100episodes-vlareplica-40k"
WANDB_PROJECT="RoboColosseum"
WANDB_ENTITY="tsangb34-national-university-of-singapore-students-union"

source "$(dirname "$0")/../utils/resumable_lerobot_train.sh"

# Resumes from the latest local/Hub checkpoint when re-submitted.
run_resumable_train "$OUTPUT_DIR" "$HUB_REPO_ID" \
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
    --policy.action_mode=both \
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
    --policy.train_action_expert_only=false \
    --policy.enable_knowledge_insulation=false \
    --policy.push_to_hub=true \
    --policy.repo_id="$HUB_REPO_ID" \
    --policy.private=false \
    --policy.tags='["molmoact2","so101","robocolosseum","stack-white_bowls","vlareplica"]' \
    --save_checkpoint=true \
    --save_freq=5000 \
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
