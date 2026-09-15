
#!/bin/bash
set -euo pipefail

export HF_HOME=/scratch/e1583535/cache
export HF_DATASETS_CACHE=/scratch/e1583535/cache/datasets

export FFMPEG_ROOT=/scratch/e1583535/opt/ffmpeg-8.0.3
export PATH="$FFMPEG_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$FFMPEG_ROOT/lib:${LD_LIBRARY_PATH:-}"

DATASET="tsangb34/robocolosseum-so101-soccer-red_bowl-40episodes"
OUTPUT_DIR="outputs/molmoact2-so101-soccer-red_bowl-40episodes"

# Hub push target (the trained policy repo name). Change to your namespace/name.
HUB_REPO_ID="tsangb34/molmoact2-so101-soccer-red_bowl-40episodes"

# wandb run identity
JOB_NAME="molmoact2-so101-soccer-red_bowl-40episodes"
WANDB_PROJECT="RoboColosseum"
WANDB_ENTITY="tsangb34-national-university-of-singapore-students-union"

lerobot-train \
    --dataset.repo_id="$DATASET" \
    --dataset.video_backend=ffmpeg \
    --policy.type=molmoact2 \
    --policy.checkpoint_path=allenai/MolmoAct2 \
    --policy.device=cuda \
    --policy.image_keys='["observation.images.top","observation.images.wrist"]' \
    --policy.joint_signs='[1,-1,1,1,1,1]' \
    --policy.joint_offsets='[0,90,90,0,0,0]' \
    --policy.model_dtype=bfloat16 \
    --policy.normalize_gripper=true \
    --policy.normalization_mapping='{"ACTION":"MEAN_STD","STATE":"MEAN_STD","VISUAL":"IDENTITY"}' \
    --policy.gradient_checkpointing=true \
    --policy.num_flow_timesteps=8 \
    --policy.freeze_embedding=true \
    --policy.optimizer_lr=1e-5 \
    --policy.optimizer_vit_lr=5e-6 \
    --policy.optimizer_connector_lr=5e-6 \
    --policy.optimizer_action_expert_lr=5e-5 \
    --policy.scheduler_warmup_steps=200 \
    --policy.scheduler_decay_steps=5000 \
    --policy.scheduler_decay_lr=1e-6 \
    --policy.push_to_hub=true \
    --policy.repo_id="$HUB_REPO_ID" \
    --policy.private=true \
    --policy.tags='["molmoact2","so101","robocolosseum","soccer-red_bowl"]' \
    --save_checkpoint_to_hub=false \
    --steps=5000 \
    --batch_size=16 \
    --num_workers=4 \
    --log_freq=20 \
    --save_checkpoint=true \
    --save_freq=1000 \
    --job_name="$JOB_NAME" \
    --wandb.enable=true \
    --wandb.project="$WANDB_PROJECT" \
    --wandb.entity="$WANDB_ENTITY" \
    --wandb.run_id="$JOB_NAME" \
    --output_dir="$OUTPUT_DIR"