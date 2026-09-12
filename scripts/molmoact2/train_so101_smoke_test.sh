
#!/bin/bash
set -euo pipefail

export HF_HOME=/scratch/e1583535/cache
export HF_DATASETS_CACHE=/scratch/e1583535/cache/datasets

export FFMPEG_ROOT=/scratch/e1583535/opt/ffmpeg-8.0.3
export PATH="$FFMPEG_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$FFMPEG_ROOT/lib:${LD_LIBRARY_PATH:-}"

# python -m lerobot.scripts.convert_dataset_v21_to_v30 \
#     --repo-id=liu-tao/so101_dataset

lerobot-train \
    --dataset.repo_id=liu-tao/so101_dataset \
    --dataset.video_backend=pyav \
    --policy.type=molmoact2 \
    --policy.checkpoint_path=allenai/MolmoAct2-SO100_101 \
    --policy.device=cuda \
    --policy.image_keys='["observation.images.top","observation.images.wrist"]' \
    --policy.joint_signs='[1,-1,1,1,1,1]' \
    --policy.joint_offsets='[0,90,90,0,0,0]' \
    --policy.model_dtype=bfloat16 \
    --policy.normalize_gripper=true \
    --policy.normalization_mapping='{"ACTION":"MEAN_STD","STATE":"MEAN_STD","VISUAL":"IDENTITY"}' \
    --policy.gradient_checkpointing=true \
    --policy.push_to_hub=false \
    --steps=10 \
    --batch_size=1 \
    --num_workers=0 \
    --log_freq=1 \
    --save_checkpoint=true \
    --wandb.enable=false \
    --output_dir=outputs/molmoact2_so101_smoke_test