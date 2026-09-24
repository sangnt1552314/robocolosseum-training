
#!/bin/bash
set -euo pipefail

export HF_HOME=/scratch/e1583535/cache
export HF_DATASETS_CACHE=/scratch/e1583535/cache/datasets

export FFMPEG_ROOT=/scratch/e1583535/opt/ffmpeg-8.0.3
export PATH="$FFMPEG_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$FFMPEG_ROOT/lib:${LD_LIBRARY_PATH:-}"

# python -m lerobot.scripts.convert_dataset_v21_to_v30 \
#     --repo-id=liu-tao/so101_dataset

DATASET="liu-tao/so101_dataset"
OUTPUT_DIR="outputs/pi05_so101_smoke_test"

lerobot-train \
    --dataset.repo_id="$DATASET" \
    --dataset.video_backend=pyav \
    --policy.type=pi05 \
    --policy.pretrained_path=lerobot/pi05_base \
    --policy.device=cuda \
    --policy.normalization_mapping='{"ACTION":"MEAN_STD","STATE":"MEAN_STD","VISUAL":"IDENTITY"}' \
    --policy.freeze_vision_encoder=false \
    --policy.train_expert_only=false \
    --policy.gradient_checkpointing=true \
    --policy.dtype=bfloat16 \
    --policy.device=cuda \
    --policy.freeze_vision_encoder=false \
    --policy.train_expert_only=false \
    --policy.push_to_hub=false \
    --output_dir="$OUTPUT_DIR" \
    --job_name=pi05_so101_smoke_test \
    --batch_size=1 \
    --num_workers=0 \
    --steps=10 \
    --log_freq=1 \
    --save_checkpoint=true \
    --save_freq=0 \
    --wandb.enable=false \
    --seed=1000