#!/bin/bash
set -euo pipefail

# pi05 recipe from the VLA-REPLICA paper (https://irvlutd.github.io/VLAReplica/):
# quantile normalization, batch 16, 1 GPU, 40K steps, chunk 32 / 32 action steps,
# default vision encoder (trainable) and default LR schedule.
# Checkpoints are saved at 20K and 40K and each is pushed to the Hub under checkpoints/<step>/.

export HF_HOME=/scratch/e1583535/cache
export HF_DATASETS_CACHE=/scratch/e1583535/cache/datasets

# Keep wandb staging/cache off the 40 GB home quota.
export WANDB_DATA_DIR=/scratch/e1583535/cache/wandb/data
export WANDB_CACHE_DIR=/scratch/e1583535/cache/wandb/cache
export WANDB_ARTIFACT_DIR=/scratch/e1583535/cache/wandb/artifacts

export FFMPEG_ROOT=/scratch/e1583535/opt/ffmpeg-8.0.3
export PATH="$FFMPEG_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$FFMPEG_ROOT/lib:${LD_LIBRARY_PATH:-}"

DATASET="Jiamo0912/robocolosseum-so101-black-screwdriver-box-to-table-100episodes"
DATASET_ROOT="/scratch/e1583535/datasets/robocolosseum-so101-black-screwdriver-box-to-table-100episodes"

JOB_NAME="pi05-so101-black-screwdriver-box-to-table-100episodes-vlareplica-40k"
MODELS_ROOT="/scratch/Projects/CFP-05/CFP05-CF-002/robocolosseum-finetuned-models"
OUTPUT_DIR="$MODELS_ROOT/$JOB_NAME"

HUB_REPO_ID="tsangb34/$JOB_NAME"

WANDB_PROJECT="RoboColosseum"
WANDB_ENTITY="tsangb34-national-university-of-singapore-students-union"

# save_freq=20000 writes checkpoints at 20K and 40K (the final step); save_checkpoint_to_hub
# pushes each one to <repo>/checkpoints/<step>/ (tagged with the step), and
# push_to_hub uploads the final model once training ends.
lerobot-train \
    --dataset.repo_id="$DATASET" \
    --dataset.root="$DATASET_ROOT" \
    --dataset.video_backend=pyav \
    --policy.type=pi05 \
    --policy.pretrained_path=lerobot/pi05_base \
    --policy.device=cuda \
    --policy.dtype=bfloat16 \
    --policy.normalization_mapping='{"ACTION":"QUANTILES","STATE":"QUANTILES","VISUAL":"IDENTITY"}' \
    --policy.chunk_size=32 \
    --policy.n_action_steps=32 \
    --policy.empty_cameras=1 \
    --policy.freeze_vision_encoder=false \
    --policy.train_expert_only=false \
    --policy.gradient_checkpointing=false \
    --policy.compile_model=false \
    --policy.push_to_hub=true \
    --policy.repo_id="$HUB_REPO_ID" \
    --policy.private=false \
    --output_dir="$OUTPUT_DIR" \
    --job_name="$JOB_NAME" \
    --batch_size=16 \
    --num_workers=4 \
    --steps=40000 \
    --log_freq=20 \
    --save_checkpoint=true \
    --save_freq=20000 \
    --save_checkpoint_to_hub=true \
    --wandb.enable=true \
    --wandb.project="$WANDB_PROJECT" \
    --wandb.entity="$WANDB_ENTITY" \
    --wandb.run_id="$JOB_NAME" \
    --wandb.disable_artifact=true

# Keep only the model weights/config/processors; drop the optimizer/scheduler state.
# rm -rf "$OUTPUT_DIR"/checkpoints/*/training_state
# echo "Final model: $(readlink -f "$OUTPUT_DIR/checkpoints/last")/pretrained_model"
