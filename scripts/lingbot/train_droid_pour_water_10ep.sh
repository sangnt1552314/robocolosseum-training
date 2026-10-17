#!/bin/bash
set -euo pipefail

# LingBot-VLA v2 6B full fine-tuning on the DROID (Franka) pour-water dataset, following
# LingBot's real-world recipe (configs/vla/real_robot/real_robot.yaml + Training_Config.md):
# per-joint meanstd norm, state-relative arm actions / absolute gripper, Muon, LR 5e-5 constant,
# VLM + vision + action expert all trained (torch.compile off; the repo template turns it on).
# Cameras: left, right, wrist (robot_configs/droid_pour_water.yaml); 7 arm joints + gripper.
# 10 epochs = 28,710 steps at global batch 32 (8 x 4 grad-accum on 1 GPU). No intermediate checkpoints:
# only the final step is saved, and only that step is uploaded to the Hub afterwards by
# upload_droid_pour_water_10ep.sh.
#
# Norm stats are computed automatically on the first run if the JSON does not exist yet.

export HF_HOME=/scratch/e1583535/cache
export HF_DATASETS_CACHE=/scratch/e1583535/cache/datasets

export FFMPEG_ROOT=/scratch/e1583535/opt/ffmpeg-7.1
export PATH="$FFMPEG_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$FFMPEG_ROOT/lib:${LD_LIBRARY_PATH:-}"

# Triton links flex_attention kernels with -lcuda, which needs an unversioned
# libcuda.so. The container's compat libcuda.so.1 is a broken symlink under -e,
# so prefer the real host driver injected by --nv and expose a symlink to it.
_cuda_stub=/scratch/e1583535/tmp/cuda-stub-lib
_real_libcuda=""
# 1) The --nv injected driver (real file, always valid when GPUs are present).
if [ -e /.singularity.d/libs/libcuda.so.1 ]; then
    _real_libcuda=/.singularity.d/libs/libcuda.so.1
fi
# 2) An ldconfig-registered libcuda.so.1 that actually exists.
if [ -z "$_real_libcuda" ]; then
    _ldc_libcuda="$(/sbin/ldconfig -p 2>/dev/null | awk '/libcuda\.so\.1/ {print $NF}' || true)"
    for _f in $_ldc_libcuda; do
        if [ -e "$_f" ]; then _real_libcuda="$_f"; break; fi
    done
fi
# 3) Scan common driver directories.
if [ -z "$_real_libcuda" ]; then
    for _d in $(echo "${LD_LIBRARY_PATH:-}" | tr ':' ' ') /usr/local/nvidia/lib64 /usr/lib/x86_64-linux-gnu /usr/lib64; do
        if [ -e "$_d/libcuda.so.1" ]; then _real_libcuda="$_d/libcuda.so.1"; break; fi
    done
fi
if [ -n "$_real_libcuda" ]; then
    mkdir -p "$_cuda_stub"
    ln -sf "$_real_libcuda" "$_cuda_stub/libcuda.so"
    export TRITON_LIBCUDA_PATH="$_cuda_stub"
    echo "TRITON_LIBCUDA_PATH=$_cuda_stub -> $_real_libcuda"
else
    echo "WARNING: could not locate a working libcuda.so.1 for Triton" >&2
fi

LINGBOT_ROOT=/scratch/e1583535/projects/lingbot-vla-v2
ROBO_ROOT=/scratch/e1583535/projects/robocolosseum-training

CONFIG=$ROBO_ROOT/scripts/lingbot/configs/droid_pour_water_10ep.yaml
DATASET=/scratch/e1583535/datasets/droid-pour-water
NORM_STATS=$ROBO_ROOT/scripts/lingbot/norm_stats/droid_pour_water.json

VENV=/scratch/e1583535/virtualenvs/lingbot-vla-v2

JOB_NAME="lingbot-vla-v2-6b-droid-pour-water-10ep"
STEPS=28710  # = max_steps / save_steps in $CONFIG (10 epochs)
MODELS_ROOT="/scratch/Projects/CFP-05/CFP05-CF-002/robocolosseum-finetuned-models"
OUTPUT_DIR="$MODELS_ROOT/$JOB_NAME"
WANDB_PROJECT="RoboColosseum"
WANDB_ENTITY="tsangb34-national-university-of-singapore-students-union"
export WANDB_PROJECT WANDB_ENTITY
export WANDB_NAME="$JOB_NAME"

# Activate LingBot environment
source "$VENV/bin/activate"

# Use exactly one GPU
export CUDA_VISIBLE_DEVICES=0
export NPROC_PER_NODE=1

# Keep HF models local/offline during training
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

# Sanity checks
test -f "$CONFIG"
test -d "$DATASET"

# Norm stats need a GPU (compute_norm_stats.py initialises NCCL), so compute them here on the
# first run instead of on the login node. Later runs/resubmissions reuse the file.
if [ ! -f "$NORM_STATS" ]; then
    echo "Norm stats not found; computing $NORM_STATS"
    bash "$ROBO_ROOT/scripts/lingbot/compute_norm_stats_droid_pour_water.sh"
fi
test -f "$NORM_STATS"
test -d /scratch/e1583535/models/lingbot-vla-v2-6b
test -d /scratch/e1583535/models/Qwen3-VL-4B-Instruct
test -f /scratch/e1583535/models/moge-2-vitb-normal/model.pt

mkdir -p "$OUTPUT_DIR"

cd "$LINGBOT_ROOT"

echo "========================================"
echo "LingBot-VLA v2 full fine-tuning"
echo "Dataset: DROID pour water"
echo "Episodes: 100"
echo "Steps: $STEPS = 10 epochs (final checkpoint only)"
echo "GPU: 1"
echo "Micro batch: 8"
echo "Gradient accumulation: 4"
echo "Global batch: 32"
echo "========================================"

bash train.sh \
  tasks/vla/train_lingbotvla.py \
  "$CONFIG" \
  --train.use_wandb true \
  --train.wandb_project "$WANDB_PROJECT" \
  --train.wandb_name "$JOB_NAME" \
  --train.output_dir "$OUTPUT_DIR"

# Push the final checkpoint to the Hub (the trainer has no push_to_hub).
bash "$ROBO_ROOT/scripts/lingbot/upload_droid_pour_water_10ep.sh" "$STEPS"
