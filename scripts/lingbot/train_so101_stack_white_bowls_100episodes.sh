#!/bin/bash
set -euo pipefail

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

CONFIG=$ROBO_ROOT/scripts/lingbot/configs/so101_stack_white_bowls_100episodes.yaml
DATASET=/scratch/e1583535/datasets/robocolosseum-so101-stack-white_bowls-100episodes
NORM_STATS=$ROBO_ROOT/scripts/lingbot/norm_stats/so101_stack_white_bowls_100episodes.json

VENV=/scratch/e1583535/virtualenvs/lingbot-vla-v2

JOB_NAME="lingbot-vla-6b-so101-stack-white_bowls-100episodes"
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
test -f "$NORM_STATS"
test -d /scratch/e1583535/models/lingbot-vla-v2-6b
test -d /scratch/e1583535/models/Qwen3-VL-4B-Instruct
test -f /scratch/e1583535/models/moge-2-vitb-normal/model.pt

mkdir -p "$ROBO_ROOT/outputs/lingbot/so101_stack_white_bowls_100episodes"

cd "$LINGBOT_ROOT"

echo "========================================"
echo "LingBot-VLA v2 full fine-tuning"
echo "Dataset: SO-101 stack white bowls"
echo "Episodes: 100"
echo "Epochs: 6"
echo "GPU: 1"
echo "Micro batch: 2"
echo "Gradient accumulation: 16"
echo "Global batch: 32"
echo "========================================"

bash train.sh \
  tasks/vla/train_lingbotvla.py \
  "$CONFIG" \
  --train.use_wandb true \
  --train.wandb_project "$WANDB_PROJECT" \
  --train.wandb_name "$JOB_NAME"