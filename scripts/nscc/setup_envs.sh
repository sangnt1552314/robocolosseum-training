#!/bin/bash
set -euo pipefail

# One-time environment setup for the NSCC ASPIRE 2B runs (login node: needs internet, no GPU).
# Builds everything under $TANSANG_ROOT (see scripts/nscc/env.sh). Each step is idempotent, so a
# failed step can simply be rerun.
#
# Usage:
#   bash scripts/nscc/setup_envs.sh [all|uv|ffmpeg|repos|lerobot|lingbot|g05|dataset|models ...]
#
#   uv       uv binary -> $TANSANG_ROOT/bin
#   ffmpeg   FFmpeg 7.1 shared libs (conda-forge, via a standalone micromamba) -> opt/ffmpeg-7.1
#   repos    lerobot v0.6.1, lingbot-vla-v2, GalaxeaVLA -> projects/
#   lerobot  venv for MolmoAct2 / pi05 / GR00T N1.7 (lerobot v0.6.1 [molmoact2,pi,groot,training,dataset]),
#            Python 3.12, torch cu128
#   leftwrist  datasets/droid-put-marker-pen-left-wrist (pi05 / GR00T camera layout, see tools/)
#   lingbot  venv for LingBot-VLA v2, mirroring tools/create_train_env.sh (conda -> uv), Python 3.12
#   g05      venv for G0.5, `uv sync` against GalaxeaVLA's uv.lock, Python 3.10
#   dataset  RoboColosseum/droid-put-marker-pen -> datasets/droid-put-marker-pen
#   models   base checkpoints that are loaded from local paths (LingBot, G0.5); MolmoAct2 is
#            pulled from the Hub into $HF_HOME at train time (pre-fetched here too)

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

mkdir -p "$TANSANG_ROOT"/{bin,projects,venvs,opt,models,datasets,cache,tmp}

# --torch-backend cu128 also pulls torchcodec's CUDA build (+cu128), which needs NVIDIA NPP
# (libnppicc.so.12) that no venv ships, so torchcodec fails to load. The frameworks decode video on
# the CPU, so replace it with the CPU wheel of the same version from PyPI.
cpu_torchcodec() {  # cpu_torchcodec <venv>
    local v
    v="$("$1/bin/python" -c 'import importlib.metadata as m; print(m.version("torchcodec").split("+")[0])')"
    uv pip install -p "$1" --index-url https://pypi.org/simple --no-deps --reinstall "torchcodec==$v"
}

step_uv() {
    if [ ! -x "$TANSANG_ROOT/bin/uv" ]; then
        curl -LsSf https://astral.sh/uv/install.sh \
            | env UV_INSTALL_DIR="$TANSANG_ROOT/bin" UV_NO_MODIFY_PATH=1 sh
    fi
    uv --version
    if [ ! -x "$VENVS_ROOT/hftools/bin/hf" ]; then
        uv venv -q -p 3.12 "$VENVS_ROOT/hftools"
        uv pip install -q -p "$VENVS_ROOT/hftools" "huggingface_hub[cli]>=0.35" hf_transfer
    fi
}

step_ffmpeg() {
    if [ -f "$FFMPEG_ROOT/lib/libavcodec.so" ]; then
        echo "FFmpeg already in $FFMPEG_ROOT"; return
    fi
    local mm="$TANSANG_ROOT/bin/micromamba"
    if [ ! -x "$mm" ]; then
        curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C "$TANSANG_ROOT" bin/micromamba
    fi
    MAMBA_ROOT_PREFIX="$TANSANG_ROOT/cache/mamba" "$mm" create -y -q -p "$FFMPEG_ROOT" \
        -c conda-forge "ffmpeg=7.1"
    "$FFMPEG_ROOT/bin/ffmpeg" -hide_banner -version | head -1
}

step_repos() {
    cd "$PROJECTS_ROOT"
    [ -d lerobot ] || git clone --depth 1 --branch v0.6.1 https://github.com/huggingface/lerobot.git lerobot
    [ -d lingbot-vla-v2 ] || git clone https://github.com/robbyant/lingbot-vla-v2.git
    [ -d GalaxeaVLA ] || git clone https://github.com/OpenGalaxea/GalaxeaVLA.git
    for r in lerobot lingbot-vla-v2 GalaxeaVLA; do
        echo "$r: $(git -C "$r" describe --tags --always)"
    done
}

step_lerobot() {
    [ -x "$LEROBOT_VENV/bin/python" ] || uv venv -q -p 3.12 "$LEROBOT_VENV"
    # torch from the cu128 index (H200 nodes); lerobot editable so the checkout is the source of truth.
    uv pip install -p "$LEROBOT_VENV" --torch-backend cu128 \
        -e "$LEROBOT_ROOT[molmoact2,pi,groot,training,dataset]"
    cpu_torchcodec "$LEROBOT_VENV"
    "$LEROBOT_VENV/bin/python" -c "import lerobot, torch, transformers, accelerate, wandb; \
print('lerobot', lerobot.__version__, 'torch', torch.__version__, torch.version.cuda, \
'transformers', transformers.__version__, 'accelerate', accelerate.__version__)"
    test -x "$LEROBOT_VENV/bin/lerobot-train"
}

step_lingbot() {
    # Same pins and order as lingbot-vla-v2/tools/create_train_env.sh, with uv instead of conda.
    local py="$LINGBOT_VENV/bin/python"
    local pip=(uv pip install -p "$LINGBOT_VENV")
    [ -x "$py" ] || uv venv -q -p 3.12 --seed "$LINGBOT_VENV"
    "${pip[@]}" setuptools wheel
    "${pip[@]}" --torch-backend cu128 \
        torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 torchdata==0.11.0 torchcodec==0.6.0
    "${pip[@]}" --torch-backend cu128 -r "$LINGBOT_ROOT/requirements.txt"
    "${pip[@]}" --no-deps numpydantic==1.9.0
    # Prebuilt flash-attn 2.8.3 wheel for torch 2.8 / cu12 / cp312 (avoids a long nvcc build).
    "${pip[@]}" --no-deps \
        "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
    "${pip[@]}" --no-deps "lerobot @ https://github.com/huggingface/lerobot/archive/refs/tags/v0.4.2.tar.gz"
    "${pip[@]}" --no-deps -e "$LINGBOT_ROOT"
    "${pip[@]}" --torch-backend cu128 -r "$LINGBOT_ROOT/requirements-depth.txt"
    "${pip[@]}" --torch-backend cu128 -r "$LINGBOT_ROOT/requirements.txt"
    "${pip[@]}" --no-deps numpydantic==1.9.0
    "$py" - "$LINGBOT_ROOT" <<'PY'
import site, sys
from pathlib import Path
pth = Path(site.getsitepackages()[0]) / "stablevla_local_depth.pth"
pth.write_text(f"{sys.argv[1]}/lingbotvla/models/vla/vision_models/morgbd_clean/3rd/utils3d\n")
print("wrote", pth)
PY
    "${pip[@]}" --no-deps -e "$LINGBOT_ROOT/lingbotvla/models/vla/vision_models/lingbot-depth"
    "${pip[@]}" --torch-backend cu128 -e "$LINGBOT_ROOT/lingbotvla/models/vla/vision_models/MoGe"
    # MoGe pulls unpinned deps; restore the training pins (as the upstream script does).
    "${pip[@]}" --torch-backend cu128 -r "$LINGBOT_ROOT/requirements.txt"
    "${pip[@]}" --no-deps numpydantic==1.9.0
    "${pip[@]}" huggingface_hub==0.34.0
    cpu_torchcodec "$LINGBOT_VENV"
    "$py" -c "import torch, flash_attn, cv2, accelerate, mlflow, trimesh, moge, mdm, utils3d, lingbotvla, wandb; \
print('torch', torch.__version__, torch.version.cuda, 'flash_attn', flash_attn.__version__); print('depth imports ok')"
}

step_g05() {
    # GalaxeaVLA's own lockfile (torch 2.7.1 cu128, flash-attn-4, fla); the venv lives outside the checkout.
    cd "$G05_ROOT"
    UV_PROJECT_ENVIRONMENT="$G05_VENV" uv sync --frozen --index-strategy unsafe-best-match
    cpu_torchcodec "$G05_VENV"
    "$G05_VENV/bin/python" -c "import torch, hydra, wandb; print('torch', torch.__version__, torch.version.cuda)"
}

step_dataset() {
    local dst="$DATASETS_ROOT/droid-put-marker-pen"
    "$VENVS_ROOT/hftools/bin/hf" download RoboColosseum/droid-put-marker-pen \
        --repo-type dataset --local-dir "$dst"
    python3 -c "import json; d=json.load(open('$dst/meta/info.json')); \
print('episodes', d['total_episodes'], 'frames', d['total_frames'])"
}

step_leftwrist() {
    # Left + wrist copy used by pi05 and GR00T (right camera removed, left listed before wrist).
    "$LEROBOT_VENV/bin/python" "$ROBO_ROOT/tools/make_droid_left_wrist_dataset.py" \
        "$DATASETS_ROOT/droid-put-marker-pen" "$DATASETS_ROOT/droid-put-marker-pen-left-wrist"
}

step_models() {
    local hf="$VENVS_ROOT/hftools/bin/hf"
    export HF_XET_HIGH_PERFORMANCE=1
    # LeRobot policies load these from the Hub cache ($HF_HOME) at train time; pre-fetch them.
    "$hf" download allenai/MolmoAct2
    "$hf" download lerobot/pi05_base
    "$hf" download google/paligemma-3b-pt-224      # pi05 tokenizer (gated; tsangb34 has access)
    "$hf" download nvidia/GR00T-N1.7-3B
    "$hf" download nvidia/Cosmos-Reason2-2B        # GR00T N1.7 backbone / processor (gated)
    # LingBot-VLA v2: local paths in scripts/lingbot/configs/*_nscc.yaml.
    "$hf" download robbyant/lingbot-vla-v2-6b --local-dir "$MODELS_DIR/lingbot-vla-v2-6b"
    "$hf" download Qwen/Qwen3-VL-4B-Instruct --local-dir "$MODELS_DIR/Qwen3-VL-4B-Instruct"
    "$hf" download Ruicheng/moge-2-vitb-normal --local-dir "$MODELS_DIR/moge-2-vitb-normal"
    # G0.5: the checked-in configs expect $G05_ROOT/checkpoints/{g05-base,action_tokenizer.pt,
    # qwen3_5_2b_base_processor} (README "Model Checkpoints"); only what fine-tuning needs.
    "$hf" download OpenGalaxea/G05 --local-dir "$G05_ROOT/checkpoints" \
        --include "g05-base/*" --include "action_tokenizer.pt" --include "qwen3_5_2b_base_processor/*"
}

STEPS=("$@")
[ ${#STEPS[@]} -gt 0 ] || STEPS=(all)
if [ "${STEPS[0]}" = all ]; then
    STEPS=(uv ffmpeg repos dataset lerobot leftwrist lingbot g05 models)
fi
for s in "${STEPS[@]}"; do
    echo "==================== setup: $s ===================="
    "step_$s"
done
echo "Setup done: ${STEPS[*]}"
