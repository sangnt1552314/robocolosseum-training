#!/bin/bash
set -euo pipefail

# Manual upload of the fine-tuned LingBot-VLA v2 6B checkpoint to the HuggingFace Hub.
#
# LingBot-VLA has no built-in push_to_hub, but the trainer already writes
# deployable HuggingFace-format weights to `hf_ckpt/` inside every checkpoint
# (config.json + model-0000N-of-00006.safetensors + tokenizer files). So there
# is no export step: we just upload the newest checkpoint's hf_ckpt plus the
# run-level config, mirroring robbyant/lingbot-vla-v2-6b-robotwin:
#
#   checkpoints/global_step_<N>/hf_ckpt/...   <- HF-format model weights
#   lingbotvla_cli.yaml                       <- resolved training config
#   model_assets/...                          <- tokenizer / processor assets
#
# The bulky DCP shards (model/, optimizer/, extra_state/) are intentionally
# skipped; they are only needed to resume training, not for inference.
#
# Usage:
#   bash scripts/lingbot/upload_so101_stack_white_bowls_100episodes.sh [STEP]
#
#   STEP  optional. Global step number to upload (e.g. 2340). Defaults to the
#         highest global_step_N under the run's checkpoints/ directory.
#
# Requires HF_TOKEN in the environment or in the repo .env file.

ROBO_ROOT=/scratch/e1583535/projects/robocolosseum-training
LINGBOT_ENV=/scratch/e1583535/virtualenvs/lingbot-vla-v2

export HF_HOME=/scratch/e1583535/cache

# The Xet upload backend throws "I/O error (os error 2)" on this filesystem;
# force the classic LFS multipart uploader instead.
export HF_HUB_DISABLE_XET=1

RUN_DIR="$ROBO_ROOT/outputs/lingbot/so101_stack_white_bowls_100episodes"
CKPT_DIR="$RUN_DIR/checkpoints"
REPO_ID="tsangb34/lingbot-vla-v2-6b-so101-stack-white_bowls-100episodes"
PRIVATE=false

# Load HF_TOKEN from the repo .env file if it is not already in the environment.
ENV_FILE="$ROBO_ROOT/.env"
if [ -z "${HF_TOKEN:-}" ] && [ -f "$ENV_FILE" ]; then
    set -a
    source "$ENV_FILE"
    set +a
fi

: "${HF_TOKEN:?HF_TOKEN not set; add it to $ENV_FILE or export it}"

test -d "$CKPT_DIR" || { echo "No checkpoints dir: $CKPT_DIR" >&2; exit 1; }

# Resolve the step: explicit arg, else the highest global_step_N on disk.
STEP="${1:-}"
if [ -z "$STEP" ]; then
    STEP="$(find "$CKPT_DIR" -mindepth 1 -maxdepth 1 -type d -name 'global_step_*' \
        | sed 's#.*/global_step_##' | sort -n | tail -n1)"
fi
test -n "$STEP" || { echo "No global_step_* checkpoints found in $CKPT_DIR" >&2; exit 1; }

HF_CKPT_DIR="$CKPT_DIR/global_step_$STEP/hf_ckpt"
test -d "$HF_CKPT_DIR" || { echo "Missing hf_ckpt for step $STEP: $HF_CKPT_DIR" >&2; exit 1; }

echo "Run dir:   $RUN_DIR"
echo "Uploading: global_step_$STEP/hf_ckpt"
echo "Repo id:   $REPO_ID (private=$PRIVATE)"

# Activate the lingbot venv if not already active (no-op if already active).
if [ -d "$LINGBOT_ENV" ] && [ -z "${VIRTUAL_ENV:-}" ]; then
    source "$LINGBOT_ENV/bin/activate"
fi

# Create (if needed) and upload only the selected files, preserving repo layout.
python - "$REPO_ID" "$RUN_DIR" "$STEP" "$PRIVATE" <<'PY'
import sys
from huggingface_hub import HfApi

repo_id, run_dir, step, private = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4].lower() == "true"

# Upload the newest checkpoint's HF weights + run config; skip DCP/optimizer shards.
allow_patterns = [
    f"checkpoints/global_step_{step}/hf_ckpt/*",
    "lingbotvla_cli.yaml",
    "model_assets/*",
]

api = HfApi()
api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
api.upload_folder(
    repo_id=repo_id,
    folder_path=run_dir,
    repo_type="model",
    allow_patterns=allow_patterns,
    commit_message=f"Upload LingBot-VLA v2 6B so101 stack-white_bowls checkpoint (global_step_{step})",
)
print(f"Uploaded global_step_{step} -> https://huggingface.co/{repo_id}")
PY

echo "Done."
