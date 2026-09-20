#!/bin/bash
set -euo pipefail

# Manual upload of the fine-tuned G0.5 (GalaxeaVLA) checkpoint to the HuggingFace Hub.
#
# G0.5 has no built-in push_to_hub (unlike the lerobot policies), so we export the
# final checkpoint with tools/export_checkpoint.py and upload the resulting package.
#
# Usage:
#   bash scripts/g05/upload_so101_stack_white_bowls_100episodes.sh [RUN_DIR]
#
#   RUN_DIR  optional. Defaults to the newest timestamped run under
#            outputs/g05/so101_stack_white_bowls/.
#
# Requires HF_TOKEN in the environment (the PBS job exports it from .env).

ROBO_ROOT=/scratch/e1583535/projects/robocolosseum-training
G05_ROOT=/scratch/e1583535/projects/GalaxeaVLA
G05_ENV=/scratch/e1583535/virtualenvs/g05

export HF_HOME=/scratch/e1583535/cache

RUN_BASE="$ROBO_ROOT/outputs/g05/so101_stack_white_bowls"
EXPORT_BASE="$ROBO_ROOT/outputs/g05/_hf_export"
REPO_ID="tsangb34/g05-so101-stack-white_bowls-100episodes"
PRIVATE=true

# Load HF_TOKEN from the repo .env file if it is not already in the environment.
ENV_FILE="$ROBO_ROOT/.env"
if [ -z "${HF_TOKEN:-}" ] && [ -f "$ENV_FILE" ]; then
    set -a
    source "$ENV_FILE"
    set +a
fi

: "${HF_TOKEN:?HF_TOKEN not set; add it to $ENV_FILE or export it}"

# Resolve run dir: explicit arg, else newest timestamped dir under RUN_BASE.
RUN_DIR="${1:-}"
if [ -z "$RUN_DIR" ]; then
    RUN_DIR="$(find "$RUN_BASE" -mindepth 1 -maxdepth 1 -type d | sort | tail -n1)"
fi
test -n "$RUN_DIR" || { echo "No run directory found under $RUN_BASE" >&2; exit 1; }
test -d "$RUN_DIR"

EXPORT_DIR="$EXPORT_BASE/$(basename "$RUN_DIR")"

echo "Run dir:   $RUN_DIR"
echo "Export to: $EXPORT_DIR"
echo "Repo id:   $REPO_ID (private=$PRIVATE)"

# Activate the g05 venv if not already active (no-op if already active).
if [ -d "$G05_ENV" ] && [ -z "${VIRTUAL_ENV:-}" ]; then
    source "$G05_ENV/bin/activate"
fi

# export_checkpoint.py prompts before overwriting an existing target; clear it first.
rm -rf "$EXPORT_DIR"

# Package the final checkpoint (highest step) + .hydra/config.yaml + dataset_stats.json
# + action_tokenizer.pt, stripping optimizer/scheduler/ema so only model weights remain.
cd "$G05_ROOT"
python tools/export_checkpoint.py "$RUN_DIR" \
    --target-base "$EXPORT_BASE" \
    --name "$(basename "$RUN_DIR")" \
    --strip \
    -y

# Create (if needed) and upload the exported package to the Hub.
python - "$REPO_ID" "$EXPORT_DIR" "$PRIVATE" <<'PY'
import sys
from huggingface_hub import HfApi

repo_id, folder, private = sys.argv[1], sys.argv[2], sys.argv[3].lower() == "true"
api = HfApi()
api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
api.upload_folder(
    repo_id=repo_id,
    folder_path=folder,
    repo_type="model",
    commit_message="Upload G0.5 fine-tuned so101 stack-white_bowls checkpoint",
)
print(f"Uploaded {folder} -> https://huggingface.co/{repo_id}")
PY

echo "Done."
