#!/bin/bash
set -euo pipefail

# Upload the fine-tuned LingBot-VLA v2 6B stack-bowls run to the HuggingFace Hub.
#
# LingBot-VLA has no built-in push_to_hub, but every save (global_step_<N>, every 3,920 steps)
# already contains deployable HuggingFace-format weights under hf_ckpt/. We upload each of them
# plus the run-level config, mirroring robbyant/lingbot-vla-v2-6b-robotwin:
#
#   checkpoints/global_step_<N>/hf_ckpt/...   <- HF-format model weights (one per saved step)
#   lingbotvla_cli.yaml                       <- resolved training config
#   model_assets/...                          <- tokenizer / processor assets
#
# The bulky DCP shards (model/, optimizer/, extra_state/) are intentionally skipped; they are
# only needed to resume training, not for inference.
#
# Usage:
#   bash scripts/lingbot/upload_so101_stack_bowls_10ep.sh [STEP]
#
#   STEP  optional. Upload only global_step_<STEP>. Defaults to every global_step_N on disk.
#
# Requires HF_TOKEN in the environment or in the repo .env file.

ROBO_ROOT=/scratch/e1583535/projects/robocolosseum-training
LINGBOT_ENV=/scratch/e1583535/virtualenvs/lingbot-vla-v2

export HF_HOME=/scratch/e1583535/cache

# Training runs with the Hub offline; the upload needs it online.
export HF_HUB_OFFLINE=0

# The Xet upload backend throws "I/O error (os error 2)" on this filesystem;
# force the classic LFS multipart uploader instead.
export HF_HUB_DISABLE_XET=1

JOB_NAME="lingbot-vla-v2-6b-so101-stack-bowls-10ep"
RUN_DIR="/scratch/Projects/CFP-05/CFP05-CF-002/robocolosseum-finetuned-models/$JOB_NAME"
CKPT_DIR="$RUN_DIR/checkpoints"
REPO_ID="tsangb34/$JOB_NAME"
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

# Resolve the steps: explicit arg, else every global_step_N that has hf_ckpt weights.
if [ -n "${1:-}" ]; then
    STEPS="$1"
else
    STEPS="$(find "$CKPT_DIR" -mindepth 2 -maxdepth 2 -type d -path '*/global_step_*/hf_ckpt' \
        | sed 's#.*/global_step_##; s#/hf_ckpt##' | sort -n | tr '\n' ' ')"
fi
test -n "${STEPS// /}" || { echo "No global_step_*/hf_ckpt found in $CKPT_DIR" >&2; exit 1; }
for STEP in $STEPS; do
    test -d "$CKPT_DIR/global_step_$STEP/hf_ckpt" || { echo "Missing hf_ckpt for step $STEP" >&2; exit 1; }
done

echo "Run dir:   $RUN_DIR"
echo "Uploading: global_step_{${STEPS% }}/hf_ckpt"
echo "Repo id:   $REPO_ID (private=$PRIVATE)"

# Activate the lingbot venv if not already active (no-op if already active).
if [ -d "$LINGBOT_ENV" ] && [ -z "${VIRTUAL_ENV:-}" ]; then
    source "$LINGBOT_ENV/bin/activate"
fi

# Create (if needed) and upload only the selected files, preserving repo layout.
python - "$REPO_ID" "$RUN_DIR" "$PRIVATE" $STEPS <<'PY'
import sys
from huggingface_hub import HfApi

repo_id, run_dir, private, steps = sys.argv[1], sys.argv[2], sys.argv[3].lower() == "true", sys.argv[4:]

api = HfApi()
api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
for step in steps:
    # One commit per checkpoint, like lerobot's per-checkpoint Hub pushes.
    api.upload_folder(
        repo_id=repo_id,
        folder_path=run_dir,
        repo_type="model",
        allow_patterns=[
            f"checkpoints/global_step_{step}/hf_ckpt/*",
            "lingbotvla_cli.yaml",
            "model_assets/*",
        ],
        commit_message=f"Upload LingBot-VLA v2 6B so101 stack-bowls checkpoint (global_step_{step})",
    )
    print(f"Uploaded global_step_{step} -> https://huggingface.co/{repo_id}")
PY

echo "Done."
