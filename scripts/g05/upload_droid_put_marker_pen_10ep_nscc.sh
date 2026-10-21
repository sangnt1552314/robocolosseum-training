#!/bin/bash
set -euo pipefail

# Upload the final G0.5 droid put-marker-pen NSCC checkpoint (8 x H200 run) to the HuggingFace Hub, then
# clean up locally. Same as upload_droid_put_marker_pen_10ep.sh with the NSCC paths (scripts/nscc/env.sh).
#
# G0.5 has no built-in push_to_hub, so the final (highest) checkpoints/step_<N>.pt is exported with
# GalaxeaVLA's tools/export_checkpoint.py (--strip: model weights only, plus .hydra/config.yaml and
# dataset_stats.json in the layout scripts/serve_policy.py expects) and uploaded to the repo root
# as the ready-to-use model. No checkpoints/step_<N>/ copies are uploaded (the 10ep runs save only
# the final step anyway).
#
# Cleanup: after every upload is verified on the Hub, the local step_*.pt files (weights +
# optimizer state), last.pt and the export staging copies are deleted. .hydra/, dataset_stats.json
# and the logs stay in the run dir. Set CLEANUP=false to keep everything.
#
# Usage:
#   bash scripts/g05/upload_droid_put_marker_pen_10ep_nscc.sh
#
# Requires HF_TOKEN in the environment or in the repo .env file.

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../nscc" && pwd)/env.sh"
G05_ENV="$G05_VENV"
export HF_HUB_OFFLINE=0

JOB_NAME="g05-droid-put-marker-pen-10ep-nscc"
RUN_DIR="$MODELS_ROOT/$JOB_NAME"
EXPORT_BASE="$RUN_DIR/_hf_export"
REPO_ID="$HF_HUB_ORG/$JOB_NAME"  # RoboColosseum/<job> (scripts/nscc/env.sh)
PRIVATE=false
CLEANUP="${CLEANUP:-true}"

# Load HF_TOKEN from the repo .env file if it is not already in the environment.
ENV_FILE="$ROBO_ROOT/.env"
if [ -z "${HF_TOKEN:-}" ] && [ -f "$ENV_FILE" ]; then
    set -a
    source "$ENV_FILE"
    set +a
fi

: "${HF_TOKEN:?HF_TOKEN not set; add it to $ENV_FILE or export it}"

test -d "$RUN_DIR/checkpoints" || { echo "No checkpoints dir: $RUN_DIR/checkpoints" >&2; exit 1; }

STEPS="$(find "$RUN_DIR/checkpoints" -maxdepth 1 -type f -name 'step_*.pt' \
    | sed 's#.*/step_##; s#\.pt$##' | sort -n | tr '\n' ' ')"
test -n "${STEPS// /}" || { echo "No step_*.pt checkpoints in $RUN_DIR/checkpoints" >&2; exit 1; }
FINAL_STEP="$(echo $STEPS | awk '{print $NF}')"

echo "Run dir: $RUN_DIR"
echo "Steps:   $STEPS(final: $FINAL_STEP)"
echo "Repo id: $REPO_ID (private=$PRIVATE)"

# Activate the g05 venv if not already active (no-op if already active).
if [ -d "$G05_ENV" ] && [ -z "${VIRTUAL_ENV:-}" ]; then
    source "$G05_ENV/bin/activate"
fi

mkdir -p "$EXPORT_BASE"

for STEP in $FINAL_STEP; do
    NAME="step_$(printf "%06d" "$STEP")"
    EXPORT_DIR="$EXPORT_BASE/$NAME"
    # export_checkpoint.py prompts before overwriting and stages under ./tmp, so start clean
    # and run it from the export dir to keep GalaxeaVLA untouched.
    rm -rf "$EXPORT_DIR"
    (cd "$EXPORT_BASE" && python "$G05_ROOT/tools/export_checkpoint.py" "$RUN_DIR" \
        --target-base "$EXPORT_BASE" \
        --name "$NAME" \
        --checkpoint "step_$STEP.pt" \
        --strip \
        -y)
done

# Upload the final package to the repo root, then verify before any cleanup.
python - "$REPO_ID" "$EXPORT_BASE" "$PRIVATE" "$FINAL_STEP" <<'PY'
import sys
from huggingface_hub import HfApi

repo_id, export_base, private, final_step = sys.argv[1], sys.argv[2], sys.argv[3].lower() == "true", sys.argv[4]

api = HfApi()
api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
expected = []
final_name = f"step_{int(final_step):06d}"
api.upload_folder(
    repo_id=repo_id,
    folder_path=f"{export_base}/{final_name}",
    repo_type="model",
    commit_message=f"Upload final G0.5 droid put-marker-pen model ({final_name})",
)
expected.append("checkpoints/model_state_dict.pt")
print(f"Uploaded final model ({final_name}) -> https://huggingface.co/{repo_id}")

files = set(api.list_repo_files(repo_id, repo_type="model"))
missing = [f for f in expected if f not in files]
if missing:
    sys.exit(f"Upload verification failed, missing on Hub: {missing}")
print("All uploads verified on the Hub.")
PY

if [ "$CLEANUP" = "true" ]; then
    echo "Cleaning up local checkpoint files..."
    rm -rf "$EXPORT_BASE"
    rm -f "$RUN_DIR"/checkpoints/step_*.pt "$RUN_DIR/last.pt"
    rmdir "$RUN_DIR/checkpoints" 2>/dev/null || true
fi

echo "Done."
