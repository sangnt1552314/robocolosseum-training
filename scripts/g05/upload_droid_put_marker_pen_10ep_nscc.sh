#!/bin/bash
set -euo pipefail

# Upload the final G0.5 droid put-marker-pen NSCC checkpoint (8 x H200 run) to the HuggingFace Hub, then
# clean up locally. Same as upload_droid_put_marker_pen_10ep.sh with the NSCC paths (scripts/nscc/env.sh).
#
# G0.5 has no built-in push_to_hub, so the final (highest) checkpoints/step_<N>.pt is exported with
# GalaxeaVLA's tools/export_checkpoint.py (--strip: model weights only, plus .hydra/config.yaml and
# dataset_stats.json in the layout scripts/serve_policy.py expects) and uploaded to the repo root
# as the ready-to-use model. No checkpoints/step_<N>/ copies are uploaded (the 10ep runs save only
# the final step anyway). The repo is private, like every repo in the RoboColosseum org.
#
# Cleanup: after the upload is verified on the Hub (file present with the exported size), the local
# step_*.pt files, last.pt and the export staging copies are deleted. .hydra/, dataset_stats.json and
# the logs stay in the run dir. Set CLEANUP=false to keep everything.
#
# Safe to rerun (e.g. as a fallback after the job): if the local checkpoint is already gone because a
# previous run uploaded and cleaned up, it only confirms the model is on the Hub.
#
# Usage:
#   bash scripts/g05/upload_droid_put_marker_pen_10ep_nscc.sh
#   EXPORT_STRIP=false bash ...   copy the checkpoint instead of loading it (the final checkpoint is
#                                 already inference-only; use this on the 16 GiB login node)
#   DRY_RUN=1 bash ...            only report what would be exported / uploaded
#
# Requires HF_TOKEN in the environment or in the repo .env file.

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../nscc" && pwd)/env.sh"
G05_ENV="$G05_VENV"
PY="$G05_ENV/bin/python"
export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 HF_DATASETS_OFFLINE=0

JOB_NAME="g05-droid-put-marker-pen-10ep-nscc"
RUN_DIR="$MODELS_ROOT/$JOB_NAME"
EXPORT_BASE="$RUN_DIR/_hf_export"
REPO_ID="$HF_HUB_ORG/$JOB_NAME"  # RoboColosseum/<job> (scripts/nscc/env.sh)
PRIVATE=true
CLEANUP="${CLEANUP:-true}"
EXPORT_STRIP="${EXPORT_STRIP:-true}"

: "${HF_TOKEN:?HF_TOKEN not set; add it to $ROBO_ROOT/.env or export it}"

STEPS="$(find "$RUN_DIR/checkpoints" -maxdepth 1 -type f -name 'step_*.pt' 2>/dev/null \
    | sed 's#.*/step_##; s#\.pt$##' | sort -n | tr '\n' ' ')"
if [ -z "${STEPS// /}" ]; then
    # Nothing local: either training never saved, or an earlier run uploaded and cleaned up.
    if "$PY" - "$REPO_ID" <<'PY'
import sys
from huggingface_hub import HfApi
try:
    files = HfApi().list_repo_files(sys.argv[1], repo_type="model")
except Exception:  # noqa: BLE001
    files = []
sys.exit(0 if "checkpoints/model_state_dict.pt" in files else 1)
PY
    then
        echo "No local step_*.pt, but $REPO_ID already has checkpoints/model_state_dict.pt; nothing to do."
        exit 0
    fi
    echo "No step_*.pt checkpoints in $RUN_DIR/checkpoints and nothing on $REPO_ID" >&2
    exit 1
fi
FINAL_STEP="$(echo $STEPS | awk '{print $NF}')"
NAME="step_$(printf "%06d" "$FINAL_STEP")"

echo "Run dir: $RUN_DIR"
echo "Steps:   $STEPS(final: $FINAL_STEP)"
echo "Repo id: $REPO_ID (private=$PRIVATE, strip=$EXPORT_STRIP)"
if [ "${DRY_RUN:-}" = "1" ]; then
    echo "DRY_RUN: would export checkpoints/step_$FINAL_STEP.pt to $EXPORT_BASE/$NAME and upload it to $REPO_ID"
    exit 0
fi

mkdir -p "$EXPORT_BASE"
EXPORT_DIR="$EXPORT_BASE/$NAME"
STRIP_ARGS=()
[ "$EXPORT_STRIP" = "true" ] && STRIP_ARGS=(--strip)
# export_checkpoint.py prompts before overwriting and stages under ./tmp, so start clean
# and run it from the export dir to keep GalaxeaVLA untouched.
rm -rf "$EXPORT_DIR"
(cd "$EXPORT_BASE" && "$PY" "$G05_ROOT/tools/export_checkpoint.py" "$RUN_DIR" \
    --target-base "$EXPORT_BASE" \
    --name "$NAME" \
    --checkpoint "step_$FINAL_STEP.pt" \
    "${STRIP_ARGS[@]}" \
    -y)

# Upload the final package to the repo root, then verify (presence and size) before any cleanup.
"$PY" - "$REPO_ID" "$EXPORT_DIR" "$PRIVATE" "$NAME" <<'PY'
import os, sys, time
from huggingface_hub import HfApi

repo_id, export_dir, private, final_name = sys.argv[1], sys.argv[2], sys.argv[3].lower() == "true", sys.argv[4]
weights = "checkpoints/model_state_dict.pt"
local_size = os.path.getsize(os.path.join(export_dir, weights))

api = HfApi()
api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
for attempt, wait in enumerate([60, 300, 900, None], start=1):
    try:
        api.upload_folder(
            repo_id=repo_id,
            folder_path=export_dir,
            repo_type="model",
            commit_message=f"Upload final G0.5 droid put-marker-pen model ({final_name})",
        )
        break
    except Exception as e:  # noqa: BLE001
        print(f"  upload attempt {attempt} failed: {e!r}")
        if wait is None:
            raise
        time.sleep(wait)
print(f"Uploaded final model ({final_name}) -> https://huggingface.co/{repo_id}")

info = {p.path: p for p in api.get_paths_info(repo_id, [weights], repo_type="model")}
remote_size = getattr(info.get(weights), "size", None)
if remote_size != local_size:
    sys.exit(f"Upload verification failed: {weights} is {remote_size} bytes on the Hub, {local_size} locally")
print(f"Verified on the Hub: {weights} ({remote_size} bytes).")
PY

if [ "$CLEANUP" = "true" ]; then
    echo "Cleaning up local checkpoint files..."
    rm -rf "$EXPORT_BASE"
    rm -f "$RUN_DIR"/checkpoints/step_*.pt "$RUN_DIR/last.pt"
    rmdir "$RUN_DIR/checkpoints" 2>/dev/null || true
fi

echo "Done."
