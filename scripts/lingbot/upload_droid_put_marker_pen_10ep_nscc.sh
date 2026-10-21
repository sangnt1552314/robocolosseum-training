#!/bin/bash
set -euo pipefail

# Upload the fine-tuned LingBot-VLA v2 6B droid-put-marker-pen NSCC run (8 x H200) to the HuggingFace Hub.
# Same as upload_droid_put_marker_pen_10ep.sh with the NSCC paths (scripts/nscc/env.sh).
#
# LingBot-VLA has no built-in push_to_hub, but every save (global_step_<N>; only the final step for these runs)
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
# The repo is private, like every repo in the RoboColosseum org. The script is idempotent (a step whose
# files are all on the Hub already is skipped) and verifies the upload. Afterwards it copies LingBot's
# TensorBoard scalars into the run's W&B run (sync_tb_to_wandb.py), since LingBot logs its training
# curves only to TensorBoard.
#
# Usage:
#   bash scripts/lingbot/upload_droid_put_marker_pen_10ep_nscc.sh [STEP]
#   DRY_RUN=1 bash ...   checks the export and lists what would be uploaded, without uploading
#
#   STEP  optional. Upload only global_step_<STEP>. Defaults to every global_step_N on disk.
#
# Requires HF_TOKEN in the environment or in the repo .env file.

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../nscc" && pwd)/env.sh"
LINGBOT_ENV="$LINGBOT_VENV"
PY="$LINGBOT_ENV/bin/python"

# Training runs with the Hub offline; the upload needs it online (huggingface_hub 0.34 also treats an
# inherited TRANSFORMERS_OFFLINE=1 as offline when HF_HUB_OFFLINE is unset).
export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 HF_DATASETS_OFFLINE=0

# The Xet upload backend throws "I/O error (os error 2)" on the Hopper filesystem;
# keep the classic LFS multipart uploader here too.
export HF_HUB_DISABLE_XET=1

JOB_NAME="lingbot-vla-v2-6b-droid-put-marker-pen-10ep-nscc"
RUN_DIR="$MODELS_ROOT/$JOB_NAME"
CKPT_DIR="$RUN_DIR/checkpoints"
REPO_ID="$HF_HUB_ORG/$JOB_NAME"  # RoboColosseum/<job> (scripts/nscc/env.sh)
PRIVATE=true

: "${HF_TOKEN:?HF_TOKEN not set; add it to $ROBO_ROOT/.env or export it}"

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

# Create (if needed) and upload only the selected files, preserving repo layout; verify afterwards.
"$PY" - "$REPO_ID" "$RUN_DIR" "$PRIVATE" $STEPS <<'PY'
import glob, json, os, struct, sys, time
from huggingface_hub import HfApi

repo_id, run_dir, private, steps = sys.argv[1], sys.argv[2], sys.argv[3].lower() == "true", sys.argv[4:]


def safetensors_complete(path):
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    end = max((v["data_offsets"][1] for k, v in header.items() if k != "__metadata__"), default=0)
    return size == 8 + n + end


api = HfApi()
api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
for step in steps:
    hf = os.path.join(run_dir, "checkpoints", f"global_step_{step}", "hf_ckpt")
    # The HF export must be complete: every shard named in the index exists and is not truncated.
    index = os.path.join(hf, "model.safetensors.index.json")
    shards = sorted(set(json.load(open(index))["weight_map"].values())) if os.path.isfile(index) else ["model.safetensors"]
    bad = [s for s in shards if not os.path.isfile(os.path.join(hf, s)) or not safetensors_complete(os.path.join(hf, s))]
    if bad:
        sys.exit(f"global_step_{step}/hf_ckpt is incomplete (missing or truncated: {bad}); not uploading.")

    local = [os.path.relpath(p, run_dir) for p in glob.glob(os.path.join(hf, "**"), recursive=True) if os.path.isfile(p)]
    local += [os.path.relpath(p, run_dir) for p in glob.glob(os.path.join(run_dir, "model_assets", "**"), recursive=True)
              if os.path.isfile(p)]
    if os.path.isfile(os.path.join(run_dir, "lingbotvla_cli.yaml")):
        local.append("lingbotvla_cli.yaml")
    if not set(local) - set(api.list_repo_files(repo_id, repo_type="model")):
        print(f"global_step_{step}: all {len(local)} files already on the Hub; skipping.")
        continue
    if os.environ.get("DRY_RUN") == "1":
        print(f"DRY_RUN: would upload {len(local)} files for global_step_{step}: {sorted(local)[:8]} ...")
        continue

    for attempt, wait in enumerate([60, 300, 900, None], start=1):
        try:
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
                commit_message=f"Upload LingBot-VLA v2 6B droid put-marker-pen checkpoint (global_step_{step})",
            )
            break
        except Exception as e:  # noqa: BLE001
            print(f"  upload attempt {attempt} failed: {e!r}")
            if wait is None:
                raise
            time.sleep(wait)
    missing = sorted(set(local) - set(api.list_repo_files(repo_id, repo_type="model")))
    if missing:
        sys.exit(f"Upload verification failed for global_step_{step}, missing on the Hub: {missing}")
    print(f"Uploaded global_step_{step} -> https://huggingface.co/{repo_id} (verified {len(local)} files)")
PY

# LingBot logs its training curves only to TensorBoard; copy them into the W&B run (idempotent).
"$PY" "$ROBO_ROOT/scripts/lingbot/sync_tb_to_wandb.py" --logdir "$RUN_DIR/runs" --run-id "$JOB_NAME" \
    ${DRY_RUN:+--dry-run} || echo "W&B backfill failed; rerun: $PY scripts/lingbot/sync_tb_to_wandb.py --logdir $RUN_DIR/runs --run-id $JOB_NAME"

echo "Done."
