#!/bin/bash
set -euo pipefail

# Fallback Hub push for the lerobot NSCC runs (MolmoAct2 / pi05 / GR00T).
#
# lerobot-train pushes the final model from rank 0 while ranks 1-7 wait in an NCCL barrier
# (accelerator.wait_for_everyone) with torch's default 10-minute NCCL timeout. If saving plus uploading
# takes longer, ranks 1-7 abort and the push dies half-way. The final checkpoint is already on disk at
# that point (checkpoints/last/pretrained_model), so this script uploads it from a single process
# (no NCCL) to the same repo, RoboColosseum/<job>.
#
# Idempotent: if every file of pretrained_model/ is already in the Hub repo, nothing is uploaded.
#
# Usage:
#   bash scripts/nscc/push_lerobot_final.sh <job name>
#   e.g. bash scripts/nscc/push_lerobot_final.sh groot-n17-droid-put-marker-pen-10ep-nscc

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"
export HF_HUB_OFFLINE=0

JOB_NAME="${1:?usage: $0 <job name>}"
FINAL="$MODELS_ROOT/$JOB_NAME/checkpoints/last/pretrained_model"
REPO_ID="$HF_HUB_ORG/$JOB_NAME"

: "${HF_TOKEN:?HF_TOKEN not set; add it to $ROBO_ROOT/.env}"
if [ ! -f "$FINAL/model.safetensors" ] || [ ! -f "$FINAL/config.json" ]; then
    echo "No final checkpoint at $FINAL (training did not finish?); nothing to push." >&2
    exit 1
fi

"$VENVS_ROOT/hftools/bin/python" - "$REPO_ID" "$FINAL" <<'PY'
import json, os, sys
from huggingface_hub import HfApi

repo_id, final = sys.argv[1], sys.argv[2]
api = HfApi()
local = sorted(
    os.path.relpath(os.path.join(root, f), final)
    for root, _, files in os.walk(final, followlinks=True) for f in files
)
try:
    remote = set(api.list_repo_files(repo_id, repo_type="model"))
except Exception:  # repo does not exist yet
    remote = set()
missing = [f for f in local if f not in remote]
if not missing:
    print(f"{repo_id}: all {len(local)} files of {final} are already on the Hub; nothing to do.")
    sys.exit(0)

print(f"{repo_id}: missing on the Hub: {missing}; uploading {final}")
api.create_repo(repo_id, repo_type="model", private=False, exist_ok=True)
if "README.md" not in remote:
    cfg = json.load(open(os.path.join(final, "train_config.json")))
    policy = cfg.get("policy", {})
    base = policy.get("pretrained_path") or policy.get("base_model_path") or policy.get("checkpoint_path")
    card = "\n".join([
        "---",
        "library_name: lerobot",
        "pipeline_tag: robotics",
        f"tags: [lerobot, robotics, {policy.get('type', 'policy')}, robocolosseum, droid, put-marker-pen]",
        *( [f"base_model: {base}"] if base else [] ),
        "---",
        "",
        f"# {repo_id.split('/')[-1]}",
        "",
        f"`{policy.get('type')}` fine-tuned on the RoboColosseum DROID put-marker-pen dataset (10 epochs, "
        "8 x H200 on NSCC ASPIRE 2B). Final checkpoint, uploaded by scripts/nscc/push_lerobot_final.sh.",
        "",
    ])
    with open(os.path.join(os.environ["TMPDIR"], f"{repo_id.split('/')[-1]}_README.md"), "w") as f:
        f.write(card)
    api.upload_file(path_or_fileobj=f.name, path_in_repo="README.md", repo_id=repo_id, repo_type="model",
                    commit_message="Add model card")
api.upload_folder(repo_id=repo_id, folder_path=final, repo_type="model",
                  commit_message="Upload final policy (fallback push from checkpoints/last/pretrained_model)")
remote = set(api.list_repo_files(repo_id, repo_type="model"))
still = [f for f in local if f not in remote]
if still:
    sys.exit(f"Upload verification failed, still missing: {still}")
print(f"Uploaded and verified -> https://huggingface.co/{repo_id}")
PY
