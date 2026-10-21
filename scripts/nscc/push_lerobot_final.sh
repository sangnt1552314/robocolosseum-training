#!/bin/bash
set -euo pipefail

# Fallback Hub push for the lerobot NSCC runs (MolmoAct2 / pi05 / GR00T).
#
# lerobot-train pushes the final model from rank 0 while ranks 1-7 wait in an NCCL barrier
# (accelerator.wait_for_everyone) with torch's default 10-minute NCCL timeout. If saving plus uploading
# takes longer, ranks 1-7 abort and the push dies half-way. The final checkpoint is already on disk at
# that point, so this script uploads it from a single process (no NCCL) to the same repo,
# RoboColosseum/<job> (private, like every repo in the org).
#
# Idempotent: if every file of the final pretrained_model/ is already in the Hub repo, nothing is
# uploaded. Outside PBS it first waits until PBS confirms that no colosseum-<job> job is queued or
# running, so it never races the job's own push.
#
# Exit codes: 0 = on the Hub (uploaded or already there), 1 = upload failed (retry later),
#             2 = no complete final checkpoint (training did not finish).
#
# Usage:
#   bash scripts/nscc/push_lerobot_final.sh <job name>
#   e.g. bash scripts/nscc/push_lerobot_final.sh groot-n17-droid-put-marker-pen-10ep-nscc

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"
export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 HF_DATASETS_OFFLINE=0
export HF_XET_HIGH_PERFORMANCE=1

JOB_NAME="${1:?usage: $0 <job name>}"
RUN_DIR="$MODELS_ROOT/$JOB_NAME"
REPO_ID="$HF_HUB_ORG/$JOB_NAME"

: "${HF_TOKEN:?HF_TOKEN not set; add it to $ROBO_ROOT/.env}"

if [ -z "${PBS_JOBID:-}" ]; then
    # qselect exits 0 with empty output only when PBS confirms there is no such queued/running job.
    for _ in $(seq 720); do
        if out="$(qselect -u "$USER" -N "colosseum-$JOB_NAME" 2>/dev/null)" && [ -z "$out" ]; then
            break
        fi
        echo "$(date '+%F %T') waiting: colosseum-$JOB_NAME is still queued/running (${out:-PBS query failed})"
        sleep 120
    done
fi

"$VENVS_ROOT/hftools/bin/python" - "$REPO_ID" "$RUN_DIR" "$JOB_NAME" <<'PY'
import glob, json, os, shutil, struct, sys, time
from huggingface_hub import HfApi

repo_id, run_dir, job_name = sys.argv[1:4]
REQUIRED = ["config.json", "train_config.json", "model.safetensors",
            "policy_preprocessor.json", "policy_postprocessor.json"]


def safetensors_complete(path):
    """True if the file size matches the header (catches a checkpoint cut off mid-write)."""
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            (n,) = struct.unpack("<Q", f.read(8))
            header = json.loads(f.read(n))
        end = max((v["data_offsets"][1] for k, v in header.items() if k != "__metadata__"), default=0)
        return size == 8 + n + end
    except Exception as e:  # noqa: BLE001
        print(f"  {path}: unreadable safetensors ({e})")
        return False


def complete(d):
    return all(os.path.isfile(os.path.join(d, f)) for f in REQUIRED) and safetensors_complete(
        os.path.join(d, "model.safetensors"))


candidates = [os.path.join(run_dir, "checkpoints", "last", "pretrained_model")]
candidates += sorted(glob.glob(os.path.join(run_dir, "checkpoints", "[0-9]*", "pretrained_model")), reverse=True)
final = next((d for d in candidates if os.path.isdir(d) and complete(d)), None)
if final is None:
    print(f"No complete final checkpoint under {run_dir}/checkpoints (training did not finish?); nothing to push.")
    sys.exit(2)
final = os.path.realpath(final)

api = HfApi()
local = sorted(os.path.relpath(os.path.join(r, f), final) for r, _, fs in os.walk(final) for f in fs)
try:
    remote = set(api.list_repo_files(repo_id, repo_type="model"))
except Exception:  # noqa: BLE001  (repo does not exist yet)
    remote = set()
missing = [f for f in local if f not in remote]
if not missing:
    print(f"{repo_id}: all {len(local)} files of {final} are already on the Hub; nothing to do.")
else:
    print(f"{repo_id}: missing on the Hub: {missing}; uploading {final}")
    api.create_repo(repo_id, repo_type="model", private=True, exist_ok=True)
    if "README.md" not in remote:
        policy = json.load(open(os.path.join(final, "train_config.json"))).get("policy", {})
        base = policy.get("pretrained_path") or policy.get("base_model_path") or policy.get("checkpoint_path")
        card = "\n".join([
            "---", "library_name: lerobot", "pipeline_tag: robotics",
            f"tags: [lerobot, robotics, {policy.get('type', 'policy')}, robocolosseum, droid, put-marker-pen]",
            *([f"base_model: {base}"] if base else []), "---", "",
            f"# {job_name}", "",
            f"`{policy.get('type')}` fine-tuned on the RoboColosseum DROID put-marker-pen dataset (10 epochs, "
            "8 x H200 on NSCC ASPIRE 2B). Final checkpoint, uploaded by scripts/nscc/push_lerobot_final.sh.", "",
        ])
        readme = os.path.join(os.environ["TMPDIR"], f"{job_name}_README.md")
        with open(readme, "w") as f:
            f.write(card)
        api.upload_file(path_or_fileobj=readme, path_in_repo="README.md", repo_id=repo_id,
                        repo_type="model", commit_message="Add model card")
    for attempt, wait in enumerate([60, 300, 900, None], start=1):
        try:
            api.upload_folder(repo_id=repo_id, folder_path=final, repo_type="model",
                              commit_message=f"Upload final policy (fallback push from {os.path.relpath(final, run_dir)})")
            break
        except Exception as e:  # noqa: BLE001
            print(f"  upload attempt {attempt} failed: {e!r}")
            if wait is None:
                sys.exit(1)
            time.sleep(wait)
    remote = set(api.list_repo_files(repo_id, repo_type="model"))
    still = [f for f in local if f not in remote]
    if still:
        print(f"Upload verification failed, still missing: {still}")
        sys.exit(1)
    print(f"Uploaded and verified -> https://huggingface.co/{repo_id}")

# A push killed mid-way (NCCL timeout) leaves lerobot's TemporaryDirectory copy behind in $TMPDIR.
for leftover in glob.glob(os.path.join(os.environ["TMPDIR"], "tmp*", repo_id)):
    shutil.rmtree(os.path.dirname(os.path.dirname(leftover)), ignore_errors=True)
    print(f"Removed leftover push staging dir {os.path.dirname(os.path.dirname(leftover))}")
PY
