#!/bin/bash
# Crash-safe wrapper around lerobot-train. Source this file, then call:
#
#   run_resumable_train "$OUTPUT_DIR" "$HUB_REPO_ID" <lerobot-train args...>
#
# The training args must include --save_checkpoint=true, --save_freq=N and
# --save_checkpoint_to_hub=true, so every checkpoint is uploaded to
# <repo>/checkpoints/<step>/ (model + optimizer + scheduler + RNG state).
#
# Each (re)submission of the same job picks one of:
#   1. local checkpoint exists  -> resume from $OUTPUT_DIR/checkpoints/last
#   2. Hub checkpoint exists    -> download the latest one and resume from it
#   3. nothing saved yet        -> fresh start with the given args
#   4. final model already on the Hub -> nothing to do
#
# While training runs, a background pruner keeps only the newest
# KEEP_LOCAL_CHECKPOINTS (default 1) finished local checkpoints; older ones were
# already pushed to the Hub. A checkpoint still being written is never touched.

# Exit codes: 0 = yes/found, 3 = no/none. Anything else is a lookup error (network,
# auth, missing module) and must abort the job rather than trigger a fresh start.
_hub_query() {
    # $1 = repo id, $2 = "latest" | "done"
    python - "$1" "$2" <<'PY'
import sys
from huggingface_hub import HfApi
from huggingface_hub.utils import RepositoryNotFoundError
from lerobot.common.train_utils import find_latest_hub_checkpoint

repo_id, what = sys.argv[1], sys.argv[2]
api = HfApi()
try:
    files = api.list_repo_files(repo_id, repo_type="model")
except RepositoryNotFoundError:
    sys.exit(3)
if what == "done":
    # The final push_to_hub writes model.safetensors at the repo root.
    sys.exit(0 if "model.safetensors" in files else 3)
latest = find_latest_hub_checkpoint(repo_id)
if latest is None:
    sys.exit(3)
print(latest)
PY
}

_prune_local_checkpoints() {
    local ckpt_root="$1/checkpoints" keep="${KEEP_LOCAL_CHECKPOINTS:-1}"
    while sleep 120; do
        [ -L "$ckpt_root/last" ] || continue
        local last_name
        last_name=$(basename "$(readlink "$ckpt_root/last")")
        [[ "$last_name" =~ ^[0-9]+$ ]] || continue
        # Finished checkpoints are numbered dirs up to `last`; newer ones may still be writing.
        local finished=() d name
        for d in "$ckpt_root"/*/; do
            name=$(basename "$d")
            if [[ "$name" =~ ^[0-9]+$ ]] && ((10#$name <= 10#$last_name)); then
                finished+=("$name")
            fi
        done
        local n_remove=$((${#finished[@]} - keep))
        ((n_remove > 0)) || continue
        # Zero-padded names sort chronologically; drop the oldest ones.
        for name in $(printf '%s\n' "${finished[@]}" | sort | head -n "$n_remove"); do
            echo "[pruner] removing local checkpoint $name (already pushed to the Hub)"
            rm -rf "${ckpt_root:?}/$name"
        done
    done
}

run_resumable_train() {
    local output_dir="$1" repo_id="$2"
    shift 2

    # wandb stages a full copy of every model artifact (6-12 GB) under ~/.local/share
    # and caches it under ~/.cache; keep both off the 40 GB home quota.
    export WANDB_DATA_DIR="${WANDB_DATA_DIR:-/scratch/e1583535/cache/wandb/data}"
    export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/scratch/e1583535/cache/wandb/cache}"
    export WANDB_ARTIFACT_DIR="${WANDB_ARTIFACT_DIR:-/scratch/e1583535/cache/wandb/artifacts}"

    local rc=0
    _hub_query "$repo_id" done || rc=$?
    if [ "$rc" -eq 0 ]; then
        echo "Final model already on the Hub at $repo_id; nothing to do."
        return 0
    elif [ "$rc" -ne 3 ]; then
        echo "ERROR: could not query the Hub for $repo_id (exit $rc); refusing to start." >&2
        return 1
    fi

    _prune_local_checkpoints "$output_dir" &
    local pruner_pid=$!
    trap "pkill -P $pruner_pid 2>/dev/null; kill $pruner_pid 2>/dev/null" EXIT

    local local_cfg="$output_dir/checkpoints/last/pretrained_model/train_config.json"
    local hub_latest=""
    rc=0
    if [ ! -f "$local_cfg" ]; then
        hub_latest=$(_hub_query "$repo_id" latest) || rc=$?
        if [ "$rc" -ne 0 ] && [ "$rc" -ne 3 ]; then
            echo "ERROR: could not query Hub checkpoints for $repo_id (exit $rc); refusing to start." >&2
            return 1
        fi
    fi

    rc=0
    if [ -f "$local_cfg" ]; then
        echo "Resuming from local checkpoint: $(readlink -f "$output_dir/checkpoints/last")"
        lerobot-train --config_path="$local_cfg" --resume=true
    elif [ -n "$hub_latest" ]; then
        echo "Resuming from Hub checkpoint: $repo_id/$hub_latest"
        rm -rf "$output_dir"
        lerobot-train --config_path="$repo_id" --resume=true --output_dir="$output_dir"
    else
        echo "No checkpoint found; starting a fresh run in $output_dir"
        # lerobot-train refuses to start fresh into an existing output_dir.
        rm -rf "$output_dir"
        lerobot-train "$@"
    fi || rc=$?

    pkill -P "$pruner_pid" 2>/dev/null || true
    kill "$pruner_pid" 2>/dev/null || true
    trap - EXIT
    return "$rc"
}
