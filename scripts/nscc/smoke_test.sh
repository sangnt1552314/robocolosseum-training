#!/bin/bash
set -uo pipefail

# Compute-node smoke test for the NSCC runs (~5 min on one 8 x H200 node; see
# nscc_pbs/smoke_test_8gpu.pbs). Checks what the login node cannot: that the three venvs' torch builds
# work with the node's driver, all 8 GPUs and NCCL work, flash-attn / Triton compile kernels (LingBot's
# flex attention), video decoding works, and the node can reach W&B and the Hub with our credentials.
# Nothing is trained, no W&B run is created and nothing is pushed. Exits non-zero if any check fails.

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

FAILED=()
check() {  # check <name> <command...>
    local name="$1"; shift
    echo "---- $name"
    if "$@"; then echo "PASS: $name"; else echo "FAIL: $name"; FAILED+=("$name"); fi
}

NCCL_TEST="$TMPDIR/nscc_nccl_test.py"
cat > "$NCCL_TEST" <<'PY'
import os, torch, torch.distributed as dist
dist.init_process_group("nccl")
rank = dist.get_rank()
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
x = torch.ones(64 * 1024 * 1024, device="cuda") * (rank + 1)
dist.all_reduce(x)
expected = sum(range(1, dist.get_world_size() + 1))
assert torch.allclose(x[:8], torch.full((8,), float(expected), device="cuda")), x[:8]
if rank == 0:
    print(f"all_reduce OK over {dist.get_world_size()} ranks ({torch.cuda.get_device_name()})")
dist.destroy_process_group()
PY

gpu_check() {  # gpu_check <venv>
    "$1/bin/python" - <<'PY'
import torch
n = torch.cuda.device_count()
print("torch", torch.__version__, "cuda", torch.version.cuda, "gpus", n, torch.cuda.get_device_name(0))
assert n == 8, f"expected 8 GPUs, got {n}"
x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
torch.cuda.synchronize(); print("matmul ok", float((x @ x).float().abs().mean()))
PY
}

nccl_check() {  # nccl_check <venv>
    "$1/bin/python" -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=8 "$NCCL_TEST"
}

echo "Host: $(hostname)  Job: ${PBS_JOBID:-interactive}  $(date)"
check "nvidia-smi sees 8 GPUs" bash -c '[ "$(nvidia-smi -L | tee /dev/stderr | wc -l)" -eq 8 ]'

for venv in "$LEROBOT_VENV" "$LINGBOT_VENV" "$G05_VENV"; do
    name="$(basename "$venv")"
    check "$name: torch sees 8 GPUs" gpu_check "$venv"
    check "$name: NCCL all_reduce on 8 GPUs" nccl_check "$venv"
done

check "lingbot: flash-attn kernel" "$LINGBOT_VENV/bin/python" - <<'PY'
import torch, flash_attn
from flash_attn import flash_attn_func
q = torch.randn(2, 256, 8, 64, device="cuda", dtype=torch.bfloat16)
print("flash_attn", flash_attn.__version__, tuple(flash_attn_func(q, q, q).shape))
PY

check "lingbot: Triton compiles flex_attention" "$LINGBOT_VENV/bin/python" - <<'PY'
import torch
from torch.nn.attention.flex_attention import flex_attention
q = torch.randn(1, 4, 256, 64, device="cuda", dtype=torch.bfloat16)
out = torch.compile(flex_attention)(q, q, q)
torch.cuda.synchronize(); print("flex_attention ok", tuple(out.shape))
PY

check "g05: imports (flash-attn-4 / fla / g05 deps)" "$G05_VENV/bin/python" - <<'PY'
import importlib
for mod in ("torch", "hydra", "wandb", "fla", "flash_attn"):
    try:
        m = importlib.import_module(mod)
        print(mod, getattr(m, "__version__", "ok"))
    except ImportError as e:
        if mod == "flash_attn":  # flash-attn-4 may expose a different top-level module
            print("flash_attn import:", e)
        else:
            raise
PY

check "lerobot: decode a frame (pyav)" "$LEROBOT_VENV/bin/python" - <<'PY'
import os
from lerobot.datasets.lerobot_dataset import LeRobotDataset
root = os.path.join(os.environ["DATASETS_ROOT"], "droid-put-marker-pen-left-wrist")
ds = LeRobotDataset("local/droid-put-marker-pen-left-wrist", root=root, video_backend="pyav")
x = ds[len(ds) // 2]
print({k: tuple(v.shape) for k, v in x.items() if k.startswith("observation.images")})
PY

check "internet: huggingface.co" curl -fsS -o /dev/null -m 30 https://huggingface.co/api/models/allenai/MolmoAct2
check "internet: api.wandb.ai" curl -sS -o /dev/null -m 30 https://api.wandb.ai/graphql
check "HF_TOKEN can push to $HF_HUB_ORG" nscc_check_hf
check "W&B key can log to $WANDB_ENTITY/$WANDB_PROJECT" nscc_check_wandb

echo "========================================"
if [ ${#FAILED[@]} -eq 0 ]; then
    echo "ALL CHECKS PASSED"
else
    printf 'FAILED: %s\n' "${FAILED[@]}"
    exit 1
fi
