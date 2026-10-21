# Shared paths for the NSCC ASPIRE 2B runs (sourced by scripts/nscc/*.sh, scripts/*/*_nscc.sh and
# nscc_pbs/*.pbs). Everything (frameworks, virtualenvs, models, datasets, caches, outputs) lives under
# $TANSANG_ROOT; nothing is read from or written to the Hopper paths (/scratch/e1583535, /app1).
#
#   $TANSANG_ROOT/
#   ├── bin/uv                          uv (python/venv manager)
#   ├── python/                         uv-managed CPython builds
#   ├── projects/{lerobot,lingbot-vla-v2,GalaxeaVLA}   framework checkouts (kept unmodified)
#   │   └── robocolosseum-training      this repo
#   ├── venvs/{lerobot-v0.6.1,lingbot-vla-v2,g05,hftools}
#   ├── opt/ffmpeg-7.1                  FFmpeg shared libs (conda-forge) for torchcodec / PyAV
#   ├── models/                         local base checkpoints (LingBot)
#   ├── datasets/droid-put-marker-pen   RoboColosseum/droid-put-marker-pen (LeRobot v3)
#   ├── cache/{hf,uv,wandb,pip}
#   ├── tmp/
#   └── outputs/robocolosseum-finetuned-models/<job name>

export TANSANG_ROOT=/scratch/users/nus/duanj1/tansang
export ROBO_ROOT="$TANSANG_ROOT/projects/robocolosseum-training"

# Load secrets (HF_TOKEN, optional WANDB_API_KEY / WANDB_ENTITY) from the repo .env (gitignored).
if [ -f "$ROBO_ROOT/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "$ROBO_ROOT/.env"
    set +a
fi

export PROJECTS_ROOT="$TANSANG_ROOT/projects"
export VENVS_ROOT="$TANSANG_ROOT/venvs"
export MODELS_DIR="$TANSANG_ROOT/models"
export DATASETS_ROOT="$TANSANG_ROOT/datasets"
export MODELS_ROOT="$TANSANG_ROOT/outputs/robocolosseum-finetuned-models"

export LEROBOT_ROOT="$PROJECTS_ROOT/lerobot"            # huggingface/lerobot v0.6.1
export LINGBOT_ROOT="$PROJECTS_ROOT/lingbot-vla-v2"     # robbyant/lingbot-vla-v2
export G05_ROOT="$PROJECTS_ROOT/GalaxeaVLA"             # OpenGalaxea/GalaxeaVLA

export LEROBOT_VENV="$VENVS_ROOT/lerobot-v0.6.1"
export LINGBOT_VENV="$VENVS_ROOT/lingbot-vla-v2"
export G05_VENV="$VENVS_ROOT/g05"

# uv: keep interpreters, caches and builds inside $TANSANG_ROOT (not ~/.local or ~/.cache).
export PATH="$TANSANG_ROOT/bin:$PATH"
export UV_PYTHON_INSTALL_DIR="$TANSANG_ROOT/python"
export UV_CACHE_DIR="$TANSANG_ROOT/cache/uv"
export PIP_CACHE_DIR="$TANSANG_ROOT/cache/pip"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-600}"   # pypi.nvidia.com wheels time out at the 30 s default

export HF_HOME="$TANSANG_ROOT/cache/hf"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"

# Keep wandb staging/cache off the 50 GB home quota. The API key comes from ~/.netrc
# (`wandb login`) or WANDB_API_KEY in the repo .env.
export WANDB_DIR="$TANSANG_ROOT/cache/wandb"
export WANDB_DATA_DIR="$TANSANG_ROOT/cache/wandb/data"
export WANDB_CACHE_DIR="$TANSANG_ROOT/cache/wandb/cache"
export WANDB_ARTIFACT_DIR="$TANSANG_ROOT/cache/wandb/artifacts"
export WANDB_CONFIG_DIR="$TANSANG_ROOT/cache/wandb/config"
# All runs log to tsangb34-national-university-of-singapore-students-union/RoboColosseum (as on
# Hopper). The key must belong to a member of that team: put WANDB_API_KEY=... in the repo .env
# (the ~/.netrc login on this machine cannot access that team). nscc_check_wandb verifies this.
export WANDB_PROJECT="${WANDB_PROJECT:-RoboColosseum}"
export WANDB_ENTITY="${WANDB_ENTITY:-tsangb34-national-university-of-singapore-students-union}"

# Hub namespace for the fine-tuned models (HF_TOKEN's account must be able to write to it;
# tsangb34 is an admin of RoboColosseum). nscc_check_hf verifies this.
export HF_HUB_ORG="${HF_HUB_ORG:-RoboColosseum}"

export TMPDIR="$TANSANG_ROOT/tmp"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export TRITON_CACHE_DIR="$TANSANG_ROOT/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$TANSANG_ROOT/cache/inductor"
# Anything else that defaults to ~/.cache or ~/.config (torch.hub, matplotlib, ...) stays here too.
export TORCH_HOME="$TANSANG_ROOT/cache/torch"
export XDG_CACHE_HOME="$TANSANG_ROOT/cache/xdg"
export MPLCONFIGDIR="$TANSANG_ROOT/cache/matplotlib"

export FFMPEG_ROOT="$TANSANG_ROOT/opt/ffmpeg-7.1"
export PATH="$FFMPEG_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$FFMPEG_ROOT/lib:${LD_LIBRARY_PATH:-}"

mkdir -p "$TMPDIR" "$WANDB_DATA_DIR" "$WANDB_CACHE_DIR" "$WANDB_ARTIFACT_DIR" "$WANDB_CONFIG_DIR" \
    "$MODELS_ROOT" "$HF_HOME" "$TORCH_HOME" "$XDG_CACHE_HOME" "$MPLCONFIGDIR"


# Free TCP port for torchrun / accelerate rendezvous (a node can host several jobs).
nscc_free_port() {
    python3 -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()'
}

# Pre-flight: fail fast (before any GPU time) unless the W&B key can write to $WANDB_ENTITY.
# Uses WANDB_API_KEY if set, else the api.wandb.ai entry in ~/.netrc (the same order as wandb).
nscc_check_wandb() {
    python3 - <<'PY'
import base64, json, netrc, os, sys, urllib.request
entity = os.environ.get("WANDB_ENTITY", "")
key = os.environ.get("WANDB_API_KEY")
if not key:
    try:
        auth = netrc.netrc().authenticators("api.wandb.ai")
        key = auth[2] if auth else None
    except (FileNotFoundError, netrc.NetrcParseError):
        key = None
if not key:
    sys.exit("W&B: no API key (set WANDB_API_KEY in the repo .env or run `wandb login`).")
query = "{ viewer { username entity teams { edges { node { name } } } } }"
req = urllib.request.Request(
    "https://api.wandb.ai/graphql",
    data=json.dumps({"query": query}).encode(),
    headers={"Content-Type": "application/json",
             "Authorization": "Basic " + base64.b64encode(f"api:{key}".encode()).decode()},
)
viewer = json.load(urllib.request.urlopen(req, timeout=60))["data"]["viewer"]
if viewer is None:
    sys.exit("W&B: the API key was rejected.")
names = {viewer["username"], viewer["entity"]} | {e["node"]["name"] for e in viewer["teams"]["edges"]}
if entity and entity not in names:
    sys.exit(f"W&B: user '{viewer['username']}' is not a member of '{entity}' (has: {sorted(names)}). "
             "Put a WANDB_API_KEY from a member of that team in the repo .env.")
project = os.environ.get("WANDB_PROJECT")
print(f"W&B: OK, user '{viewer['username']}' -> {entity or viewer['entity']}/{project}")
PY
}

# Pre-flight: fail fast unless HF_TOKEN can push to $HF_HUB_ORG.
nscc_check_hf() {
    python3 - <<'PY'
import json, os, sys, urllib.request
org, token = os.environ.get("HF_HUB_ORG", ""), os.environ.get("HF_TOKEN")
if not token:
    sys.exit("HF: HF_TOKEN is not set (add it to the repo .env).")
req = urllib.request.Request("https://huggingface.co/api/whoami-v2",
                             headers={"Authorization": f"Bearer {token}"})
me = json.load(urllib.request.urlopen(req, timeout=60))
role = me.get("auth", {}).get("accessToken", {}).get("role")
org_roles = {o["name"]: o.get("roleInOrg") for o in me.get("orgs", [])}
if org != me["name"] and org_roles.get(org) not in ("admin", "write", "contributor"):
    sys.exit(f"HF: '{me['name']}' cannot write to '{org}' (org roles: {org_roles}).")
if role not in ("write", "fineGrained"):
    sys.exit(f"HF: token role is '{role}', need a write token.")
print(f"HF: OK, '{me['name']}' ({org_roles.get(org, 'owner')}, token {role}) -> {org}/<job>")
PY
}
