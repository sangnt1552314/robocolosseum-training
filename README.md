# RoboColosseum Training

Fine-tuning scripts for benchmarking vision-language-action (VLA) policies on RoboColosseum
tasks, run as PBS jobs on NUS Hopper. Each model is trained with its official or recommended
framework and its default fine-tuning recipe, so results stay comparable across models.

## Models

| Model | Framework | Base checkpoint | Recipe |
| --- | --- | --- | --- |
| MolmoAct2 | LeRobot (`--policy.type=molmoact2`) | `allenai/MolmoAct2` | full fine-tune, batch 16, 40K steps |
| π₀.₅ | LeRobot (`--policy.type=pi05`) | `lerobot/pi05_base` | full fine-tune, batch 16, 40K steps |
| GR00T N1.7 | LeRobot (`--policy.type=groot`) | `nvidia/GR00T-N1.7-3B` | default recipe (frozen LLM + vision, tuned projector / DiT / VLLN), batch 32, 40K steps |
| LingBot-VLA v2 6B | LingBot-VLA | `lingbot-vla-v2-6b` | full fine-tune, batch 16, 10 epochs |
| G0.5 | GalaxeaVLA | `g05-base` | native SO-100 recipe, batch 16, 10 epochs |

The training script header of each run documents its exact settings and any deviation from
the upstream recipe.

## Tasks

| Task | Robot | Dataset | MolmoAct2 | π₀.₅ | GR00T | LingBot | G0.5 |
| --- | --- | --- | :-: | :-: | :-: | :-: | :-: |
| Black screwdriver box to table | SO-101 | `Jiamo0912/robocolosseum-so101-black-screwdriver-box-to-table-100episodes` | ✓ | ✓ | ✓ | ✓ | ✓ |
| Cube drawer | SO-101 | `Jiamo0912/so101-cube-drawer` | ✓ | ✓ | ✓ | ✓ | ✓ |
| Upright bottle | SO-101 | `Jiamo0912/so101-upright-bottle` | ✓ | ✓ | ✓ | ✓ | ✓ |
| Stack bowls | SO-101 | `Jiamo0912/so101-stack-bowls` | ✓ | ✓ | ✓ | ✓ | ✓ |
| Stack cubes | SO-101 | `Jiamo0912/so101-stack-cubes` | ✓ | ✓ | ✓ | ✓ | ✓ |
| Carry box | Unitree G1 | `g1-carry-box` (local only) | | | ✓ | | |
| Open drawer | DROID Franka | `droid-open-drawer` (local only) | ✓ | ✓ | ✓ | ✓ | ✓ |
| Pour water | DROID Franka | `droid-pour-water` (local only) | ✓ | ✓ | ✓ | ✓ | ✓ |
| Towel in basket | DROID Franka | `droid-towel-in-basket` (local only) | ✓ | ✓ | ✓ | ✓ | ✓ |
| Put marker pen | DROID Franka | `droid-put-marker-pen` (local only, `_10ep` only) | ✓ | ✓ | ✓ | ✓ | ✓ |

Datasets are LeRobot v3 and are read from local copies under `/scratch/e1583535/datasets/`.

### DROID Franka runs (`*_droid_<task>_5ep`, `*_droid_<task>_10ep`)

The DROID tasks use one shared recipe: 5 epochs per model (`_5ep`), and each model's default LR (as
suggested by its own repo, not rescaled for batch size), default chunk size, and only the final
checkpoint. The `_10ep` variants are the same recipe for 10 epochs; they save no intermediate
checkpoints and push only the final, ready-to-use model to the Hub repo root
(`tsangb34/<model>-droid-<task>-10ep`; LingBot keeps its `checkpoints/global_step_<N>/hf_ckpt/` layout).

| Model | Effective batch | Cameras | Notes |
| --- | --- | --- | --- |
| MolmoAct2 | 64 (1 GPU) | left, right, wrist | `franka_droid` setup_type / control mode, gripper not normalized |
| π₀.₅ | 32 | left, wrist (+ masked empty slot) | trains on `droid-<task>-left-wrist` |
| GR00T N1.7 | 32 | left, wrist | full fine-tune (LLM, vision, projector, DiT, VLLN); trains on `droid-<task>-left-wrist` |
| G0.5 | 16 | left, wrist (+ dummy slot) | native so100 recipe with the Franka data config in `scripts/g05/configs/` (via `hydra.searchpath`) |
| LingBot-VLA v2 | 32 (8 x 4 grad-accum) | left, right, wrist | robot configs `scripts/lingbot/configs/robot_configs/droid_<task>.yaml` |

`droid-<task>-left-wrist` is built by `tools/make_droid_left_wrist_dataset.py`. The script
removes the right camera and lists left before wrist in `info.json`, which sets the camera order
for GR00T and π₀.₅. The pi05 and groot scripts build it on first use if it is missing.

### NSCC ASPIRE 2B runs (`*_droid_put_marker_pen_10ep_nscc`, 8 x H200)

The same `_10ep` recipes, ported to one full ASPIRE 2B node (project `51001013`, `normal` queue,
`select=1:ncpus=96:ngpus=8:mem=1500gb`). The global batch, LR and schedule of each model stay the
same, so each run is still exactly 10 epochs. Only the per-GPU split changes:

| Model | Per GPU x GPUs (x accum) | Global batch | Steps | Launcher |
| --- | --- | --- | --- | --- |
| MolmoAct2 | 8 x 8 | 64 | 8,080 | `accelerate launch lerobot-train` (lerobot v0.6.1) |
| π₀.₅ | 4 x 8 | 32 | 16,160 | `accelerate launch lerobot-train` (lerobot v0.6.1) |
| GR00T N1.7 | 4 x 8 | 32 | 16,160 | `accelerate launch lerobot-train` (lerobot v0.6.1) |
| LingBot-VLA v2 | 4 x 8 (x 1) | 32 | 16,150 | `torchrun tasks/vla/train_lingbotvla.py` (FSDP2) |
| G0.5 | 2 x 8 | 16 | 10 epochs (derived) | `scripts/run/finetune.sh 8` |

There are no Singularity images or `/scratch/e1583535` paths here. Everything lives under
`/scratch/users/nus/duanj1/tansang`, as set in `scripts/nscc/env.sh`: framework checkouts in
`projects/`, uv virtualenvs in `venvs/`, FFmpeg 7.1 in `opt/`, LingBot base models in `models/`,
the datasets in `datasets/` (`droid-put-marker-pen`, plus the `-left-wrist` copy for π₀.₅ and
GR00T), and outputs in `outputs/robocolosseum-finetuned-models/<job>`. Each final model is pushed
to the RoboColosseum org as `RoboColosseum/<job>`. Job names end in `-nscc`, so these runs don't
collide with the Hopper runs in the same W&B project.

```bash
# one-time, on the login node (idempotent; rerun any step that fails)
bash scripts/nscc/setup_envs.sh all      # uv, ffmpeg, repos, datasets, 3 venvs, base models
# .env in the repo root: HF_TOKEN=... and WANDB_API_KEY=... (see below)
qsub nscc_pbs/molmoact2_droid_put_marker_pen_10ep.pbs
qsub nscc_pbs/pi05_droid_put_marker_pen_10ep.pbs
qsub nscc_pbs/groot_n17_droid_put_marker_pen_10ep.pbs
qsub nscc_pbs/lingbot_droid_put_marker_pen_10ep.pbs
qsub nscc_pbs/g05_droid_put_marker_pen_10ep.pbs
qstat -u $USER; tail -f logs/<job>/stderr.<jobid>.log
```

Every run logs to W&B `tsangb34-national-university-of-singapore-students-union/RoboColosseum`
(run id = job name), as on Hopper. The key has to belong to a member of that team, so put
`WANDB_API_KEY=...` in `.env`. Without it the job falls back to the `~/.netrc` login. Before
training, each PBS job runs two checks from `scripts/nscc/env.sh` and exits early if either fails:
`nscc_check_wandb` (the key can write to the team) and `nscc_check_hf` (`HF_TOKEN` can push to
`RoboColosseum`). `WANDB_ENTITY`, `WANDB_PROJECT` and `HF_HUB_ORG` can be overridden in `.env`.

Things that differ from Hopper on bare-metal ASPIRE 2B nodes:

- **torchcodec**: the `+cu128` torchcodec wheels need NVIDIA NPP (`libnppicc.so.12`), which the
  Hopper images shipped but no venv here does. `setup_envs.sh` therefore installs the CPU torchcodec
  build from PyPI in all three venvs (`cpu_torchcodec`). LingBot and G0.5 decode video with it; the
  lerobot runs use PyAV.
- **LingBot data parallelism**: LingBot's DDP path fails on 8 GPUs at the first backward ("Expected to
  mark a variable ready only once ... model.action_out_proj.bias"; with `enable_fp32` it wraps DDP with
  an fp32 `MixedPrecision`, which registers each parameter's reducer hook twice). On Hopper's single GPU
  the model was never wrapped. The NSCC config uses LingBot's own multi-GPU setup from
  `configs/vla/real_robot/real_robot.yaml` (`fsdp2`, `vlm_fsdp`, `module_fsdp_enable`; Muon becomes
  `DistributedMuon`). This also brings upstream's multi-GPU precision: the VLM, vision tower and
  projections run in bf16 and the action expert in fp32, where the Hopper run was fp32 throughout.
  Global batch, steps and LR are unchanged.
- **LingBot W&B curves**: LingBot writes its training scalars only to TensorBoard (`<output>/runs`), on
  Hopper too, so its W&B run shows only the config and system metrics while training. When training
  ends, the upload script copies the TensorBoard scalars into the same W&B run
  (`scripts/lingbot/sync_tb_to_wandb.py`, idempotent; it can also be run by hand).
- **Private repos**: every repo in the RoboColosseum org is private, so these runs push privately too
  (`--policy.private=true`, `PRIVATE=true` in the upload scripts). The five target repos were created
  private in advance, and `create_repo(..., exist_ok=True)` keeps an existing repo's visibility.
- **Hub push of the lerobot runs**: lerobot pushes the final model from rank 0 while the other 7 ranks
  wait in an NCCL barrier with torch's 10-minute default timeout. The final checkpoint is saved before
  the push, so after `accelerate` exits, the train scripts run `scripts/nscc/push_lerobot_final.sh <job>`.
  It uploads `checkpoints/last/pretrained_model` from a single process if any file is missing on
  `RoboColosseum/<job>`, and is a no-op otherwise. It can also be run by hand.
- **Fallback watcher**: `scripts/nscc/fallback_watcher.sh <pbs id>:<job> ...` runs detached on the login
  node (`setsid nohup ... >> logs/fallback_watcher.log &`). When PBS reports a job finished, it reruns that
  model's idempotent upload script from a single process, so a model still reaches the Hub if the job
  died after saving its final checkpoint.
- **Reruns**: only the final step is checkpointed and lerobot refuses an existing output dir, so the
  lerobot scripts stop before launching if `outputs/.../<job>` exists. Move it aside and give W&B a new
  run id: `qsub -v WANDB_RUN_ID=<job>-r2 nscc_pbs/<job>.pbs`.
- **Queues**: `normal` routes 8-GPU jobs to `g2` (walltime 2-24 h) or `gdev` (at most 2 h, used by
  `nscc_pbs/smoke_test_8gpu.pbs`). PBS gives every 8-GPU job the whole node (192 cores, 2203 GB).
- **Inode quota**: the scratch file-count quota is 200,000 (see `myquota`). The three venvs plus the uv
  cache come to about 210k inodes, so the quota is exceeded. `uv cache clean` frees about 29k (the
  venvs hardlink the rest of it); ask NSCC for a higher file quota if it starts being enforced.

## Repository Structure

```text
robocolosseum-training/
├── pbs/                          # Hopper PBS jobs, one per (model, task)
│   ├── <model>_<task>_<len>.pbs  # official runs
│   ├── 40k_step_experiment_runs/ # VLA-REPLICA recipe comparison runs
│   ├── 4k_step_runs/             # short runs
│   ├── experiments/              # ablations (normalization, action-expert-only, ...)
│   └── sandbox/                  # smoke tests and early trials
├── scripts/
│   ├── molmoact2/                # train_so101_<task>_40k.sh
│   ├── pi05/                     # train_so101_<task>_40k.sh
│   ├── groot/                    # train_so101_<task>_40k.sh, train_g1_carry_box_40k.sh
│   ├── lingbot/                  # train / upload / compute_norm_stats scripts, configs/, norm_stats/
│   ├── g05/                      # train / upload scripts, configs/
│   └── utils/                    # resumable_lerobot_train.sh (resume from local or Hub checkpoints)
├── tools/
│   ├── lerobot_train_pyav_single_thread.py  # lerobot-train with single-threaded PyAV decoding
│   └── upload_dataset_lerobot_v3.py         # push a local LeRobot v3 dataset to the Hub
├── logs/                         # job stdout/stderr, not committed
└── outputs/                      # local outputs, not committed
```

Official runs live at the top level of `scripts/<model>/` and `pbs/`. The subdirectories hold
experiments and are not part of the benchmark.

## Naming

| Item | Pattern | Example |
| --- | --- | --- |
| Training script | `scripts/<model>/train_<robot>_<task>_<len>.sh` | `scripts/groot/train_so101_stack_cubes_40k.sh` |
| PBS job | `pbs/<model>_<task>_<len>.pbs` | `pbs/groot_n17_stack_cubes_40k.pbs` |
| Job / run name | `<model>-<robot>-<task>-<len>` | `groot-n17-so101-stack-cubes-40k` |
| Checkpoints | `/scratch/Projects/CFP-05/CFP05-CF-002/robocolosseum-finetuned-models/<job name>` | |
| Hub model | `tsangb34/<job name>` | `tsangb34/groot-n17-so101-stack-cubes-40k` |
| Logs | `logs/<job name>/{stdout,stderr}.<jobid>.log` | |

Every run logs to the `RoboColosseum` W&B project, using the job name as the run ID.

## Environments

Each PBS job runs inside a Hopper Singularity image and activates a model-specific virtualenv:

| Model | Singularity image | Virtualenv | Framework checkout |
| --- | --- | --- | --- |
| MolmoAct2, π₀.₅, GR00T | `pytorch_2.7.0_cuda_12.8.sif` | `/scratch/e1583535/virtualenvs/lerobot-cu128` | `/scratch/e1583535/projects/lerobot` (LeRobot fork, editable install) |
| LingBot-VLA v2 | `pytorch_2.6.0_cuda_12.8.sif` | `/scratch/e1583535/virtualenvs/lingbot-vla-v2` | `/scratch/e1583535/projects/lingbot-vla-v2` |
| G0.5 | `pytorch_2.3.0_cuda_12.4_ngc_24.04.sif` | `/scratch/e1583535/virtualenvs/g05` | `/scratch/e1583535/projects/GalaxeaVLA` |

The framework repos are kept unmodified. Any customization lives in this repository, as
Hydra/CLI overrides, configs under `scripts/<model>/configs/`, or wrappers in `tools/`.

Video decoding uses the FFmpeg builds under `/scratch/e1583535/opt/`. The training scripts set
`PATH` and `LD_LIBRARY_PATH` for them.

## Running a Job

1. Create a `.env` file in the repository root containing your Hub token (`HF_TOKEN=...`). Every
   run pushes its checkpoints to the Hub, and the PBS jobs exit early when the token is missing.
2. Submit from the repository root:

   ```bash
   qsub pbs/groot_n17_stack_cubes_40k.pbs
   ```

3. Follow progress in `logs/<job name>/stderr.<jobid>.log` or in W&B.

All official jobs request one GPU (`select=1:ngpus=1`), and PBS assigns 12 CPUs per GPU.

### Model-specific notes

- **LeRobot models (MolmoAct2, π₀.₅, GR00T)**: `--save_checkpoint_to_hub=true` pushes each
  checkpoint to `<repo>/checkpoints/<step>/`, and the final model goes to the repo root.
- **LingBot-VLA**: on the first run, the training script calls
  `compute_norm_stats_so101_<task>.sh` to create `norm_stats/so101_<task>.json`. This needs a
  GPU. When training ends, `upload_so101_<task>_10ep.sh` pushes every checkpoint to the Hub. If
  the upload fails, rerun that script by hand.
- **G0.5**: when training ends, `upload_so101_<task>_10ep.sh` exports and uploads every
  checkpoint, then deletes the local copies after it has checked they are on the Hub.
- **LingBot / G0.5 step counts**: both train for 10 epochs and hard-code the step count and
  checkpoint interval for each dataset. Recompute both when you add a task. See the header of an
  existing script for how.
- **GR00T on the G1** (`train_g1_carry_box_40k.sh`):
  - Uses `new_embodiment` with absolute actions. The 29-D joint state and the 23-D
    pivot/end-effector/gripper action don't line up, so state-relative actions can't be used.
  - Launches through `tools/lerobot_train_pyav_single_thread.py`. The dataset's AV1 videos
    decode slowly with PyAV's default threading.

## Adding a Task

1. Copy the closest existing official script and PBS file for each model.
2. Change the dataset path or repo ID, the job name, and the log directory.
3. For LingBot and G0.5, also recompute the step count and checkpoint interval.
4. Check that the cameras (`observation.images.*`) and state/action layout in the dataset's
   `meta/info.json` match what the script assumes. The SO-101 scripts expect `front` and
   `wrist` cameras and 6-D joint positions.
