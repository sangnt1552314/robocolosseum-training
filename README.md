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

Datasets are LeRobot v3 and are read from local copies under `/scratch/e1583535/datasets/`.

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
