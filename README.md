# RoboColosseum Training

## Models

| Model | Framework | Status |
| --- | --- | --- |
| MolmoAct2 | Hugging Face LeRobot | In progress |
| π₀.₅ | Hugging Face LeRobot | Planned |
| GR00T N1.7 | Isaac-GR00T | Planned |
| LingBot-VLA v2 | LingBot-VLA | Planned |
| G0.5 | GalaxeaVLA | Planned |

The goal is to keep the training workflow and data interface as consistent as possible across models while using the official or recommended training framework for each policy.

## Repository Structure

```text
robocolosseum-training/
├── pbs/
│   └── molmoact2_so101.pbs
│
├── scripts/
│   ├── molmoact2/
│   │   └── train_so101.sh
│   │
│   ├── pi05/                   # Planned
│   │   ├── train_so101.sh
│   │   └── train_droid.sh
│   │
│   ├── groot/                  # Planned
│   ├── lingbot/                # Planned
│   └── g05/                    # Planned
│
├── configs/                    # Planned
├── tools/                      # Planned
├── outputs/                    # Training checkpoints, not committed
└── README.md
```

### Directory Description

- `pbs/`: PBS job scripts for running training jobs on NUS Hopper.
- `scripts/`: Model-specific fine-tuning scripts.
- `configs/`: Model and experiment configurations. Planned.
- `tools/`: Dataset validation, statistics, conversion, and utility scripts. Planned.
- `outputs/`: Locally generated checkpoints and training outputs. This directory should not be committed to Git.

Training scripts are organized by model rather than by task. For example:

```text
scripts/
├── molmoact2/
├── pi05/
├── groot/
├── lingbot/
└── g05/
```

Each model directory can contain training scripts for different robot embodiments, such as:

```text
train_so101.sh
train_droid.sh
```

## Install LeRobot

MolmoAct2 and π₀.₅ use **Hugging Face LeRobot** as the training framework.

LeRobot should be cloned and installed separately from this repository.

Recommended directory layout:

```text
~/projects/
├── lerobot/
└── robocolosseum-training/
```

Clone LeRobot:

```bash
git clone https://github.com/huggingface/lerobot.git
cd lerobot
```

Install it in the Python environment used for training:

```bash
pip install -e .
```

Verify the installation:

```bash
lerobot-train --help
```

## Fine-Tuning Example

### MolmoAct2 on SO-101

The MolmoAct2 SO-101 fine-tuning script is:

```text
scripts/molmoact2/train_so101.sh
```

Example:

```bash
lerobot-train \
	--policy.type=molmoact2 \
	--dataset.repo_id=YOUR_DATASET \
	--output_dir=outputs/molmoact2_so101
```

Replace `YOUR_DATASET` with the RoboColosseum training dataset.

The corresponding Hopper PBS job is:

```text
pbs/molmoact2_so101.pbs
```

Submit it with:

```bash
qsub pbs/molmoact2_so101.pbs
```

Additional models and embodiments will follow the same repository organization where possible, with model-specific training commands kept under their corresponding directory in `scripts/`.
