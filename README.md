# AMP_mjlab

[中文 README](README_zh.md)

Deployment integration code is in [ccrpRepo/wbc_fsm](https://github.com/ccrpRepo/wbc_fsm), under `MJAmp State`.

G1 AMP motion control project built on top of mjlab + rsl_rl.

Key features of this repository:

- A single policy learns both locomotion (walk/run) and recovery (fall-and-get-up)
- AMP discriminator regularizes motion style and priors
- Training and deployment pipelines are consistent, with direct ONNX policy export support
- A dedicated HoST-based stand-up task (`Unitree-G1-HoST-StandUp`) learns to rise from a prone posture and stay stable

## Core Idea

Instead of training separate policies for locomotion and recovery and switching between them, this project learns both capabilities in one unified policy.

Implementation highlights:

- Motion data split:
  - Walk/Run data: `src/assets/motions/g1/amp/WalkandRun`
  - Recovery data: `src/assets/motions/g1/amp/Recovery`
- Delayed termination/reset:
  - A subset of environments does not reset immediately after termination
  - These environments receive a recovery window and reset states sampled from recovery clips
- Unified AMP training:
  - One actor-critic + One AMP discriminator
  - Velocity tracking, perturbation robustness, and recovery are learned together

This reduces discontinuities caused by policy switching and yields more consistent behavior.

## Requirements

- Linux
- Python 3.11 (recommended)
- Working MuJoCo and GPU driver setup

## Quick Start

### 1. Install

```bash
conda activate mjlab
cd AMP_mjlab
python -m pip install -e .
```

### 2. Apply mjlab Patch (Optional)

If you do not apply this patch, remove `history_ordering` configuration from the code.

What this patch does:

- It adds an option for how observation history is flattened: by time (`time`) or by term (`term`).
- Default mjlab behavior supports only `term` ordering.

Patch file:

- `mjlab_patch/mjlab/managers/observation_manager.py`

Example command:

```bash
cp mjlab_patch/mjlab/managers/observation_manager.py \
  /home/crp/miniconda3/envs/mjlab/lib/python3.11/site-packages/mjlab/managers/observation_manager.py
```

### 3. List Available Tasks

```bash
python scripts/list_envs.py --keyword AMP
```

Main tasks:

- `Unitree-G1-AMP-Rough`
- `Unitree-G1-AMP-Flat`
- `Unitree-G1-HoST-StandUp` (see [HoST Stand-up Recovery](#host-stand-up-recovery))

## Training

```bash
python scripts/train.py Unitree-G1-AMP-Flat --env.scene.num-envs=4096
```

Logs are saved by default to:

- `logs/rsl_rl/g1_amp_locomotion/<time_stamp_run>/`

## Training Curve Note (Important)

- Around `2w` iterations (about 20k), the policy often suddenly learns fall-recovery behavior.
- As a result, multiple metrics in `logs` may show abrupt jumps. This is expected and not necessarily a training failure.

![Training log transition example](logs.png)

## Evaluation and Visualization

Replay with a trained checkpoint:

```bash
python scripts/play.py Unitree-G1-AMP-Rough \
  --checkpoint-file logs/rsl_rl/g1_amp_locomotion/<run_dir>/model_<iter>.pt
```

Note: ONNX export is enabled by default in both training and play workflows.

## HoST Stand-up Recovery

In addition to the unified AMP policy, this repository also ships a dedicated
stand-up recovery task migrated from [OpenRobotLab/HoST](https://github.com/OpenRobotLab/HoST)
(RSS 2025, *Learning Humanoid Standing-up Control across Diverse Postures*).

- Task: `Unitree-G1-HoST-StandUp`
- 23-DoF G1 on flat terrain, starting from a prone posture and learning to stand up and stay stable
- Uses the same mjlab manager architecture, ONNX export, and train/play entry points

```bash
python scripts/list_envs.py --keyword HoST
python scripts/train.py Unitree-G1-HoST-StandUp --env.scene.num-envs=4096
python scripts/play.py Unitree-G1-HoST-StandUp --checkpoint-file <ckpt>
```

Headless evaluation, recording, and resume fine-tuning:

```bash
CKPT=<checkpoint.pt> python scripts/eval_stand.py
CKPT=<checkpoint.pt> python scripts/eval_stand_perjoint.py
CKPT=<checkpoint.pt> python scripts/record_stand.py
CKPT=<checkpoint.pt> LOG_DIR=<log_dir> python scripts/resume_finetune.py
```

For migration details, known differences, and unimplemented items, see
[`src/tasks/host_recovery/README.md`](src/tasks/host_recovery/README.md).

## Motion Data Preparation

CSV-to-NPZ conversion script:

```bash
python scripts/csv_to_npz.py --help
```

Recommended data layout:

- Raw CSV: `motion_data_csv/amp`
- Converted NPZ: `src/assets/motions/g1/amp/WalkandRun` and `src/assets/motions/g1/amp/Recovery`

If valid NPZ files exist in these folders, training config loads them automatically.

## Repository Structure

- `src/tasks/amp_loco`: AMP locomotion/recovery task implementation
- `src/tasks/amp_loco/config/g1`: G1 task registration, env configs, RL configs
- `src/tasks/amp_loco/mdp`: rewards, observations, events, termination logic
- `src/tasks/host_recovery`: HoST stand-up recovery task implementation
- `src/tasks/host_recovery/config/g1`: G1 stand-up registration, env/RL configs
- `src/tasks/host_recovery/mdp`: stand-up rewards, observations, events, curriculum
- `scripts/train.py`: training entry point
- `scripts/play.py`: playback entry point
- `scripts/eval_stand.py` / `scripts/eval_stand_perjoint.py`: headless stand-up evaluation
- `scripts/record_stand.py`: headless stand-up video recording
- `scripts/resume_finetune.py`: resume stand-up fine-tuning
- `scripts/csv_to_npz.py`: motion data conversion tool
- `mjlab_patch`: required local patch for mjlab

## Highlights

- One policy unifies walk/run and recovery skills
- AMP + velocity objective jointly optimize style and task performance
- Delayed reset with recovery sampling explicitly improves recovery ability
- Dedicated HoST stand-up task learns to rise from prone and stay upright
- End-to-end pipeline supports ONNX export for deployment

## Acknowledgements

- Thanks to [unitreerobotics/unitree_rl_mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab) for open-sourcing their work and inspiration.
- Thanks to [Open-X-Humanoid/TienKung-Lab](https://github.com/Open-X-Humanoid/TienKung-Lab); the rsl_rl AMP part in this project references their implementation.
- Thanks to [OpenRobotLab/HoST](https://github.com/OpenRobotLab/HoST); the stand-up recovery task in this project is migrated from its framework.
