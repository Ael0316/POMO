# POMO

This repository contains a POMO-based Traveling Salesman Problem (TSP) training and evaluation pipeline, including preference-optimization (PO) training, reinforcement learning (RL) training, and TSPLIB evaluation scripts.

## Project Overview

The current codebase focuses on TSP with problem size 100 and provides:

- `train.py`: the main training entrypoint driven by `configs/train_full.json`
- `test.py`: the standardized TSPLIB evaluation entrypoint
- `train_n100_po.py`: legacy script for PO training on TSP100
- `train_n100_rl.py`: legacy script for RL training on TSP100
- `test_n100.py`: legacy validation script for TSP100

Training artifacts are written under `result/`, and evaluation logs are written under `result_lib/`.

## Repository Structure

```text
POMO/
├── configs/
│   └── train_full.json
├── TSPEnv.py
├── TSPLocalSearch.py
├── TSPModel.py
├── TSPTester.py
├── TSPTester_LIB.py
├── TSPTrainer.py
├── test.py
├── test_n100.py
├── train.py
├── train_n100_po.py
├── train_n100_rl.py
└── tsplib_utils.py
```

## Requirements

This repository depends on Python 3 and common scientific computing packages such as:

- `torch`
- `numpy`
- `pytz`

## Important Dependency Note

This repository is not fully self-contained yet.

Several scripts import shared modules from parent directories via:

```python
sys.path.insert(0, "..")
sys.path.insert(0, "../..")
```

In particular, the following modules are expected to exist outside this repository:

- `utils.utils`
- `TSProblemDef.py`

If you plan to run training or evaluation directly from this repository, make sure those shared dependencies are available in the expected parent directories, or vendor them into this project before running the scripts.

## Training

The recommended training entrypoint is `train.py`, which reads configuration from `configs/train_full.json`.

Run training with:

```bash
python train.py
```

Use a custom config file:

```bash
python train.py --config_path configs/train_full.json
```

Run a small debug job:

```bash
python train.py --debug
```

Force CPU training:

```bash
python train.py --use_cuda false
```

## Evaluation

The recommended evaluation entrypoint is `test.py`.

Example:

```bash
python test.py \
  --data_path /path/to/tsplib_dir \
  --model_dir ./result/saved_tsp100_po \
  --epoch 2010
```

You can also point directly to a checkpoint:

```bash
python test.py \
  --data_path /path/to/tsplib_dir \
  --checkpoint_path /path/to/checkpoint.pt
```

Optional JSON output:

```bash
python test.py \
  --data_path /path/to/tsplib_dir \
  --checkpoint_path /path/to/checkpoint.pt \
  --output_json ./result_lib/eval.json
```

## Configuration

The main training configuration lives in [`configs/train_full.json`](./configs/train_full.json).

It currently defines:

- TSP size: `100`
- POMO size: `100`
- training epochs: `2010`
- batch size: `64`
- loss type: `po_loss`
- optional sparse backtracking and RL-mix settings

## Git Ignore

The repository ignores runtime or local-only files through `.gitignore`, including:

- `.codex`
- `__pycache__/`
- `result/`
- `result_lib/`

## Status

This README documents the current repository layout and usage based on the checked-in scripts. If you later vendor the missing shared utilities into this repository, it would be a good next step to add a pinned `requirements.txt` or `environment.yml` for reproducible setup.
