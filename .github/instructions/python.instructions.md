---
description: "Use when writing or editing Python source code for the B-WIM axle detection project. Covers model conventions, evaluation rules, training patterns, and data handling."
applyTo: "**/*.py"
---

# Python Coding Rules — B-WIM Axle Detection

## Environment
- Always run Python via the `.venv` in the workspace root: `d:\Thesis\.venv\Scripts\python.exe`
- Never use system Python or a globally installed interpreter.

## Class Imbalance — Non-negotiable Rules
- Loss must always be `BCEWithLogitsLoss`. Never use plain `BCELoss` on sigmoid outputs.
- `pos_weight` must be set to `~287` (the dataset's negative-to-positive ratio). Never omit it in full training runs.
- Exception: the sanity-check overfit cell uses plain `BCEWithLogitsLoss()` without `pos_weight` intentionally — do not add it there.

## Evaluation (`src/evaluate.py`)
- Axle-level evaluation uses a **±5 sample tolerance window** with **greedy matching** (closest first). Do not change this.
- Peak extraction (`pulses_to_peaks`) must use `min_distance=5` for both baseline and neural network outputs to keep the comparison fair.
- Never default to `threshold=0.5` without justification — a threshold sweep on the validation set should select the optimal cut-off.
- Metrics: Precision, Recall, F1, and MATE (Mean Absolute Timing Error in samples). All four must be reported together.

## Model Conventions
- CNN checkpoint: `checkpoints/cnn_best.pt`
- TCN checkpoint: `checkpoints/tcn_best.pt`
- Training logs: `checkpoints/cnn_log.csv`, `checkpoints/tcn_log.csv`
- Log columns must be: `epoch, train_loss, val_loss, val_f1, val_precision, val_recall, val_mate, lr`
- Checkpoint dict must contain keys: `state_dict`, `epoch`, `val_f1`

## Training CLI
- Always invoke training as a module: `python -m src.train --model <cnn|tcn> --epochs N --batch_size 64`
- Never run `src/train.py` directly.

## Data
- Dataset file path: `axle_data.json/axle_data.json` (relative to workspace root) or `../axle_data.json/axle_data.json` from inside `notebooks/`
- Never commit `axle_data.json` or anything under `checkpoints/` to git.
- Signal shape into models: `[B, 1, 1300]` — batch, channel, time. Target shape: `[B, 1, 1300]`.

## Code Style
- Use `torch.no_grad()` context manager for all inference calls.
- Prefer `model.eval()` before evaluation loops; restore with `model.train()` after.
- Use `f-strings` for all print/logging statements.
- Do not add docstrings or type annotations to code that isn't being changed.
