# Axle Detection B-WIM Project Instructions

This file contains the core context, architectural decisions, and data structure information for the Axle Detection project. 
Any AI assistant joining this workspace should read this first.

## Project Context
- **Goal:** Detect truck axles (wheel pairs) from Bridge Weigh-In-Motion (B-WIM) strain sensor signals. Find the exact sample positions where axles occur.
- **Why it matters:** Accurate axle detection is required to calculate the gross vehicle weight (GVW) of heavy goods vehicles. If the axle detector gets confused by noise, the weight math will be entirely wrong.

## Data Structure (`axle_data.json`)
- **Total records:** 32,141 vehicles.
- **`signal`:** 1,300-sample array. A 1D strain signal covering the entire bridge crossing.
- **`pulses`:** 1,300-sample binary array. Contains `1.0` at the exact sample where an axle is present, `0.0` everywhere else.
- **Class Imbalance:** Extreme. Only ~0.348% of samples are positive (axles). This is a negative:positive ratio of **287:1**.
- **Axle counts:** Vehicles have between 2 and 11 axles (mean ~4.5).

## Methodology & Models
We are comparing a traditional "old-school" math baseline with two Deep Learning architectures.
1. **Baseline (`src/baseline.py`)**: Uses SciPy's `savgol_filter` and `find_peaks`.
2. **CNN (`src/models/cnn.py`)**: A 1D U-Net approach (encoder/decoder with skip connections).
3. **TCN (`src/models/tcn.py`)**: A Temporal Convolutional Network with dilated residual blocks (non-causal, symmetric padding) and `num_blocks=9` to ensure the receptive field covers the entire 1300-sample sequence.

## Critical Implementation Details
- **Loss Function:** Because of the 287:1 imbalance, we use `BCEWithLogitsLoss`. The `pos_weight` parameter MUST be set to ~287 so the network doesn't "cheat" by guessing 0 everywhere.
- **Threshold Sweep:** Because axles are so rare, the model's confidence logic will be low. Right before final evaluation, `src/train.py` sweeps probability thresholds (0.1 to 0.9) on the validation set to find the optimal cut-off, rather than defaulting to 0.5.
- **Evaluation (`src/evaluate.py`):** Metrics (Precision, Recall, F1, and MATE - Mean Absolute Timing Error) are calculated at the **axle level** using a ±5 sample tolerance window greedy-match, not strict sample-by-sample classification.

## Tech Stack
- Python 3.12 via `.venv`
- PyTorch (for CNN/TCN)
- SciPy / NumPy / Matplotlib
- Jupyter Notebooks for exploration and comparison

## Rules for AI Assistants
1. Do not commit or track `axle_data.json` or `checkpoints/` to git.
2. If modifying `src/evaluate.py`, ensure the post-processing (finding peaks via `min_distance=5`) remains identical for both the baseline and the neural network models so the comparison remains fair.
3. If running training, always execute via CLI: `python -m src.train --model tcn --epochs 50 --batch_size 64`
