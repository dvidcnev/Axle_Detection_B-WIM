# Axle Detection — Command Reference

All commands are run from `D:\Thesis`. Open a terminal there first:
```
cd D:\Thesis
```

---

## One-time setup (first time only)

```powershell
# Create the virtual environment and install all packages
.\setup_env.ps1
```

---

## Training

### Start a fresh training run

```powershell
# Train TCN (recommended — best results)
python scripts\run_training.py --model tcn

# Train CNN
python scripts\run_training.py --model cnn

# Train both sequentially (CNN first, then TCN)
python scripts\run_training.py --model both
```

### Recommended full runs (what the thesis uses)

```powershell
# TCN — give it room to converge (takes ~3-4 hours)
python scripts\run_training.py --model tcn --epochs 100 --patience 25

# CNN — it converges fast, 50 epochs is enough
python scripts\run_training.py --model cnn --epochs 50 --patience 20
```

### Resume after a crash or cancellation

```powershell
# Picks up from the last saved checkpoint, keeps the log
python scripts\run_training.py --model tcn --resume --epochs 100 --patience 25
python scripts\run_training.py --model cnn --resume --epochs 50 --patience 20
```

### Run in a separate window (survives terminal close)

```powershell
Start-Process powershell -ArgumentList '-NoExit', '-Command', `
  'cd D:\Thesis; $env:PYTHONPATH="D:\Thesis"; python scripts\run_training.py --model tcn --epochs 100 --patience 25'
```

### Cross-validation (K-fold)

```powershell
# 5-fold CV for TCN (recommended)
$env:PYTHONPATH = "D:\Thesis"; .venv\Scripts\python.exe -m src.train --model tcn --epochs 50 --folds 5

# 5-fold CV for CNN
$env:PYTHONPATH = "D:\Thesis"; .venv\Scripts\python.exe -m src.train --model cnn --epochs 50 --folds 5
```

Each fold trains independently. When all folds finish, the script prints mean ± std across folds and copies the best fold's checkpoint to `checkpoints/<model>_best.pt`.

---

### All training flags

| Flag | Default | Meaning |
|---|---|---|
| `--model` | `tcn` | Which model: `cnn`, `tcn`, or `both` |
| `--epochs` | `50` | How many epochs to run |
| `--lr` | `0.001` | Learning rate (try `1e-4` for fine-tuning) |
| `--batch_size` | `64` | Samples per batch |
| `--patience` | `10` | Stop early if F1 doesn't improve for this many epochs |
| `--resume` | off | Continue from saved checkpoint instead of starting fresh |
| `--seed` | `42` | Random seed (change for reproducibility experiments) |

---

## Monitoring (open in a second terminal while training runs)

```powershell
# Watch both models
.venv\Scripts\python.exe scripts\watch_training.py

# Watch one model
.venv\Scripts\python.exe scripts\watch_training.py --model tcn

# Match your epoch count so the progress bar is correct
.venv\Scripts\python.exe scripts\watch_training.py --model tcn --epochs 100

# Change refresh rate (default 10 seconds)
.venv\Scripts\python.exe scripts\watch_training.py --interval 5
```

---

## Evaluation & comparison

```powershell
# Run full test-set evaluation: Baseline vs CNN vs TCN
# Outputs results to console + saves plots to outputs/
$env:PYTHONPATH = "D:\Thesis"; .venv\Scripts\python.exe scripts\compare.py
```

Output files saved to `outputs/`:
- `comparison_bar.png` — Precision / Recall / F1 / MATE bar chart
- `f1_vs_tolerance.png` — F1 score across different timing tolerances
- `training_curves.png` — Loss and F1 over epochs

---

## Inspect checkpoints

```powershell
# See what's saved
Get-ChildItem D:\Thesis\checkpoints\

# Check best epoch and F1 stored in a checkpoint
cmd /c "cd /d D:\Thesis && .venv\Scripts\python.exe -c ""import torch; c=torch.load('checkpoints/tcn_best.pt',map_location='cpu',weights_only=False); print('epoch:', c['epoch'], '| val F1:', round(c['val_f1'],4))"""
cmd /c "cd /d D:\Thesis && .venv\Scripts\python.exe -c ""import torch; c=torch.load('checkpoints/cnn_best.pt',map_location='cpu',weights_only=False); print('epoch:', c['epoch'], '| val F1:', round(c['val_f1'],4))"""

# Read the training log CSV directly
Get-Content D:\Thesis\checkpoints\tcn_log.csv
Get-Content D:\Thesis\checkpoints\cnn_log.csv

# Just the last 5 epochs
Get-Content D:\Thesis\checkpoints\tcn_log.csv | Select-Object -Last 5
```

---

## Typical workflow start-to-finish

```powershell
# 1. Start training in one terminal
.venv\Scripts\python.exe scripts\run_training.py --model tcn --epochs 100 --patience 25

# 2. Open a second terminal and watch it live
.venv\Scripts\python.exe scripts\watch_training.py --model tcn --epochs 100

# 3. After training finishes, run evaluation
$env:PYTHONPATH = "D:\Thesis"; .venv\Scripts\python.exe scripts\compare.py

# 4. Check results
Get-Content D:\Thesis\outputs\eval_output.txt
```

---

## Quick diagnostics

```powershell
# Check GPU is available
cmd /c "cd /d D:\Thesis && .venv\Scripts\python.exe -c ""import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"""

# Check all packages are installed
.venv\Scripts\pip.exe list | Select-String "torch|numpy|scipy|scikit|tqdm|matplotlib|pandas"
```
