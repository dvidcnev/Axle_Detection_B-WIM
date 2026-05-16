---
description: "Use when creating, editing, or running Jupyter notebooks in the B-WIM thesis project. Covers cell structure, figure export, kernel selection, and documentation conventions."
applyTo: "**/*.ipynb"
---

# Notebook Rules — B-WIM Axle Detection

## Kernel
- Always use the `.venv (Python 3.13)` kernel located at `d:\Thesis\.venv\Scripts\python.exe`.
- If the kernel fails to start, reinstall ipykernel: `.venv\Scripts\pip install --force-reinstall ipykernel`

## Cell 1 — Setup (every notebook must have this)
```python
import sys, os
sys.path.insert(0, os.path.abspath('..'))
%matplotlib inline
plt.rcParams.update({'figure.dpi': 120, 'font.size': 11})
JSON_PATH = '../axle_data.json/axle_data.json'
CKPT_DIR  = '../checkpoints'
DEVICE    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED      = 42
```

## Notebook Structure
| Notebook | Purpose |
|---|---|
| `01_data_exploration.ipynb` | Data stats, class imbalance, signal examples |
| `02_baseline.ipynb` | Classical baseline — parameter tuning and test evaluation |
| `03_cnn_model.ipynb` | CNN architecture, overfit check, training curves, visual inspection |
| `04_tcn_model.ipynb` | TCN architecture, receptive field diagram, training curves, visual inspection |
| `05_comparison.ipynb` | Side-by-side comparison table and plots of all three methods |
| `06_error_analysis.ipynb` | Failure modes, error by axle count, spacing distribution |

## Figures
- Save all publication-quality figures to `d:\Thesis\latex-format\images\` as `.pdf` (vector) for LaTeX inclusion.
- Use `plt.savefig('../latex-format/images/<name>.pdf', bbox_inches='tight')` after `plt.show()`.
- Figure naming convention: `<model>_<description>.pdf` — e.g., `tcn_training_curves.pdf`, `comparison_f1_bar.pdf`

## Evaluation Consistency
- Use the **same** `evaluate_model` function from `src/evaluate.py` for all models.
- Always pass `tolerance=5` and the optimal threshold (not hardcoded 0.5) when reporting final test numbers.

## Visual Inspection Cells
- Always show **6** randomly selected test examples (seed `SEED=42`).
- Panel layout: signal (blue), model probability scaled to signal amplitude (orange), ground truth (green vlines), predictions (red dashed vlines).
- Title each panel with the record index, axle count, and ✓/✗ match indicator.

## Documentation
- Every notebook must have a markdown cell at the top with: title, architecture summary (if applicable), and the goals of that notebook as a numbered list.
- Section headings use `## N. Section name` numbering to match the thesis chapter structure.
