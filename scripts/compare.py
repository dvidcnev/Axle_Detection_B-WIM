import os
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from src.dataset import build_datasets
from src.models import AxleUNet, AxleTCN
from src.baseline import signal_to_pulse_peaks
from src.evaluate import axle_level_metrics, evaluate_model, print_metrics

JSON_PATH = 'axle_data.json/axle_data.json'
CKPT_DIR  = 'checkpoints'
DEVICE    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED      = 42

os.makedirs('outputs', exist_ok=True)

# load data
_, _, test_ds = build_datasets(JSON_PATH, seed=SEED)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

# load cnn
cnn_model = AxleUNet().to(DEVICE)
cnn_ckpt  = torch.load(os.path.join(CKPT_DIR, 'cnn_best.pt'), map_location=DEVICE)
cnn_model.load_state_dict(cnn_ckpt['state_dict'])

# load tcn
tcn_model = AxleTCN().to(DEVICE)
tcn_ckpt  = torch.load(os.path.join(CKPT_DIR, 'tcn_best.pt'), map_location=DEVICE)
tcn_model.load_state_dict(tcn_ckpt['state_dict'])

# baseline preds
test_signals = [test_ds[i][0].squeeze(0).numpy() for i in range(len(test_ds))]
test_pulses  = [test_ds[i][1].numpy() for i in range(len(test_ds))]

# Evaluate at tolerance=5
TOL = 5
pred_base = [signal_to_pulse_peaks(s) for s in test_signals]
base_metrics = axle_level_metrics(test_pulses, pred_base, tolerance=TOL)
cnn_metrics  = evaluate_model(cnn_model, test_loader, DEVICE, tolerance=TOL)
tcn_metrics  = evaluate_model(tcn_model, test_loader, DEVICE, tolerance=TOL)

print('Baseline:'); print_metrics(base_metrics, prefix='Baseline')
print('CNN:'); print_metrics(cnn_metrics, prefix='CNN')
print('TCN:'); print_metrics(tcn_metrics, prefix='TCN')

# Summary table
import pandas as pd
rows = []
for name, m in [('Peak Detection (baseline)', base_metrics), ('1D U-Net CNN', cnn_metrics), ('TCN', tcn_metrics)]:
    rows.append({
        'Method': name,
        'Precision': m['precision'],
        'Recall': m['recall'],
        'F1': m['f1'],
        'MATE': m['mate'],
        'TP': m['tp'], 'FP': m['fp'], 'FN': m['fn']
    })
df = pd.DataFrame(rows).set_index('Method')
print('\nSummary:')
print(df)

def save_bar_chart(base_metrics, cnn_metrics, tcn_metrics, tol):
    labels = ['Baseline','CNN','TCN']
    metrics_list = [base_metrics, cnn_metrics, tcn_metrics]
    x = np.arange(len(labels))
    width = 0.22
    fig, axes = plt.subplots(1,2,figsize=(12,4))
    # scores
    axes[0].bar(x - width, [m['precision'] for m in metrics_list], width, label='Precision')
    axes[0].bar(x, [m['recall'] for m in metrics_list], width, label='Recall')
    axes[0].bar(x + width, [m['f1'] for m in metrics_list], width, label='F1')
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels)
    axes[0].set_ylim(0,1.05); axes[0].set_title(f'Scores (tol={tol})')
    axes[0].legend()
    # MATE
    axes[1].bar(labels, [m['mate'] for m in metrics_list], color=['steelblue','orange','green'])
    axes[1].set_title('MATE (samples)')
    plt.tight_layout()
    fig.savefig('outputs/comparison_bar.png')

save_bar_chart(base_metrics, cnn_metrics, tcn_metrics, TOL)

# F1 vs tolerance
tolerances = [1,2,3,5,8,10,15,20]
results = {'Baseline': [], 'CNN': [], 'TCN': []}
for tol in tolerances:
    results['Baseline'].append(axle_level_metrics(test_pulses, pred_base, tolerance=tol)['f1'])
    results['CNN'].append(evaluate_model(cnn_model, test_loader, DEVICE, tolerance=tol)['f1'])
    results['TCN'].append(evaluate_model(tcn_model, test_loader, DEVICE, tolerance=tol)['f1'])

plt.figure(figsize=(8,4))
for name, f1s in results.items():
    plt.plot(tolerances, f1s, marker='o', label=name)
plt.xlabel('Tolerance'); plt.ylabel('F1'); plt.title('F1 vs tolerance'); plt.legend(); plt.grid(True)
plt.tight_layout(); plt.savefig('outputs/f1_vs_tolerance.png')

# Training curves overlay

def read_log(path):
    epochs, train_l, val_l, val_f1 = [], [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            epochs.append(int(row['epoch'])); train_l.append(float(row['train_loss'])); val_l.append(float(row['val_loss'])); val_f1.append(float(row['val_f1']))
    return epochs, train_l, val_l, val_f1

cnn_e, cnn_tl, cnn_vl, cnn_f1 = read_log(os.path.join(CKPT_DIR, 'cnn_log.csv'))
tcn_e, tcn_tl, tcn_vl, tcn_f1 = read_log(os.path.join(CKPT_DIR, 'tcn_log.csv'))
fig, (ax1, ax2) = plt.subplots(1,2,figsize=(13,4))
ax1.plot(cnn_e, cnn_vl, label='CNN val loss', color='steelblue')
ax1.plot(tcn_e, tcn_vl, label='TCN val loss', color='tomato')
ax1.set_xlabel('Epoch'); ax1.set_ylabel('BCE Loss'); ax1.set_title('Validation loss'); ax1.legend()
ax2.plot(cnn_e, cnn_f1, label='CNN val F1', color='steelblue', marker='.')
ax2.plot(tcn_e, tcn_f1, label='TCN val F1', color='tomato', marker='.')
ax2.set_xlabel('Epoch'); ax2.set_ylabel('F1'); ax2.set_title('Validation F1'); ax2.legend()
plt.tight_layout(); fig.savefig('outputs/training_curves.png')

print('\nSaved plots to outputs/ (comparison_bar.png, f1_vs_tolerance.png, training_curves.png)')
