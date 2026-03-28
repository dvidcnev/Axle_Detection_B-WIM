"""
train.py
--------
Training loop for the 1D CNN (AxleUNet) and TCN (AxleTCN) models.

Usage (from project root):
    python -m src.train --model cnn --epochs 50 --batch_size 64
    python -m src.train --model tcn --epochs 50 --batch_size 64

Features
--------
* BCEWithLogitsLoss with automatic pos_weight to handle class imbalance
* AdamW optimiser with ReduceLROnPlateau scheduler
* Early stopping on validation F1 (patience = 10 epochs)
* Mixed-precision training (AMP) when GPU is available
* Saves best checkpoint to checkpoints/<model>_best.pt
* Logs per-epoch train loss, val loss, and val F1 to stdout + CSV
"""

import argparse
import csv
import os
import sys
import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Make sure src/ is on the path when running as __main__
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset  import build_datasets
from src.evaluate import evaluate_model, print_metrics
from src.models   import AxleUNet, AxleTCN


# ---------------------------------------------------------------------------
# Loss: BCEWithLogitsLoss with automatic pos_weight
# ---------------------------------------------------------------------------

def build_loss(train_dataset, device: torch.device) -> nn.BCEWithLogitsLoss:
    """Compute pos_weight from the training set to handle class imbalance."""
    pos = 0.0
    total = 0.0
    for _, pulse in tqdm(train_dataset, desc="Computing pos_weight", leave=False):
        pos   += float(pulse.sum())
        total += float(pulse.numel())
    neg = total - pos
    pw  = neg / (pos + 1e-8)
    print(f"  pos_weight = {pw:.2f}  (positives: {pos:.0f} / {total:.0f} = {100*pos/total:.3f}%)")
    return nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw, device=device))


# ---------------------------------------------------------------------------
# One epoch of training
# ---------------------------------------------------------------------------

def train_one_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device:    torch.device,
    scaler:    Optional[torch.cuda.amp.GradScaler],
) -> float:
    model.train()
    total_loss = 0.0

    for signals, pulses in tqdm(loader, desc="  train", leave=False):
        signals = signals.to(device)   # [B, 1, L]
        pulses  = pulses.to(device)    # [B, L]

        optimizer.zero_grad()

        if scaler is not None:
            with torch.autocast(device_type="cuda"):
                logits = model(signals)
                loss   = criterion(logits, pulses)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(signals)
            loss   = criterion(logits, pulses)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item() * signals.size(0)

    return total_loss / len(loader.dataset)


# ---------------------------------------------------------------------------
# One epoch of validation (loss only)
# ---------------------------------------------------------------------------

@torch.no_grad()
def val_loss_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    for signals, pulses in loader:
        signals = signals.to(device)
        pulses  = pulses.to(device)
        logits  = model(signals)
        loss    = criterion(logits, pulses)
        total_loss += loss.item() * signals.size(0)
    return total_loss / len(loader.dataset)


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def train(
    model_name:  str   = "cnn",
    json_path:   str   = "axle_data.json/axle_data.json",
    epochs:      int   = 50,
    batch_size:  int   = 64,
    lr:          float = 1e-3,
    patience:    int   = 10,
    num_workers: int   = 0,
    seed:        int   = 42,
    checkpoint_dir: str = "checkpoints",
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"Model  : {model_name.upper()}")

    # --- Data ---
    train_ds, val_ds, test_ds = build_datasets(json_path)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=(device.type == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=(device.type == "cuda"))
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers)

    # --- Model ---
    if model_name.lower() == "cnn":
        model = AxleUNet().to(device)
    elif model_name.lower() == "tcn":
        model = AxleTCN().to(device)
    else:
        raise ValueError(f"Unknown model '{model_name}'. Choose 'cnn' or 'tcn'.")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params : {total_params:,}")

    # --- Loss ---
    criterion = build_loss(train_ds, device)

    # --- Optimiser & scheduler ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    # --- Mixed precision ---
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    # --- Logging ---
    os.makedirs(checkpoint_dir, exist_ok=True)
    log_path  = os.path.join(checkpoint_dir, f"{model_name}_log.csv")
    ckpt_path = os.path.join(checkpoint_dir, f"{model_name}_best.pt")

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_f1",
                         "val_precision", "val_recall", "val_mate", "lr"])

    # --- Training loop ---
    best_val_f1  = -1.0
    no_improve   = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss   = val_loss_epoch(model, val_loader, criterion, device)
        val_metrics = evaluate_model(model, val_loader, device)

        val_f1 = val_metrics["f1"]
        scheduler.step(val_f1)
        current_lr = optimizer.param_groups[0]["lr"]

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"train_loss={train_loss:.5f}  val_loss={val_loss:.5f} | ",
            end="",
        )
        print_metrics(val_metrics, prefix=f"val")
        print(f"  lr={current_lr:.2e}  time={elapsed:.1f}s")

        # Save log
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, f"{train_loss:.6f}", f"{val_loss:.6f}",
                f"{val_f1:.6f}", f"{val_metrics['precision']:.6f}",
                f"{val_metrics['recall']:.6f}", f"{val_metrics['mate']:.4f}",
                f"{current_lr:.2e}",
            ])

        # Checkpoint
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            no_improve  = 0
            torch.save({
                "epoch":      epoch,
                "model_name": model_name,
                "state_dict": model.state_dict(),
                "val_f1":     val_f1,
                "val_metrics": val_metrics,
            }, ckpt_path)
            print(f"  ✓ Saved best checkpoint (val F1={val_f1:.4f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    # --- Final test evaluation ---
    print("\n--- Loading best checkpoint for test evaluation ---")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    
    # Threshold sweep on validation set
    print("Sweeping probability thresholds on validation set...")
    from src.evaluate import get_model_predictions, find_best_threshold
    val_targets, val_preds = get_model_predictions(model, val_loader, device)
    best_thresh, val_best_metrics = find_best_threshold(val_targets, val_preds)
    print(f"Optimal threshold found: {best_thresh:.2f} (Val F1={val_best_metrics['f1']:.4f})")

    # Evaluate on test set with optimal threshold
    test_metrics = evaluate_model(model, test_loader, device, threshold=best_thresh)
    print(f"Test results (thresh={best_thresh:.2f}):")
    print_metrics(test_metrics, prefix="test")
    print(f"Log saved to: {log_path}")
    return test_metrics


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CNN or TCN for axle detection")
    parser.add_argument("--model",       type=str,   default="cnn",
                        choices=["cnn", "tcn"])
    parser.add_argument("--json_path",   type=str,   default="axle_data.json/axle_data.json")
    parser.add_argument("--epochs",      type=int,   default=50)
    parser.add_argument("--batch_size",  type=int,   default=64)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--patience",    type=int,   default=10)
    parser.add_argument("--num_workers", type=int,   default=0)
    parser.add_argument("--seed",        type=int,   default=42)
    args = parser.parse_args()

    train(
        model_name   = args.model,
        json_path    = args.json_path,
        epochs       = args.epochs,
        batch_size   = args.batch_size,
        lr           = args.lr,
        patience     = args.patience,
        num_workers  = args.num_workers,
        seed         = args.seed,
    )
