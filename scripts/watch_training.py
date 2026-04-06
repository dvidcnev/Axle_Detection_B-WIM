"""
watch_training.py
-----------------
Live dashboard for monitoring CNN and TCN training progress.

Reads checkpoints/<model>_log.csv every few seconds and prints a
formatted table that refreshes in-place.

Usage (from project root):
    python scripts/watch_training.py              # watch both models
    python scripts/watch_training.py --model cnn  # watch only CNN
    python scripts/watch_training.py --model tcn  # watch only TCN
    python scripts/watch_training.py --interval 5 # refresh every 5 s
"""

import argparse
import csv
import os
import sys
import time
from datetime import datetime

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
CHECKPOINT_DIR = os.path.normpath(CHECKPOINT_DIR)

COLS = ["epoch", "train_loss", "val_loss", "val_f1",
        "val_precision", "val_recall", "val_mate", "lr"]

# ANSI colours (skip on systems that don't support them)
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"


def _supports_ansi() -> bool:
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def colour(text: str, code: str) -> str:
    return f"{code}{text}{RESET}" if _supports_ansi() else text


def read_log(model_name: str) -> list[dict]:
    path = os.path.join(CHECKPOINT_DIR, f"{model_name}_log.csv")
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def bar(value: float, width: int = 20, lo: float = 0.0, hi: float = 1.0) -> str:
    """ASCII progress bar for a 0-1 metric."""
    clamped = max(lo, min(hi, value))
    filled  = int(round((clamped - lo) / (hi - lo) * width))
    return "[" + "█" * filled + "░" * (width - filled) + "]"


def render(models: list[str], total_epochs: int) -> None:
    """Render a snapshot of all requested model logs."""
    now = datetime.now().strftime("%H:%M:%S")
    lines = []
    lines.append(colour(f"\n{'─'*72}", CYAN))
    lines.append(colour(f"  Axle Detection — Training Monitor   {now}", BOLD))
    lines.append(colour(f"{'─'*72}", CYAN))

    for model_name in models:
        rows = read_log(model_name)
        path = os.path.join(CHECKPOINT_DIR, f"{model_name}_log.csv")

        header = colour(f"\n  [{model_name.upper()}]", BOLD + YELLOW)
        if not rows:
            if not os.path.exists(path):
                lines.append(f"{header}  {colour('log not found — training not started yet', YELLOW)}")
            else:
                lines.append(f"{header}  {colour('log empty — waiting for first epoch…', YELLOW)}")
            continue

        last   = rows[-1]
        epoch  = int(last["epoch"])
        t_loss = float(last["train_loss"])
        v_loss = float(last["val_loss"])
        v_f1   = float(last["val_f1"])
        v_prec = float(last["val_precision"])
        v_rec  = float(last["val_recall"])
        v_mate = float(last["val_mate"])
        lr     = last["lr"]

        best_f1_row = max(rows, key=lambda r: float(r["val_f1"]))
        best_f1     = float(best_f1_row["val_f1"])
        best_epoch  = int(best_f1_row["epoch"])

        f1_bar = bar(v_f1)
        f1_str = colour(f"{v_f1:.4f}", GREEN if v_f1 >= best_f1 - 1e-6 else RESET)

        lines.append(header)
        lines.append(
            f"    Progress : epoch {epoch}/{total_epochs}  "
            + colour(f"{'█' * epoch + '░' * (total_epochs - epoch)}", CYAN)[:total_epochs + 2]
        )
        lines.append(f"    Loss     : train={t_loss:.5f}   val={v_loss:.5f}   lr={lr}")
        lines.append(
            f"    F1       : {f1_str}  {f1_bar}  "
            f"(best={colour(f'{best_f1:.4f}', GREEN)} @ epoch {best_epoch})"
        )
        lines.append(f"    Precision: {v_prec:.4f}   Recall: {v_rec:.4f}   MATE: {v_mate:.2f} samples")

        # Trend: last 5 val_f1 values
        if len(rows) >= 2:
            recent = [float(r["val_f1"]) for r in rows[-5:]]
            trend  = "  ↑" if recent[-1] > recent[0] else ("  ↓" if recent[-1] < recent[0] else "  →")
            trend_vals = "  ".join(f"{v:.4f}" for v in recent)
            lines.append(f"    Trend    : {trend_vals}{colour(trend, GREEN if '↑' in trend else YELLOW)}")

    lines.append(colour(f"\n{'─'*72}", CYAN))
    lines.append("  Press Ctrl+C to stop watching.\n")

    # Move cursor to top and overwrite (only when in a real terminal)
    if _supports_ansi():
        sys.stdout.write("\033[H\033[J")  # clear screen
    print("\n".join(lines))
    sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser(description="Live training dashboard")
    parser.add_argument("--model",    type=str, default="both",
                        choices=["cnn", "tcn", "both"],
                        help="Which model log(s) to watch (default: both)")
    parser.add_argument("--epochs",   type=int, default=50,
                        help="Total epochs configured (for the progress bar, default: 50)")
    parser.add_argument("--interval", type=int, default=10,
                        help="Refresh interval in seconds (default: 10)")
    args = parser.parse_args()

    models = ["cnn", "tcn"] if args.model == "both" else [args.model]

    print(f"Watching {', '.join(m.upper() for m in models)} "
          f"in {CHECKPOINT_DIR}  (refresh every {args.interval}s)…")

    try:
        while True:
            render(models, args.epochs)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nMonitor stopped.")


if __name__ == "__main__":
    main()
