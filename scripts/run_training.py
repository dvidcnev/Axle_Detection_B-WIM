"""
run_training.py
---------------
Convenient launcher for CNN and/or TCN training.

Usage (from project root):
    python scripts/run_training.py --model cnn
    python scripts/run_training.py --model tcn
    python scripts/run_training.py --model both
    python scripts/run_training.py --model tcn --epochs 100 --batch_size 32

All extra flags are forwarded directly to src/train.py.
After launching, open a second terminal and run:
    python scripts/watch_training.py
to monitor live progress.
"""

import argparse
import subprocess
import sys
import os

# Always run from the project root regardless of where this script is called from
PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))


def _venv_python() -> str:
    """Return the .venv Python if it exists, else fall back to the current interpreter."""
    venv_py = os.path.join(PROJECT_ROOT, ".venv", "Scripts", "python.exe")
    if os.path.isfile(venv_py):
        return venv_py
    # Warn when the venv is missing so the user knows what to do
    print(
        "[WARNING] .venv not found — using system Python.\n"
        "          Run  .\\setup_env.ps1  from the project root first.\n"
    )
    return sys.executable


DEFAULTS = {
    "epochs":      50,
    "batch_size":  64,
    "lr":          1e-3,
    "patience":    10,
    "num_workers": 0,
    "seed":        42,
    "json_path":   "axle_data.json/axle_data.json",
}


def build_cmd(model_name: str, extra_args: list[str]) -> list[str]:
    return [_venv_python(), "-m", "src.train", "--model", model_name] + extra_args


def run(cmd: list[str]) -> int:
    display = " ".join(cmd)
    print(f"\n{'='*60}")
    print(f"  Running: {display}")
    print(f"  cwd    : {PROJECT_ROOT}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Launch axle-detection model training.",
        epilog=(
            "Extra flags (e.g. --epochs, --lr, --batch_size) are forwarded "
            "straight to src/train.py. Defaults match the project spec.\n\n"
            "Examples:\n"
            "  python scripts/run_training.py --model tcn\n"
            "  python scripts/run_training.py --model both --epochs 100\n"
            "  python scripts/run_training.py --model cnn --lr 5e-4 --patience 15"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model", type=str, default="tcn",
        choices=["cnn", "tcn", "both"],
        help="Model to train (default: tcn). 'both' trains CNN first, then TCN.",
    )

    # Expose the most-used flags directly so --help is informative
    parser.add_argument("--epochs",      type=int,   default=DEFAULTS["epochs"],
                        help=f"Number of training epochs (default: {DEFAULTS['epochs']})")
    parser.add_argument("--batch_size",  type=int,   default=DEFAULTS["batch_size"],
                        help=f"Batch size (default: {DEFAULTS['batch_size']})")
    parser.add_argument("--lr",          type=float, default=DEFAULTS["lr"],
                        help=f"Learning rate (default: {DEFAULTS['lr']})")
    parser.add_argument("--patience",    type=int,   default=DEFAULTS["patience"],
                        help=f"Early-stopping patience in epochs (default: {DEFAULTS['patience']})")
    parser.add_argument("--num_workers", type=int,   default=DEFAULTS["num_workers"],
                        help=f"DataLoader workers (default: {DEFAULTS['num_workers']})")
    parser.add_argument("--seed",        type=int,   default=DEFAULTS["seed"],
                        help=f"Random seed (default: {DEFAULTS['seed']})")
    parser.add_argument("--json_path",   type=str,   default=DEFAULTS["json_path"],
                        help=f"Path to axle_data.json (default: {DEFAULTS['json_path']})")

    args = parser.parse_args()

    # Build the forwarded argument list (excludes --model which we handle ourselves)
    forward = [
        "--epochs",      str(args.epochs),
        "--batch_size",  str(args.batch_size),
        "--lr",          str(args.lr),
        "--patience",    str(args.patience),
        "--num_workers", str(args.num_workers),
        "--seed",        str(args.seed),
        "--json_path",   args.json_path,
    ]

    models = ["cnn", "tcn"] if args.model == "both" else [args.model]

    print(f"\nAxle Detection Training Launcher")
    print(f"  Model(s)   : {', '.join(m.upper() for m in models)}")
    print(f"  Epochs     : {args.epochs}")
    print(f"  Batch size : {args.batch_size}")
    print(f"  LR         : {args.lr}")
    print(f"  Patience   : {args.patience}")
    print(f"\nTip: open a second terminal and run:")
    print(f"     python scripts/watch_training.py --model {args.model} --epochs {args.epochs}")

    exit_codes = {}
    for model_name in models:
        cmd = build_cmd(model_name, forward)
        rc  = run(cmd)
        exit_codes[model_name] = rc
        if rc != 0:
            print(f"\n[WARNING] {model_name.upper()} training exited with code {rc}.")
            if args.model == "both":
                print("  Skipping remaining models due to error.")
                break

    print(f"\n{'='*60}")
    for m, rc in exit_codes.items():
        status = "OK" if rc == 0 else f"FAILED (code {rc})"
        print(f"  {m.upper():4s}  {status}")
    print(f"{'='*60}")
    sys.exit(max(exit_codes.values()))


if __name__ == "__main__":
    main()
