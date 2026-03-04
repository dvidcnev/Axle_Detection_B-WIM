"""
evaluate.py
-----------
Evaluation utilities for B-WIM axle detection.

Metrics are computed at the **axle level** using a tolerance window:
a predicted peak that falls within ±tolerance samples of a ground-truth
peak counts as a true positive.

Functions
---------
pulses_to_peaks(pulse, threshold)
    Convert a probability/binary pulse vector to a list of peak sample indices.

axle_level_metrics(targets, predictions, tolerance)
    Precision, recall, F1, mean absolute timing error over a batch.

evaluate_model(model, loader, device, threshold, tolerance)
    Run a PyTorch model on a DataLoader and return an aggregated metrics dict.

print_metrics(metrics)
    Pretty-print a metrics dict.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import List, Dict, Tuple


# ---------------------------------------------------------------------------
# Peak extraction
# ---------------------------------------------------------------------------

def pulses_to_peaks(
    pulse: np.ndarray,
    threshold: float = 0.5,
    min_distance: int = 5,
) -> np.ndarray:
    """
    Extract peak positions from a probability / binary pulse vector.

    Parameters
    ----------
    pulse : 1-D array of shape (N,)
    threshold : float
        Values above this are considered positive.
    min_distance : int
        Minimum sample distance between two accepted peaks (non-maximum
        suppression along the 1-D sequence).

    Returns
    -------
    peaks : sorted array of integer sample indices
    """
    from scipy.signal import find_peaks

    sig = np.asarray(pulse, dtype=np.float32)
    # find_peaks needs the curve above threshold; use it as height
    peaks, _ = find_peaks(sig, height=threshold, distance=min_distance)
    return peaks


# ---------------------------------------------------------------------------
# Tolerance-window matching (greedy, Hungarian-free for speed)
# ---------------------------------------------------------------------------

def _match_peaks(
    pred_peaks: np.ndarray,
    true_peaks: np.ndarray,
    tolerance: int,
) -> Tuple[int, int, int]:
    """
    Greedy nearest-neighbour matching within tolerance.

    Returns (tp, fp, fn).
    """
    if len(true_peaks) == 0:
        return 0, len(pred_peaks), 0
    if len(pred_peaks) == 0:
        return 0, 0, len(true_peaks)

    matched_true = set()
    matched_pred = set()

    for pi, pp in enumerate(pred_peaks):
        dists = np.abs(true_peaks - pp)
        best  = int(np.argmin(dists))
        if dists[best] <= tolerance and best not in matched_true:
            matched_true.add(best)
            matched_pred.add(pi)

    tp = len(matched_true)
    fp = len(pred_peaks) - tp
    fn = len(true_peaks)  - tp
    return tp, fp, fn


# ---------------------------------------------------------------------------
# Batch metrics
# ---------------------------------------------------------------------------

def axle_level_metrics(
    targets: List[np.ndarray],
    predictions: List[np.ndarray],
    tolerance: int = 5,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute precision, recall, F1, and mean absolute timing error (MATE)
    over a batch of (target, prediction) pulse pairs.

    Parameters
    ----------
    targets : list of 1-D arrays (ground-truth pulse vectors)
    predictions : list of 1-D arrays (predicted probability / binary vectors)
    tolerance : int
        Half-window in samples; a prediction within ±tolerance of a true
        axle position counts as a true positive.
    threshold : float
        Binarisation threshold applied to predicted probabilities.

    Returns
    -------
    dict with keys: precision, recall, f1, mate, tp, fp, fn
    """
    total_tp = total_fp = total_fn = 0
    timing_errors: List[float] = []

    for target, pred in zip(targets, predictions):
        true_peaks = pulses_to_peaks(target,  threshold=threshold)
        pred_peaks = pulses_to_peaks(pred,    threshold=threshold)

        tp, fp, fn = _match_peaks(pred_peaks, true_peaks, tolerance)
        total_tp += tp
        total_fp += fp
        total_fn += fn

        # Mean absolute timing error for matched pairs
        matched_true = set()
        for pp in pred_peaks:
            dists  = np.abs(true_peaks - pp)
            best   = int(np.argmin(dists)) if len(dists) > 0 else -1
            if best >= 0 and dists[best] <= tolerance and best not in matched_true:
                matched_true.add(best)
                timing_errors.append(float(dists[best]))

    precision = total_tp / (total_tp + total_fp + 1e-8)
    recall    = total_tp / (total_tp + total_fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    mate      = float(np.mean(timing_errors)) if timing_errors else 0.0

    return {
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
        "mate":      mate,       # mean absolute timing error (samples)
        "tp":        total_tp,
        "fp":        total_fp,
        "fn":        total_fn,
    }


# ---------------------------------------------------------------------------
# Model evaluation on a DataLoader
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
    tolerance: int = 5,
) -> Dict[str, float]:
    """
    Run model inference on all batches in *loader* and return aggregated
    axle-level metrics.

    Parameters
    ----------
    model : nn.Module  (outputs raw logits of shape [B, L])
    loader : DataLoader
    device : torch.device
    threshold : float   applied after sigmoid
    tolerance : int     matching window in samples

    Returns
    -------
    metrics dict  (same as axle_level_metrics)
    """
    model.eval()
    all_targets: List[np.ndarray] = []
    all_preds:   List[np.ndarray] = []

    for signals, pulses in loader:
        signals = signals.to(device)     # [B, 1, L]
        logits  = model(signals)         # [B, L]
        probs   = torch.sigmoid(logits).cpu().numpy()  # [B, L]
        targets = pulses.numpy()                        # [B, L]

        for b in range(len(targets)):
            all_targets.append(targets[b])
            all_preds.append(probs[b])

    return axle_level_metrics(
        all_targets, all_preds,
        tolerance=tolerance,
        threshold=threshold,
    )


# ---------------------------------------------------------------------------
# Pretty-print
# ---------------------------------------------------------------------------

def print_metrics(metrics: Dict[str, float], prefix: str = "") -> None:
    label = f"[{prefix}] " if prefix else ""
    print(
        f"{label}"
        f"Precision: {metrics['precision']:.4f}  "
        f"Recall: {metrics['recall']:.4f}  "
        f"F1: {metrics['f1']:.4f}  "
        f"MATE: {metrics['mate']:.2f} samples  "
        f"(TP={metrics['tp']}, FP={metrics['fp']}, FN={metrics['fn']})"
    )
