"""Traditional peak-detection baseline for B-WIM axle detection."""

import numpy as np
from scipy.signal import find_peaks, savgol_filter
from typing import List, Tuple


def signal_to_pulse_peaks(
    signal: np.ndarray,
    height_factor: float = 0.15,
    distance: int = 20,
    prominence_factor: float = 0.05,
    smooth: bool = True,
    smooth_window: int = 11,
    smooth_poly: int = 2,
) -> np.ndarray:
    """Return a binary pulse vector with 1.0 at detected axle peak positions."""
    sig = signal.astype(np.float32).copy()

    if smooth and len(sig) >= smooth_window:
        sig = savgol_filter(sig, window_length=smooth_window, polyorder=smooth_poly)

    sig_range = float(sig.max() - sig.min()) + 1e-8

    peaks, _ = find_peaks(
        sig,
        height=sig.min() + height_factor * sig_range,
        distance=distance,
        prominence=prominence_factor * sig_range,
    )

    pulse = np.zeros(len(signal), dtype=np.float32)
    pulse[peaks] = 1.0
    return pulse


def predict_batch(
    signals: List[np.ndarray],
    **kwargs,
) -> List[np.ndarray]:
    """Run peak detection on a list of signals."""
    return [signal_to_pulse_peaks(s, **kwargs) for s in signals]


def tune_threshold(
    signals: List[np.ndarray],
    targets: List[np.ndarray],
    tolerance: int = 5,
    height_factors: Tuple = (0.03, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0,.35, 0.40, 0.45, 0.50),
    distances: Tuple = (10, 15, 20, 30, 40, 50),
) -> dict:
    """Grid-search height_factor and distance on a validation set; return best params."""
    from .evaluate import axle_level_metrics

    best_f1 = -1.0
    best_params = {}

    for hf in height_factors:
        for dist in distances:
            preds = [signal_to_pulse_peaks(s, height_factor=hf, distance=dist) for s in signals]
            f1 = axle_level_metrics(targets, preds, tolerance=tolerance)["f1"]
            if f1 > best_f1:
                best_f1 = f1
                best_params = {"height_factor": hf, "distance": dist, "f1": f1}

    print(f"Best baseline params: {best_params}")
    return best_params
