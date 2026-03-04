"""
baseline.py
-----------
Traditional peak-detection baseline for B-WIM axle detection.

Converts a 1300-sample strain signal into a 1300-sample binary pulse vector
by finding local maxima using scipy.signal.find_peaks.

The output pulse vector has 1.0 at every detected peak position and 0.0
elsewhere, matching the format of the ground-truth pulses used by the
neural network models.
"""

import numpy as np
from scipy.signal import find_peaks, savgol_filter
from typing import List, Tuple, Optional


# ---------------------------------------------------------------------------
# Core peak → pulse conversion
# ---------------------------------------------------------------------------

def signal_to_pulse_peaks(
    signal: np.ndarray,
    height_factor: float = 0.15,
    distance: int = 20,
    prominence_factor: float = 0.05,
    smooth: bool = True,
    smooth_window: int = 11,
    smooth_poly: int = 2,
) -> np.ndarray:
    """
    Detect axle peaks in a single bridge strain signal.

    Parameters
    ----------
    signal : 1-D array of shape (N,)
        Raw (or normalised) bridge strain signal.
    height_factor : float
        Minimum peak height as a fraction of (max - min) of the signal.
    distance : int
        Minimum number of samples between consecutive peaks.
    prominence_factor : float
        Minimum peak prominence as a fraction of (max - min).
    smooth : bool
        Apply Savitzky-Golay smoothing before peak detection.
    smooth_window : int
        Window length for Savitzky-Golay filter (must be odd).
    smooth_poly : int
        Polynomial order for Savitzky-Golay filter.

    Returns
    -------
    pulse : np.ndarray of shape (N,), dtype float32
        Binary vector: 1.0 at detected peak positions, 0.0 elsewhere.
    """
    sig = signal.astype(np.float32).copy()

    if smooth and len(sig) >= smooth_window:
        sig = savgol_filter(sig, window_length=smooth_window, polyorder=smooth_poly)

    sig_range = float(sig.max() - sig.min()) + 1e-8
    height     = sig.min() + height_factor * sig_range
    prominence = prominence_factor * sig_range

    peaks, _ = find_peaks(
        sig,
        height=height,
        distance=distance,
        prominence=prominence,
    )

    pulse = np.zeros(len(signal), dtype=np.float32)
    pulse[peaks] = 1.0
    return pulse


# ---------------------------------------------------------------------------
# Batch prediction (list of signals)
# ---------------------------------------------------------------------------

def predict_batch(
    signals: List[np.ndarray],
    **kwargs,
) -> List[np.ndarray]:
    """Run peak detection on a list of signals."""
    return [signal_to_pulse_peaks(s, **kwargs) for s in signals]


# ---------------------------------------------------------------------------
# Parameter grid search on a small validation set
# ---------------------------------------------------------------------------

def tune_threshold(
    signals: List[np.ndarray],
    targets: List[np.ndarray],
    tolerance: int = 5,
    height_factors: Tuple = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30),
    distances: Tuple = (10, 15, 20, 30),
) -> dict:
    """
    Grid-search the two most impactful hyperparameters on a small val set.

    Returns the best parameter dict and its F1 score.
    """
    from .evaluate import axle_level_metrics

    best_f1 = -1.0
    best_params = {}

    for hf in height_factors:
        for dist in distances:
            preds = [
                signal_to_pulse_peaks(s, height_factor=hf, distance=dist)
                for s in signals
            ]
            metrics = axle_level_metrics(targets, preds, tolerance=tolerance)
            f1 = metrics["f1"]
            if f1 > best_f1:
                best_f1 = f1
                best_params = {"height_factor": hf, "distance": dist, "f1": f1}

    print(f"Best baseline params: {best_params}")
    return best_params


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    np.random.seed(0)
    t = np.linspace(0, 1, 1300)
    # Synthetic signal: two Gaussian peaks
    sig = (np.exp(-((t - 0.3) ** 2) / 0.001)
           + 1.5 * np.exp(-((t - 0.7) ** 2) / 0.0015)
           + np.random.normal(0, 0.05, 1300))

    pulse = signal_to_pulse_peaks(sig)
    peaks = np.where(pulse == 1.0)[0]

    print(f"Detected {len(peaks)} peak(s) at sample positions: {peaks.tolist()}")

    plt.figure(figsize=(12, 3))
    plt.plot(sig, label="signal")
    plt.vlines(peaks, sig.min(), sig.max(), colors="r", label="detected peaks")
    plt.legend()
    plt.tight_layout()
    plt.show()
