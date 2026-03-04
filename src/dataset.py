"""
dataset.py
----------
PyTorch Dataset for B-WIM axle detection.
Loads axle_data.json, normalises the signals (z-score, fit on train split only),
and returns (signal, pulse) tensor pairs.

Expected JSON structure (list of records):
  [
    {"signal": [float, ...],   # length 1300
     "pulses": [float, ...],   # length 1300  (binary 0/1 or soft values)
     ...                       # optional extra keys (metadata)
    },
    ...
  ]
"""

import json
import os
from typing import List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Helper: load JSON lazily (streams to avoid holding everything in RAM)
# ---------------------------------------------------------------------------

def load_json_records(path: str) -> List[dict]:
    """Load the full JSON file.  Returns a list of record dicts."""
    with open(path, "r") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    # If top-level is a dict with a records key, unwrap it
    for key in ("records", "data", "samples", "events"):
        if key in data:
            return data[key]
    raise ValueError(
        f"Unexpected JSON structure: top-level keys are {list(data.keys())}"
    )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class AxleDataset(Dataset):
    """
    Parameters
    ----------
    records : list of dict
        Pre-loaded list of record dicts (signal + pulses).
    signal_key : str
        Key in each record that holds the bridge strain signal.
    pulse_key : str
        Key in each record that holds the target pulse sequence.
    signal_len : int
        Expected/enforced length of every signal (default 1300).
    mean : float or None
        If provided, used for z-score normalisation (fit on train set).
    std : float or None
        If provided, used for z-score normalisation (fit on train set).
    """

    def __init__(
        self,
        records: List[dict],
        signal_key: str = "signal",
        pulse_key: str = "pulses",
        signal_len: int = 1300,
        mean: Optional[float] = None,
        std: Optional[float] = None,
    ):
        self.records = records
        self.signal_key = signal_key
        self.pulse_key = pulse_key
        self.signal_len = signal_len
        self.mean = mean
        self.std = std

    # ------------------------------------------------------------------
    # Normalisation stats (call on train subset only)
    # ------------------------------------------------------------------

    @staticmethod
    def compute_stats(records: List[dict], signal_key: str = "signal") -> Tuple[float, float]:
        """Compute global mean and std across all training signals."""
        all_vals = np.concatenate([np.array(r[signal_key], dtype=np.float32) for r in records])
        return float(all_vals.mean()), float(all_vals.std()) + 1e-8

    # ------------------------------------------------------------------
    # __len__ / __getitem__
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        rec = self.records[idx]

        signal = np.array(rec[self.signal_key], dtype=np.float32)
        pulse  = np.array(rec[self.pulse_key],  dtype=np.float32)

        # Pad or truncate to signal_len
        if len(signal) < self.signal_len:
            signal = np.pad(signal, (0, self.signal_len - len(signal)))
            pulse  = np.pad(pulse,  (0, self.signal_len - len(pulse)))
        else:
            signal = signal[:self.signal_len]
            pulse  = pulse[:self.signal_len]

        # Z-score normalisation
        if self.mean is not None and self.std is not None:
            signal = (signal - self.mean) / self.std

        # Shape: [1, signal_len] for Conv1d (channels-first)
        signal_t = torch.from_numpy(signal).unsqueeze(0)  # [1, 1300]
        pulse_t  = torch.from_numpy(pulse)                 # [1300]

        return signal_t, pulse_t

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------

    def get_metadata(self, idx: int) -> dict:
        """Return all non-signal/pulse keys for a record."""
        rec = self.records[idx]
        return {k: v for k, v in rec.items()
                if k not in (self.signal_key, self.pulse_key)}


# ---------------------------------------------------------------------------
# Factory: build train / val / test datasets from a JSON file
# ---------------------------------------------------------------------------

def build_datasets(
    json_path: str,
    signal_key: str = "signal",
    pulse_key: str = "pulses",
    signal_len: int = 1300,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    # test_ratio is implicitly 1 - train - val
    seed: int = 42,
) -> Tuple["AxleDataset", "AxleDataset", "AxleDataset"]:
    """
    Load JSON, split, fit normalisation stats on train, return three Datasets.

    Returns
    -------
    train_ds, val_ds, test_ds
    """
    assert train_ratio + val_ratio < 1.0, "train + val must be < 1"
    assert os.path.isfile(json_path), f"JSON not found: {json_path}"

    records = load_json_records(json_path)
    n = len(records)
    print(f"Loaded {n:,} records from {os.path.basename(json_path)}")

    # Detect keys if defaults don't exist
    if signal_key not in records[0]:
        candidates = [k for k in records[0] if "signal" in k.lower() or "strain" in k.lower()]
        if candidates:
            signal_key = candidates[0]
            print(f"  Auto-detected signal_key = '{signal_key}'")
        else:
            raise KeyError(f"Cannot find signal key. Available keys: {list(records[0].keys())}")

    if pulse_key not in records[0]:
        candidates = [k for k in records[0]
                      if any(w in k.lower() for w in ("pulse", "label", "target", "axle"))]
        if candidates:
            pulse_key = candidates[0]
            print(f"  Auto-detected pulse_key  = '{pulse_key}'")
        else:
            raise KeyError(f"Cannot find pulse key. Available keys: {list(records[0].keys())}")

    indices = list(range(n))
    test_ratio = 1.0 - train_ratio - val_ratio

    train_idx, temp_idx = train_test_split(
        indices, test_size=(val_ratio + test_ratio), random_state=seed, shuffle=True
    )
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=test_ratio / (val_ratio + test_ratio),
        random_state=seed,
    )

    train_records = [records[i] for i in train_idx]
    val_records   = [records[i] for i in val_idx]
    test_records  = [records[i] for i in test_idx]

    print(f"  Train: {len(train_records):,}  |  Val: {len(val_records):,}  |  Test: {len(test_records):,}")

    # Fit normalisation stats on train only
    mean, std = AxleDataset.compute_stats(train_records, signal_key)
    print(f"  Signal stats (train) — mean: {mean:.4f}, std: {std:.4f}")

    kwargs = dict(signal_key=signal_key, pulse_key=pulse_key,
                  signal_len=signal_len, mean=mean, std=std)
    train_ds = AxleDataset(train_records, **kwargs)
    val_ds   = AxleDataset(val_records,   **kwargs)
    test_ds  = AxleDataset(test_records,  **kwargs)

    return train_ds, val_ds, test_ds


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "axle_data.json/axle_data.json"
    train_ds, val_ds, test_ds = build_datasets(path)
    x, y = train_ds[0]
    print(f"Signal shape: {x.shape}  |  Pulse shape: {y.shape}")
    print(f"Signal range: [{x.min():.3f}, {x.max():.3f}]")
    print(f"Pulse unique values: {y.unique().tolist()}")
