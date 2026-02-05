"""Reference vector backend."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Set

import numpy as np
import pandas as pd


@dataclass
class ReferenceVectorBackend:
    """Backend that returns a fixed reference theory vector."""

    ref_path: Path
    ref_format: str = "auto"
    name: str = "reference"
    supports_probes: Set[str] | None = None
    supports_w0wa: bool = False

    def _load(self) -> np.ndarray:
        path = Path(self.ref_path)
        fmt = self.ref_format
        if fmt == "auto":
            if path.suffix == ".npy":
                fmt = "npy"
            elif path.suffix == ".npz":
                fmt = "npz"
            elif path.suffix == ".csv":
                fmt = "csv"
            else:
                raise ValueError("Unsupported reference format.")
        if fmt == "npy":
            return np.load(path)
        if fmt == "npz":
            data = np.load(path)
            if "theory" in data:
                return data["theory"]
            if len(data.files) == 1:
                return data[data.files[0]]
            raise ValueError("NPZ must contain 'theory' or a single array.")
        if fmt == "csv":
            arr = np.loadtxt(path, delimiter=",")
            if arr.ndim == 1:
                return arr
            if arr.shape[1] == 1:
                return arr[:, 0]
            raise ValueError("CSV must contain a single column of floats.")
        raise ValueError("Unsupported reference format.")

    def predict(self, theta: np.ndarray, block_index: pd.DataFrame) -> np.ndarray:
        ref = np.asarray(self._load(), dtype=float)
        if len(ref) == len(block_index):
            return ref
        if "i" not in block_index.columns:
            raise ValueError("block_index must include column 'i' for subsetting.")
        idx = block_index["i"].to_numpy()
        if not np.issubdtype(idx.dtype, np.integer):
            idx = idx.astype(int)
        if idx.min() < 0 or idx.max() >= len(ref):
            raise ValueError("block_index i values out of bounds for reference vector.")
        return ref[idx]
