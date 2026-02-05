"""Block split helpers for PPD-style audits."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def make_split_indices(n: int, *, split: str) -> Tuple[np.ndarray, np.ndarray]:
    """Create deterministic split indices."""
    if split == "first_half":
        mid = n // 2
        idx_A = np.arange(0, mid, dtype=int)
        idx_B = np.arange(mid, n, dtype=int)
    elif split == "alternating":
        idx_A = np.arange(0, n, 2, dtype=int)
        idx_B = np.arange(1, n, 2, dtype=int)
    else:
        raise ValueError("split must be 'first_half' or 'alternating'.")
    return idx_A, idx_B


def sub_cov(cov: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Return covariance submatrix."""
    return cov[np.ix_(idx, idx)]


def cross_cov(cov: np.ndarray, idxA: np.ndarray, idxB: np.ndarray) -> np.ndarray:
    """Return cross-covariance block."""
    return cov[np.ix_(idxA, idxB)]
