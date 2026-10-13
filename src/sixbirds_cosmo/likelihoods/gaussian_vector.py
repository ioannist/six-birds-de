"""Gaussian vector likelihood helpers."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.linalg import cho_solve
from sixbirds_cosmo.numerics import factor_covariance


def apply_mask(y: np.ndarray, cov: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Apply an index mask to a data vector and covariance."""
    y = np.asarray(y, dtype=float)
    cov = np.asarray(cov, dtype=float)
    mask = np.asarray(mask)
    if mask.dtype == bool:
        if mask.shape != y.shape:
            raise ValueError("boolean mask must match the data vector")
        mask = np.flatnonzero(mask)
    elif not np.issubdtype(mask.dtype, np.integer):
        raise ValueError("mask must contain boolean flags or integer indices")
    if mask.ndim != 1 or np.any(mask < 0) or np.any(mask >= len(y)) or len(np.unique(mask)) != len(mask):
        raise ValueError("mask must contain distinct in-range indices")
    y_masked = y[mask]
    cov_masked = cov[np.ix_(mask, mask)]
    return y_masked, cov_masked


def chi2_gaussian(resid: np.ndarray, cov: np.ndarray, *, jitter: float = 0.0) -> Tuple[float, float]:
    """Compute chi2 with an explicitly chosen covariance jitter."""
    resid = np.asarray(resid, dtype=float)
    cov = np.asarray(cov, dtype=float)
    if resid.ndim != 1 or cov.shape != (len(resid), len(resid)) or not np.all(np.isfinite(resid)):
        raise ValueError("residual must be finite and match covariance dimensions")
    cho = factor_covariance(cov, jitter=jitter)
    sol = cho_solve(cho, resid)
    return float(resid @ sol), jitter
