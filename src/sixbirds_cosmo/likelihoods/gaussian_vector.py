"""Gaussian vector likelihood helpers."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.linalg import cho_factor, cho_solve


def apply_mask(y: np.ndarray, cov: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Apply an index mask to a data vector and covariance."""
    y = np.asarray(y, dtype=float)
    cov = np.asarray(cov, dtype=float)
    mask = np.asarray(mask, dtype=int)
    y_masked = y[mask]
    cov_masked = cov[np.ix_(mask, mask)]
    return y_masked, cov_masked


def chi2_gaussian(resid: np.ndarray, cov: np.ndarray, *, jitter: float = 0.0) -> Tuple[float, float]:
    """Compute chi2 via Cholesky solve with optional jitter ladder."""
    resid = np.asarray(resid, dtype=float)
    cov = np.asarray(cov, dtype=float)
    jitter_levels = [jitter, 1e-15, 1e-12, 1e-10, 1e-8, 1e-6]
    last_exc: Exception | None = None
    for j in jitter_levels:
        try:
            cho = cho_factor(cov + j * np.eye(cov.shape[0]), lower=True)
            sol = cho_solve(cho, resid)
            chi2 = float(resid @ sol)
            return chi2, j
        except Exception as exc:  # pragma: no cover - defensive
            last_exc = exc
            continue
    raise RuntimeError("Cholesky failed up to jitter=1e-6.") from last_exc
