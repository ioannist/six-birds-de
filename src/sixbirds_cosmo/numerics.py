"""Numerical primitives that preserve the specified mathematical problem."""

from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, solve_triangular


def validated_indices(indices, n: int, *, allow_empty: bool = False) -> np.ndarray:
    """Keep row identity explicit: never truncate floats or reinterpret booleans."""
    idx = np.asarray(indices)
    if idx.ndim != 1 or not np.issubdtype(idx.dtype, np.integer):
        raise ValueError("indices must be a one-dimensional integer vector")
    if not allow_empty and len(idx) == 0:
        raise ValueError("indices must be nonempty")
    if np.any(idx < 0) or np.any(idx >= n) or len(np.unique(idx)) != len(idx):
        raise ValueError("indices must be distinct and in range")
    return idx


def factor_covariance(cov: np.ndarray, *, jitter: float = 0.0):
    """Factor a finite symmetric positive definite covariance.

    Jitter changes the likelihood and is used only when explicitly requested.
    An indefinite matrix is never silently projected to a different covariance.
    """
    cov = np.asarray(cov, dtype=float)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1] or cov.shape[0] == 0:
        raise ValueError("cov must be a nonempty square matrix")
    if not np.all(np.isfinite(cov)):
        raise ValueError("cov must be finite")
    scale = float(np.max(np.abs(cov)))
    if np.max(np.abs(cov - cov.T)) > 1e-10 * scale:
        raise ValueError("cov must be symmetric")
    if not np.isfinite(jitter) or jitter < 0:
        raise ValueError("jitter must be finite and nonnegative")
    return cho_factor((cov + cov.T) / 2 + jitter * np.eye(len(cov)), lower=True)


def generalized_least_squares(resid: np.ndarray, basis: np.ndarray, cov: np.ndarray):
    """Whiten and fit linear coefficients without squaring the condition number.

    Reject nonidentifiable coefficients: an arbitrary solution may predict a
    different held-out vector along an unconstrained direction.
    """
    resid = np.asarray(resid, dtype=float)
    basis = np.asarray(basis, dtype=float)
    if resid.ndim != 1 or basis.ndim != 2 or basis.shape[0] != len(resid):
        raise ValueError("residual and basis dimensions do not match")
    if np.shape(cov) != (len(resid), len(resid)):
        raise ValueError("covariance dimension does not match residual")
    if not np.all(np.isfinite(resid)) or not np.all(np.isfinite(basis)):
        raise ValueError("residual and basis must be finite")
    cho = factor_covariance(cov)
    r_white = solve_triangular(cho[0], resid, lower=True)
    b_white = solve_triangular(cho[0], basis, lower=True)
    coefficients, _, rank, _ = np.linalg.lstsq(b_white, r_white, rcond=None)
    if rank != basis.shape[1]:
        raise ValueError("training basis is rank deficient; held-out coefficients are not identifiable")
    error = r_white - b_white @ coefficients
    return coefficients, float(error @ error)
