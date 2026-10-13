"""PPD-style probe split helpers for LSS."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.linalg import cho_solve
from sixbirds_cosmo.numerics import factor_covariance, generalized_least_squares, validated_indices


def get_probe_split_indices(block_index: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Return indices for shear vs (clustering+ggl)."""
    idx_shear = block_index.loc[block_index["probe"] == "shear", "i"].to_numpy(dtype=int)
    idx_other = block_index.loc[
        block_index["probe"].isin(["clustering", "ggl"]), "i"
    ].to_numpy(dtype=int)
    return np.sort(idx_shear), np.sort(idx_other)


def _zscore_x(x: np.ndarray) -> np.ndarray:
    finite = np.isfinite(x)
    if not np.any(finite):
        return np.zeros_like(x, dtype=float)
    xv = x[finite]
    mean = float(np.mean(xv))
    std = float(np.std(xv))
    if std == 0:
        return np.zeros_like(x, dtype=float)
    z = np.zeros_like(x, dtype=float)
    z[finite] = (xv - mean) / std
    return z


def build_surrogate_theory(
    y: np.ndarray,
    cov: np.ndarray,
    block_index: pd.DataFrame,
    *,
    frac_sigma: float,
    beta_x: float,
    kappa_probe: Dict[str, float],
) -> np.ndarray:
    """Construct an observed-vector-centered injection for protocol tests.

    This is not an independent prediction: all entries of y, including test
    entries, are used. Residuals and fitted coefficients are independent of the
    measured central values. Use a separate fixed reference for predictive tests.
    """
    y = np.asarray(y, dtype=float)
    cov = np.asarray(cov, dtype=float)
    _check_problem(y, cov)
    if len(block_index) != len(y):
        raise ValueError("block metadata must match the vector length")
    sigma = np.sqrt(np.diag(cov))
    x = block_index["x"].to_numpy(dtype=float)
    x_z = _zscore_x(x)
    kappa = np.array([kappa_probe.get(p, 0.0) for p in block_index["probe"].to_numpy()])
    t0 = y + frac_sigma * sigma * (1.0 + kappa + beta_x * x_z)
    return t0


def _normalize_basis(B: np.ndarray) -> np.ndarray:
    Bn = B.copy()
    for j in range(B.shape[1]):
        rms = float(np.sqrt(np.mean(B[:, j] ** 2)))
        if rms > 0:
            Bn[:, j] /= rms
    return Bn


def build_basis(y: np.ndarray, cov: np.ndarray, block_index: pd.DataFrame, *, model: str) -> np.ndarray:
    _check_problem(np.asarray(y), np.asarray(cov))
    if len(block_index) != len(y):
        raise ValueError("block metadata must match the vector length")
    sigma = np.sqrt(np.diag(cov))
    x_z = _zscore_x(block_index["x"].to_numpy(dtype=float))
    if model == "lcdm_like":
        B = np.column_stack([sigma])
    elif model == "rewrite_like":
        B = np.column_stack([sigma, sigma * x_z])
    else:
        raise ValueError(f"Unknown model kind: {model}")
    return _normalize_basis(B)


def fit_linear_correction(
    y: np.ndarray,
    cov: np.ndarray,
    t0: np.ndarray,
    B: np.ndarray,
    idx_train: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """Fit linear parameters minimizing chi2 on training indices."""
    y = np.asarray(y, dtype=float)
    t0 = np.asarray(t0, dtype=float)
    _check_problem(y, np.asarray(cov), t0)
    idx = validated_indices(idx_train, len(y))
    resid = y - t0
    cov_train = cov[np.ix_(idx, idx)]
    B_train = B[idx]
    r_train = resid[idx]

    return generalized_least_squares(r_train, B_train, cov_train)


def eval_metrics(y: np.ndarray, cov: np.ndarray, t: np.ndarray, idx: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)
    cov = np.asarray(cov, dtype=float)
    _check_problem(y, cov, t)
    idx = validated_indices(idx, len(y))
    resid = (y - t)[idx]
    cov_sub = cov[np.ix_(idx, idx)]
    L = factor_covariance(cov_sub)
    chi2 = float(resid @ cho_solve(L, resid))
    n = resid.size
    rmse = float(np.sqrt(np.mean(resid**2))) if n else 0.0
    rmse_weighted = float(np.sqrt(chi2 / n)) if n else 0.0
    bias = float(np.mean(resid)) if n else 0.0
    return {
        "chi2": chi2,
        "chi2_over_n": float(chi2 / n) if n else 0.0,
        "rmse": rmse,
        "rmse_weighted": rmse_weighted,
        "bias": bias,
    }


def _check_problem(y, cov, theory=None):
    if y.ndim != 1 or not len(y) or not np.all(np.isfinite(y)) or cov.shape != (len(y), len(y)) or not np.all(np.isfinite(cov)):
        raise ValueError("data and covariance must be finite and have matching vector dimensions")
    if theory is not None and (theory.shape != y.shape or not np.all(np.isfinite(theory))):
        raise ValueError("theory must be a finite vector with the same shape as data")
