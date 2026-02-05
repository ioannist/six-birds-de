"""Posterior predictive-style audit utilities."""
from __future__ import annotations

import numpy as np


def interp_at(z_src: np.ndarray, y_src: np.ndarray, z_eval: np.ndarray) -> np.ndarray:
    """Interpolate y(z) at requested z_eval with edge fill."""
    z_src = np.asarray(z_src, dtype=float)
    y_src = np.asarray(y_src, dtype=float)
    z_eval = np.asarray(z_eval, dtype=float)
    if z_src.shape != y_src.shape:
        raise ValueError("z_src and y_src must have the same shape")
    order = np.argsort(z_src)
    z_sorted = z_src[order]
    y_sorted = y_src[order]
    return np.interp(z_eval, z_sorted, y_sorted, left=y_sorted[0], right=y_sorted[-1])


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root-mean-square error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean signed bias (pred - true)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(y_pred - y_true))


def predict_varH_lcdm(z_eval: np.ndarray) -> np.ndarray:
    """Homogeneous FLRW predicts zero variance proxy."""
    z_eval = np.asarray(z_eval, dtype=float)
    return np.zeros_like(z_eval)


def predict_varH_rewrite_from_template(
    z_eval: np.ndarray,
    *,
    A_fit: float,
    A_ref: float,
    z_ref: np.ndarray,
    varH_ref: np.ndarray,
    A_cut: float = 0.1,
) -> np.ndarray:
    """Predict var_H using a calibrated template and fitted amplitude."""
    z_eval = np.asarray(z_eval, dtype=float)
    if A_fit < A_cut:
        return np.zeros_like(z_eval)
    if A_ref <= 0:
        return np.zeros_like(z_eval)
    var_ref_eval = interp_at(z_ref, varH_ref, z_eval)
    return (A_fit / A_ref) * var_ref_eval
