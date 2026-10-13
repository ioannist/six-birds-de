"""Posterior predictive-style audit utilities."""
from __future__ import annotations

import numpy as np


def interp_at(z_src: np.ndarray, y_src: np.ndarray, z_eval: np.ndarray) -> np.ndarray:
    """Interpolate within the declared source range; no hidden extrapolation."""
    z_src = np.asarray(z_src, dtype=float)
    y_src = np.asarray(y_src, dtype=float)
    z_eval = np.asarray(z_eval, dtype=float)
    if z_src.ndim != 1 or len(z_src) < 2 or z_src.shape != y_src.shape or not np.all(np.isfinite(z_src)) or not np.all(np.isfinite(y_src)) or not np.all(np.isfinite(z_eval)):
        raise ValueError("source must consist of matching finite vectors with at least two points")
    order = np.argsort(z_src)
    z_sorted = z_src[order]
    y_sorted = y_src[order]
    if np.any(np.diff(z_sorted) <= 0) or np.any(z_eval < z_sorted[0]) or np.any(z_eval > z_sorted[-1]):
        raise ValueError("source redshifts must be distinct and cover requested redshifts")
    return np.interp(z_eval, z_sorted, y_sorted, left=y_sorted[0], right=y_sorted[-1])


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root-mean-square error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    _check_scored_vectors(y_true, y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean signed bias (pred - true)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    _check_scored_vectors(y_true, y_pred)
    return float(np.mean(y_pred - y_true))


def _check_scored_vectors(y_true, y_pred):
    if y_true.shape != y_pred.shape or not y_true.size or not np.all(np.isfinite(y_true)) or not np.all(np.isfinite(y_pred)):
        raise ValueError("scored arrays must be nonempty, finite and have the same shape")


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
    if not np.all(np.isfinite([A_fit, A_ref, A_cut])) or A_fit < 0 or A_ref <= 0 or A_cut < 0:
        raise ValueError("template requires finite nonnegative fit/cut and positive reference amplitude")
    if A_fit < A_cut:
        return np.zeros_like(z_eval)
    if np.any(np.asarray(varH_ref) < 0):
        raise ValueError("variance template must be nonnegative")
    var_ref_eval = interp_at(z_ref, varH_ref, z_eval)
    return (A_fit / A_ref) * var_ref_eval
