"""Fit effective CPL w0-wa to a rewrite background H(z)."""

from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.optimize import least_squares


def _as_array(z: np.ndarray | float) -> tuple[np.ndarray, bool]:
    z_arr = np.asarray(z, dtype=float)
    if not np.all(np.isfinite(z_arr)) or np.any(z_arr <= -1):
        raise ValueError("Redshift must be finite and greater than -1.")
    return z_arr, z_arr.ndim == 0


def _reshape_output(values: np.ndarray, shape: tuple[int, ...], scalar: bool) -> np.ndarray | float:
    if scalar:
        return float(values.reshape(()))
    return values.reshape(shape)


def E_w0wa(z: np.ndarray | float, om: float, w0: float, wa: float) -> np.ndarray | float:
    """Dimensionless E(z) for flat CPL w0-wa model."""
    z_arr, scalar = _as_array(z)
    if np.any(z_arr < -1.0):
        raise ValueError("Redshift z must be >= -1.")
    ol = 1.0 - om
    f = (1.0 + z_arr) ** (3.0 * (1.0 + w0 + wa)) * np.exp(-3.0 * wa * z_arr / (1.0 + z_arr))
    E2 = om * (1.0 + z_arr) ** 3 + ol * f
    if np.any(E2 <= 0.0) or np.any(~np.isfinite(E2)):
        raise ValueError("Non-positive E^2 encountered in CPL model.")
    Ez = np.sqrt(E2)
    return _reshape_output(Ez, z_arr.shape, scalar)


def H_w0wa(z: np.ndarray | float, H0: float, om: float, w0: float, wa: float) -> np.ndarray | float:
    """H(z) for flat CPL w0-wa model."""
    if not np.isfinite(H0) or H0 <= 0:
        raise ValueError("H0 must be finite and positive.")
    return H0 * E_w0wa(z, om, w0, wa)


def compare_H_models(z_grid: np.ndarray, H_rw: np.ndarray, H_approx: np.ndarray) -> Dict[str, float]:
    """Compare two H(z) arrays with fractional error metrics."""
    frac = (H_approx - H_rw) / H_rw
    rms = float(np.sqrt(np.mean(frac**2)))
    max_abs = float(np.max(np.abs(frac)))
    p95 = float(np.percentile(np.abs(frac), 95))
    return {"rms_frac_err": rms, "max_frac_err": max_abs, "p95_frac_err": p95}


def fit_effective_w0wa(
    z_grid: np.ndarray,
    H_rw: np.ndarray,
    *,
    H0: float | None = None,
    om: float | None = None,
    weights: np.ndarray | None = None,
    w0_init: float = -1.0,
    wa_init: float = 0.0,
    bounds: tuple[tuple[float, float], tuple[float, float]] | None = None,
) -> Dict[str, float]:
    """Fit w0-wa by minimizing squared error in log E(z)."""
    z_grid = np.asarray(z_grid, dtype=float)
    H_rw = np.asarray(H_rw, dtype=float)
    if z_grid.ndim != 1 or len(z_grid) < 2 or H_rw.shape != z_grid.shape or not np.all(np.isfinite(z_grid)) or not np.all(np.isfinite(H_rw)) or np.any(H_rw <= 0):
        raise ValueError("fit grids must be matching finite vectors with positive H")
    if om is None:
        raise ValueError("om must be provided for effective w0-wa fit.")
    if not np.isfinite(om) or not 0 <= om < 1:
        raise ValueError("flat CPL fit requires 0 <= om < 1")
    if H0 is None:
        zero = np.flatnonzero(z_grid == 0)
        if len(zero) != 1:
            raise ValueError("H0 must be provided unless the grid contains z=0 exactly once")
        H0 = float(H_rw[zero[0]])
    if not np.isfinite(H0) or H0 <= 0:
        raise ValueError("H0 must be finite and positive")
    E_rw = H_rw / H0
    if weights is None:
        weights = np.ones_like(E_rw)
    weights = np.asarray(weights, dtype=float)
    if weights.shape != z_grid.shape or not np.all(np.isfinite(weights)) or np.any(weights < 0) or np.count_nonzero(weights) < 2:
        raise ValueError("weights must match the grid with at least two positive entries")
    sqrt_w = np.sqrt(weights)

    def residuals(theta: np.ndarray) -> np.ndarray:
        w0, wa = float(theta[0]), float(theta[1])
        try:
            E_model = E_w0wa(z_grid, om, w0, wa)
        except Exception:
            return np.full_like(E_rw, 1e6, dtype=float)
        if np.any(E_model <= 0) or np.any(~np.isfinite(E_model)):
            return np.full_like(E_rw, 1e6, dtype=float)
        return sqrt_w * (np.log(E_model) - np.log(E_rw))

    if bounds is None:
        bounds = ((-2.5, -5.0), (0.0, 5.0))

    result = least_squares(residuals, np.array([w0_init, wa_init]), bounds=bounds)
    w0_hat, wa_hat = float(result.x[0]), float(result.x[1])
    H_approx = H_w0wa(z_grid, H0, om, w0_hat, wa_hat)
    comp = compare_H_models(z_grid, H_rw, H_approx)

    return {
        "bridge_kind": "H_only_flat_CPL_approximation",
        "w0_hat": w0_hat,
        "wa_hat": wa_hat,
        "success": bool(result.success),
        "message": str(result.message),
        "rms_frac_err": comp["rms_frac_err"],
        "max_frac_err": comp["max_frac_err"],
        "z_min": float(np.min(z_grid)),
        "z_max": float(np.max(z_grid)),
        "n_grid": int(z_grid.size),
    }
