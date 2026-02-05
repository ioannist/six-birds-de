"""Rewrite-term utilities for distance fitting."""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import least_squares


def prepare_proxy_shape(
    z_proxy: np.ndarray, var_proxy: np.ndarray, *, eps: float = 1e-15
) -> dict[str, Any]:
    """Prepare a normalized proxy shape g(z) from heterogeneity variance."""
    z_proxy = np.asarray(z_proxy, dtype=float)
    var_proxy = np.asarray(var_proxy, dtype=float)
    if z_proxy.shape != var_proxy.shape:
        raise ValueError("z_proxy and var_proxy must have the same shape")

    mask = z_proxy >= 0.0
    if not np.any(mask):
        raise ValueError("z_proxy must include non-negative values")

    z_f = z_proxy[mask]
    v_f = var_proxy[mask]
    order = np.argsort(z_f)
    z_sorted = z_f[order]
    v_sorted = v_f[order]

    scale = float(np.max(v_sorted))
    g_sorted = v_sorted / (scale + eps)
    return {
        "z_sorted": z_sorted,
        "g_sorted": g_sorted,
        "scale": scale,
    }


def _interp_proxy(z: np.ndarray, z_proxy: np.ndarray, g_proxy: np.ndarray) -> np.ndarray:
    return np.interp(z, z_proxy, g_proxy, left=g_proxy[0], right=g_proxy[-1])


def H_rewrite(
    z: np.ndarray,
    H0: float,
    A: float,
    z_proxy: np.ndarray,
    g_proxy: np.ndarray,
    *,
    eps: float = 1e-15,
) -> np.ndarray:
    """Compute rewrite-modified H(z) with clipping for nonphysical H^2."""
    z = np.asarray(z, dtype=float)
    g = _interp_proxy(z, z_proxy, g_proxy)
    H2 = H0**2 * (1.0 + z) ** 3 - A * g
    H2 = np.clip(H2, eps, None)
    return np.sqrt(H2)


def D_rewrite(
    z: np.ndarray,
    H0: float,
    A: float,
    z_proxy: np.ndarray,
    g_proxy: np.ndarray,
    *,
    n_grid: int = 4000,
) -> np.ndarray:
    """Comoving distance for the rewrite model."""
    z = np.asarray(z, dtype=float)
    z_max = float(np.max(z))
    z_grid = np.linspace(0.0, z_max, int(n_grid))
    H_grid = H_rewrite(z_grid, H0, A, z_proxy, g_proxy)
    dist_grid = cumulative_trapezoid(1.0 / H_grid, z_grid, initial=0.0)
    return np.interp(z, z_grid, dist_grid)


def fit_rewrite(
    z: np.ndarray,
    D_obs: np.ndarray,
    sigma: np.ndarray,
    proxy: dict,
    *,
    H0_init: float = 1.0,
    A_init: float = 0.1,
) -> dict[str, Any]:
    """Fit rewrite model with penalty for nonphysical H^2 regions."""
    z = np.asarray(z, dtype=float)
    D_obs = np.asarray(D_obs, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    z_proxy = np.asarray(proxy["z_sorted"], dtype=float)
    g_proxy = np.asarray(proxy["g_sorted"], dtype=float)

    z_max = float(np.max(z))
    z_grid = np.linspace(0.0, z_max, 4000)

    def residuals(p: np.ndarray) -> np.ndarray:
        H0, A = p
        D_model = D_rewrite(z, H0, A, z_proxy, g_proxy)
        data_resid = (D_model - D_obs) / sigma
        g_grid = _interp_proxy(z_grid, z_proxy, g_proxy)
        H2_grid = H0**2 * (1.0 + z_grid) ** 3 - A * g_grid
        min_H2 = float(np.min(H2_grid))
        penalty = 0.0 if min_H2 > 0.0 else 1e6 * abs(min_H2)
        return np.concatenate([data_resid, np.array([penalty])])

    res = least_squares(
        residuals,
        x0=[H0_init, A_init],
        bounds=([1e-6, 0.0], [1e6, np.inf]),
    )

    H0_opt, A_opt = res.x
    D_model = D_rewrite(z, H0_opt, A_opt, z_proxy, g_proxy)
    chi2 = float(np.sum(((D_model - D_obs) / sigma) ** 2))
    k = 2
    aic = chi2 + 2 * k
    return {
        "params": {"H0": float(H0_opt), "A": float(A_opt)},
        "chi2": chi2,
        "aic": aic,
        "success": bool(res.success),
    }


def fit_table_row(model: str, k: int, params: dict, chi2: float, aic: float) -> dict[str, Any]:
    """Build a summary row for a fit table."""
    return {
        "model": model,
        "k": int(k),
        "H0": params.get("H0"),
        "omega_lambda": params.get("omega_lambda"),
        "A": params.get("A"),
        "chi2": float(chi2),
        "aic": float(aic),
    }
