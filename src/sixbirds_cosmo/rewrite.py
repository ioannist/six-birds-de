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
    if z_proxy.ndim != 1 or not len(z_proxy) or not np.all(np.isfinite(z_proxy)) or not np.all(np.isfinite(var_proxy)):
        raise ValueError("proxy must consist of finite nonempty vectors")
    if np.any(var_proxy < 0) or not np.isfinite(eps) or eps <= 0:
        raise ValueError("variance must be nonnegative and eps positive")

    mask = z_proxy >= 0.0
    if not np.any(mask):
        raise ValueError("z_proxy must include non-negative values")

    z_f = z_proxy[mask]
    v_f = var_proxy[mask]
    order = np.argsort(z_f)
    z_sorted = z_f[order]
    v_sorted = v_f[order]
    if np.any(np.diff(z_sorted) <= 0):
        raise ValueError("proxy redshifts must be distinct")

    scale = float(np.max(v_sorted))
    g_sorted = v_sorted / (scale + eps)
    return {
        "z_sorted": z_sorted,
        "g_sorted": g_sorted,
        "scale": scale,
    }


def _interp_proxy(z: np.ndarray, z_proxy: np.ndarray, g_proxy: np.ndarray) -> np.ndarray:
    z_proxy = np.asarray(z_proxy, dtype=float)
    g_proxy = np.asarray(g_proxy, dtype=float)
    if z_proxy.ndim != 1 or len(z_proxy) < 2 or g_proxy.shape != z_proxy.shape:
        raise ValueError("proxy grid and shape must be matching vectors with at least two points")
    if not np.all(np.isfinite(z_proxy)) or not np.all(np.isfinite(g_proxy)) or np.any(g_proxy < 0) or np.any(np.diff(z_proxy) <= 0):
        raise ValueError("proxy must be finite, nonnegative and strictly ordered")
    if not np.all(np.isfinite(z)) or np.any(z < 0) or np.any(z < z_proxy[0]) or np.any(z > z_proxy[-1]):
        raise ValueError("requested redshifts must be nonnegative and within proxy support")
    return np.interp(z, z_proxy, g_proxy, left=g_proxy[0], right=g_proxy[-1])


def positivity_ratio_max(z_max: float, z_proxy: np.ndarray, g_proxy: np.ndarray) -> float:
    """Maximize g(z)/(1+z)^3 over the whole piecewise-linear proxy interval.

    Endpoints alone or a numerical grid can miss the maximum. On a segment
    g=s*z+b the only interior stationary point is (s-3*b)/(2*s).
    Thus A/H0^2 < 1/this maximum guarantees H^2>0 everywhere on [0,z_max].
    """
    z_proxy = np.asarray(z_proxy, dtype=float)
    g_proxy = np.asarray(g_proxy, dtype=float)
    knots = np.unique(np.concatenate(([0.0, z_max], z_proxy[(z_proxy > 0) & (z_proxy < z_max)])))
    values = _interp_proxy(knots, z_proxy, g_proxy)
    candidates = list(knots)
    for left, right, g_left, g_right in zip(knots[:-1], knots[1:], values[:-1], values[1:]):
        slope = (g_right - g_left) / (right - left)
        intercept = g_left - slope * left
        if slope != 0:
            point = (slope - 3 * intercept) / (2 * slope)
            if left < point < right:
                candidates.append(point)
    candidates = np.asarray(candidates)
    return float(np.max(_interp_proxy(candidates, z_proxy, g_proxy) / (1 + candidates)**3))


def H_rewrite(
    z: np.ndarray,
    H0: float,
    A: float,
    z_proxy: np.ndarray,
    g_proxy: np.ndarray,
    *,
    eps: float = 1e-15,
) -> np.ndarray:
    """Synthetic subtractive family H^2 = H0^2(1+z)^3 - A*g(z).

    H0 is the matter coefficient's square root, not the present Hubble rate
    unless A*g(0)=0. This differs from the additive public-data family.
    Nonpositive H^2 is rejected rather than replaced by a clipped model.
    """
    z = np.asarray(z, dtype=float)
    if not np.isfinite(H0) or H0 <= 0 or not np.isfinite(A) or A < 0:
        raise ValueError("H0 must be positive and A nonnegative, both finite")
    g = _interp_proxy(z, z_proxy, g_proxy)
    H2 = H0**2 * (1.0 + z) ** 3 - A * g
    if np.any(H2 <= 0) or not np.all(np.isfinite(H2)):
        raise ValueError("Non-positive H^2 encountered in synthetic rewrite model")
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
    if z.ndim != 1 or not len(z) or not np.all(np.isfinite(z)) or np.any(z < 0) or n_grid < 2:
        raise ValueError("distance redshifts must be a finite nonnegative vector and n_grid >= 2")
    z_max = float(np.max(z))
    ratio_max = positivity_ratio_max(z_max, z_proxy, g_proxy)
    if H0 <= 0 or A < 0 or not np.isfinite(H0) or not np.isfinite(A) or H0**2 <= A * ratio_max:
        raise ValueError("H^2 must be positive on the entire distance interval")
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
    """Fit two parameters within the physical domain, without clipping/penalties."""
    z = np.asarray(z, dtype=float)
    D_obs = np.asarray(D_obs, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    z_proxy = np.asarray(proxy["z_sorted"], dtype=float)
    g_proxy = np.asarray(proxy["g_sorted"], dtype=float)
    if z.ndim != 1 or not len(z) or D_obs.shape != z.shape or sigma.shape != z.shape:
        raise ValueError("fit inputs must be matching nonempty vectors")
    if not np.all(np.isfinite(D_obs)) or not np.all(np.isfinite(sigma)) or np.any(sigma <= 0):
        raise ValueError("data must be finite and uncertainties finite and positive")

    z_max = float(np.max(z))
    ratio_max = positivity_ratio_max(z_max, z_proxy, g_proxy)
    if ratio_max <= 0:
        raise ValueError("rewrite amplitude is unidentifiable for a zero proxy")
    if not np.isfinite(H0_init) or H0_init <= 0 or not np.isfinite(A_init) or A_init < 0:
        raise ValueError("initial H0 must be positive and A nonnegative")

    def residuals(p: np.ndarray) -> np.ndarray:
        H0, fraction = p
        A = fraction * H0**2 / ratio_max
        D_model = D_rewrite(z, H0, A, z_proxy, g_proxy)
        data_resid = (D_model - D_obs) / sigma
        return data_resid

    res = least_squares(
        residuals,
        x0=[H0_init, min(A_init * ratio_max / H0_init**2, 0.9)],
        bounds=([1e-6, 0.0], [1e6, 1.0 - 1e-8]),
    )

    H0_opt, fraction_opt = res.x
    A_opt = fraction_opt * H0_opt**2 / ratio_max
    D_model = D_rewrite(z, H0_opt, A_opt, z_proxy, g_proxy)
    chi2 = float(np.sum(((D_model - D_obs) / sigma) ** 2))
    k = 2
    aic = chi2 + 2 * k
    return {
        "params": {"H0": float(H0_opt), "A": float(A_opt)},
        "chi2": chi2,
        "aic": aic,
        "success": bool(res.success),
        "H_present": float(H_rewrite(np.array([0.0]), H0_opt, A_opt, z_proxy, g_proxy)[0]),
        "positivity_fraction": float(fraction_opt),
        "family": "subtractive_generator_proxy",
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
