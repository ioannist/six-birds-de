"""Distance-redshift inference utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import least_squares


@dataclass(frozen=True)
class FitResult:
    params: dict[str, float]
    chi2: float
    aic: float
    success: bool


def make_redshift_from_a(a: np.ndarray, *, normalize_to_last: bool = True) -> np.ndarray:
    """Convert scale factor history into redshift array."""
    a = np.asarray(a, dtype=float)
    if a.ndim != 1:
        raise ValueError("a must be a 1D array")
    if normalize_to_last:
        denom = a[-1]
        if denom <= 0:
            raise ValueError("a[-1] must be positive for normalization")
        a_norm = a / denom
    else:
        a_norm = a
    eps = np.finfo(float).eps
    a_safe = np.clip(a_norm, eps, None)
    z = 1.0 / a_safe - 1.0
    return z


def distance_from_Hz(z: np.ndarray, H: np.ndarray, z_eval: np.ndarray) -> np.ndarray:
    """Compute comoving distance proxy from H(z)."""
    z = np.asarray(z, dtype=float)
    H = np.asarray(H, dtype=float)
    z_eval = np.asarray(z_eval, dtype=float)
    if z.shape != H.shape:
        raise ValueError("z and H must have the same shape")

    mask = z >= 0.0
    if not np.any(mask):
        raise ValueError("z must include non-negative values")
    z = z[mask]
    H = H[mask]

    order = np.argsort(z)
    z = z[order]
    H = H[order]

    z_max_eval = float(np.max(z_eval))
    z_max_eval = max(z_max_eval, float(z[-1]))
    z_grid = np.linspace(0.0, z_max_eval, 4000)

    H_safe = np.interp(z_grid, z, H, left=H[0], right=H[-1])
    eps = np.finfo(float).eps
    H_safe = np.clip(H_safe, eps, None)
    invH = 1.0 / H_safe
    dist_grid = cumulative_trapezoid(invH, z_grid, initial=0.0)

    return np.interp(z_eval, z_grid, dist_grid)


def generate_mock_distance(
    z_eval: np.ndarray,
    D_true: np.ndarray,
    *,
    noise_frac: float,
    noise_floor: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate noisy mock distances with Gaussian errors."""
    z_eval = np.asarray(z_eval, dtype=float)
    D_true = np.asarray(D_true, dtype=float)
    if z_eval.shape != D_true.shape:
        raise ValueError("z_eval and D_true must have the same shape")
    rng = np.random.default_rng(seed)
    sigma = noise_floor + noise_frac * D_true
    D_obs = D_true + rng.normal(0.0, sigma)
    return D_obs, sigma


def D_matter_only(z: np.ndarray, H0: float) -> np.ndarray:
    """Matter-only (Omega_Lambda=0) comoving distance in flat FLRW."""
    z = np.asarray(z, dtype=float)
    if H0 <= 0:
        raise ValueError("H0 must be positive")
    return (2.0 / H0) * (1.0 - 1.0 / np.sqrt(1.0 + z))


def D_lcdm(z: np.ndarray, H0: float, omega_lambda: float, *, n_grid: int = 4000) -> np.ndarray:
    """Flat LambdaCDM comoving distance via numerical integration."""
    z = np.asarray(z, dtype=float)
    if H0 <= 0:
        raise ValueError("H0 must be positive")
    if not (0.0 <= omega_lambda <= 0.95):
        raise ValueError("omega_lambda must be in [0, 0.95]")

    z_max = float(np.max(z))
    z_grid = np.linspace(0.0, z_max, int(n_grid))
    omega_m = 1.0 - omega_lambda
    E = np.sqrt(omega_m * (1.0 + z_grid) ** 3 + omega_lambda)
    invE = 1.0 / E
    dist_grid = cumulative_trapezoid(invE, z_grid, initial=0.0) / H0
    return np.interp(z, z_grid, dist_grid)


def fit_matter_only(
    z: np.ndarray,
    D_obs: np.ndarray,
    sigma: np.ndarray,
    *,
    H0_init: float | None = None,
) -> FitResult:
    """Fit matter-only model with free H0."""
    z = np.asarray(z, dtype=float)
    D_obs = np.asarray(D_obs, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    if H0_init is None:
        H0_init = 1.0

    def residuals(p: np.ndarray) -> np.ndarray:
        H0 = p[0]
        model = D_matter_only(z, H0)
        return (model - D_obs) / sigma

    res = least_squares(residuals, x0=[H0_init], bounds=([1e-6], [1e6]))
    chi2 = float(np.sum(res.fun ** 2))
    k = 1
    aic = chi2 + 2 * k
    return FitResult(
        params={"H0": float(res.x[0])},
        chi2=chi2,
        aic=aic,
        success=bool(res.success),
    )


def fit_lcdm(
    z: np.ndarray,
    D_obs: np.ndarray,
    sigma: np.ndarray,
    *,
    H0_init: float | None = None,
    omega_lambda_init: float = 0.7,
) -> FitResult:
    """Fit flat LambdaCDM with free H0 and Omega_Lambda."""
    z = np.asarray(z, dtype=float)
    D_obs = np.asarray(D_obs, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    if H0_init is None:
        H0_init = 1.0

    def residuals(p: np.ndarray) -> np.ndarray:
        H0, omega_lambda = p
        model = D_lcdm(z, H0, omega_lambda)
        return (model - D_obs) / sigma

    res = least_squares(
        residuals,
        x0=[H0_init, omega_lambda_init],
        bounds=([1e-6, 0.0], [1e6, 0.95]),
    )
    chi2 = float(np.sum(res.fun ** 2))
    k = 2
    aic = chi2 + 2 * k
    return FitResult(
        params={"H0": float(res.x[0]), "omega_lambda": float(res.x[1])},
        chi2=chi2,
        aic=aic,
        success=bool(res.success),
    )


def compare_models(fitA: FitResult, fitB: FitResult) -> dict[str, float]:
    """Compare two fits via delta chi2 and delta AIC."""
    return {
        "delta_chi2": float(fitA.chi2 - fitB.chi2),
        "delta_aic": float(fitA.aic - fitB.aic),
    }
