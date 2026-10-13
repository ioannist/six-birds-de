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
    if a.ndim != 1 or not len(a) or not np.all(np.isfinite(a)) or np.any(a <= 0):
        raise ValueError("a must be a finite nonempty positive 1D array")
    if normalize_to_last:
        denom = a[-1]
        if denom <= 0:
            raise ValueError("a[-1] must be positive for normalization")
        a_norm = a / denom
    else:
        a_norm = a
    z = 1.0 / a_norm - 1.0
    return z


def distance_from_Hz(z: np.ndarray, H: np.ndarray, z_eval: np.ndarray) -> np.ndarray:
    """Compute chi(z)=integral dz/H(z), in units c=1 (not luminosity distance).

    The source must cover [0,max(z_eval)]; extrapolation does not supply missing
    generator evolution. H must be finite and strictly positive throughout.
    """
    z = np.asarray(z, dtype=float)
    H = np.asarray(H, dtype=float)
    z_eval = np.asarray(z_eval, dtype=float)
    if z.ndim != 1 or len(z) < 2 or z.shape != H.shape:
        raise ValueError("z and H must be matching vectors with at least two points")
    if not np.all(np.isfinite(z)) or not np.all(np.isfinite(H)) or np.any(H <= 0):
        raise ValueError("source redshifts must be finite and H finite and positive")
    if not np.all(np.isfinite(z_eval)) or np.any(z_eval < 0):
        raise ValueError("evaluation redshifts must be finite and nonnegative")
    if z_eval.size == 0:
        return np.empty_like(z_eval)

    mask = z >= 0.0
    if not np.any(mask):
        raise ValueError("z must include non-negative values")
    z = z[mask]
    H = H[mask]

    order = np.argsort(z)
    z = z[order]
    H = H[order]
    if np.any(np.diff(z) <= 0) or z[0] != 0.0 or np.max(z_eval) > z[-1]:
        raise ValueError("source redshifts must be distinct and cover [0,max(z_eval)]")

    z_max_eval = float(np.max(z_eval))
    z_max_eval = max(z_max_eval, float(z[-1]))
    z_grid = np.linspace(0.0, z_max_eval, 4000)

    H_safe = np.interp(z_grid, z, H, left=H[0], right=H[-1])
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
    if not np.all(np.isfinite(D_true)) or np.any(D_true < 0) or not np.isfinite(noise_frac) or noise_frac < 0 or not np.isfinite(noise_floor) or noise_floor <= 0:
        raise ValueError("distances must be nonnegative and noise scale finite and positive")
    rng = np.random.default_rng(seed)
    sigma = noise_floor + noise_frac * D_true
    D_obs = D_true + rng.normal(0.0, sigma)
    return D_obs, sigma


def D_matter_only(z: np.ndarray, H0: float) -> np.ndarray:
    """Matter-only (Omega_Lambda=0) comoving distance in flat FLRW."""
    z = np.asarray(z, dtype=float)
    if not np.isfinite(H0) or H0 <= 0 or not np.all(np.isfinite(z)) or np.any(z < 0):
        raise ValueError("H0 must be positive")
    return (2.0 / H0) * (1.0 - 1.0 / np.sqrt(1.0 + z))


def D_lcdm(z: np.ndarray, H0: float, omega_lambda: float, *, n_grid: int = 4000) -> np.ndarray:
    """Flat LambdaCDM comoving distance via numerical integration."""
    z = np.asarray(z, dtype=float)
    if not np.isfinite(H0) or H0 <= 0 or z.ndim != 1 or not len(z) or not np.all(np.isfinite(z)) or np.any(z < 0) or n_grid < 2:
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
    _validate_fit_data(z, D_obs, sigma)
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
    _validate_fit_data(z, D_obs, sigma)
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


def _validate_fit_data(z: np.ndarray, data: np.ndarray, sigma: np.ndarray) -> None:
    if z.ndim != 1 or not len(z) or z.shape != data.shape or z.shape != sigma.shape:
        raise ValueError("fit data must be matching nonempty vectors")
    if not np.all(np.isfinite(z)) or np.any(z < 0) or not np.all(np.isfinite(data)) or not np.all(np.isfinite(sigma)) or np.any(sigma <= 0):
        raise ValueError("fit data must be finite, z nonnegative and sigma positive")
