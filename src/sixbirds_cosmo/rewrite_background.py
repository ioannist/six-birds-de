"""Rewrite-background cosmology utilities."""

from __future__ import annotations

import numpy as np
from scipy.integrate import cumulative_trapezoid, quad
from scipy.optimize import brentq
from sixbirds_cosmo.cosmo_background import _positive_scale


def _as_array(z: np.ndarray | float) -> tuple[np.ndarray, bool]:
    z_arr = np.asarray(z, dtype=float)
    if not np.all(np.isfinite(z_arr)) or np.any(z_arr <= -1):
        raise ValueError("Redshift must be finite and greater than -1.")
    return z_arr, z_arr.ndim == 0


def _reshape_output(values: np.ndarray, shape: tuple[int, ...], scalar: bool) -> np.ndarray | float:
    if scalar:
        return float(values.reshape(()))
    return values.reshape(shape)


def g_powerlaw(z: np.ndarray | float, m: float) -> np.ndarray | float:
    """Proxy shape g(z) = (1+z)^(-m)."""
    z_arr, scalar = _as_array(z)
    if np.any(z_arr < -1.0):
        raise ValueError("Redshift z must be >= -1.")
    g = (1.0 + z_arr) ** (-m)
    return _reshape_output(g, z_arr.shape, scalar)


def E_rewrite(z: np.ndarray | float, om: float, A: float, m: float) -> np.ndarray | float:
    """Dimensionless E(z) for rewrite background with curvature closure."""
    z_arr, scalar = _as_array(z)
    if not np.all(np.isfinite([om, A, m])) or min(om, A, m) < 0:
        raise ValueError("rewrite om, amplitude and shape must be finite and nonnegative")
    if np.any(z_arr < -1.0):
        raise ValueError("Redshift z must be >= -1.")
    ok = 1.0 - om - A
    g = (1.0 + z_arr) ** (-m)
    E2 = om * (1.0 + z_arr) ** 3 + ok * (1.0 + z_arr) ** 2 + A * g
    if np.any(E2 <= 0.0) or np.any(~np.isfinite(E2)):
        raise ValueError("Non-positive E^2 encountered in rewrite model.")
    Ez = np.sqrt(E2)
    return _reshape_output(Ez, z_arr.shape, scalar)


def rewrite_interval_minimum(z_max: float, om: float, A: float, m: float) -> tuple[float, float]:
    """Return the E^2 minimum over [0,z_max] in the declared positive family.

    Writing u=1+z gives f'(u)=u*D(u), where
    D=3*om*u+2*ok-m*A*u^(-m-2). Its derivative is nonnegative, so endpoints
    and its unique possible interior zero exhaust all minimum candidates.
    This is a floating evaluation of an analytic criterion, not an interval
    arithmetic certificate for arbitrarily near-degenerate coefficients.
    """
    if not np.isfinite(z_max) or z_max < 0 or not np.all(np.isfinite([om, A, m])) or min(om, A, m) < 0:
        raise ValueError("interval and rewrite coefficients must be finite and nonnegative")
    ok = 1 - om - A
    def derivative_factor(u):
        return 3 * om * u + 2 * ok - m * A * u**(-m-2)
    points = [1., 1. + z_max]
    if derivative_factor(points[0]) < 0 < derivative_factor(points[-1]):
        points.append(brentq(derivative_factor, points[0], points[-1]))
    u = np.array(points)
    values = om * u**3 + ok * u**2 + A * u**(-m)
    i = int(np.argmin(values))
    return float(u[i] - 1), float(values[i])


def H_rewrite(z: np.ndarray | float, H0: float, om: float, A: float, m: float) -> np.ndarray | float:
    """H(z) in km/s/Mpc for rewrite background."""
    _positive_scale(H0)
    return H0 * E_rewrite(z, om, A, m)


def comoving_distance_rewrite(
    z: np.ndarray | float,
    H0: float,
    om: float,
    A: float,
    m: float,
    *,
    c_km_s: float = 299792.458,
    method: str = "trapz",
    n_grid: int = 8192,
    rtol: float = 1e-10,
    atol: float = 0.0,
) -> np.ndarray | float:
    """Comoving distance D_C(z) in Mpc for rewrite background."""
    _positive_scale(H0)
    _positive_scale(c_km_s)
    if n_grid < 2:
        raise ValueError("n_grid must be at least two.")
    z_arr, scalar = _as_array(z)
    if np.any(z_arr < 0.0):
        raise ValueError("Redshift z must be >= 0 for distance calculations.")

    shape = z_arr.shape
    z_flat = z_arr.reshape(-1)
    z_max = float(np.max(z_flat)) if z_flat.size else 0.0
    _, minimum = rewrite_interval_minimum(z_max, om, A, m)
    if not np.isfinite(minimum) or minimum <= 0:
        raise ValueError("rewrite E^2 must be positive on the entire distance interval")

    if method == "trapz":
        z_max = float(np.max(z_flat)) if z_flat.size else 0.0
        if z_max == 0.0:
            dc_flat = np.zeros_like(z_flat)
        else:
            z_grid = np.linspace(0.0, z_max, n_grid)
            invE = 1.0 / E_rewrite(z_grid, om, A, m)
            integral = cumulative_trapezoid(invE, z_grid, initial=0.0)
            dc_grid = (c_km_s / H0) * integral
            dc_flat = np.interp(z_flat, z_grid, dc_grid)
    elif method == "quad":
        dc_values = []
        for z_i in z_flat:
            if z_i == 0.0:
                dc_values.append(0.0)
                continue
            integral, _ = quad(
                lambda zz: 1.0 / E_rewrite(zz, om, A, m),
                0.0,
                z_i,
                epsrel=rtol,
                epsabs=atol,
            )
            dc_values.append((c_km_s / H0) * integral)
        dc_flat = np.array(dc_values, dtype=float)
    else:
        raise ValueError("method must be 'trapz' or 'quad'.")

    return _reshape_output(dc_flat, shape, scalar)


def transverse_comoving_distance_rewrite(
    z: np.ndarray | float,
    H0: float,
    om: float,
    A: float,
    m: float,
    **kwargs,
) -> np.ndarray | float:
    """Transverse comoving distance D_M(z) in Mpc for rewrite background."""
    z_arr, scalar = _as_array(z)
    ok = 1.0 - om - A
    dc = comoving_distance_rewrite(z_arr, H0, om, A, m, **kwargs)
    dc_arr = np.asarray(dc, dtype=float)

    if abs(ok) < 1e-12:
        dm = dc_arr
    else:
        c_km_s = kwargs.get("c_km_s", 299792.458)
        sqrt_ok = np.sqrt(abs(ok))
        arg = sqrt_ok * (H0 / c_km_s) * dc_arr
        if ok > 0:
            dm = (c_km_s / H0) / sqrt_ok * np.sinh(arg)
        else:
            dm = (c_km_s / H0) / sqrt_ok * np.sin(arg)

    return _reshape_output(dm, z_arr.shape, scalar)


def luminosity_distance_rewrite(
    z: np.ndarray | float, H0: float, om: float, A: float, m: float, **kwargs
) -> np.ndarray | float:
    """Luminosity distance D_L(z) in Mpc for rewrite background."""
    z_arr, scalar = _as_array(z)
    dm = transverse_comoving_distance_rewrite(z_arr, H0, om, A, m, **kwargs)
    dl = np.asarray(dm, dtype=float) * (1.0 + z_arr)
    return _reshape_output(dl, z_arr.shape, scalar)


def distance_modulus_rewrite(
    z: np.ndarray | float, H0: float, om: float, A: float, m: float, **kwargs
) -> np.ndarray | float:
    """Distance modulus mu(z) for rewrite background."""
    dl = luminosity_distance_rewrite(z, H0, om, A, m, **kwargs)
    dl_arr = np.asarray(dl, dtype=float)
    if np.any(dl_arr <= 0) or not np.all(np.isfinite(dl_arr)):
        raise ValueError("Distance modulus requires a finite positive luminosity distance.")
    mu = 5.0 * np.log10(dl_arr) + 25.0
    return _reshape_output(mu, dl_arr.shape, dl_arr.ndim == 0)
