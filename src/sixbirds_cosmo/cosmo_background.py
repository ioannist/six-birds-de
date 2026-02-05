"""Background cosmology utilities.

Conventions:
- H0 is in km/s/Mpc.
- Distances are returned in Mpc.
- Speed of light c defaults to 299792.458 km/s.
- Models include matter, curvature, and Lambda (no radiation).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.integrate import cumulative_trapezoid, quad


def _as_array(z: np.ndarray | float) -> Tuple[np.ndarray, bool]:
    z_arr = np.asarray(z, dtype=float)
    return z_arr, z_arr.ndim == 0


def _reshape_output(values: np.ndarray, shape: tuple[int, ...], scalar: bool) -> np.ndarray | float:
    if scalar:
        return float(values.reshape(()))
    return values.reshape(shape)


def E_lcdm(z: np.ndarray | float, om: float, ol: float) -> np.ndarray | float:
    """Dimensionless expansion rate E(z) for LCDM with curvature."""
    z_arr, scalar = _as_array(z)
    if np.any(z_arr < -1.0):
        raise ValueError("Redshift z must be >= -1.")
    ok = 1.0 - om - ol
    ez = np.sqrt(om * (1.0 + z_arr) ** 3 + ok * (1.0 + z_arr) ** 2 + ol)
    return _reshape_output(ez, z_arr.shape, scalar)


def H_lcdm(z: np.ndarray | float, H0: float, om: float, ol: float) -> np.ndarray | float:
    """Hubble rate H(z) in km/s/Mpc for LCDM with curvature."""
    return H0 * E_lcdm(z, om, ol)


def comoving_distance(
    z: np.ndarray | float,
    H0: float,
    om: float,
    ol: float,
    *,
    c_km_s: float = 299792.458,
    method: str = "trapz",
    rtol: float = 1e-10,
    atol: float = 0.0,
    n_grid: int = 8192,
) -> np.ndarray | float:
    """Comoving distance D_C(z) in Mpc."""
    z_arr, scalar = _as_array(z)
    if np.any(z_arr < 0.0):
        raise ValueError("Redshift z must be >= 0 for distance calculations.")

    shape = z_arr.shape
    z_flat = z_arr.reshape(-1)

    if method == "trapz":
        z_max = float(np.max(z_flat)) if z_flat.size else 0.0
        if z_max == 0.0:
            dc_flat = np.zeros_like(z_flat)
        else:
            z_grid = np.linspace(0.0, z_max, n_grid)
            invE = 1.0 / E_lcdm(z_grid, om, ol)
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
                lambda zz: 1.0 / E_lcdm(zz, om, ol), 0.0, z_i, epsrel=rtol, epsabs=atol
            )
            dc_values.append((c_km_s / H0) * integral)
        dc_flat = np.array(dc_values, dtype=float)
    else:
        raise ValueError("method must be 'trapz' or 'quad'.")

    return _reshape_output(dc_flat, shape, scalar)


def transverse_comoving_distance(
    z: np.ndarray | float,
    H0: float,
    om: float,
    ol: float,
    **kwargs,
) -> np.ndarray | float:
    """Transverse comoving distance D_M(z) in Mpc."""
    z_arr, scalar = _as_array(z)
    ok = 1.0 - om - ol
    dc = comoving_distance(z_arr, H0, om, ol, **kwargs)
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


def angular_diameter_distance(
    z: np.ndarray | float, H0: float, om: float, ol: float, **kwargs
) -> np.ndarray | float:
    """Angular diameter distance D_A(z) in Mpc."""
    z_arr, scalar = _as_array(z)
    dm = transverse_comoving_distance(z_arr, H0, om, ol, **kwargs)
    da = np.asarray(dm, dtype=float) / (1.0 + z_arr)
    return _reshape_output(da, z_arr.shape, scalar)


def luminosity_distance(
    z: np.ndarray | float, H0: float, om: float, ol: float, **kwargs
) -> np.ndarray | float:
    """Luminosity distance D_L(z) in Mpc."""
    z_arr, scalar = _as_array(z)
    dm = transverse_comoving_distance(z_arr, H0, om, ol, **kwargs)
    dl = np.asarray(dm, dtype=float) * (1.0 + z_arr)
    return _reshape_output(dl, z_arr.shape, scalar)


def E_wcdm(z: np.ndarray | float, om: float, w: float) -> np.ndarray | float:
    """Dimensionless expansion rate E(z) for flat wCDM (no radiation)."""
    z_arr, scalar = _as_array(z)
    if np.any(z_arr < -1.0):
        raise ValueError("Redshift z must be >= -1.")
    ol = 1.0 - om
    ez = np.sqrt(om * (1.0 + z_arr) ** 3 + ol * (1.0 + z_arr) ** (3.0 * (1.0 + w)))
    return _reshape_output(ez, z_arr.shape, scalar)


def H_wcdm(z: np.ndarray | float, H0: float, om: float, w: float) -> np.ndarray | float:
    """Hubble rate H(z) in km/s/Mpc for flat wCDM."""
    return H0 * E_wcdm(z, om, w)


def luminosity_distance_wcdm(
    z: np.ndarray | float,
    H0: float,
    om: float,
    w: float,
    *,
    c_km_s: float = 299792.458,
    method: str = "trapz",
    rtol: float = 1e-10,
    atol: float = 0.0,
    n_grid: int = 8192,
) -> np.ndarray | float:
    """Luminosity distance D_L(z) in Mpc for flat wCDM."""
    z_arr, scalar = _as_array(z)
    if np.any(z_arr < 0.0):
        raise ValueError("Redshift z must be >= 0 for distance calculations.")

    shape = z_arr.shape
    z_flat = z_arr.reshape(-1)

    if method == "trapz":
        z_max = float(np.max(z_flat)) if z_flat.size else 0.0
        if z_max == 0.0:
            dc_flat = np.zeros_like(z_flat)
        else:
            z_grid = np.linspace(0.0, z_max, n_grid)
            invE = 1.0 / E_wcdm(z_grid, om, w)
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
                lambda zz: 1.0 / E_wcdm(zz, om, w), 0.0, z_i, epsrel=rtol, epsabs=atol
            )
            dc_values.append((c_km_s / H0) * integral)
        dc_flat = np.array(dc_values, dtype=float)
    else:
        raise ValueError("method must be 'trapz' or 'quad'.")

    dl = dc_flat * (1.0 + z_flat)
    return _reshape_output(dl, shape, scalar)


def distance_modulus(
    z: np.ndarray | float, H0: float, om: float, ol: float, **kwargs
) -> np.ndarray | float:
    """Distance modulus mu(z) from luminosity distance."""
    dl = luminosity_distance(z, H0, om, ol, **kwargs)
    dl_arr = np.asarray(dl, dtype=float)
    mu = 5.0 * np.log10(dl_arr) + 25.0
    return _reshape_output(mu, dl_arr.shape, dl_arr.ndim == 0)
