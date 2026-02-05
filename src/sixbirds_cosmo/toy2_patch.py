"""Toy model 2: patch cosmology with backreaction-like closure proxy."""
from __future__ import annotations

from typing import Any

import numpy as np


def _resolve_params(params: dict) -> dict[str, float]:
    required = ["dt", "g4pi", "rho_mean", "delta_rho", "f_void", "kappa_scale"]
    missing = [key for key in required if key not in params]
    if missing:
        raise ValueError(f"params missing required keys: {missing}")

    dt = float(params["dt"])
    g4pi = float(params["g4pi"])
    rho_mean = float(params["rho_mean"])
    delta_rho = float(params["delta_rho"])
    f_void = float(params["f_void"])
    kappa_scale = float(params["kappa_scale"])

    if dt <= 0:
        raise ValueError("dt must be positive")
    if g4pi <= 0:
        raise ValueError("g4pi must be positive")
    if rho_mean <= 0:
        raise ValueError("rho_mean must be positive")
    if not (0.0 <= delta_rho < 1.0):
        raise ValueError("delta_rho must be in [0, 1)")
    if not (0.0 < f_void < 1.0):
        raise ValueError("f_void must be in (0, 1)")
    if kappa_scale < 0:
        raise ValueError("kappa_scale must be >= 0")

    rho_void = rho_mean * (1.0 - delta_rho)
    rho_wall = (rho_mean - f_void * rho_void) / (1.0 - f_void)

    return {
        "dt": dt,
        "g4pi": g4pi,
        "rho_mean": rho_mean,
        "delta_rho": delta_rho,
        "f_void": f_void,
        "kappa_scale": kappa_scale,
        "rho_void": rho_void,
        "rho_wall": rho_wall,
    }


def _init_rho0(N: int, rho_void: float, rho_wall: float, f_void: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_void = int(round(f_void * N))
    idx = np.arange(N)
    rng.shuffle(idx)
    rho0 = np.full(N, rho_wall, dtype=float)
    rho0[idx[:n_void]] = rho_void
    return rho0


def simulate_patches(N: int, steps: int, params: dict, seed: int) -> dict[str, Any]:
    """Simulate patch dynamics and compute domain-averaged quantities."""
    if N <= 0 or steps <= 0:
        raise ValueError("N and steps must be positive")

    resolved = _resolve_params(params)
    dt = resolved["dt"]
    g4pi = resolved["g4pi"]
    rho_mean = resolved["rho_mean"]
    delta_rho = resolved["delta_rho"]
    f_void = resolved["f_void"]
    kappa_scale = resolved["kappa_scale"]
    rho_void = resolved["rho_void"]
    rho_wall = resolved["rho_wall"]

    rho0 = _init_rho0(int(N), rho_void, rho_wall, f_void, seed)
    kappa = kappa_scale * np.maximum(0.0, (rho_mean - rho0) / rho_mean)

    H0 = np.sqrt((2.0 / 3.0) * g4pi * rho0 + kappa)
    a = np.ones(int(N), dtype=float)
    adot = H0.copy()

    def accel(a_in: np.ndarray) -> np.ndarray:
        return -(g4pi / 3.0) * rho0 / (a_in ** 2)

    t = np.linspace(0.0, dt * steps, steps + 1)
    aD = np.zeros(steps + 1, dtype=float)
    HD = np.zeros(steps + 1, dtype=float)
    rhoD = np.zeros(steps + 1, dtype=float)
    ddot_aD = np.zeros(steps + 1, dtype=float)
    Q = np.zeros(steps + 1, dtype=float)
    var_H = np.zeros(steps + 1, dtype=float)

    def record(i: int, a_in: np.ndarray, adot_in: np.ndarray) -> None:
        V = np.sum(a_in ** 3)
        dotV = np.sum(3.0 * a_in ** 2 * adot_in)
        ddotV = np.sum(3.0 * (2.0 * a_in * (adot_in ** 2) + a_in ** 2 * accel(a_in)))
        aD_i = (V / N) ** (1.0 / 3.0)
        ddot_aD_over_aD = (1.0 / 3.0) * (ddotV / V) - (2.0 / 9.0) * (dotV / V) ** 2
        aD[i] = aD_i
        HD[i] = np.sum(a_in ** 2 * adot_in) / V
        rhoD[i] = np.sum(rho0) / V
        ddot_aD[i] = ddot_aD_over_aD * aD_i
        Q[i] = 3.0 * ddot_aD_over_aD + g4pi * rhoD[i]
        H_i = adot_in / a_in
        w = (a_in ** 3) / V
        var_H[i] = np.sum(w * (H_i ** 2)) - (np.sum(w * H_i) ** 2)

    record(0, a, adot)

    for i in range(steps):
        k1a = adot
        k1v = accel(a)

        k2a = adot + 0.5 * dt * k1v
        k2v = accel(a + 0.5 * dt * k1a)

        k3a = adot + 0.5 * dt * k2v
        k3v = accel(a + 0.5 * dt * k2a)

        k4a = adot + dt * k3v
        k4v = accel(a + dt * k3a)

        a = a + (dt / 6.0) * (k1a + 2.0 * k2a + 2.0 * k3a + k4a)
        adot = adot + (dt / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)

        record(i + 1, a, adot)

    params_used = {
        "dt": dt,
        "g4pi": g4pi,
        "rho_mean": rho_mean,
        "delta_rho": delta_rho,
        "f_void": f_void,
        "kappa_scale": kappa_scale,
        "rho_void": rho_void,
        "rho_wall": rho_wall,
        "seed": int(seed),
        "N": int(N),
        "steps": int(steps),
    }

    return {
        "t": t,
        "aD": aD,
        "HD": HD,
        "rhoD": rhoD,
        "ddot_aD": ddot_aD,
        "Q": Q,
        "var_H": var_H,
        "params_used": params_used,
        "rho0": rho0,
        "kappa": kappa,
    }
