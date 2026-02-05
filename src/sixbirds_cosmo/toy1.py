"""Toy model 1: nonlinear dynamics with coarse-graining route mismatch."""
from __future__ import annotations

from typing import Any

import numpy as np

from sixbirds_cosmo.core import Completion, Lens, PackagingOperator, idempotence_defect, route_mismatch


def _resolve_params(params: dict) -> dict[str, float | int | str]:
    if "alpha" not in params or "beta" not in params or "dt" not in params:
        raise ValueError("params must include alpha, beta, and dt")
    x0_dist = params.get("x0_dist", "uniform")
    if x0_dist != "uniform":
        raise ValueError("only x0_dist='uniform' is supported")
    if "x0_low" not in params or "x0_high" not in params:
        raise ValueError("params must include x0_low and x0_high for uniform init")

    return {
        "alpha": float(params["alpha"]),
        "beta": float(params["beta"]),
        "dt": float(params["dt"]),
        "x0_dist": str(x0_dist),
        "x0_low": float(params["x0_low"]),
        "x0_high": float(params["x0_high"]),
    }


def simulate_micro(n: int, steps: int, params: dict, seed: int) -> dict[str, Any]:
    """Simulate the micro dynamics and report mismatch and idempotence signals."""
    if n <= 0 or steps <= 0:
        raise ValueError("n and steps must be positive")

    resolved = _resolve_params(params)
    alpha = float(resolved["alpha"])
    beta = float(resolved["beta"])
    dt = float(resolved["dt"])
    x0_low = float(resolved["x0_low"])
    x0_high = float(resolved["x0_high"])

    rng = np.random.default_rng(seed)
    x = rng.uniform(x0_low, x0_high, size=int(n)).astype(float)

    lens = Lens(lambda x_in: float(np.mean(x_in)))
    completion = Completion(lambda y: np.full(int(n), y, dtype=float))
    E = PackagingOperator(lens, completion)

    def T(x_in: np.ndarray) -> np.ndarray:
        return x_in + dt * (alpha * x_in - beta * (x_in ** 2))

    rm = np.zeros(int(steps), dtype=float)
    delta = np.zeros(int(steps), dtype=float)
    x_mean = np.zeros(int(steps) + 1, dtype=float)
    x_var = np.zeros(int(steps) + 1, dtype=float)

    x_mean[0] = float(np.mean(x))
    x_var[0] = float(np.var(x))

    for t in range(int(steps)):
        rm[t] = float(route_mismatch(T, E, x, space="y"))
        delta[t] = float(idempotence_defect(E, x, space="x"))
        x = T(x)
        x_mean[t + 1] = float(np.mean(x))
        x_var[t + 1] = float(np.var(x))

    params_used = {
        **resolved,
        "seed": int(seed),
        "n": int(n),
        "steps": int(steps),
    }

    return {
        "rm": rm,
        "delta": delta,
        "x_mean": x_mean,
        "x_var": x_var,
        "params_used": params_used,
    }
