"""MAP fitting utilities."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
from scipy.optimize import minimize, minimize_scalar

from sixbirds_cosmo.likelihoods.base import Dataset, as_theta_dict


def fit_map(
    dataset: Dataset,
    theta0: np.ndarray,
    *,
    bounds: list[tuple[float, float]] | None = None,
    method: str | None = None,
    options: dict | None = None,
) -> Dict[str, Any]:
    """Fit maximum a-posteriori parameters by minimizing negative loglike."""
    theta0 = np.asarray(theta0, dtype=float)

    if bounds is not None:
        method = method or "L-BFGS-B"
    else:
        method = method or "BFGS"

    def nll(theta: np.ndarray) -> float:
        value = -float(dataset.loglike(theta))
        # A finite barrier lets the optimizer traverse unsupported points without
        # inf-inf arithmetic. Reported scores still use the actual likelihood.
        return value if np.isfinite(value) else 1e100

    if theta0.size == 1 and bounds is not None and method == "L-BFGS-B":
        # A bounded scalar search avoids undefined finite differences across the
        # released alpha support and handles its piecewise-linear cusp minimum.
        scalar_options = {"xatol": 1e-10, **(options or {})}
        scalar = minimize_scalar(lambda val: nll(np.array([val])), bounds=bounds[0],
                                 method="bounded", options=scalar_options)
        result = scalar
        theta_hat = np.array([scalar.x])
        candidates = [scalar.x, bounds[0][0], bounds[0][1]]
        if bounds[0][0] <= theta0[0] <= bounds[0][1]:
            candidates.append(theta0[0])
        theta_hat[0] = min(candidates, key=lambda val: nll(np.array([val])))
    else:
        result = minimize(nll, theta0, method=method, bounds=bounds, options=options)
        theta_hat = np.asarray(result.x, dtype=float)

    loglike_hat = float(dataset.loglike(theta_hat))
    k_explicit = int(theta_hat.size)
    k = k_explicit + dataset.n_profiled_parameters
    n_data = dataset.n_data

    chi2_hat = None
    if dataset.loglike_is_minus_half_chi2:
        chi2_hat = -2.0 * loglike_hat

    aic = 2.0 * k - 2.0 * loglike_hat
    bic = None
    if n_data is not None and n_data > 0:
        bic = k * float(np.log(n_data)) - 2.0 * loglike_hat

    return {
        "theta_hat": theta_hat.tolist(),
        "loglike_hat": loglike_hat,
        "success": bool(result.success and np.isfinite(loglike_hat)),
        "message": str(result.message),
        "nfev": int(result.nfev) if result.nfev is not None else None,
        "nit": int(result.nit) if getattr(result, "nit", None) is not None else None,
        "k": k,
        "k_explicit": k_explicit,
        "k_profiled": dataset.n_profiled_parameters,
        "n_data": n_data,
        "chi2_hat": chi2_hat,
        "aic": aic,
        "bic": bic,
    }


def format_fit_summary(
    fit_result: Dict[str, Any], *, param_names: list[str] | None = None
) -> Dict[str, Any]:
    """Format a fit result into a serializable summary dict."""
    summary = {
        "theta_hat": fit_result.get("theta_hat"),
        "loglike_hat": fit_result.get("loglike_hat"),
        "chi2_hat": fit_result.get("chi2_hat"),
        "aic": fit_result.get("aic"),
        "bic": fit_result.get("bic"),
        "k": fit_result.get("k"),
        "n_data": fit_result.get("n_data"),
        "success": fit_result.get("success"),
        "message": fit_result.get("message"),
    }

    if param_names is not None and fit_result.get("theta_hat") is not None:
        summary["theta_hat_named"] = as_theta_dict(
            np.asarray(fit_result["theta_hat"], dtype=float), param_names
        )

    return summary
