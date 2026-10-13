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
    if theta0.ndim != 1 or len(theta0) == 0 or not np.all(np.isfinite(theta0)):
        raise ValueError("optimizer start must be a finite nonempty vector")
    if type(dataset.n_profiled_parameters) is not int or dataset.n_profiled_parameters < 0:
        raise ValueError("profiled parameter count must be a nonnegative integer")
    if dataset.param_names is not None and len(dataset.param_names) != len(theta0):
        raise ValueError("parameter names and optimizer dimensions must agree")
    if bounds is not None and (len(bounds) != len(theta0) or any(lo >= hi for lo, hi in bounds)):
        raise ValueError("optimizer bounds must match parameters with lower < upper")

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
        # A search may terminate at a worse point near a support boundary.
        # Retain a better supplied feasible start without inventing its score.
        if nll(theta0) < nll(theta_hat):
            theta_hat = theta0.copy()

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


def fit_map_multistart(dataset: Dataset, starts, *, bounds, **kwargs) -> Dict[str, Any]:
    """Compare explicitly supplied starts without claiming global optimality.

    Keep unsuccessful candidates in the receipt, and fail if the best finite
    candidate itself did not converge instead of hiding a lower failed fit.
    """
    if not starts:
        raise ValueError("at least one optimizer start is required")
    candidates = [fit_map(dataset, np.asarray(start, dtype=float), bounds=bounds, **kwargs)
                  for start in starts]
    finite = [i for i, fit in enumerate(candidates) if np.isfinite(fit["loglike_hat"])]
    if not finite:
        raise RuntimeError("no optimizer start returned a finite likelihood")
    selected = max(finite, key=lambda i: candidates[i]["loglike_hat"])
    result = dict(candidates[selected])
    result["optimization_scope"] = "best_of_declared_starts"
    result["selected_start"] = selected
    result["starts"] = [{"theta0": list(map(float, start)), **fit}
                        for start, fit in zip(starts, candidates)]
    return result
