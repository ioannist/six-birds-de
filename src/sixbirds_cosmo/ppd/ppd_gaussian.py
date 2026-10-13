"""Gaussian PPD utilities with simple parametric theory stubs."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from scipy.linalg import cho_solve
from sixbirds_cosmo.numerics import factor_covariance, generalized_least_squares, validated_indices


def make_directions(y: np.ndarray, cov: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Construct two deterministic direction vectors."""
    sigma = np.sqrt(np.diag(cov))
    n = len(y)
    d1 = sigma.copy()
    ramp = (np.arange(n, dtype=float) - n / 2.0) / (n / 2.0)
    d2 = sigma * ramp

    def normalize(vec: np.ndarray) -> np.ndarray:
        rms = float(np.sqrt(np.mean(vec**2)))
        return vec / rms if rms > 0 else vec

    return normalize(d1), normalize(d2)


def _cho_with_jitter(cov: np.ndarray) -> Tuple[Tuple[np.ndarray, bool], float]:
    return factor_covariance(cov), 0.0


def _fit_linear_amplitudes(
    r0_A: np.ndarray, D_A: np.ndarray, cov_A: np.ndarray
) -> Tuple[np.ndarray, float, float]:
    amps, chi2_A = generalized_least_squares(-r0_A, D_A, cov_A)
    return amps, chi2_A, 0.0


def fit_amplitudes_on_block(
    y: np.ndarray,
    cov: np.ndarray,
    idxA: np.ndarray,
    model_kind: str,
    *,
    r0_amp: float = 0.1,
) -> Dict[str, object]:
    """Fit stub amplitudes on block A."""
    y = np.asarray(y, dtype=float)
    cov = np.asarray(cov, dtype=float)
    idxA = validated_indices(idxA, len(y))
    d1, d2 = make_directions(y, cov)
    r0 = r0_amp * (d1 + d2)

    r0_A = r0[idxA]
    cov_A = cov[np.ix_(idxA, idxA)]

    if model_kind == "lcdm_stub":
        D_A = d1[idxA][:, None]
        amps, chi2_A, jitter_used = _fit_linear_amplitudes(r0_A, D_A, cov_A)
        return {
            "amps": {"a": float(amps[0])},
            "chi2_A": chi2_A,
            "jitter_used": jitter_used,
            "d1": d1,
            "d2": d2,
            "r0": r0,
        }

    if model_kind == "rewrite_stub":
        D_A = np.column_stack([d1[idxA], d2[idxA]])
        amps, chi2_A, jitter_used = _fit_linear_amplitudes(r0_A, D_A, cov_A)
        return {
            "amps": {"a": float(amps[0]), "b": float(amps[1])},
            "chi2_A": chi2_A,
            "jitter_used": jitter_used,
            "d1": d1,
            "d2": d2,
            "r0": r0,
        }

    raise ValueError("model_kind must be 'lcdm_stub' or 'rewrite_stub'.")


def eval_block_chi2(y: np.ndarray, cov: np.ndarray, idx: np.ndarray, residual_full: np.ndarray) -> float:
    """Compute chi2 on a block for a residual vector."""
    idx = validated_indices(idx, len(y))
    cov_block = cov[np.ix_(idx, idx)]
    resid_block = residual_full[idx]
    cho, _ = _cho_with_jitter(cov_block)
    return float(resid_block @ cho_solve(cho, resid_block))


def ppd_fit_predict(
    y: np.ndarray,
    cov: np.ndarray,
    idxA: np.ndarray,
    idxB: np.ndarray,
    model_kind: str,
    *,
    r0_amp: float = 0.1,
) -> Dict[str, object]:
    """Fit on A, score the prescribed injection on B (not survey prediction).

    y is used only for its length: the residual r0 is prescribed from covariance
    and metadata. The two-template model can cancel it exactly by construction.
    """
    idxA = validated_indices(idxA, len(y))
    idxB = validated_indices(idxB, len(y))
    if np.intersect1d(idxA, idxB).size:
        raise ValueError("training and held-out indices must be disjoint")
    fit = fit_amplitudes_on_block(y, cov, idxA, model_kind, r0_amp=r0_amp)
    d1 = fit["d1"]
    d2 = fit["d2"]
    r0 = fit["r0"]
    amps = fit["amps"]

    if model_kind == "lcdm_stub":
        a = amps["a"]
        resid_full = -(r0 + a * d1)
    else:
        a = amps["a"]
        b = amps["b"]
        resid_full = -(r0 + a * d1 + b * d2)

    chi2_A = eval_block_chi2(y, cov, idxA, resid_full)
    chi2_B = eval_block_chi2(y, cov, idxB, resid_full)
    rmse_B = float(np.sqrt(np.mean(resid_full[idxB] ** 2)))

    return {
        "audit_kind": "injected_residual_protocol",
        "independent_prediction": False,
        "amps": amps,
        "chi2_A": chi2_A,
        "chi2_B": chi2_B,
        "rmse_B": rmse_B,
        "jitter_used": fit["jitter_used"],
    }
