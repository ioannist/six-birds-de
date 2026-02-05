#!/usr/bin/env python3
"""Compare rewrite-background vs LCDM on SN and SN+BAO."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from sixbirds_cosmo import manifest
from sixbirds_cosmo.data_registry import fetch
from sixbirds_cosmo.cosmo_background import (
    distance_modulus,
    transverse_comoving_distance,
)
from sixbirds_cosmo.datasets.des_sn5yr import (
    chi2_profiled_over_M,
    extract_distances_covmat,
    load_des_sn5yr_covariance,
    load_des_sn5yr_hubble_diagram,
    prepare_cov_solve,
)
from sixbirds_cosmo.datasets.des_y6_bao import (
    load_alpha_likelihood_curve,
    load_bao_fiducial_info,
)
from sixbirds_cosmo.infer.fit import fit_map
from sixbirds_cosmo.likelihoods.base import Dataset
from sixbirds_cosmo.rewrite_background import (
    distance_modulus_rewrite,
    transverse_comoving_distance_rewrite,
)
from sixbirds_cosmo.reporting import write_run_markdown_stub


def combine_curves(curves: list[dict]) -> dict:
    grids = [curve["alpha_grid"] for curve in curves]
    union = np.unique(np.concatenate(grids))
    sum_delta = np.zeros_like(union, dtype=float)
    for curve in curves:
        delta = np.interp(union, curve["alpha_grid"], curve["delta_chi2_grid"])
        sum_delta += delta
    sum_delta = sum_delta - np.min(sum_delta)
    return {"alpha_grid": union, "delta_chi2_grid": sum_delta}


def estimate_sigma(alpha_grid: np.ndarray, delta_chi2: np.ndarray) -> float:
    idx_min = int(np.argmin(delta_chi2))
    left = None
    right = None
    for i in range(idx_min, 0, -1):
        if delta_chi2[i] <= 1.0 and delta_chi2[i - 1] > 1.0:
            x0, x1 = alpha_grid[i - 1], alpha_grid[i]
            y0, y1 = delta_chi2[i - 1], delta_chi2[i]
            left = x0 + (1.0 - y0) * (x1 - x0) / (y1 - y0)
            break
    for i in range(idx_min, len(alpha_grid) - 1):
        if delta_chi2[i] <= 1.0 and delta_chi2[i + 1] > 1.0:
            x0, x1 = alpha_grid[i], alpha_grid[i + 1]
            y0, y1 = delta_chi2[i], delta_chi2[i + 1]
            right = x0 + (1.0 - y0) * (x1 - x0) / (y1 - y0)
            break
    if left is None or right is None:
        return float("nan")
    return 0.5 * (right - left)


def hit_bounds(theta: list[float], bounds: list[tuple[float, float]], tol: float = 1e-6) -> list[bool]:
    hits = []
    for val, (lo, hi) in zip(theta, bounds):
        hits.append(abs(val - lo) <= tol or abs(val - hi) <= tol)
    return hits


def chi2_sn_lcdm(z: np.ndarray, mu: np.ndarray, cho, om: float, ol: float, H0: float) -> tuple[float, float]:
    mu_theory = distance_modulus(z, H0, om, ol, method="trapz")
    resid0 = mu - np.asarray(mu_theory, dtype=float)
    chi2, M_hat = chi2_profiled_over_M(resid0, cho)
    return chi2, M_hat


def chi2_sn_rewrite(
    z: np.ndarray, mu: np.ndarray, cho, om: float, A: float, m: float, H0: float
) -> tuple[float, float]:
    mu_theory = distance_modulus_rewrite(z, H0, om, A, m, method="trapz")
    resid0 = mu - np.asarray(mu_theory, dtype=float)
    chi2, M_hat = chi2_profiled_over_M(resid0, cho)
    return chi2, M_hat


def alpha_pred_lcdm(z_eff: float, H0_fid: float, om: float, ol: float, dm_fid: float) -> float:
    dm = transverse_comoving_distance(z_eff, H0_fid, om, ol, method="trapz")
    return float(dm / dm_fid)


def alpha_pred_rewrite(z_eff: float, H0_fid: float, om: float, A: float, m: float, dm_fid: float) -> float:
    dm = transverse_comoving_distance_rewrite(z_eff, H0_fid, om, A, m, method="trapz")
    return float(dm / dm_fid)


def chi2_bao(alpha_pred: float, alpha_grid: np.ndarray, delta_chi2: np.ndarray) -> tuple[float, bool]:
    alpha_min = float(alpha_grid.min())
    alpha_max = float(alpha_grid.max())
    clamped = False
    if alpha_pred < alpha_min:
        alpha_eval = alpha_min
        clamped = True
    elif alpha_pred > alpha_max:
        alpha_eval = alpha_max
        clamped = True
    else:
        alpha_eval = alpha_pred
    chi2 = float(np.interp(alpha_eval, alpha_grid, delta_chi2))
    return chi2, clamped


def grid_contour(ax, X, Y, chi2_grid, levels):
    chi2_min = np.nanmin(chi2_grid)
    delta = chi2_grid - chi2_min
    ax.contour(X, Y, delta, levels=levels)
    ax.set_xlabel("om")


def bin_residuals(z: np.ndarray, resid: np.ndarray, n_bins: int = 20) -> pd.DataFrame:
    order = np.argsort(z)
    z_sorted = z[order]
    resid_sorted = resid[order]
    bins = np.array_split(np.arange(len(z_sorted)), n_bins)
    rows = []
    for idx in bins:
        if len(idx) == 0:
            continue
        z_bin = z_sorted[idx]
        r_bin = resid_sorted[idx]
        rows.append(
            {
                "z_mean": float(np.mean(z_bin)),
                "resid_mean": float(np.mean(r_bin)),
                "resid_stderr": float(np.std(r_bin, ddof=1) / np.sqrt(len(r_bin)))
                if len(r_bin) > 1
                else 0.0,
                "n_bin": int(len(r_bin)),
            }
        )
    return pd.DataFrame(rows)


def fit_minimize(objective, theta0, bounds, n_data: int, k: int) -> dict:
    result = minimize(objective, np.asarray(theta0, dtype=float), method="L-BFGS-B", bounds=bounds)
    theta_hat = np.asarray(result.x, dtype=float)
    chi2 = float(result.fun)
    loglike_hat = -0.5 * chi2
    return {
        "theta_hat": theta_hat.tolist(),
        "loglike_hat": loglike_hat,
        "success": bool(result.success),
        "message": str(result.message),
        "nfev": int(result.nfev) if result.nfev is not None else None,
        "nit": int(result.nit) if result.nit is not None else None,
        "k": k,
        "n_data": n_data,
        "chi2_hat": chi2,
        "aic": 2.0 * k - 2.0 * loglike_hat,
        "bic": k * float(np.log(n_data)) - 2.0 * loglike_hat,
    }


def main() -> None:
    seed = 123
    run_dir = manifest.create_run_dir("rewrite_background_vs_lcdm", seed=seed)

    raw_sn = Path(fetch("des_sn5yr_distances"))
    raw_bao = Path(fetch("des_y6_bao_release"))

    extract_distances_covmat(raw_sn)
    cov_sys_tmp = load_des_sn5yr_covariance(raw_sn, kind="stat+sys", mu_err=None)
    hd = load_des_sn5yr_hubble_diagram(raw_sn, n_cov=cov_sys_tmp["N"])
    mu_err = hd["mu_err"]
    cov_sys = load_des_sn5yr_covariance(raw_sn, kind="stat+sys", mu_err=mu_err)

    z_sn = hd["z"]
    mu_sn = hd["mu"]
    cov = cov_sys["cov"]
    cho_info = prepare_cov_solve(cov)
    cho = cho_info["cho"]

    fid_info = load_bao_fiducial_info(raw_bao)
    z_eff = fid_info["z_eff"]
    curves = {
        "acf": load_alpha_likelihood_curve(raw_bao, "acf"),
        "aps": load_alpha_likelihood_curve(raw_bao, "aps"),
        "pcf": load_alpha_likelihood_curve(raw_bao, "pcf"),
    }
    combined = combine_curves([curves["acf"], curves["aps"], curves["pcf"]])
    alpha_grid = combined["alpha_grid"]
    delta_combined = combined["delta_chi2_grid"]
    alpha_hat = float(alpha_grid[np.argmin(delta_combined)])
    sigma_est = estimate_sigma(alpha_grid, delta_combined)

    H0_fid = 67.4
    om_fid = 0.315
    ol_fid = 0.685
    dm_fid = transverse_comoving_distance(z_eff, H0_fid, om_fid, ol_fid, method="trapz")

    # Fit functions
    @dataclass
    class SNOnlyLCDM(Dataset):
        name: str = "sn_lcdm"
        n_data: int = len(z_sn)
        param_names: list[str] = field(default_factory=lambda: ["om", "ol"])
        loglike_is_minus_half_chi2: bool = True
        last_M_hat: float | None = None

        def loglike(self, theta):
            om, ol = float(theta[0]), float(theta[1])
            try:
                chi2, M_hat = chi2_sn_lcdm(z_sn, mu_sn, cho, om, ol, H0=70.0)
            except Exception:
                return -0.5 * 1e12
            if not np.isfinite(chi2):
                return -0.5 * 1e12
            self.last_M_hat = M_hat
            return -0.5 * chi2

    @dataclass
    class SNOnlyRewrite(Dataset):
        name: str = "sn_rw"
        n_data: int = len(z_sn)
        param_names: list[str] = field(default_factory=lambda: ["om", "A", "m"])
        loglike_is_minus_half_chi2: bool = True
        last_M_hat: float | None = None

        def loglike(self, theta):
            om, A, m = float(theta[0]), float(theta[1]), float(theta[2])
            try:
                chi2, M_hat = chi2_sn_rewrite(z_sn, mu_sn, cho, om, A, m, H0=70.0)
            except Exception:
                return -0.5 * 1e12
            if not np.isfinite(chi2):
                return -0.5 * 1e12
            self.last_M_hat = M_hat
            return -0.5 * chi2

    @dataclass
    class SNBAOLCDM(Dataset):
        name: str = "snbao_lcdm"
        n_data: int = len(z_sn) + 1
        param_names: list[str] = field(default_factory=lambda: ["om", "ol"])
        loglike_is_minus_half_chi2: bool = True
        last_M_hat: float | None = None
        last_alpha_pred: float | None = None
        alpha_clamped: bool = False

        def loglike(self, theta):
            om, ol = float(theta[0]), float(theta[1])
            try:
                chi2_sn, M_hat = chi2_sn_lcdm(z_sn, mu_sn, cho, om, ol, H0=70.0)
                alpha = alpha_pred_lcdm(z_eff, H0_fid, om, ol, dm_fid)
                chi2_bao, clamped = chi2_bao(alpha, alpha_grid, delta_combined)
            except Exception:
                return -0.5 * 1e12
            if not (np.isfinite(chi2_sn) and np.isfinite(alpha) and np.isfinite(chi2_bao)):
                return -0.5 * 1e12
            self.last_M_hat = M_hat
            self.last_alpha_pred = alpha
            self.alpha_clamped = clamped
            return -0.5 * (chi2_sn + chi2_bao)

    @dataclass
    class SNBAORewrite(Dataset):
        name: str = "snbao_rw"
        n_data: int = len(z_sn) + 1
        param_names: list[str] = field(default_factory=lambda: ["om", "A", "m"])
        loglike_is_minus_half_chi2: bool = True
        last_M_hat: float | None = None
        last_alpha_pred: float | None = None
        alpha_clamped: bool = False

        def loglike(self, theta):
            om, A, m = float(theta[0]), float(theta[1]), float(theta[2])
            try:
                chi2_sn, M_hat = chi2_sn_rewrite(z_sn, mu_sn, cho, om, A, m, H0=70.0)
                alpha = alpha_pred_rewrite(z_eff, H0_fid, om, A, m, dm_fid)
                chi2_bao, clamped = chi2_bao(alpha, alpha_grid, delta_combined)
            except Exception:
                return -0.5 * 1e12
            if not (np.isfinite(chi2_sn) and np.isfinite(alpha) and np.isfinite(chi2_bao)):
                return -0.5 * 1e12
            self.last_M_hat = M_hat
            self.last_alpha_pred = alpha
            self.alpha_clamped = clamped
            return -0.5 * (chi2_sn + chi2_bao)

    sn_lcdm = SNOnlyLCDM()
    sn_rw = SNOnlyRewrite()
    snbao_lcdm = SNBAOLCDM()
    snbao_rw = SNBAORewrite()

    fit_lcdm_sn = fit_map(sn_lcdm, np.array([0.3, 0.7]), bounds=[(0.05, 1.5), (-0.5, 2.0)])
    fit_rw_sn = fit_map(sn_rw, np.array([0.3, 0.7, 0.0]), bounds=[(0.05, 1.5), (0.0, 2.0), (0.0, 6.0)])
    fit_lcdm_snbao = fit_map(snbao_lcdm, np.array([0.3, 0.7]), bounds=[(0.05, 1.5), (-0.5, 2.0)])
    fit_rw_snbao = fit_map(snbao_rw, np.array([0.3, 0.7, 0.0]), bounds=[(0.05, 1.5), (0.0, 2.0), (0.0, 6.0)])

    def bad_fit(fit: dict) -> bool:
        loglike = fit.get("loglike_hat")
        if loglike is None or not np.isfinite(loglike):
            return True
        chi2 = -2.0 * loglike
        return chi2 > 1e10 or not fit.get("success", False)

    if bad_fit(fit_lcdm_snbao):
        def objective(theta):
            om, ol = float(theta[0]), float(theta[1])
            try:
                chi2_sn, _ = chi2_sn_lcdm(z_sn, mu_sn, cho, om, ol, H0=70.0)
                alpha = alpha_pred_lcdm(z_eff, H0_fid, om, ol, dm_fid)
                chi2_b, _ = chi2_bao(alpha, alpha_grid, delta_combined)
            except Exception:
                return 1e12
            if not (np.isfinite(chi2_sn) and np.isfinite(alpha) and np.isfinite(chi2_b)):
                return 1e12
            return chi2_sn + chi2_b

        fit_lcdm_snbao = fit_minimize(
            objective, [0.3, 0.7], bounds=[(0.05, 1.5), (-0.5, 2.0)], n_data=len(z_sn) + 1, k=2
        )

    if bad_fit(fit_rw_snbao):
        def objective(theta):
            om, A, m = float(theta[0]), float(theta[1]), float(theta[2])
            try:
                chi2_sn, _ = chi2_sn_rewrite(z_sn, mu_sn, cho, om, A, m, H0=70.0)
                alpha = alpha_pred_rewrite(z_eff, H0_fid, om, A, m, dm_fid)
                chi2_b, _ = chi2_bao(alpha, alpha_grid, delta_combined)
            except Exception:
                return 1e12
            if not (np.isfinite(chi2_sn) and np.isfinite(alpha) and np.isfinite(chi2_b)):
                return 1e12
            return chi2_sn + chi2_b

        fit_rw_snbao = fit_minimize(
            objective, [0.3, 0.7, 0.0], bounds=[(0.05, 1.5), (0.0, 2.0), (0.0, 6.0)], n_data=len(z_sn) + 1, k=3
        )

    # compute chi2 and AIC/BIC
    def fit_stats(fit, k_total, n_data):
        chi2 = -2.0 * fit["loglike_hat"]
        aic = chi2 + 2 * k_total
        bic = chi2 + k_total * np.log(n_data)
        return chi2, aic, bic

    chi2_lcdm_sn, aic_lcdm_sn, bic_lcdm_sn = fit_stats(fit_lcdm_sn, 3, len(z_sn))
    chi2_rw_sn, aic_rw_sn, bic_rw_sn = fit_stats(fit_rw_sn, 4, len(z_sn))
    chi2_lcdm_snbao, aic_lcdm_snbao, bic_lcdm_snbao = fit_stats(fit_lcdm_snbao, 3, len(z_sn) + 1)
    chi2_rw_snbao, aic_rw_snbao, bic_rw_snbao = fit_stats(fit_rw_snbao, 4, len(z_sn) + 1)

    delta_aic_sn = aic_rw_sn - aic_lcdm_sn
    delta_aic_snbao = aic_rw_snbao - aic_lcdm_snbao

    theta_lcdm_sn = fit_lcdm_sn["theta_hat"]
    theta_rw_sn = fit_rw_sn["theta_hat"]
    theta_lcdm_snbao = fit_lcdm_snbao["theta_hat"]
    theta_rw_snbao = fit_rw_snbao["theta_hat"]

    _, M_hat_lcdm_sn = chi2_sn_lcdm(z_sn, mu_sn, cho, theta_lcdm_sn[0], theta_lcdm_sn[1], H0=70.0)
    _, M_hat_rw_sn = chi2_sn_rewrite(z_sn, mu_sn, cho, theta_rw_sn[0], theta_rw_sn[1], theta_rw_sn[2], H0=70.0)
    _, M_hat_lcdm_snbao = chi2_sn_lcdm(
        z_sn, mu_sn, cho, theta_lcdm_snbao[0], theta_lcdm_snbao[1], H0=70.0
    )
    _, M_hat_rw_snbao = chi2_sn_rewrite(
        z_sn, mu_sn, cho, theta_rw_snbao[0], theta_rw_snbao[1], theta_rw_snbao[2], H0=70.0
    )

    alpha_pred_lcdm_snbao = alpha_pred_lcdm(z_eff, H0_fid, theta_lcdm_snbao[0], theta_lcdm_snbao[1], dm_fid)
    alpha_pred_rw_snbao = alpha_pred_rewrite(
        z_eff, H0_fid, theta_rw_snbao[0], theta_rw_snbao[1], theta_rw_snbao[2], dm_fid
    )
    _, alpha_clamped_lcdm = chi2_bao(alpha_pred_lcdm_snbao, alpha_grid, delta_combined)
    _, alpha_clamped_rw = chi2_bao(alpha_pred_rw_snbao, alpha_grid, delta_combined)

    A_hat_sn = theta_rw_sn[1]
    A_hat_snbao = theta_rw_snbao[1]
    if not (abs(A_hat_sn) > 1e-4 or abs(A_hat_snbao) > 1e-4):
        raise AssertionError("Rewrite amplitude is too close to zero in both fits.")
    if min(delta_aic_sn, delta_aic_snbao) > 10.0:
        raise AssertionError("Rewrite AIC is not within 10 of LCDM for any dataset.")

    # Contours
    om_grid = np.linspace(0.05, 1.5, 40)
    ol_grid = np.linspace(-0.5, 2.0, 40)
    OM, OL = np.meshgrid(om_grid, ol_grid)
    chi2_sn_lcdm_grid = np.full_like(OM, np.nan)
    chi2_snbao_lcdm_grid = np.full_like(OM, np.nan)
    for i in range(OM.shape[0]):
        for j in range(OM.shape[1]):
            om, ol = float(OM[i, j]), float(OL[i, j])
            try:
                chi2_sn_val, _ = chi2_sn_lcdm(z_sn, mu_sn, cho, om, ol, H0=70.0)
                alpha = alpha_pred_lcdm(z_eff, H0_fid, om, ol, dm_fid)
                chi2_bao_val, _ = chi2_bao(alpha, alpha_grid, delta_combined)
                chi2_sn_lcdm_grid[i, j] = chi2_sn_val
                chi2_snbao_lcdm_grid[i, j] = chi2_sn_val + chi2_bao_val
            except Exception:
                continue

    A_grid = np.linspace(0.0, 2.0, 40)
    OM2, AA = np.meshgrid(om_grid, A_grid)
    m_sn = fit_rw_sn["theta_hat"][2]
    m_snbao = fit_rw_snbao["theta_hat"][2]

    chi2_sn_rw_grid = np.full_like(OM2, np.nan)
    chi2_snbao_rw_grid = np.full_like(OM2, np.nan)
    for i in range(OM2.shape[0]):
        for j in range(OM2.shape[1]):
            om, A = float(OM2[i, j]), float(AA[i, j])
            try:
                chi2_sn_val, _ = chi2_sn_rewrite(z_sn, mu_sn, cho, om, A, m_sn, H0=70.0)
                chi2_sn_rw_grid[i, j] = chi2_sn_val
            except Exception:
                continue
            try:
                chi2_sn_val_bao, _ = chi2_sn_rewrite(z_sn, mu_sn, cho, om, A, m_snbao, H0=70.0)
                alpha = alpha_pred_rewrite(z_eff, H0_fid, om, A, m_snbao, dm_fid)
                chi2_bao_val, _ = chi2_bao(alpha, alpha_grid, delta_combined)
                chi2_snbao_rw_grid[i, j] = chi2_sn_val_bao + chi2_bao_val
            except Exception:
                continue

    # contour plots
    fig1, axes = plt.subplots(1, 2, figsize=(10, 4))
    grid_contour(axes[0], OM, OL, chi2_sn_lcdm_grid, levels=[2.30, 6.17])
    axes[0].set_ylabel("ol")
    axes[0].set_title("LCDM SN-only")
    grid_contour(axes[1], OM2, AA, chi2_sn_rw_grid, levels=[2.30, 6.17])
    axes[1].set_ylabel("A")
    axes[1].set_title(f"Rewrite SN-only (m={m_sn:.3f})")
    plot_sn_only = run_dir / "contours_sn_only.png"
    manifest.save_fig(fig1, plot_sn_only)
    plt.close(fig1)

    fig2, axes2 = plt.subplots(1, 2, figsize=(10, 4))
    grid_contour(axes2[0], OM, OL, chi2_snbao_lcdm_grid, levels=[2.30, 6.17])
    axes2[0].set_ylabel("ol")
    axes2[0].set_title("LCDM SN+BAO")
    grid_contour(axes2[1], OM2, AA, chi2_snbao_rw_grid, levels=[2.30, 6.17])
    axes2[1].set_ylabel("A")
    axes2[1].set_title(f"Rewrite SN+BAO (m={m_snbao:.3f})")
    plot_sn_bao = run_dir / "contours_sn_bao.png"
    manifest.save_fig(fig2, plot_sn_bao)
    plt.close(fig2)

    # mu residuals compare
    mu_lcdm_sn = distance_modulus(z_sn, 70.0, fit_lcdm_sn["theta_hat"][0], fit_lcdm_sn["theta_hat"][1])
    mu_lcdm_snbao = distance_modulus(z_sn, 70.0, fit_lcdm_snbao["theta_hat"][0], fit_lcdm_snbao["theta_hat"][1])
    mu_rw_sn = distance_modulus_rewrite(
        z_sn, 70.0, fit_rw_sn["theta_hat"][0], fit_rw_sn["theta_hat"][1], fit_rw_sn["theta_hat"][2]
    )
    mu_rw_snbao = distance_modulus_rewrite(
        z_sn, 70.0, fit_rw_snbao["theta_hat"][0], fit_rw_snbao["theta_hat"][1], fit_rw_snbao["theta_hat"][2]
    )

    resid_lcdm_sn = mu_sn - (mu_lcdm_sn + M_hat_lcdm_sn)
    resid_rw_sn = mu_sn - (mu_rw_sn + M_hat_rw_sn)
    resid_lcdm_snbao = mu_sn - (mu_lcdm_snbao + M_hat_lcdm_snbao)
    resid_rw_snbao = mu_sn - (mu_rw_snbao + M_hat_rw_snbao)

    df_lcdm_sn = bin_residuals(z_sn, resid_lcdm_sn)
    df_rw_sn = bin_residuals(z_sn, resid_rw_sn)
    df_lcdm_snbao = bin_residuals(z_sn, resid_lcdm_snbao)
    df_rw_snbao = bin_residuals(z_sn, resid_rw_snbao)

    fig3, ax3 = plt.subplots()
    ax3.errorbar(df_lcdm_sn["z_mean"], df_lcdm_sn["resid_mean"], yerr=df_lcdm_sn["resid_stderr"], fmt="o", label="LCDM SN")
    ax3.errorbar(df_rw_sn["z_mean"], df_rw_sn["resid_mean"], yerr=df_rw_sn["resid_stderr"], fmt="s", label="Rewrite SN")
    ax3.errorbar(df_lcdm_snbao["z_mean"], df_lcdm_snbao["resid_mean"], yerr=df_lcdm_snbao["resid_stderr"], fmt="^", label="LCDM SN+BAO")
    ax3.errorbar(df_rw_snbao["z_mean"], df_rw_snbao["resid_mean"], yerr=df_rw_snbao["resid_stderr"], fmt="v", label="Rewrite SN+BAO")
    ax3.axhline(0.0, color="black", linewidth=1.0)
    ax3.set_xlabel("zHD")
    ax3.set_ylabel("mu residual")
    ax3.set_title("SN residuals comparison")
    ax3.legend()

    plot_mu = run_dir / "mu_residuals_compare.png"
    manifest.save_fig(fig3, plot_mu)
    plt.close(fig3)

    # BAO alpha curve with predictions
    fig4, ax4 = plt.subplots()
    ax4.plot(alpha_grid, delta_combined, label="combined")
    ax4.axvline(alpha_hat, color="black", linestyle="--", label="alpha_hat")
    ax4.axvline(alpha_pred_lcdm_snbao, color="blue", linestyle=":", label="lcdm pred")
    ax4.axvline(alpha_pred_rw_snbao, color="red", linestyle="-.", label="rewrite pred")
    ax4.set_xlabel("alpha")
    ax4.set_ylabel("delta chi2")
    ax4.set_title("BAO alpha with predictions")
    ax4.legend()

    plot_alpha = run_dir / "bao_alpha_curve_with_preds.png"
    manifest.save_fig(fig4, plot_alpha)
    plt.close(fig4)

    # Summary table
    summary_rows = [
        {
            "dataset": "SN",
            "model": "lcdm",
            "k_total": 3,
            "chi2": chi2_lcdm_sn,
            "aic_total": aic_lcdm_sn,
            "bic_total": bic_lcdm_sn,
            "om": fit_lcdm_sn["theta_hat"][0],
            "ol": fit_lcdm_sn["theta_hat"][1],
            "M_hat": M_hat_lcdm_sn,
        },
        {
            "dataset": "SN",
            "model": "rewrite",
            "k_total": 4,
            "chi2": chi2_rw_sn,
            "aic_total": aic_rw_sn,
            "bic_total": bic_rw_sn,
            "om": fit_rw_sn["theta_hat"][0],
            "A": fit_rw_sn["theta_hat"][1],
            "m": fit_rw_sn["theta_hat"][2],
            "M_hat": M_hat_rw_sn,
        },
        {
            "dataset": "SN+BAO",
            "model": "lcdm",
            "k_total": 3,
            "chi2": chi2_lcdm_snbao,
            "aic_total": aic_lcdm_snbao,
            "bic_total": bic_lcdm_snbao,
            "om": fit_lcdm_snbao["theta_hat"][0],
            "ol": fit_lcdm_snbao["theta_hat"][1],
            "M_hat": M_hat_lcdm_snbao,
            "alpha_pred": alpha_pred_lcdm_snbao,
        },
        {
            "dataset": "SN+BAO",
            "model": "rewrite",
            "k_total": 4,
            "chi2": chi2_rw_snbao,
            "aic_total": aic_rw_snbao,
            "bic_total": bic_rw_snbao,
            "om": fit_rw_snbao["theta_hat"][0],
            "A": fit_rw_snbao["theta_hat"][1],
            "m": fit_rw_snbao["theta_hat"][2],
            "M_hat": M_hat_rw_snbao,
            "alpha_pred": alpha_pred_rw_snbao,
        },
    ]

    df_summary = pd.DataFrame(summary_rows)
    summary_path = run_dir / "summary.csv"
    manifest.save_table(df_summary, summary_path)

    metrics = {
        "alpha_hat": alpha_hat,
        "alpha_sigma_est": sigma_est,
        "delta_aic_sn": delta_aic_sn,
        "delta_aic_snbao": delta_aic_snbao,
        "A_hat_sn": fit_rw_sn["theta_hat"][1],
        "m_hat_sn": fit_rw_sn["theta_hat"][2],
        "A_hat_snbao": fit_rw_snbao["theta_hat"][1],
        "m_hat_snbao": fit_rw_snbao["theta_hat"][2],
        "lcdm_sn_bounds_hit": hit_bounds(fit_lcdm_sn["theta_hat"], [(0.05, 1.5), (-0.5, 2.0)]),
        "rw_sn_bounds_hit": hit_bounds(fit_rw_sn["theta_hat"], [(0.05, 1.5), (0.0, 2.0), (0.0, 6.0)]),
        "lcdm_snbao_bounds_hit": hit_bounds(fit_lcdm_snbao["theta_hat"], [(0.05, 1.5), (-0.5, 2.0)]),
        "rw_snbao_bounds_hit": hit_bounds(fit_rw_snbao["theta_hat"], [(0.05, 1.5), (0.0, 2.0), (0.0, 6.0)]),
        "alpha_pred_lcdm_snbao": alpha_pred_lcdm_snbao,
        "alpha_pred_rw_snbao": alpha_pred_rw_snbao,
        "alpha_clamped_lcdm_snbao": alpha_clamped_lcdm,
        "alpha_clamped_rw_snbao": alpha_clamped_rw,
    }

    config = {
        "seed": seed,
        "bounds_lcdm": [(0.05, 1.5), (-0.5, 2.0)],
        "bounds_rewrite": [(0.05, 1.5), (0.0, 2.0), (0.0, 6.0)],
        "fid_params": {"om_fid": om_fid, "ol_fid": ol_fid, "H0_fid": H0_fid, "z_eff": z_eff},
    }

    manifest.write_config(run_dir, config)
    manifest.write_metrics(run_dir, metrics)
    manifest.write_provenance(run_dir, "rewrite_background_vs_lcdm", seed)
    write_run_markdown_stub(run_dir)

    print(json.dumps({"run_dir": str(run_dir)}, indent=2))


if __name__ == "__main__":
    main()
