#!/usr/bin/env python3
"""PPD-style audit using SN (train) and BAO alpha (test), and vice versa."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sixbirds_cosmo import manifest
from sixbirds_cosmo.data_registry import fetch
from sixbirds_cosmo.datasets.des_sn5yr import (
    DESSN5YRDistanceDataset,
    extract_distances_covmat,
    load_des_sn5yr_covariance,
    load_des_sn5yr_hubble_diagram,
)
from sixbirds_cosmo.datasets.des_y6_bao import (
    DESY6BAOAlphaDataset,
    load_alpha_likelihood_curve,
    load_bao_fiducial_info,
)
from sixbirds_cosmo.infer.fit import fit_map
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
        raise ValueError("Could not estimate sigma from delta_chi2 curve.")
    return 0.5 * (right - left)


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


def main() -> None:
    seed = 123
    run_dir = manifest.create_run_dir("ppd_background", seed=seed)

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
    sn_sigma_median = float(np.median(np.sqrt(np.diag(cov))))

    sn_dataset = DESSN5YRDistanceDataset(
        name="des_sn5yr",
        z=z_sn,
        mu=mu_sn,
        cov=cov,
        cov_kind="stat+sys",
        model_kind="flat_lcdm",
        n_data=len(z_sn),
        param_names=["om"],
    )

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

    bao_dataset = DESY6BAOAlphaDataset(
        name="des_y6_bao_alpha",
        n_data=1,
        param_names=["om"],
        alpha_grid=alpha_grid,
        delta_chi2_grid=delta_combined,
        z_eff=z_eff,
        model_kind="flat_lcdm",
    )

    fit_sn = fit_map(sn_dataset, np.array([0.3]), bounds=[(0.05, 0.8)])
    sn_dataset.loglike(np.array(fit_sn["theta_hat"]))
    om_hat_sn = fit_sn["theta_hat"][0]
    chi2_hat_sn = fit_sn["chi2_hat"]
    chi2_over_n_sn = chi2_hat_sn / len(z_sn)
    M_hat_sn = sn_dataset.last_M_hat

    fit_bao = fit_map(bao_dataset, np.array([0.3]), bounds=[(0.05, 1.5)])
    om_hat_bao = fit_bao["theta_hat"][0]
    chi2_hat_bao = fit_bao["chi2_hat"]
    alpha_pred_bao = bao_dataset.alpha_pred(om_hat_bao)

    alpha_pred_from_sn = bao_dataset.alpha_pred(om_hat_sn)
    delta_chi2_pred_sn = float(np.interp(alpha_pred_from_sn, alpha_grid, delta_combined))
    bias_alpha = float(alpha_pred_from_sn - alpha_hat)
    rmse_alpha = abs(bias_alpha)
    zscore_alpha = bias_alpha / sigma_est

    sn_dataset.loglike(np.array([om_hat_bao]))
    M_hat_bao = sn_dataset.last_M_hat
    chi2_pred_sn_from_bao = sn_dataset.loglike(np.array([om_hat_bao])) * -2.0
    chi2_over_n_pred = chi2_pred_sn_from_bao / len(z_sn)

    from sixbirds_cosmo.cosmo_background import luminosity_distance

    mu_theory_bao = 5.0 * np.log10(
        luminosity_distance(z_sn, sn_dataset.H0_fixed, om_hat_bao, 1.0 - om_hat_bao)
    ) + 25.0
    resid_bao = mu_sn - (mu_theory_bao + M_hat_bao)
    bias_mu_bao = float(np.mean(resid_bao))
    rmse_mu_bao = float(np.sqrt(np.mean(resid_bao**2)))
    rmse_mu_weighted = float(np.sqrt(chi2_pred_sn_from_bao / len(z_sn)))

    mu_theory_sn = 5.0 * np.log10(
        luminosity_distance(z_sn, sn_dataset.H0_fixed, om_hat_sn, 1.0 - om_hat_sn)
    ) + 25.0
    resid_sn = mu_sn - (mu_theory_sn + M_hat_sn)

    df_sn = bin_residuals(z_sn, resid_sn, n_bins=20)
    df_bao = bin_residuals(z_sn, resid_bao, n_bins=20)
    df_binned = df_sn.rename(columns={"resid_mean": "resid_sn_mean", "resid_stderr": "resid_sn_stderr"})
    df_binned["resid_bao_mean"] = df_bao["resid_mean"].values
    df_binned["resid_bao_stderr"] = df_bao["resid_stderr"].values

    fig1, ax1 = plt.subplots()
    ax1.plot(alpha_grid, delta_combined, label="combined")
    ax1.axvline(alpha_hat, color="black", linestyle="--", label="alpha_hat")
    ax1.axvline(alpha_pred_from_sn, color="blue", linestyle=":", label="alpha_pred_sn")
    if alpha_pred_bao is not None:
        ax1.axvline(alpha_pred_bao, color="red", linestyle="-.", label="alpha_pred_bao")
    ax1.set_xlabel("alpha")
    ax1.set_ylabel("delta chi2")
    ax1.set_title("PPD BAO alpha")
    ax1.legend()

    plot_alpha = run_dir / "ppd_bao_alpha.png"
    manifest.save_fig(fig1, plot_alpha)
    plt.close(fig1)

    fig2, ax2 = plt.subplots()
    ax2.errorbar(
        df_binned["z_mean"],
        df_binned["resid_sn_mean"],
        yerr=df_binned["resid_sn_stderr"],
        fmt="o",
        label="SN-fit",
    )
    ax2.errorbar(
        df_binned["z_mean"],
        df_binned["resid_bao_mean"],
        yerr=df_binned["resid_bao_stderr"],
        fmt="s",
        label="BAO-fit",
    )
    ax2.axhline(0.0, color="black", linewidth=1.0)
    ax2.set_xlabel("zHD")
    ax2.set_ylabel("mu residual")
    ax2.set_title("PPD SN residuals")
    ax2.legend()

    plot_sn = run_dir / "ppd_sn_residuals.png"
    manifest.save_fig(fig2, plot_sn)
    plt.close(fig2)

    summary_rows = [
        {
            "train_probe": "SN",
            "test_probe": "BAO",
            "model": "flat_lcdm",
            "om_hat_train": om_hat_sn,
            "test_metric_rmse": rmse_alpha,
            "test_metric_bias": bias_alpha,
            "alpha_pred": alpha_pred_from_sn,
            "alpha_hat": alpha_hat,
            "sigma_est": sigma_est,
            "zscore": zscore_alpha,
            "delta_chi2_at_alpha_pred": delta_chi2_pred_sn,
        },
        {
            "train_probe": "BAO",
            "test_probe": "SN",
            "model": "flat_lcdm",
            "om_hat_train": om_hat_bao,
            "test_metric_rmse": rmse_mu_bao,
            "test_metric_bias": bias_mu_bao,
            "chi2_pred": chi2_pred_sn_from_bao,
            "chi2_over_n": chi2_over_n_pred,
            "rmse_mu": rmse_mu_bao,
            "rmse_mu_weighted": rmse_mu_weighted,
        },
    ]

    df_summary = pd.DataFrame(summary_rows)
    summary_path = run_dir / "summary.csv"
    manifest.save_table(df_summary, summary_path)

    binned_path = run_dir / "sn_residuals_binned.csv"
    manifest.save_table(df_binned, binned_path)

    metrics = {
        "sn_n": len(z_sn),
        "sn_sigma_median": sn_sigma_median,
        "bao_alpha_hat": alpha_hat,
        "bao_sigma_est": sigma_est,
        "bao_z_eff": z_eff,
        "om_hat_sn": om_hat_sn,
        "chi2_hat_sn": chi2_hat_sn,
        "chi2_over_n_sn": chi2_over_n_sn,
        "M_hat_sn": M_hat_sn,
        "om_hat_bao": om_hat_bao,
        "chi2_hat_bao": chi2_hat_bao,
        "alpha_pred_bao": alpha_pred_bao,
        "alpha_pred_from_sn": alpha_pred_from_sn,
        "delta_chi2_at_alpha_pred_from_sn": delta_chi2_pred_sn,
        "bias_alpha_sn_to_bao": bias_alpha,
        "rmse_alpha_sn_to_bao": rmse_alpha,
        "zscore_alpha_sn_to_bao": zscore_alpha,
        "chi2_pred_sn_from_bao": chi2_pred_sn_from_bao,
        "chi2_over_n_pred_sn_from_bao": chi2_over_n_pred,
        "bias_mu_bao_to_sn": bias_mu_bao,
        "rmse_mu_bao_to_sn": rmse_mu_bao,
        "rmse_mu_weighted_bao_to_sn": rmse_mu_weighted,
        "curve_modes": {
            "acf": curves["acf"]["notes"],
            "aps": curves["aps"]["notes"],
            "pcf": curves["pcf"]["notes"],
        },
    }

    config = {
        "seed": seed,
        "sn_bounds": [0.05, 0.8],
        "bao_bounds": [0.05, 1.5],
        "z_eff": z_eff,
        "z_eff_source": fid_info.get("z_eff_source"),
    }

    manifest.write_config(run_dir, config)
    manifest.write_metrics(run_dir, metrics)
    manifest.write_provenance(run_dir, "ppd_background", seed)
    write_run_markdown_stub(run_dir)

    print(json.dumps({"run_dir": str(run_dir)}, indent=2))


if __name__ == "__main__":
    main()
