#!/usr/bin/env python3
"""Fit DES-SN5YR distance modulus data with flat LCDM and wCDM."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sixbirds_cosmo import manifest
from sixbirds_cosmo.data_registry import fetch
from sixbirds_cosmo.datasets.des_sn5yr import (
    DESSN5YRDistanceDataset,
    extract_distances_covmat,
    load_des_sn5yr_covariance,
    load_des_sn5yr_hubble_diagram,
    prepare_cov_solve,
)
from sixbirds_cosmo.infer.fit import fit_map
from sixbirds_cosmo.reporting import write_run_markdown_stub


def main() -> None:
    seed = 123
    run_dir = manifest.create_run_dir("fit_des_sn5yr", seed=seed)

    raw_dir = Path(fetch("des_sn5yr_distances"))
    extract_distances_covmat(raw_dir)

    cov_sys_tmp = load_des_sn5yr_covariance(raw_dir, kind="stat+sys", mu_err=None)
    hd = load_des_sn5yr_hubble_diagram(raw_dir, n_cov=cov_sys_tmp["N"])

    z = hd["z"]
    mu = hd["mu"]
    mu_err = hd["mu_err"]

    cov_stat = load_des_sn5yr_covariance(raw_dir, kind="stat", mu_err=mu_err)
    cov_sys = load_des_sn5yr_covariance(raw_dir, kind="stat+sys", mu_err=mu_err)

    cov = cov_sys["cov"]
    cho_info = prepare_cov_solve(cov)

    bounds_lcdm = [(0.05, 0.8)]
    bounds_wcdm = [(0.05, 0.8), (-2.5, -0.3)]

    lcdm = DESSN5YRDistanceDataset(
        name="des_sn5yr_lcdm",
        z=z,
        mu=mu,
        cov=cov,
        cov_kind="stat+sys",
        model_kind="flat_lcdm",
        n_data=len(z),
        param_names=["om"],
    )
    wcdm = DESSN5YRDistanceDataset(
        name="des_sn5yr_wcdm",
        z=z,
        mu=mu,
        cov=cov,
        cov_kind="stat+sys",
        model_kind="flat_wcdm",
        n_data=len(z),
        param_names=["om", "w"],
    )

    fit_lcdm = fit_map(lcdm, np.array([0.3]), bounds=bounds_lcdm)
    fit_wcdm = fit_map(wcdm, np.array([0.3, -1.0]), bounds=bounds_wcdm)

    # recompute M_hat using prepared cho for reporting
    lcdm.cho = cho_info["cho"]
    wcdm.cho = cho_info["cho"]
    lcdm.loglike(np.array(fit_lcdm["theta_hat"]))
    wcdm.loglike(np.array(fit_wcdm["theta_hat"]))

    n_data = len(z)
    if fit_lcdm["chi2_hat"] / n_data >= 1000:
        raise AssertionError("LCDM chi2/n too large; covariance decode likely wrong.")
    if fit_wcdm["chi2_hat"] / n_data >= 1000:
        raise AssertionError("wCDM chi2/n too large; covariance decode likely wrong.")

    k_lcdm = 1
    k_wcdm = 2
    k_total_lcdm = k_lcdm + 1
    k_total_wcdm = k_wcdm + 1

    aic_total_lcdm = 2 * k_total_lcdm - 2 * fit_lcdm["loglike_hat"]
    aic_total_wcdm = 2 * k_total_wcdm - 2 * fit_wcdm["loglike_hat"]
    bic_total_lcdm = k_total_lcdm * np.log(n_data) - 2 * fit_lcdm["loglike_hat"]
    bic_total_wcdm = k_total_wcdm * np.log(n_data) - 2 * fit_wcdm["loglike_hat"]

    sigma_median = float(np.median(np.sqrt(np.diag(cov))))

    metrics = {
        "hubble_diagram_file": hd["filename"],
        "hubble_diagram_columns": hd["columns"],
        "mu_err_col": hd["mu_err_col"],
        "cov_stat_file": cov_stat["filename"],
        "cov_stat_format": cov_stat["format_detected"],
        "cov_stat_matrix_kind": cov_stat.get("matrix_kind_detected"),
        "cov_stat_index_base": cov_stat.get("index_base_detected"),
        "cov_sys_file": cov_sys["filename"],
        "cov_sys_format": cov_sys["format_detected"],
        "cov_sys_matrix_kind": cov_sys.get("matrix_kind_detected"),
        "cov_sys_index_base": cov_sys.get("index_base_detected"),
        "n_data": n_data,
        "lcdm_om": fit_lcdm["theta_hat"][0],
        "lcdm_chi2": fit_lcdm["chi2_hat"],
        "lcdm_loglike": fit_lcdm["loglike_hat"],
        "lcdm_M_hat": lcdm.last_M_hat,
        "lcdm_aic_total": aic_total_lcdm,
        "lcdm_bic_total": bic_total_lcdm,
        "wcdm_om": fit_wcdm["theta_hat"][0],
        "wcdm_w": fit_wcdm["theta_hat"][1],
        "wcdm_chi2": fit_wcdm["chi2_hat"],
        "wcdm_loglike": fit_wcdm["loglike_hat"],
        "wcdm_M_hat": wcdm.last_M_hat,
        "wcdm_aic_total": aic_total_wcdm,
        "wcdm_bic_total": bic_total_wcdm,
        "delta_aic_total": aic_total_lcdm - aic_total_wcdm,
        "k_cosmo_lcdm": k_lcdm,
        "k_cosmo_wcdm": k_wcdm,
        "k_total_lcdm": k_total_lcdm,
        "k_total_wcdm": k_total_wcdm,
        "cov_jitter": cho_info["jitter"],
        "cov_psd_correction": cho_info.get("psd_correction"),
        "cov_min_eig": cho_info.get("min_eig"),
        "sigma_median_stat_sys": sigma_median,
    }

    config = {
        "seed": seed,
        "bounds_lcdm": bounds_lcdm,
        "bounds_wcdm": bounds_wcdm,
        "H0_fixed": lcdm.H0_fixed,
    }

    manifest.write_config(run_dir, config)
    manifest.write_metrics(run_dir, metrics)
    manifest.write_provenance(run_dir, "fit_des_sn5yr", seed)

    def mu_theory_lcdm(om):
        from sixbirds_cosmo.cosmo_background import luminosity_distance

        dl = luminosity_distance(z, lcdm.H0_fixed, om, 1.0 - om)
        return 5.0 * np.log10(np.asarray(dl)) + 25.0

    def mu_theory_wcdm(om, w):
        from sixbirds_cosmo.cosmo_background import luminosity_distance_wcdm

        dl = luminosity_distance_wcdm(z, lcdm.H0_fixed, om, w)
        return 5.0 * np.log10(np.asarray(dl)) + 25.0

    mu_lcdm = mu_theory_lcdm(fit_lcdm["theta_hat"][0]) + lcdm.last_M_hat
    mu_wcdm = (
        mu_theory_wcdm(fit_wcdm["theta_hat"][0], fit_wcdm["theta_hat"][1])
        + wcdm.last_M_hat
    )

    resid_lcdm = mu - mu_lcdm
    resid_wcdm = mu - mu_wcdm

    fig, ax = plt.subplots()
    ax.scatter(z, resid_lcdm, s=6, label="LCDM", alpha=0.6, zorder=2)
    ax.scatter(z, resid_wcdm, s=8, label="wCDM", alpha=0.4, zorder=3)
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_xlabel("zHD")
    ax.set_ylabel("mu_obs - mu_model")
    ax.set_title("DES-SN5YR residuals")
    ax.legend()

    plot_path = run_dir / "sn5yr_mu_residuals.png"
    manifest.save_fig(fig, plot_path)
    plt.close(fig)

    summary_path = run_dir / "summary.csv"
    with summary_path.open("w") as handle:
        handle.write("model,k,om,w,chi2,aic_total\n")
        handle.write(
            f"lcdm,{k_total_lcdm},{fit_lcdm['theta_hat'][0]},,"
            f"{fit_lcdm['chi2_hat']},{aic_total_lcdm}\n"
        )
        handle.write(
            f"wcdm,{k_total_wcdm},{fit_wcdm['theta_hat'][0]},{fit_wcdm['theta_hat'][1]},"
            f"{fit_wcdm['chi2_hat']},{aic_total_wcdm}\n"
        )

    write_run_markdown_stub(run_dir)

    print(json.dumps({"run_dir": str(run_dir)}, indent=2))


if __name__ == "__main__":
    main()
