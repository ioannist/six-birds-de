#!/usr/bin/env python3
"""Sanity checks for DES-SN5YR covariance decoding."""

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
    extract_distances_covmat,
    load_des_sn5yr_covariance,
    load_des_sn5yr_hubble_diagram,
    prepare_cov_solve,
)
from sixbirds_cosmo.reporting import write_run_markdown_stub


def covariance_metrics(kind: str, raw_dir: Path) -> dict:
    cov_tmp = load_des_sn5yr_covariance(raw_dir, kind=kind, mu_err=None)
    hd = load_des_sn5yr_hubble_diagram(raw_dir, n_cov=cov_tmp["N"])
    mu_err = hd["mu_err"]
    cov_info = load_des_sn5yr_covariance(raw_dir, kind=kind, mu_err=mu_err)

    cov = cov_info["cov"]
    n_mu = len(hd["mu"])
    n_cov = cov.shape[0]

    symmetry_max_abs = float(np.max(np.abs(cov - cov.T)))
    diag = np.diag(cov)
    sigma = np.sqrt(diag)

    chol_success = True
    jitter_used = 0.0
    psd_correction = False
    min_eig = None
    try:
        chol = prepare_cov_solve(cov)
        jitter_used = chol["jitter"]
        psd_correction = chol.get("psd_correction", False)
        min_eig = chol.get("min_eig")
    except Exception:
        chol_success = False
        jitter_used = float("inf")

    corr = cov / np.outer(sigma, sigma)
    corr_offdiag = corr - np.eye(corr.shape[0])

    metrics = {
        "n_mu": n_mu,
        "n_cov": n_cov,
        "triplet_min_i": cov_info.get("triplet_min_i"),
        "triplet_max_i": cov_info.get("triplet_max_i"),
        "triplet_min_j": cov_info.get("triplet_min_j"),
        "triplet_max_j": cov_info.get("triplet_max_j"),
        "index_base_detected": cov_info.get("index_base_detected"),
        "symmetry_max_abs": symmetry_max_abs,
        "diag_min": float(np.min(diag)),
        "diag_median": float(np.median(diag)),
        "diag_max": float(np.max(diag)),
        "sigma_min": float(np.min(sigma)),
        "sigma_median": float(np.median(sigma)),
        "sigma_max": float(np.max(sigma)),
        "chol_success_no_jitter": chol_success and jitter_used == 0.0,
        "chol_jitter_used": jitter_used,
        "psd_correction": psd_correction,
        "min_eig": min_eig,
        "corr_offdiag_max_abs": float(np.max(np.abs(corr_offdiag))),
        "corr_diag_median": float(np.median(np.diag(corr))),
        "matrix_kind_detected": cov_info.get("matrix_kind_detected"),
    }

    fig, ax = plt.subplots()
    ax.hist(sigma, bins=40)
    ax.set_xlabel("sigma")
    ax.set_ylabel("count")
    ax.set_title(f"SN5YR sigma histogram ({kind})")

    fig2, ax2 = plt.subplots()
    n_block = min(100, cov.shape[0])
    ax2.imshow(corr[:n_block, :n_block], origin="lower")
    ax2.set_title(f"SN5YR corr block ({kind})")

    return metrics, fig, fig2


def main() -> None:
    seed = 123
    run_dir = manifest.create_run_dir("sn5yr_cov_sanity", seed=seed)

    raw_dir = Path(fetch("des_sn5yr_distances"))
    extract_distances_covmat(raw_dir)

    metrics = {
        "stat": {},
        "stat+sys": {},
    }

    for kind in ["stat", "stat+sys"]:
        metrics_kind, fig_hist, fig_corr = covariance_metrics(kind, raw_dir)

        # Assertions
        if metrics_kind["n_mu"] != metrics_kind["n_cov"]:
            raise AssertionError("n_mu != n_cov")
        if metrics_kind["symmetry_max_abs"] > 1e-8:
            raise AssertionError("covariance not symmetric")
        if metrics_kind["sigma_median"] < 1e-3 or metrics_kind["sigma_median"] > 5.0:
            raise AssertionError("sigma_median out of plausible range")
        if metrics_kind["chol_jitter_used"] > 1e-6:
            raise AssertionError("excessive jitter required")

        sigma_path = run_dir / f"sn5yr_sigma_hist_{kind}.png"
        corr_path = run_dir / f"sn5yr_corr_block_{kind}.png"
        manifest.save_fig(fig_hist, sigma_path)
        plt.close(fig_hist)
        manifest.save_fig(fig_corr, corr_path)
        plt.close(fig_corr)

        metrics_kind["plot_sigma_path"] = str(sigma_path)
        metrics_kind["plot_corr_path"] = str(corr_path)
        metrics[kind] = metrics_kind

    manifest.write_config(run_dir, {"seed": seed})
    manifest.write_metrics(run_dir, metrics)
    manifest.write_provenance(run_dir, "sn5yr_cov_sanity", seed)
    write_run_markdown_stub(run_dir)

    print(json.dumps({"run_dir": str(run_dir)}, indent=2))


if __name__ == "__main__":
    main()
