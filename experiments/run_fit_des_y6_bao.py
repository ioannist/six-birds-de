#!/usr/bin/env python3
"""Fit DES Y6 BAO alpha-likelihood curves with background models."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sixbirds_cosmo import manifest
from sixbirds_cosmo.data_registry import fetch
from sixbirds_cosmo.datasets.des_y6_bao import (
    DESY6BAOAlphaDataset,
    load_alpha_likelihood_curve,
    load_bao_fiducial_info,
    load_des_y6_bao_dataset,
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


def estimate_sigma(alpha_grid: np.ndarray, delta_chi2: np.ndarray) -> float | None:
    idx_min = int(np.argmin(delta_chi2))
    left = None
    right = None

    for i in range(idx_min, 0, -1):
        if delta_chi2[i] <= 1.0 and delta_chi2[i - 1] > 1.0:
            x0, x1 = alpha_grid[i - 1], alpha_grid[i]
            y0, y1 = delta_chi2[i - 1], delta_chi2[i]
            left = x0 + (1.0 - y0) * (x1 - x0) / (y1 - y0)
            break
        if delta_chi2[i] > 1.0 and delta_chi2[i - 1] <= 1.0:
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
        if delta_chi2[i] > 1.0 and delta_chi2[i + 1] <= 1.0:
            x0, x1 = alpha_grid[i], alpha_grid[i + 1]
            y0, y1 = delta_chi2[i], delta_chi2[i + 1]
            right = x0 + (1.0 - y0) * (x1 - x0) / (y1 - y0)
            break

    if left is None or right is None:
        return None
    return 0.5 * (right - left)


def main() -> None:
    seed = 123
    run_dir = manifest.create_run_dir("fit_des_y6_bao", seed=seed)

    dataset_dir = Path(fetch("des_y6_bao_release"))
    load_des_y6_bao_dataset(dataset_dir)

    fid_info = load_bao_fiducial_info(dataset_dir)
    z_eff = fid_info["z_eff"]

    curves = {}
    for kind in ["acf", "aps", "pcf", "fid"]:
        curves[kind] = load_alpha_likelihood_curve(dataset_dir, kind)

    combined = combine_curves([curves["acf"], curves["aps"], curves["pcf"]])

    alpha_grid = combined["alpha_grid"]
    delta_combined = combined["delta_chi2_grid"]
    alpha_hat = float(alpha_grid[np.argmin(delta_combined)])
    sigma_est = estimate_sigma(alpha_grid, delta_combined)

    assert alpha_grid.size > 50
    assert np.min(delta_combined) <= 1e-6
    assert 0.8 < alpha_hat < 1.2

    bounds = [(0.05, 1.5)]
    theta0 = np.array([0.3])

    ocdm = DESY6BAOAlphaDataset(
        name="des_y6_bao_alpha_ocdm",
        n_data=1,
        param_names=["om"],
        alpha_grid=alpha_grid,
        delta_chi2_grid=delta_combined,
        z_eff=z_eff,
        model_kind="ocdm",
    )
    flat = DESY6BAOAlphaDataset(
        name="des_y6_bao_alpha_flat",
        n_data=1,
        param_names=["om"],
        alpha_grid=alpha_grid,
        delta_chi2_grid=delta_combined,
        z_eff=z_eff,
        model_kind="flat_lcdm",
    )

    fit_ocdm = fit_map(ocdm, theta0, bounds=bounds)
    fit_flat = fit_map(flat, theta0, bounds=bounds)

    assert np.isfinite(fit_ocdm["chi2_hat"])
    assert np.isfinite(fit_flat["chi2_hat"])
    assert fit_ocdm["chi2_hat"] < 1e4
    assert fit_flat["chi2_hat"] < 1e4

    delta_aic = fit_ocdm["aic"] - fit_flat["aic"]

    metrics = {
        "z_eff": z_eff,
        "alpha_hat_combined": alpha_hat,
        "alpha_sigma_est_combined": sigma_est,
        "ocdm_om": fit_ocdm["theta_hat"][0],
        "ocdm_chi2": fit_ocdm["chi2_hat"],
        "ocdm_aic": fit_ocdm["aic"],
        "flat_lcdm_om": fit_flat["theta_hat"][0],
        "flat_lcdm_chi2": fit_flat["chi2_hat"],
        "flat_lcdm_aic": fit_flat["aic"],
        "delta_aic": delta_aic,
        "curve_modes": {
            "acf": curves["acf"]["notes"],
            "aps": curves["aps"]["notes"],
            "pcf": curves["pcf"]["notes"],
            "fid": curves["fid"]["notes"],
        },
        "curve_files": {
            "acf": curves["acf"]["filename"],
            "aps": curves["aps"]["filename"],
            "pcf": curves["pcf"]["filename"],
            "fid": curves["fid"]["filename"],
        },
        "fid_params_used": {
            "om_fid": ocdm.om_fid,
            "ol_fid": ocdm.ol_fid,
            "H0_fid": ocdm.H0_fid,
        },
    }

    config = {
        "seed": seed,
        "bounds": bounds,
        "theta0": theta0.tolist(),
        "z_eff": z_eff,
        "z_eff_source": fid_info.get("z_eff_source"),
        "fit_curve": "combined_acf_aps_pcf",
    }

    manifest.write_config(run_dir, config)
    manifest.write_metrics(run_dir, metrics)
    manifest.write_provenance(run_dir, "fit_des_y6_bao", seed)

    fig, ax = plt.subplots()
    ax.plot(curves["acf"]["alpha_grid"], curves["acf"]["delta_chi2_grid"], label="acf")
    ax.plot(curves["aps"]["alpha_grid"], curves["aps"]["delta_chi2_grid"], label="aps")
    ax.plot(curves["pcf"]["alpha_grid"], curves["pcf"]["delta_chi2_grid"], label="pcf")
    ax.plot(alpha_grid, delta_combined, label="combined", linewidth=2.0)

    ax.axvline(alpha_hat, color="black", linestyle="--", linewidth=1.0)
    alpha_ocdm = ocdm.alpha_pred(fit_ocdm["theta_hat"][0])
    alpha_flat = flat.alpha_pred(fit_flat["theta_hat"][0])
    if alpha_ocdm is not None:
        ax.plot(alpha_ocdm, 0.0, marker="o", label="ocdm fit")
    if alpha_flat is not None:
        ax.plot(alpha_flat, 0.0, marker="x", label="flat_lcdm fit")

    ax.set_xlabel("alpha")
    ax.set_ylabel("delta chi2")
    ax.set_title("DES Y6 BAO alpha likelihood")
    ax.legend()

    plot_path = run_dir / "alpha_likelihood.png"
    manifest.save_fig(fig, plot_path)
    plt.close(fig)

    summary_path = run_dir / "summary.csv"
    with summary_path.open("w") as handle:
        handle.write("model,k,om,chi2,aic\n")
        handle.write(
            f"ocdm,{fit_ocdm['k']},{fit_ocdm['theta_hat'][0]},"
            f"{fit_ocdm['chi2_hat']},{fit_ocdm['aic']}\n"
        )
        handle.write(
            f"flat_lcdm,{fit_flat['k']},{fit_flat['theta_hat'][0]},"
            f"{fit_flat['chi2_hat']},{fit_flat['aic']}\n"
        )

    write_run_markdown_stub(run_dir)

    print(json.dumps({"run_dir": str(run_dir)}, indent=2))


if __name__ == "__main__":
    main()
