import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from sixbirds_cosmo import manifest
from sixbirds_cosmo.infer_distance import (
    D_lcdm,
    D_matter_only,
    compare_models,
    distance_from_Hz,
    fit_lcdm,
    fit_matter_only,
    generate_mock_distance,
    make_redshift_from_a,
)
from sixbirds_cosmo.reporting import write_run_markdown_stub
from sixbirds_cosmo.rewrite import D_rewrite, fit_rewrite, fit_table_row, prepare_proxy_shape
from sixbirds_cosmo.toy2_patch import simulate_patches


def _build_mock_dataset(sim, seed: int, noise_frac: float, noise_floor: float):
    z = make_redshift_from_a(sim["aD"], normalize_to_last=True)
    z_max_avail = float(np.max(z))
    z_max = min(2.0, 0.9 * z_max_avail)
    if z_max <= 0.05:
        raise ValueError("z_max too small to build evaluation grid")
    z_eval = np.linspace(0.05, z_max, 40)
    D_true = distance_from_Hz(z, sim["HD"], z_eval)
    D_obs, sigma = generate_mock_distance(
        z_eval, D_true, noise_frac=noise_frac, noise_floor=noise_floor, seed=seed + 1
    )
    return z, z_eval, D_true, D_obs, sigma, z_max


def main() -> None:
    seed = 468
    noise_frac = 0.005
    noise_floor = 1e-4

    base_params = {
        "dt": 0.02,
        "g4pi": 1.0,
        "rho_mean": 1.0,
        "delta_rho": 0.9,
        "f_void": 0.3,
        "kappa_scale": 5.0,
    }

    run_dir = manifest.create_run_dir("rewrite_vs_lambda", seed=seed)

    sim = simulate_patches(N=500, steps=1200, params=base_params, seed=seed)
    z_proxy = make_redshift_from_a(sim["aD"], normalize_to_last=True)
    proxy = prepare_proxy_shape(z_proxy=z_proxy, var_proxy=sim["var_H"])

    z, z_eval, D_true, D_obs, sigma, z_max = _build_mock_dataset(
        sim, seed, noise_frac, noise_floor
    )

    fit_A = fit_matter_only(z_eval, D_obs, sigma)
    fit_B = fit_lcdm(z_eval, D_obs, sigma)
    fit_R = fit_rewrite(z_eval, D_obs, sigma, proxy)

    D_A = D_matter_only(z_eval, fit_A.params["H0"])
    D_B = D_lcdm(z_eval, fit_B.params["H0"], fit_B.params["omega_lambda"])
    D_R = D_rewrite(z_eval, fit_R["params"]["H0"], fit_R["params"]["A"], proxy["z_sorted"], proxy["g_sorted"])

    fig, ax = plt.subplots()
    ax.errorbar(z_eval, D_obs, yerr=sigma, fmt="o", label="mock data")
    ax.plot(z_eval, D_A, label="matter-only")
    ax.plot(z_eval, D_B, label="LambdaCDM")
    ax.plot(z_eval, D_R, label="rewrite")
    ax.plot(z_eval, D_true, linestyle="--", label="true")
    ax.set_xlabel("z")
    ax.set_ylabel("D(z)")
    ax.legend()
    manifest.save_fig(fig, run_dir / "rewrite_distance_fit.png")
    plt.close(fig)

    delta_grid = [0.3, 0.6, 0.9]
    sweep_rows = []
    A_list = []
    for delta in delta_grid:
        params = {
            "dt": base_params["dt"],
            "g4pi": base_params["g4pi"],
            "rho_mean": base_params["rho_mean"],
            "delta_rho": delta,
            "f_void": base_params["f_void"],
            "kappa_scale": base_params["kappa_scale"],
        }
        sweep_seed = seed + int(delta * 1000)
        sim_sweep = simulate_patches(N=300, steps=900, params=params, seed=sweep_seed)
        z_proxy_sweep = make_redshift_from_a(sim_sweep["aD"], normalize_to_last=True)
        proxy_sweep = prepare_proxy_shape(z_proxy=z_proxy_sweep, var_proxy=sim_sweep["var_H"])
        z_s, z_eval_s, D_true_s, D_obs_s, sigma_s, z_max_s = _build_mock_dataset(
            sim_sweep, sweep_seed, noise_frac, noise_floor
        )
        fit_s = fit_rewrite(z_eval_s, D_obs_s, sigma_s, proxy_sweep)
        A_best = fit_s["params"]["A"]
        A_list.append(A_best)
        sweep_rows.append(
            {
                "delta_rho": float(delta),
                "A_best": float(A_best),
                "proxy_scale": float(proxy_sweep["scale"]),
                "z_max": float(z_max_s),
            }
        )

    spearman = spearmanr(delta_grid, A_list)
    spearman_rho = float(spearman.correlation)
    spearman_pvalue = float(spearman.pvalue)
    A_monotone = all(A_list[i + 1] > A_list[i] for i in range(len(A_list) - 1))

    fig2, ax2 = plt.subplots()
    ax2.plot(delta_grid, A_list, marker="o")
    ax2.set_xlabel("delta_rho")
    ax2.set_ylabel("A_best")
    manifest.save_fig(fig2, run_dir / "A_vs_delta_rho.png")
    plt.close(fig2)

    metrics = {
        "chi2_matter_only": fit_A.chi2,
        "aic_matter_only": fit_A.aic,
        "chi2_lcdm": fit_B.chi2,
        "aic_lcdm": fit_B.aic,
        "chi2_rewrite": fit_R["chi2"],
        "aic_rewrite": fit_R["aic"],
        "lcdm_omega_lambda": fit_B.params["omega_lambda"],
        "rewrite_A": fit_R["params"]["A"],
        "delta_aic_lcdm_minus_rewrite": fit_B.aic - fit_R["aic"],
        "delta_chi2_rewrite_minus_lcdm": fit_R["chi2"] - fit_B.chi2,
        "sweep_rows": sweep_rows,
        "spearman_rho": spearman_rho,
        "spearman_pvalue": spearman_pvalue,
        "A_monotone_increasing": A_monotone,
    }

    summary_rows = [
        fit_table_row("matter_only", 1, {"H0": fit_A.params["H0"], "omega_lambda": 0.0, "A": 0.0}, fit_A.chi2, fit_A.aic),
        fit_table_row("lcdm", 2, {"H0": fit_B.params["H0"], "omega_lambda": fit_B.params["omega_lambda"], "A": 0.0}, fit_B.chi2, fit_B.aic),
        fit_table_row("rewrite", 2, {"H0": fit_R["params"]["H0"], "omega_lambda": 0.0, "A": fit_R["params"]["A"]}, fit_R["chi2"], fit_R["aic"]),
    ]

    summary_df = pd.DataFrame(summary_rows)
    manifest.save_table(summary_df, run_dir / "summary.csv")

    sweep_df = pd.DataFrame(sweep_rows)
    manifest.save_table(sweep_df, run_dir / "sweep.csv")

    config = {
        "seed": seed,
        "noise_frac": noise_frac,
        "noise_floor": noise_floor,
        "base_params": base_params,
        "delta_grid": delta_grid,
    }

    manifest.write_config(run_dir, config)
    manifest.write_metrics(run_dir, metrics)
    manifest.write_provenance(run_dir, "rewrite_vs_lambda", seed)
    write_run_markdown_stub(run_dir)

    assert fit_R["chi2"] <= fit_B.chi2 + 5.0
    assert spearman_rho >= 0.8
    assert A_monotone is True

    print(f"Run dir: {run_dir}")


if __name__ == "__main__":
    main()
