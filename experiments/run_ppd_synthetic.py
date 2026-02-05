import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sixbirds_cosmo import manifest
from sixbirds_cosmo.audit_ppd import (
    bias,
    interp_at,
    predict_varH_lcdm,
    predict_varH_rewrite_from_template,
    rmse,
)
from sixbirds_cosmo.infer_distance import (
    distance_from_Hz,
    fit_lcdm,
    generate_mock_distance,
    make_redshift_from_a,
)
from sixbirds_cosmo.reporting import write_run_markdown_stub
from sixbirds_cosmo.rewrite import fit_rewrite, prepare_proxy_shape
from sixbirds_cosmo.toy2_patch import simulate_patches


def _build_distance_dataset(sim, seed: int, noise_frac: float, noise_floor: float):
    z = make_redshift_from_a(sim["aD"], normalize_to_last=True)
    z_max_avail = float(np.max(z))
    z_max = min(2.0, 0.9 * z_max_avail)
    if z_max <= 0.05:
        raise ValueError("z_max too small to build evaluation grid")
    z_eval = np.linspace(0.05, z_max, 40)
    D_true = distance_from_Hz(z, sim["HD"], z_eval)
    D_obs, sigma_D = generate_mock_distance(
        z_eval, D_true, noise_frac=noise_frac, noise_floor=noise_floor, seed=seed + 1
    )
    return z, z_eval, D_true, D_obs, sigma_D, z_max


def main() -> None:
    seed_master = 123
    seed_template = 468
    test_seeds = [123, 321, 999]
    seed_homo = 123

    N = 500
    steps = 1200
    dt = 0.02
    g4pi = 1.0
    rho_mean = 1.0
    f_void = 0.3
    kappa_scale = 5.0

    noise_frac = 0.005
    noise_floor = 1e-4

    run_dir = manifest.create_run_dir("ppd_synthetic", seed=seed_master)

    # Calibration template (heterogeneous)
    params_hetero = {
        "dt": dt,
        "g4pi": g4pi,
        "rho_mean": rho_mean,
        "delta_rho": 0.9,
        "f_void": f_void,
        "kappa_scale": kappa_scale,
    }
    sim_template = simulate_patches(N=N, steps=steps, params=params_hetero, seed=seed_template)
    z_ref = make_redshift_from_a(sim_template["aD"], normalize_to_last=True)
    varH_ref = sim_template["var_H"]
    proxy_ref = prepare_proxy_shape(z_ref, varH_ref)
    z_t, z_eval_t, D_true_t, D_obs_t, sigma_D_t, z_max_t = _build_distance_dataset(
        sim_template, seed_template, noise_frac, noise_floor
    )
    fit_template = fit_rewrite(z_eval_t, D_obs_t, sigma_D_t, proxy_ref)
    A_ref = float(fit_template["params"]["A"])

    summary_rows = []
    hetero_rmse_rows = []
    hetero_bias_lcdm = []
    hetero_bias_rewrite = []

    plot_seed = 123
    plot_payload = None

    # Heterogeneous test seeds
    for seed in test_seeds:
        sim = simulate_patches(N=N, steps=steps, params=params_hetero, seed=seed)
        z, z_eval, D_true, D_obs, sigma_D, z_max = _build_distance_dataset(
            sim, seed, noise_frac, noise_floor
        )
        fit_l = fit_lcdm(z_eval, D_obs, sigma_D)
        fit_r = fit_rewrite(z_eval, D_obs, sigma_D, proxy_ref)

        var_true = interp_at(z, sim["var_H"], z_eval)
        var_pred_l = predict_varH_lcdm(z_eval)
        var_pred_r = predict_varH_rewrite_from_template(
            z_eval,
            A_fit=fit_r["params"]["A"],
            A_ref=A_ref,
            z_ref=z_ref,
            varH_ref=varH_ref,
            A_cut=0.1,
        )

        rmse_l = rmse(var_true, var_pred_l)
        rmse_r = rmse(var_true, var_pred_r)
        bias_l = bias(var_true, var_pred_l)
        bias_r = bias(var_true, var_pred_r)

        hetero_rmse_rows.append(
            {
                "seed": seed,
                "rmse_lcdm": rmse_l,
                "rmse_rewrite": rmse_r,
                "ratio": rmse_r / rmse_l if rmse_l > 0 else np.inf,
            }
        )
        hetero_bias_lcdm.append(bias_l)
        hetero_bias_rewrite.append(bias_r)

        summary_rows.append(
            {
                "scenario": "heterogeneous",
                "delta_rho": params_hetero["delta_rho"],
                "seed": seed,
                "model": "lcdm",
                "rmse": rmse_l,
                "bias": bias_l,
                "H0_fit": fit_l.params["H0"],
                "omega_lambda_fit": fit_l.params["omega_lambda"],
                "A_fit": 0.0,
            }
        )
        summary_rows.append(
            {
                "scenario": "heterogeneous",
                "delta_rho": params_hetero["delta_rho"],
                "seed": seed,
                "model": "rewrite",
                "rmse": rmse_r,
                "bias": bias_r,
                "H0_fit": fit_r["params"]["H0"],
                "omega_lambda_fit": 0.0,
                "A_fit": fit_r["params"]["A"],
            }
        )

        if seed == plot_seed:
            plot_payload = (z_eval, var_true, var_pred_l, var_pred_r)

    # Homogeneous test
    params_homo = {
        "dt": dt,
        "g4pi": g4pi,
        "rho_mean": rho_mean,
        "delta_rho": 0.0,
        "f_void": f_void,
        "kappa_scale": kappa_scale,
    }
    sim_homo = simulate_patches(N=N, steps=steps, params=params_homo, seed=seed_homo)
    z_h, z_eval_h, D_true_h, D_obs_h, sigma_D_h, z_max_h = _build_distance_dataset(
        sim_homo, seed_homo, noise_frac, noise_floor
    )
    fit_l_h = fit_lcdm(z_eval_h, D_obs_h, sigma_D_h)
    fit_r_h = fit_rewrite(z_eval_h, D_obs_h, sigma_D_h, proxy_ref)
    var_true_h = interp_at(z_h, sim_homo["var_H"], z_eval_h)
    var_pred_l_h = predict_varH_lcdm(z_eval_h)
    var_pred_r_h = predict_varH_rewrite_from_template(
        z_eval_h,
        A_fit=fit_r_h["params"]["A"],
        A_ref=A_ref,
        z_ref=z_ref,
        varH_ref=varH_ref,
        A_cut=0.1,
    )
    homo_rmse_lcdm = rmse(var_true_h, var_pred_l_h)
    homo_rmse_rewrite = rmse(var_true_h, var_pred_r_h)

    summary_rows.append(
        {
            "scenario": "homogeneous",
            "delta_rho": params_homo["delta_rho"],
            "seed": seed_homo,
            "model": "lcdm",
            "rmse": homo_rmse_lcdm,
            "bias": bias(var_true_h, var_pred_l_h),
            "H0_fit": fit_l_h.params["H0"],
            "omega_lambda_fit": fit_l_h.params["omega_lambda"],
            "A_fit": 0.0,
        }
    )
    summary_rows.append(
        {
            "scenario": "homogeneous",
            "delta_rho": params_homo["delta_rho"],
            "seed": seed_homo,
            "model": "rewrite",
            "rmse": homo_rmse_rewrite,
            "bias": bias(var_true_h, var_pred_r_h),
            "H0_fit": fit_r_h["params"]["H0"],
            "omega_lambda_fit": 0.0,
            "A_fit": fit_r_h["params"]["A"],
        }
    )

    # Plots
    if plot_payload is None:
        raise RuntimeError("plot payload not generated")
    z_eval_p, var_true_p, var_pred_l_p, var_pred_r_p = plot_payload
    fig1, ax1 = plt.subplots()
    ax1.plot(z_eval_p, var_true_p, label="varH true")
    ax1.plot(z_eval_p, var_pred_l_p, label="LCDM pred")
    ax1.plot(z_eval_p, var_pred_r_p, label="rewrite pred")
    ax1.set_xlabel("z")
    ax1.set_ylabel("var_H")
    ax1.legend()
    manifest.save_fig(fig1, run_dir / "ppd_varH_hetero_seed123.png")
    plt.close(fig1)

    fig2, ax2 = plt.subplots()
    seeds = [row["seed"] for row in hetero_rmse_rows]
    rmse_l = [row["rmse_lcdm"] for row in hetero_rmse_rows]
    rmse_r = [row["rmse_rewrite"] for row in hetero_rmse_rows]
    ax2.plot(seeds, rmse_l, marker="o", label="LCDM")
    ax2.plot(seeds, rmse_r, marker="o", label="rewrite")
    ax2.set_xlabel("seed")
    ax2.set_ylabel("RMSE(var_H)")
    ax2.legend()
    manifest.save_fig(fig2, run_dir / "ppd_rmse_across_seeds.png")
    plt.close(fig2)

    hetero_rmse_ratio = [row["ratio"] for row in hetero_rmse_rows]
    hetero_rmse_ratio_median = float(np.median(hetero_rmse_ratio))
    metrics = {
        "A_ref": A_ref,
        "seed_template": seed_template,
        "hetero_rmse_rows": hetero_rmse_rows,
        "hetero_rmse_ratio_median": hetero_rmse_ratio_median,
        "hetero_bias_lcdm_mean": float(np.mean(hetero_bias_lcdm)),
        "hetero_bias_rewrite_mean": float(np.mean(hetero_bias_rewrite)),
        "homo_rmse_lcdm": homo_rmse_lcdm,
        "homo_rmse_rewrite": homo_rmse_rewrite,
        "noise_frac": noise_frac,
        "noise_floor": noise_floor,
        "test_seeds": test_seeds,
        "seed_homo": seed_homo,
    }

    summary_df = pd.DataFrame(summary_rows)
    manifest.save_table(summary_df, run_dir / "summary.csv")

    manifest.write_config(
        run_dir,
        {
            "seed_master": seed_master,
            "seed_template": seed_template,
            "test_seeds": test_seeds,
            "seed_homo": seed_homo,
            "base_params": {
                "N": N,
                "steps": steps,
                "dt": dt,
                "g4pi": g4pi,
                "rho_mean": rho_mean,
                "f_void": f_void,
                "kappa_scale": kappa_scale,
            },
            "noise_frac": noise_frac,
            "noise_floor": noise_floor,
        },
    )
    manifest.write_metrics(run_dir, metrics)
    manifest.write_provenance(run_dir, "ppd_synthetic", seed_master)
    write_run_markdown_stub(run_dir)

    # Assertions
    for row in hetero_rmse_rows:
        # LCDM bias should be nonzero in hetero case
        seed = row["seed"]
        bias_l = next(
            r["bias"]
            for r in summary_rows
            if r["scenario"] == "heterogeneous" and r["seed"] == seed and r["model"] == "lcdm"
        )
        assert abs(bias_l) > 1e-8

    assert hetero_rmse_ratio_median < 0.2

    print(f"Run dir: {run_dir}")


if __name__ == "__main__":
    main()
