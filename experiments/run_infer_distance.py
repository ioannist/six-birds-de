import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sixbirds_cosmo import manifest
from sixbirds_cosmo.infer_distance import (
    compare_models,
    distance_from_Hz,
    fit_lcdm,
    fit_matter_only,
    generate_mock_distance,
    make_redshift_from_a,
    D_lcdm,
    D_matter_only,
)
from sixbirds_cosmo.reporting import write_run_markdown_stub
from sixbirds_cosmo.toy2_patch import simulate_patches


def main() -> None:
    seed = 123
    N = 500
    steps = 1200
    dt = 0.02
    g4pi = 1.0
    rho_mean = 1.0
    delta_rho = 0.9
    f_void = 0.3
    kappa_scale = 5.0

    run_dir = manifest.create_run_dir("infer_distance", seed=seed)

    params = {
        "dt": dt,
        "g4pi": g4pi,
        "rho_mean": rho_mean,
        "delta_rho": delta_rho,
        "f_void": f_void,
        "kappa_scale": kappa_scale,
    }

    sim = simulate_patches(N=N, steps=steps, params=params, seed=seed)
    aD = sim["aD"]
    HD = sim["HD"]

    z = make_redshift_from_a(aD, normalize_to_last=True)
    z_max_avail = float(np.max(z))
    z_max = min(2.0, 0.9 * z_max_avail)
    if z_max <= 0.05:
        raise ValueError("z_max too small to build evaluation grid")
    z_eval = np.linspace(0.05, z_max, 40)

    D_true = distance_from_Hz(z, HD, z_eval)

    noise_frac = 0.005
    noise_floor = 1e-4
    D_obs, sigma = generate_mock_distance(
        z_eval, D_true, noise_frac=noise_frac, noise_floor=noise_floor, seed=seed + 1
    )

    fit_A = fit_matter_only(z_eval, D_obs, sigma)
    fit_B = fit_lcdm(z_eval, D_obs, sigma)
    comparison = compare_models(fit_A, fit_B)

    delta_chi2 = comparison["delta_chi2"]
    delta_aic = comparison["delta_aic"]

    assert fit_B.params["omega_lambda"] > 0.1
    assert delta_aic > 10.0

    D_A = D_matter_only(z_eval, fit_A.params["H0"])
    D_B = D_lcdm(z_eval, fit_B.params["H0"], fit_B.params["omega_lambda"])

    fig, ax = plt.subplots()
    ax.errorbar(z_eval, D_obs, yerr=sigma, fmt="o", label="mock data")
    ax.plot(z_eval, D_A, label="matter-only fit")
    ax.plot(z_eval, D_B, label="LambdaCDM fit")
    ax.plot(z_eval, D_true, linestyle="--", label="true")
    ax.set_xlabel("z")
    ax.set_ylabel("D(z)")
    ax.legend()
    manifest.save_fig(fig, run_dir / "distance_fit.png")
    plt.close(fig)

    metrics = {
        "matter_only_H0": fit_A.params["H0"],
        "lcdm_H0": fit_B.params["H0"],
        "lcdm_omega_lambda": fit_B.params["omega_lambda"],
        "chi2_matter_only": fit_A.chi2,
        "aic_matter_only": fit_A.aic,
        "chi2_lcdm": fit_B.chi2,
        "aic_lcdm": fit_B.aic,
        "delta_chi2": delta_chi2,
        "delta_aic": delta_aic,
        "noise_frac": noise_frac,
        "noise_floor": noise_floor,
        "z_max": z_max,
        "n_points": int(len(z_eval)),
    }

    config = {
        "seed": seed,
        "N": N,
        "steps": steps,
        "dt": dt,
        "g4pi": g4pi,
        "rho_mean": rho_mean,
        "delta_rho": delta_rho,
        "f_void": f_void,
        "kappa_scale": kappa_scale,
        "noise_frac": noise_frac,
        "noise_floor": noise_floor,
    }

    summary_rows = [
        {
            "model": "matter_only",
            "k": 1,
            "H0": fit_A.params["H0"],
            "omega_lambda": 0.0,
            "chi2": fit_A.chi2,
            "aic": fit_A.aic,
        },
        {
            "model": "lcdm",
            "k": 2,
            "H0": fit_B.params["H0"],
            "omega_lambda": fit_B.params["omega_lambda"],
            "chi2": fit_B.chi2,
            "aic": fit_B.aic,
        },
    ]

    summary_df = pd.DataFrame(summary_rows)
    manifest.save_table(summary_df, run_dir / "summary.csv")

    manifest.write_config(run_dir, config)
    manifest.write_metrics(run_dir, metrics)
    manifest.write_provenance(run_dir, "infer_distance", seed)
    write_run_markdown_stub(run_dir)

    print(f"Run dir: {run_dir}")


if __name__ == "__main__":
    main()
