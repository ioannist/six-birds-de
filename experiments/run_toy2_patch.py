import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sixbirds_cosmo import manifest
from sixbirds_cosmo.reporting import write_run_markdown_stub
from sixbirds_cosmo.toy2_patch import simulate_patches


def _late_slice(values: np.ndarray, frac: float = 0.2) -> np.ndarray:
    start = int((1.0 - frac) * len(values))
    return values[start:]


def main() -> None:
    seed = 123
    N = 500
    steps = 1200
    dt = 0.02
    g4pi = 1.0
    rho_mean = 1.0

    homogeneous_params = {
        "dt": dt,
        "g4pi": g4pi,
        "rho_mean": rho_mean,
        "delta_rho": 0.0,
        "f_void": 0.3,
        "kappa_scale": 0.0,
    }

    hetero_params = {
        "dt": dt,
        "g4pi": g4pi,
        "rho_mean": rho_mean,
        "delta_rho": 0.9,
        "f_void": 0.3,
        "kappa_scale": 5.0,
    }

    run_dir = manifest.create_run_dir("toy2_patch", seed=seed)

    homo = simulate_patches(N=N, steps=steps, params=homogeneous_params, seed=seed)
    hetero = simulate_patches(N=N, steps=steps, params=hetero_params, seed=seed)

    t = homo["t"]

    fig1, ax1 = plt.subplots()
    ax1.plot(t, homo["aD"], label="homogeneous")
    ax1.plot(t, hetero["aD"], label="heterogeneous")
    ax1.set_xlabel("t")
    ax1.set_ylabel("a_D")
    ax1.legend()
    manifest.save_fig(fig1, run_dir / "aD_vs_time.png")
    plt.close(fig1)

    fig2, ax2 = plt.subplots()
    ax2.plot(t, homo["ddot_aD"], label="homogeneous")
    ax2.plot(t, hetero["ddot_aD"], label="heterogeneous")
    ax2.set_xlabel("t")
    ax2.set_ylabel("ddot a_D")
    ax2.legend()
    manifest.save_fig(fig2, run_dir / "ddot_aD_vs_time.png")
    plt.close(fig2)

    fig3, ax3 = plt.subplots()
    ax3.plot(t, homo["Q"], label="homogeneous")
    ax3.plot(t, hetero["Q"], label="heterogeneous")
    ax3.set_xlabel("t")
    ax3.set_ylabel("Q")
    ax3.legend()
    manifest.save_fig(fig3, run_dir / "Q_vs_time.png")
    plt.close(fig3)

    homogeneous_Q_max_abs = float(np.max(np.abs(homo["Q"])))
    hetero_Q_peak = float(np.max(hetero["Q"]))
    hetero_Q_mean_late = float(np.mean(_late_slice(hetero["Q"])))

    hetero_frac_ddot_pos = float(np.mean(hetero["ddot_aD"] > 0.0))
    hetero_frac_ddot_pos_late = float(np.mean(_late_slice(hetero["ddot_aD"]) > 0.0))

    assert homogeneous_Q_max_abs < 1e-12
    assert hetero_Q_mean_late > 1e-6
    assert hetero_Q_peak > 1e-3

    metrics = {
        "homogeneous_Q_max_abs": homogeneous_Q_max_abs,
        "hetero_Q_peak": hetero_Q_peak,
        "hetero_Q_mean_late": hetero_Q_mean_late,
        "hetero_frac_ddot_aD_pos": hetero_frac_ddot_pos,
        "hetero_frac_ddot_aD_pos_late": hetero_frac_ddot_pos_late,
    }

    config = {
        "seed": seed,
        "N": N,
        "steps": steps,
        "dt": dt,
        "g4pi": g4pi,
        "rho_mean": rho_mean,
        "homogeneous": homogeneous_params,
        "heterogeneous": hetero_params,
    }

    summary_rows = [
        {
            "scenario": "homogeneous",
            "delta_rho": homogeneous_params["delta_rho"],
            "f_void": homogeneous_params["f_void"],
            "kappa_scale": homogeneous_params["kappa_scale"],
            "Q_peak": float(np.max(homo["Q"])),
            "Q_mean_late": float(np.mean(_late_slice(homo["Q"]))),
            "frac_ddot_pos": float(np.mean(homo["ddot_aD"] > 0.0)),
            "frac_ddot_pos_late": float(np.mean(_late_slice(homo["ddot_aD"]) > 0.0)),
        },
        {
            "scenario": "heterogeneous",
            "delta_rho": hetero_params["delta_rho"],
            "f_void": hetero_params["f_void"],
            "kappa_scale": hetero_params["kappa_scale"],
            "Q_peak": hetero_Q_peak,
            "Q_mean_late": hetero_Q_mean_late,
            "frac_ddot_pos": hetero_frac_ddot_pos,
            "frac_ddot_pos_late": hetero_frac_ddot_pos_late,
        },
    ]

    summary_df = pd.DataFrame(summary_rows)
    manifest.save_table(summary_df, run_dir / "summary.csv")

    manifest.write_config(run_dir, config)
    manifest.write_metrics(run_dir, metrics)
    manifest.write_provenance(run_dir, "toy2_patch", seed)
    write_run_markdown_stub(run_dir)

    print(f"Run dir: {run_dir}")


if __name__ == "__main__":
    main()
