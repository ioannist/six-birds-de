import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sixbirds_cosmo import manifest
from sixbirds_cosmo.reporting import write_run_markdown_stub
from sixbirds_cosmo.toy1 import simulate_micro


def main() -> None:
    seed = 123
    steps = 200
    dt = 0.5
    alpha = 0.0
    beta_fixed = 2.0
    n_list = [10, 100, 1000]
    beta_grid = [0.0, 0.25, 0.5, 1.0, 2.0]

    run_dir = manifest.create_run_dir("toy1", seed=seed)

    base_params = {
        "alpha": alpha,
        "beta": beta_fixed,
        "dt": dt,
        "x0_dist": "uniform",
        "x0_low": 0.0,
        "x0_high": 1.0,
    }

    rm_by_n_summary = []
    delta_max_overall = 0.0

    fig1, ax1 = plt.subplots()
    t = np.arange(steps)
    for n in n_list:
        result = simulate_micro(n=n, steps=steps, params=base_params, seed=seed + n)
        rm = result["rm"]
        delta = result["delta"]
        rm_max = float(np.max(rm))
        rm_mean_late = float(np.mean(rm[int(0.8 * steps) :]))
        delta_max = float(np.max(delta))
        delta_max_overall = max(delta_max_overall, delta_max)
        rm_by_n_summary.append(
            {
                "n": n,
                "beta": beta_fixed,
                "rm_max": rm_max,
                "rm_mean_late": rm_mean_late,
                "delta_max": delta_max,
            }
        )
        ax1.plot(t, rm, label=f"n={n}")

    ax1.set_xlabel("t")
    ax1.set_ylabel("route mismatch RM(t)")
    ax1.legend()
    manifest.save_fig(fig1, run_dir / "rm_vs_time_by_n.png")
    plt.close(fig1)

    rm_vs_beta_table = []
    rm_summary_stat = "rm_max"
    for beta in beta_grid:
        params = {**base_params, "beta": beta}
        result = simulate_micro(n=1000, steps=steps, params=params, seed=seed)
        rm = result["rm"]
        delta = result["delta"]
        rm_max = float(np.max(rm))
        rm_mean_late = float(np.mean(rm[int(0.8 * steps) :]))
        delta_max = float(np.max(delta))
        delta_max_overall = max(delta_max_overall, delta_max)
        rm_vs_beta_table.append(
            {
                "beta": beta,
                "rm_max": rm_max,
                "rm_mean_late": rm_mean_late,
                "delta_max": delta_max,
            }
        )

    fig2, ax2 = plt.subplots()
    ax2.plot(
        [row["beta"] for row in rm_vs_beta_table],
        [row[rm_summary_stat] for row in rm_vs_beta_table],
        marker="o",
    )
    ax2.set_xlabel("beta")
    ax2.set_ylabel(rm_summary_stat)
    manifest.save_fig(fig2, run_dir / "rm_summary_vs_beta.png")
    plt.close(fig2)

    beta_linear_rm_max = next(row["rm_max"] for row in rm_vs_beta_table if row["beta"] == 0.0)
    beta_small_rm_max = next(row["rm_max"] for row in rm_vs_beta_table if row["beta"] == 0.5)
    beta_large_rm_max = next(row["rm_max"] for row in rm_vs_beta_table if row["beta"] == 2.0)

    assert beta_linear_rm_max < 1e-12
    assert beta_large_rm_max > 1e-3

    metrics = {
        "beta_linear_case_rm_max": beta_linear_rm_max,
        "beta_small_rm_max": beta_small_rm_max,
        "beta_large_rm_max": beta_large_rm_max,
        "delta_max_overall": delta_max_overall,
        "rm_vs_beta_table": rm_vs_beta_table,
        "rm_by_n_summary": rm_by_n_summary,
        "rm_summary_stat": rm_summary_stat,
    }

    config = {
        "seed": seed,
        "steps": steps,
        "dt": dt,
        "alpha": alpha,
        "beta_fixed": beta_fixed,
        "n_list": n_list,
        "beta_grid": beta_grid,
        "x0_dist": base_params["x0_dist"],
        "x0_low": base_params["x0_low"],
        "x0_high": base_params["x0_high"],
    }

    manifest.write_config(run_dir, config)
    manifest.write_metrics(run_dir, metrics)
    manifest.write_provenance(run_dir, "toy1", seed)

    rows = []
    for row in rm_by_n_summary:
        rows.append(
            {
                "scenario": "by_n",
                "n": row["n"],
                "beta": row["beta"],
                "rm_max": row["rm_max"],
                "rm_mean_late": row["rm_mean_late"],
                "delta_max": row["delta_max"],
            }
        )
    for row in rm_vs_beta_table:
        rows.append(
            {
                "scenario": "by_beta",
                "n": 1000,
                "beta": row["beta"],
                "rm_max": row["rm_max"],
                "rm_mean_late": row["rm_mean_late"],
                "delta_max": row["delta_max"],
            }
        )

    summary_df = pd.DataFrame(rows)
    manifest.save_table(summary_df, run_dir / "summary.csv")

    write_run_markdown_stub(run_dir)

    print(f"Run dir: {run_dir}")


if __name__ == "__main__":
    main()
