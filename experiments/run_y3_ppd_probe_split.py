import argparse
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from sixbirds_cosmo import manifest
from sixbirds_cosmo.datasets.des_y3_2pt import build_y3_block_index, load_des_y3_2pt_fits
from sixbirds_cosmo.lss.config import load_and_resolve_lss_config, load_yaml_config
from sixbirds_cosmo.lss.ppd_probe_split import (
    build_basis,
    build_surrogate_theory,
    eval_metrics,
    fit_linear_correction,
    get_probe_split_indices,
)
from sixbirds_cosmo.reporting import write_run_markdown_stub


def _save_residuals_csv(path: Path, block: pd.DataFrame, t: np.ndarray, y: np.ndarray) -> None:
    df = block[["i", "probe", "stat", "x", "value"]].copy()
    df["theory"] = t
    df["residual"] = y - t
    df.to_csv(path, index=False)


def _plot_residuals(path: Path, label_a: str, resid_a: np.ndarray, label_b: str, resid_b: np.ndarray) -> None:
    fig, ax = plt.subplots()
    ax.plot(np.arange(resid_a.size), resid_a, label=label_a)
    ax.plot(np.arange(resid_b.size), resid_b, label=label_b)
    ax.set_xlabel("index")
    ax.set_ylabel("residual")
    ax.legend()
    manifest.save_fig(fig, path)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/lss/y3_ppd_probe_split.yaml")
    args = parser.parse_args()

    raw_cfg = load_yaml_config(Path(args.config))
    resolved = load_and_resolve_lss_config(Path(args.config))
    resolved["ppd"] = raw_cfg.get("ppd", {})
    resolved["surrogate_theory"] = raw_cfg.get("surrogate_theory", {})
    resolved["rewrite_bridge"] = raw_cfg.get("rewrite_bridge", {})

    seed = resolved.get("seed", 0)
    exp_name = resolved.get("run", {}).get("exp_name", "y3_ppd_probe_split")
    run_dir = manifest.create_run_dir(exp_name, seed=seed)

    manifest.write_config(run_dir, resolved)

    fits_path = Path(resolved["dataset"]["resolved_path"])
    data = load_des_y3_2pt_fits(fits_path)
    block_index = build_y3_block_index(fits_path)

    y = data["data_vector"]
    cov = data["cov"]

    idx_A, idx_B = get_probe_split_indices(block_index)
    n_data = int(y.size)
    n_A = int(idx_A.size)
    n_B = int(idx_B.size)

    surrogate = resolved.get("surrogate_theory", {})
    frac_sigma = float(surrogate.get("frac_sigma", 0.1))
    beta_x = float(surrogate.get("beta_x", 0.0))
    kappa_probe = dict(surrogate.get("kappa_probe", {}))

    t0 = build_surrogate_theory(
        y,
        cov,
        block_index,
        frac_sigma=frac_sigma,
        beta_x=beta_x,
        kappa_probe=kappa_probe,
    )

    results = []

    for model in ["lcdm_like", "rewrite_like"]:
        B = build_basis(y, cov, block_index, model=model)

        # Train on A, test on B
        p_hat_A, chi2_A = fit_linear_correction(y, cov, t0, B, idx_A)
        t_A = t0 + B @ p_hat_A
        metrics_B = eval_metrics(y, cov, t_A, idx_B)
        results.append(
            {
                "direction": "A_to_B",
                "model": model,
                "chi2_train": chi2_A,
                "chi2_test": metrics_B["chi2"],
                "chi2_over_n_test": metrics_B["chi2_over_n"],
                "rmse_test": metrics_B["rmse"],
                "rmse_weighted_test": metrics_B["rmse_weighted"],
                "bias_test": metrics_B["bias"],
            }
        )

        # Train on B, test on A
        p_hat_B, chi2_B = fit_linear_correction(y, cov, t0, B, idx_B)
        t_B = t0 + B @ p_hat_B
        metrics_A = eval_metrics(y, cov, t_B, idx_A)
        results.append(
            {
                "direction": "B_to_A",
                "model": model,
                "chi2_train": chi2_B,
                "chi2_test": metrics_A["chi2"],
                "chi2_over_n_test": metrics_A["chi2_over_n"],
                "rmse_test": metrics_A["rmse"],
                "rmse_weighted_test": metrics_A["rmse_weighted"],
                "bias_test": metrics_A["bias"],
            }
        )

    summary = pd.DataFrame(results)
    summary_path = run_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)

    # residual CSVs and plots for A->B and B->A
    block_A = block_index.loc[block_index["i"].isin(idx_A)].copy().sort_values("i")
    block_B = block_index.loc[block_index["i"].isin(idx_B)].copy().sort_values("i")

    # For A->B
    B_lcdm = build_basis(y, cov, block_index, model="lcdm_like")
    p_hat_A_lcdm, _ = fit_linear_correction(y, cov, t0, B_lcdm, idx_A)
    t_A_lcdm = t0 + B_lcdm @ p_hat_A_lcdm
    resid_B_lcdm = (y - t_A_lcdm)[idx_B]

    B_rw = build_basis(y, cov, block_index, model="rewrite_like")
    p_hat_A_rw, _ = fit_linear_correction(y, cov, t0, B_rw, idx_A)
    t_A_rw = t0 + B_rw @ p_hat_A_rw
    resid_B_rw = (y - t_A_rw)[idx_B]

    _save_residuals_csv(run_dir / "residuals_trainA_testB.csv", block_B, t_A_lcdm[idx_B], y[idx_B])
    _plot_residuals(
        run_dir / "ppd_residuals_testB.png",
        "lcdm_like",
        resid_B_lcdm,
        "rewrite_like",
        resid_B_rw,
    )

    # For B->A
    p_hat_B_lcdm, _ = fit_linear_correction(y, cov, t0, B_lcdm, idx_B)
    t_B_lcdm = t0 + B_lcdm @ p_hat_B_lcdm
    resid_A_lcdm = (y - t_B_lcdm)[idx_A]

    p_hat_B_rw, _ = fit_linear_correction(y, cov, t0, B_rw, idx_B)
    t_B_rw = t0 + B_rw @ p_hat_B_rw
    resid_A_rw = (y - t_B_rw)[idx_A]

    _save_residuals_csv(run_dir / "residuals_trainB_testA.csv", block_A, t_B_lcdm[idx_A], y[idx_A])
    _plot_residuals(
        run_dir / "ppd_residuals_testA.png",
        "lcdm_like",
        resid_A_lcdm,
        "rewrite_like",
        resid_A_rw,
    )

    metrics = {
        "n_data": n_data,
        "n_A": n_A,
        "n_B": n_B,
        "surrogate_frac_sigma": frac_sigma,
        "beta_x": beta_x,
        "kappa_probe": kappa_probe,
        "rewrite_bridge_w0": resolved.get("rewrite_bridge", {}).get("w0"),
        "rewrite_bridge_wa": resolved.get("rewrite_bridge", {}).get("wa"),
    }
    # Add per-direction metrics
    for row in results:
        key = f"{row['direction']}_{row['model']}"
        metrics[f"{key}_chi2_train"] = row["chi2_train"]
        metrics[f"{key}_chi2_test"] = row["chi2_test"]
        metrics[f"{key}_chi2_over_n_test"] = row["chi2_over_n_test"]
        metrics[f"{key}_rmse_test"] = row["rmse_test"]
        metrics[f"{key}_rmse_weighted_test"] = row["rmse_weighted_test"]
        metrics[f"{key}_bias_test"] = row["bias_test"]

    manifest.write_metrics(run_dir, metrics)
    manifest.write_provenance(run_dir, exp_name, seed=seed)
    write_run_markdown_stub(run_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
