import argparse
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from sixbirds_cosmo import manifest
from sixbirds_cosmo.data_registry import fetch
from sixbirds_cosmo.datasets.kids450_vuitert18 import load_kids450_vuitert18_bandpowers
from sixbirds_cosmo.lss.config import load_and_resolve_lss_config, load_yaml_config
from sixbirds_cosmo.lss.ppd_probe_split import (
    build_basis,
    build_surrogate_theory,
    eval_metrics,
    fit_linear_correction,
)
from sixbirds_cosmo.reporting import write_run_markdown_stub


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
    parser.add_argument("--config", default="configs/lss/kids450_ppd_probe_split.yaml")
    args = parser.parse_args()

    raw_cfg = load_yaml_config(Path(args.config))
    resolved = load_and_resolve_lss_config(Path(args.config))
    resolved["ppd"] = raw_cfg.get("ppd", {})
    resolved["surrogate_theory"] = raw_cfg.get("surrogate_theory", {})

    seed = resolved.get("seed", 0)
    exp_name = resolved.get("run", {}).get("exp_name", "kids450_ppd_probe_split")
    run_dir = manifest.create_run_dir(exp_name, seed=seed)

    manifest.write_config(run_dir, resolved)

    raw_dir = Path(fetch(resolved["dataset"]["key"]))
    data = load_kids450_vuitert18_bandpowers(raw_dir)
    y = data["data_vector"]
    cov = data["cov"]
    block_index = data["block_index"]

    idx_A = block_index.loc[block_index["probe"] == "shear", "i"].to_numpy(dtype=int)
    idx_B = block_index.loc[block_index["probe"].isin(["ggl", "clustering"]), "i"].to_numpy(
        dtype=int
    )

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

        # A -> B
        p_hat_A, chi2_A = fit_linear_correction(y, cov, t0, B, idx_A)
        t_A = t0 + B @ p_hat_A
        metrics_B = eval_metrics(y, cov, t_A, idx_B)
        results.append(
            {
                "direction": "A_to_B",
                "model": model,
                "chi2_train": chi2_A,
                "chi2_test": metrics_B["chi2"],
                "chi2_test_over_n": metrics_B["chi2_over_n"],
                "rmse_test": metrics_B["rmse"],
                "rmse_weighted_test": metrics_B["rmse_weighted"],
                "bias_test": metrics_B["bias"],
            }
        )

        # B -> A
        p_hat_B, chi2_B = fit_linear_correction(y, cov, t0, B, idx_B)
        t_B = t0 + B @ p_hat_B
        metrics_A = eval_metrics(y, cov, t_B, idx_A)
        results.append(
            {
                "direction": "B_to_A",
                "model": model,
                "chi2_train": chi2_B,
                "chi2_test": metrics_A["chi2"],
                "chi2_test_over_n": metrics_A["chi2_over_n"],
                "rmse_test": metrics_A["rmse"],
                "rmse_weighted_test": metrics_A["rmse_weighted"],
                "bias_test": metrics_A["bias"],
            }
        )

    summary = pd.DataFrame(results)
    summary_path = run_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)

    # residual plots
    B_lcdm = build_basis(y, cov, block_index, model="lcdm_like")
    p_hat_A_lcdm, _ = fit_linear_correction(y, cov, t0, B_lcdm, idx_A)
    t_A_lcdm = t0 + B_lcdm @ p_hat_A_lcdm
    resid_B_lcdm = (y - t_A_lcdm)[idx_B]

    B_rw = build_basis(y, cov, block_index, model="rewrite_like")
    p_hat_A_rw, _ = fit_linear_correction(y, cov, t0, B_rw, idx_A)
    t_A_rw = t0 + B_rw @ p_hat_A_rw
    resid_B_rw = (y - t_A_rw)[idx_B]

    _plot_residuals(
        run_dir / "ppd_residuals_testB.png",
        "lcdm_like",
        resid_B_lcdm,
        "rewrite_like",
        resid_B_rw,
    )

    p_hat_B_lcdm, _ = fit_linear_correction(y, cov, t0, B_lcdm, idx_B)
    t_B_lcdm = t0 + B_lcdm @ p_hat_B_lcdm
    resid_A_lcdm = (y - t_B_lcdm)[idx_A]

    p_hat_B_rw, _ = fit_linear_correction(y, cov, t0, B_rw, idx_B)
    t_B_rw = t0 + B_rw @ p_hat_B_rw
    resid_A_rw = (y - t_B_rw)[idx_A]

    _plot_residuals(
        run_dir / "ppd_residuals_testA.png",
        "lcdm_like",
        resid_A_lcdm,
        "rewrite_like",
        resid_A_rw,
    )

    metrics = {
        "dataset_name": "kids450_vuitert18_3x2pt_bandpowers",
        "n_data": int(y.size),
        "n_A": int(idx_A.size),
        "n_B": int(idx_B.size),
        "ell_bins": data["meta"]["ell_bins"],
        "cov_chol_jitter_used": data["meta"]["cov_chol_jitter_used"],
    }

    for row in results:
        key = f"{row['direction']}_{row['model']}"
        metrics[f"{key}_chi2_test"] = row["chi2_test"]
        metrics[f"{key}_rmse_test"] = row["rmse_test"]
        metrics[f"{key}_rmse_weighted_test"] = row["rmse_weighted_test"]
        metrics[f"{key}_bias_test"] = row["bias_test"]

    manifest.write_metrics(run_dir, metrics)
    manifest.write_provenance(run_dir, exp_name, seed=seed)
    write_run_markdown_stub(run_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
