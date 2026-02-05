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
from sixbirds_cosmo.lss.masks import mask_scale_cut
from sixbirds_cosmo.lss.ppd_probe_split import (
    build_basis,
    build_surrogate_theory,
    eval_metrics,
    fit_linear_correction,
    get_probe_split_indices,
)
from sixbirds_cosmo.reporting import write_run_markdown_stub


def _unique_sorted(values: np.ndarray) -> np.ndarray:
    vals = np.unique(np.round(values, 12))
    return np.sort(vals)


def _plot_chi2(path: Path, x_vals: np.ndarray, series: dict[str, np.ndarray]) -> None:
    fig, ax = plt.subplots()
    for label, y in series.items():
        ax.plot(x_vals, y, label=label)
    ax.set_xlabel("x_min")
    ax.set_ylabel("chi2_test")
    ax.legend()
    manifest.save_fig(fig, path)
    plt.close(fig)


def _plot_params(path: Path, x_vals: np.ndarray, series: dict[str, np.ndarray]) -> None:
    fig, ax = plt.subplots()
    for label, y in series.items():
        ax.plot(x_vals, y, label=label)
    ax.set_xlabel("x_min")
    ax.set_ylabel("parameter")
    ax.legend()
    manifest.save_fig(fig, path)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/lss/y3_scale_cut_sweep.yaml")
    args = parser.parse_args()

    raw_cfg = load_yaml_config(Path(args.config))
    resolved = load_and_resolve_lss_config(Path(args.config))
    resolved["sweep"] = raw_cfg.get("sweep", {})
    resolved["surrogate_theory"] = raw_cfg.get("surrogate_theory", {})
    resolved["ppd"] = raw_cfg.get("ppd", {})

    seed = resolved.get("seed", 0)
    exp_name = resolved.get("run", {}).get("exp_name", "y3_scale_cut_sweep")
    run_dir = manifest.create_run_dir(exp_name, seed=seed)

    manifest.write_config(run_dir, resolved)

    fits_path = Path(resolved["dataset"]["resolved_path"])
    data = load_des_y3_2pt_fits(fits_path)
    block_index = build_y3_block_index(fits_path)

    y_full = data["data_vector"]
    cov_full = data["cov"]

    x_all = block_index["x"].to_numpy(dtype=float)
    finite_x = x_all[np.isfinite(x_all)]
    sweep_cfg = resolved.get("sweep", {})
    quantiles = sweep_cfg.get("quantiles", [0.0])
    min_points = int(sweep_cfg.get("min_points_per_probe", 50))
    force_zero = bool(sweep_cfg.get("force_include_zero", True))

    cuts = np.array([np.quantile(finite_x, q) for q in quantiles], dtype=float)
    if force_zero:
        cuts = np.append(cuts, 0.0)
    cuts = _unique_sorted(cuts)

    surrogate = resolved.get("surrogate_theory", {})
    frac_sigma = float(surrogate.get("frac_sigma", 0.1))
    beta_x = float(surrogate.get("beta_x", 0.0))
    kappa_probe = dict(surrogate.get("kappa_probe", {}))

    model_list = resolved.get("ppd", {}).get("models", [])
    if not model_list:
        model_list = [
            {"name": "lcdm_like", "k": 1},
            {"name": "rewrite_like", "k": 2},
        ]

    rows = []
    x_used = []

    for x_min in cuts:
        rules = {
            ("shear", "xip"): {"x_min": float(x_min), "x_max": float(np.inf)},
            ("shear", "xim"): {"x_min": float(x_min), "x_max": float(np.inf)},
            ("clustering", "wtheta"): {"x_min": float(x_min), "x_max": float(np.inf)},
            ("ggl", "gammat"): {"x_min": float(x_min), "x_max": float(np.inf)},
        }
        kept = mask_scale_cut(block_index, rules=rules)
        kept = np.sort(np.asarray(kept, dtype=int))
        if kept.size == 0:
            continue

        y = y_full[kept]
        cov = cov_full[np.ix_(kept, kept)]

        block_sub = block_index.loc[block_index["i"].isin(kept)].copy()
        block_sub = block_sub.sort_values("i").reset_index(drop=True)
        block_sub["i_orig"] = block_sub["i"]
        block_sub["i"] = np.arange(len(block_sub), dtype=int)

        idx_A, idx_B = get_probe_split_indices(block_sub)
        if idx_A.size < min_points or idx_B.size < min_points:
            continue

        t0 = build_surrogate_theory(
            y,
            cov,
            block_sub,
            frac_sigma=frac_sigma,
            beta_x=beta_x,
            kappa_probe=kappa_probe,
        )

        for model_cfg in model_list:
            model = model_cfg.get("name", "lcdm_like")
            k = int(model_cfg.get("k", 1))
            B = build_basis(y, cov, block_sub, model=model)

            # A -> B
            p_hat_A, chi2_A = fit_linear_correction(y, cov, t0, B, idx_A)
            t_A = t0 + B @ p_hat_A
            metrics_B = eval_metrics(y, cov, t_A, idx_B)
            rows.append(
                {
                    "x_min": float(x_min),
                    "n_total": int(len(y)),
                    "n_A": int(idx_A.size),
                    "n_B": int(idx_B.size),
                    "direction": "A_to_B",
                    "model": model,
                    "k": k,
                    "chi2_train": chi2_A,
                    "chi2_test": metrics_B["chi2"],
                    "chi2_test_over_n": metrics_B["chi2_over_n"],
                    "rmse_test": metrics_B["rmse"],
                    "rmse_weighted_test": metrics_B["rmse_weighted"],
                    "bias_test": metrics_B["bias"],
                    "p0": float(p_hat_A[0]) if p_hat_A.size > 0 else np.nan,
                    "p1": float(p_hat_A[1]) if p_hat_A.size > 1 else np.nan,
                }
            )

            # B -> A
            p_hat_B, chi2_B = fit_linear_correction(y, cov, t0, B, idx_B)
            t_B = t0 + B @ p_hat_B
            metrics_A = eval_metrics(y, cov, t_B, idx_A)
            rows.append(
                {
                    "x_min": float(x_min),
                    "n_total": int(len(y)),
                    "n_A": int(idx_A.size),
                    "n_B": int(idx_B.size),
                    "direction": "B_to_A",
                    "model": model,
                    "k": k,
                    "chi2_train": chi2_B,
                    "chi2_test": metrics_A["chi2"],
                    "chi2_test_over_n": metrics_A["chi2_over_n"],
                    "rmse_test": metrics_A["rmse"],
                    "rmse_weighted_test": metrics_A["rmse_weighted"],
                    "bias_test": metrics_A["bias"],
                    "p0": float(p_hat_B[0]) if p_hat_B.size > 0 else np.nan,
                    "p1": float(p_hat_B[1]) if p_hat_B.size > 1 else np.nan,
                }
            )

        x_used.append(float(x_min))

    if not rows:
        raise AssertionError("No sweep points satisfied min_points_per_probe.")

    summary = pd.DataFrame(rows)
    summary_path = run_dir / "sweep_summary.csv"
    summary.to_csv(summary_path, index=False)

    x_used_arr = np.array(sorted(set(x_used)), dtype=float)

    # Plot chi2_test vs x_min for A->B and B->A
    for direction, plot_name in [
        ("A_to_B", "chi2_test_vs_xmin_AtoB.png"),
        ("B_to_A", "chi2_test_vs_xmin_BtoA.png"),
    ]:
        series = {}
        for model in [m.get("name") for m in model_list]:
            subset = summary[(summary["direction"] == direction) & (summary["model"] == model)]
            series[model] = subset.sort_values("x_min")["chi2_test"].to_numpy(dtype=float)
        _plot_chi2(run_dir / plot_name, x_used_arr, series)

    # Plot parameters vs x_min
    series = {}
    for model in [m.get("name") for m in model_list]:
        subset = summary[(summary["direction"] == "A_to_B") & (summary["model"] == model)]
        series[f"{model}_p0"] = subset.sort_values("x_min")["p0"].to_numpy(dtype=float)
        if model == "rewrite_like":
            series[f"{model}_p1"] = subset.sort_values("x_min")["p1"].to_numpy(dtype=float)
    _plot_params(run_dir / "params_vs_xmin.png", x_used_arr, series)

    metrics = {
        "n_data_full": int(y_full.size),
        "x_min_values_used": x_used_arr.tolist(),
    }

    nonflat = False
    for direction in ["A_to_B", "B_to_A"]:
        for model in [m.get("name") for m in model_list]:
            subset = summary[(summary["direction"] == direction) & (summary["model"] == model)]
            chis = subset["chi2_test"].to_numpy(dtype=float)
            if chis.size:
                delta = float(np.max(chis) - np.min(chis))
                if delta > 1e-4:
                    nonflat = True
                metrics[f"delta_chi2_{direction}_{model}"] = delta
                min_idx = int(np.argmin(chis))
                xmin_min = float(subset.sort_values("x_min")["x_min"].iloc[min_idx])
                metrics[f"xmin_min_chi2_{direction}_{model}"] = xmin_min
                metrics[f"chi2_min_{direction}_{model}"] = float(np.min(chis))

    metrics["nonflat_check_passed"] = nonflat
    if not nonflat:
        raise AssertionError("Nonflatness check failed: chi2 curves are flat within threshold.")

    manifest.write_metrics(run_dir, metrics)
    manifest.write_provenance(run_dir, exp_name, seed=seed)
    write_run_markdown_stub(run_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
