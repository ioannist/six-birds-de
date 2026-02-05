import argparse
from dataclasses import dataclass
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
)
from sixbirds_cosmo.reporting import write_run_markdown_stub


@dataclass
class Scenario:
    scenario_id: int
    split_kind: str
    boot_id: int
    cov_jitter_eps: float
    frac_sigma: float
    beta_x: float


def _block_key(row: pd.Series) -> str:
    b1 = row.get("bin1")
    b2 = row.get("bin2")
    def fmt(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "nan"
        return str(int(v)) if isinstance(v, (int, np.integer, float)) else str(v)
    return f"{row.get('probe')}|{row.get('stat')}|{fmt(b1)}|{fmt(b2)}"


def _split_indices(block_sub: pd.DataFrame, split_kind: str) -> tuple[np.ndarray, np.ndarray]:
    if split_kind == "shear_vs_rest":
        idx_A = block_sub.loc[block_sub["probe"] == "shear", "i"].to_numpy(dtype=int)
        idx_B = block_sub.loc[
            block_sub["probe"].isin(["clustering", "ggl"]), "i"
        ].to_numpy(dtype=int)
        return np.sort(idx_A), np.sort(idx_B)

    bin1 = block_sub["bin1"].to_numpy(dtype=float)
    bin2 = block_sub["bin2"].to_numpy(dtype=float)
    b = np.nanmax(np.vstack([bin1, bin2]), axis=0)
    b = np.where(np.isfinite(b), b, 0)

    if split_kind == "evenodd_bins":
        idx_A = block_sub.loc[(b % 2 == 0), "i"].to_numpy(dtype=int)
        idx_B = block_sub.loc[(b % 2 == 1), "i"].to_numpy(dtype=int)
        return np.sort(idx_A), np.sort(idx_B)

    if split_kind == "lowhigh_bins":
        idx_A = block_sub.loc[(b <= 1), "i"].to_numpy(dtype=int)
        idx_B = block_sub.loc[(b >= 2), "i"].to_numpy(dtype=int)
        return np.sort(idx_A), np.sort(idx_B)

    raise ValueError(f"Unknown split_kind: {split_kind}")


def _sample_scenarios(seed: int, splits, n_boot, jitters, frac_list, beta_list, max_scenarios):
    scenarios = []
    scenario_id = 0
    # ensure each split has at least one scenario
    for split in splits:
        scenarios.append(
            Scenario(
                scenario_id=scenario_id,
                split_kind=split,
                boot_id=0,
                cov_jitter_eps=jitters[0],
                frac_sigma=frac_list[0],
                beta_x=beta_list[0],
            )
        )
        scenario_id += 1

    full = []
    for split in splits:
        for boot_id in range(n_boot):
            for jitter in jitters:
                for frac_sigma in frac_list:
                    for beta_x in beta_list:
                        full.append((split, boot_id, jitter, frac_sigma, beta_x))

    # remove the ones already included
    existing = {(s.split_kind, s.boot_id, s.cov_jitter_eps, s.frac_sigma, s.beta_x) for s in scenarios}
    full = [item for item in full if item not in existing]

    rng = np.random.default_rng(seed)
    remaining_slots = max(0, max_scenarios - len(scenarios))
    if remaining_slots > 0 and len(full) > 0:
        pick = rng.choice(len(full), size=min(remaining_slots, len(full)), replace=False)
        for idx in pick:
            split, boot_id, jitter, frac_sigma, beta_x = full[idx]
            scenarios.append(
                Scenario(
                    scenario_id=scenario_id,
                    split_kind=split,
                    boot_id=boot_id,
                    cov_jitter_eps=jitter,
                    frac_sigma=frac_sigma,
                    beta_x=beta_x,
                )
            )
            scenario_id += 1

    return scenarios, len(full) + len(existing)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/lss/y3_ppd_robustness.yaml")
    args = parser.parse_args()

    raw_cfg = load_yaml_config(Path(args.config))
    resolved = load_and_resolve_lss_config(Path(args.config))
    resolved["robustness"] = raw_cfg.get("robustness", {})
    resolved["surrogate_theory"] = raw_cfg.get("surrogate_theory", {})
    resolved["models"] = raw_cfg.get("models", [])
    resolved["ppd"] = raw_cfg.get("ppd", {})

    seed = resolved.get("seed", 0)
    exp_name = resolved.get("run", {}).get("exp_name", "y3_ppd_robustness")
    run_dir = manifest.create_run_dir(exp_name, seed=seed)

    manifest.write_config(run_dir, resolved)

    fits_path = Path(resolved["dataset"]["resolved_path"])
    data = load_des_y3_2pt_fits(fits_path)
    block_index = build_y3_block_index(fits_path)

    y_full = data["data_vector"]
    cov_full = data["cov"]

    rob = resolved["robustness"]
    splits = rob.get("splits", ["shear_vs_rest"])
    n_boot = int(rob.get("n_boot", 1))
    block_keep_frac = float(rob.get("block_keep_frac", 0.8))
    min_points_train = int(rob.get("min_points_train", 80))
    min_points_test = int(rob.get("min_points_test", 80))
    jitters = [float(j) for j in rob.get("cov_jitter_eps", [0.0])]
    hyper = rob.get("hyper_sweep", {})
    frac_list = [float(x) for x in hyper.get("frac_sigma", [0.1])]
    beta_list = [float(x) for x in hyper.get("beta_x", [0.1])]
    max_scenarios = int(rob.get("max_scenarios", 30))

    scenarios, n_scenarios_requested = _sample_scenarios(
        seed, splits, n_boot, jitters, frac_list, beta_list, max_scenarios
    )

    rng = np.random.default_rng(seed)

    # block keys
    block_keys = block_index.apply(_block_key, axis=1)
    unique_blocks = np.unique(block_keys)

    rows = []
    n_ok = 0
    n_skipped = 0
    n_failed = 0

    models = resolved.get("models", [])
    if not models:
        models = [
            {"name": "lcdm_like", "k": 1},
            {"name": "rewrite_like", "k": 2},
        ]

    directions = resolved.get("ppd", {}).get("directions", ["AtoB", "BtoA"])

    for scenario in scenarios:
        # subsample blocks
        n_blocks_total = len(unique_blocks)
        n_keep = max(1, int(np.ceil(block_keep_frac * n_blocks_total)))
        keep_blocks = rng.choice(unique_blocks, size=n_keep, replace=False)
        keep_mask = np.isin(block_keys, keep_blocks)

        y = y_full[keep_mask]
        cov = cov_full[np.ix_(keep_mask, keep_mask)]
        block_sub = block_index.loc[keep_mask].copy().reset_index(drop=True)
        block_sub["i"] = np.arange(len(block_sub), dtype=int)

        # jitter
        diag_med = float(np.median(np.diag(cov))) if cov.size else 0.0
        jitter_abs = scenario.cov_jitter_eps * diag_med
        cov_pert = cov + jitter_abs * np.eye(cov.shape[0])

        try:
            idx_A, idx_B = _split_indices(block_sub, scenario.split_kind)
        except Exception as exc:
            for direction in directions:
                for model in models:
                    rows.append(
                        {
                            "scenario_id": scenario.scenario_id,
                            "split_kind": scenario.split_kind,
                            "boot_id": scenario.boot_id,
                            "cov_jitter_eps": scenario.cov_jitter_eps,
                            "frac_sigma": scenario.frac_sigma,
                            "beta_x": scenario.beta_x,
                            "n_total": int(len(block_sub)),
                            "n_A": 0,
                            "n_B": 0,
                            "direction": direction,
                            "model": model["name"],
                            "k": model["k"],
                            "chi2_train": np.nan,
                            "chi2_test": np.nan,
                            "chi2_test_over_n": np.nan,
                            "rmse_test": np.nan,
                            "rmse_weighted_test": np.nan,
                            "bias_test": np.nan,
                            "p0": np.nan,
                            "p1": np.nan,
                            "status": "failed",
                            "error": f"split_error: {exc}",
                        }
                    )
            n_failed += 1
            continue

        if idx_A.size < min_points_train or idx_B.size < min_points_test:
            for direction in directions:
                for model in models:
                    rows.append(
                        {
                            "scenario_id": scenario.scenario_id,
                            "split_kind": scenario.split_kind,
                            "boot_id": scenario.boot_id,
                            "cov_jitter_eps": scenario.cov_jitter_eps,
                            "frac_sigma": scenario.frac_sigma,
                            "beta_x": scenario.beta_x,
                            "n_total": int(len(block_sub)),
                            "n_A": int(idx_A.size),
                            "n_B": int(idx_B.size),
                            "direction": direction,
                            "model": model["name"],
                            "k": model["k"],
                            "chi2_train": np.nan,
                            "chi2_test": np.nan,
                            "chi2_test_over_n": np.nan,
                            "rmse_test": np.nan,
                            "rmse_weighted_test": np.nan,
                            "bias_test": np.nan,
                            "p0": np.nan,
                            "p1": np.nan,
                            "status": "skipped",
                            "error": "insufficient points",
                        }
                    )
            n_skipped += 1
            continue

        try:
            t0 = build_surrogate_theory(
                y,
                cov_pert,
                block_sub,
                frac_sigma=scenario.frac_sigma,
                beta_x=scenario.beta_x,
                kappa_probe=resolved["surrogate_theory"].get("kappa_probe", {}),
            )

            for direction in directions:
                if direction == "AtoB":
                    idx_train, idx_test = idx_A, idx_B
                else:
                    idx_train, idx_test = idx_B, idx_A

                for model in models:
                    B = build_basis(y, cov_pert, block_sub, model=model["name"])
                    p_hat, chi2_train = fit_linear_correction(y, cov_pert, t0, B, idx_train)
                    t = t0 + B @ p_hat
                    metrics = eval_metrics(y, cov_pert, t, idx_test)

                    rows.append(
                        {
                            "scenario_id": scenario.scenario_id,
                            "split_kind": scenario.split_kind,
                            "boot_id": scenario.boot_id,
                            "cov_jitter_eps": scenario.cov_jitter_eps,
                            "frac_sigma": scenario.frac_sigma,
                            "beta_x": scenario.beta_x,
                            "n_total": int(len(block_sub)),
                            "n_A": int(idx_A.size),
                            "n_B": int(idx_B.size),
                            "direction": direction,
                            "model": model["name"],
                            "k": model["k"],
                            "chi2_train": chi2_train,
                            "chi2_test": metrics["chi2"],
                            "chi2_test_over_n": metrics["chi2_over_n"],
                            "rmse_test": metrics["rmse"],
                            "rmse_weighted_test": metrics["rmse_weighted"],
                            "bias_test": metrics["bias"],
                            "p0": float(p_hat[0]) if p_hat.size > 0 else np.nan,
                            "p1": float(p_hat[1]) if p_hat.size > 1 else np.nan,
                            "status": "ok",
                            "error": "",
                        }
                    )

            n_ok += 1

        except Exception as exc:
            for direction in directions:
                for model in models:
                    rows.append(
                        {
                            "scenario_id": scenario.scenario_id,
                            "split_kind": scenario.split_kind,
                            "boot_id": scenario.boot_id,
                            "cov_jitter_eps": scenario.cov_jitter_eps,
                            "frac_sigma": scenario.frac_sigma,
                            "beta_x": scenario.beta_x,
                            "n_total": int(len(block_sub)),
                            "n_A": int(idx_A.size),
                            "n_B": int(idx_B.size),
                            "direction": direction,
                            "model": model["name"],
                            "k": model["k"],
                            "chi2_train": np.nan,
                            "chi2_test": np.nan,
                            "chi2_test_over_n": np.nan,
                            "rmse_test": np.nan,
                            "rmse_weighted_test": np.nan,
                            "bias_test": np.nan,
                            "p0": np.nan,
                            "p1": np.nan,
                            "status": "failed",
                            "error": str(exc),
                        }
                    )
            n_failed += 1

    runs_df = pd.DataFrame(rows)
    runs_path = run_dir / "robustness_runs.csv"
    runs_df.to_csv(runs_path, index=False)

    # summary
    ok_df = runs_df[runs_df["status"] == "ok"].copy()
    summary_rows = []
    for split_kind in splits:
        for direction in directions:
            subset = ok_df[(ok_df["split_kind"] == split_kind) & (ok_df["direction"] == direction)]
            if subset.empty:
                summary_rows.append(
                    {
                        "split_kind": split_kind,
                        "direction": direction,
                        "median_chi2_lcdm": np.nan,
                        "median_chi2_rewrite": np.nan,
                        "median_improvement": np.nan,
                        "iqr_improvement": np.nan,
                        "n_ok": 0,
                        "n_skipped": int(n_skipped),
                        "n_failed": int(n_failed),
                    }
                )
                continue
            lcdm = subset[subset["model"] == "lcdm_like"]
            rw = subset[subset["model"] == "rewrite_like"]
            # align by scenario_id
            merged = pd.merge(
                lcdm,
                rw,
                on=["scenario_id", "split_kind", "direction"],
                suffixes=("_lcdm", "_rw"),
            )
            improvement = merged["chi2_test_lcdm"] - merged["chi2_test_rw"]
            summary_rows.append(
                {
                    "split_kind": split_kind,
                    "direction": direction,
                    "median_chi2_lcdm": float(lcdm["chi2_test"].median()),
                    "median_chi2_rewrite": float(rw["chi2_test"].median()),
                    "median_improvement": float(improvement.median()) if len(improvement) else np.nan,
                    "iqr_improvement": float(improvement.quantile(0.75) - improvement.quantile(0.25))
                    if len(improvement)
                    else np.nan,
                    "n_ok": int(len(merged)),
                    "n_skipped": int(n_skipped),
                    "n_failed": int(n_failed),
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = run_dir / "robustness_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    # canonical histogram
    canonical = ok_df[(ok_df["split_kind"] == "shear_vs_rest") & (ok_df["direction"] == "AtoB")]
    lcdm = canonical[canonical["model"] == "lcdm_like"]
    rw = canonical[canonical["model"] == "rewrite_like"]
    merged = pd.merge(
        lcdm,
        rw,
        on=["scenario_id", "split_kind", "direction"],
        suffixes=("_lcdm", "_rw"),
    )
    improvement = merged["chi2_test_lcdm"] - merged["chi2_test_rw"]

    fig, ax = plt.subplots()
    ax.hist(improvement.to_numpy(dtype=float), bins=10)
    ax.set_xlabel("chi2_test_lcdm - chi2_test_rewrite")
    ax.set_ylabel("count")
    manifest.save_fig(fig, run_dir / "improvement_hist.png")
    plt.close(fig)

    metrics = {
        "n_scenarios_requested": int(n_scenarios_requested),
        "n_scenarios_run": int(len(scenarios)),
        "n_ok": int(len(ok_df["scenario_id"].unique())),
        "n_skipped": int(n_skipped),
        "n_failed": int(n_failed),
        "canonical_split": "shear_vs_rest",
        "canonical_direction": "AtoB",
        "improvement_median_canonical": float(improvement.median()) if len(improvement) else np.nan,
        "improvement_iqr_canonical": float(improvement.quantile(0.75) - improvement.quantile(0.25))
        if len(improvement)
        else np.nan,
        "improvement_n_canonical": int(len(improvement)),
        "max_scenarios": int(max_scenarios),
        "sampling_applied": len(scenarios) < n_scenarios_requested,
    }

    manifest.write_metrics(run_dir, metrics)
    manifest.write_provenance(run_dir, exp_name, seed=seed)
    write_run_markdown_stub(run_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
