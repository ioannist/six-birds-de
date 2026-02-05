import argparse
import math
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import minimize  # noqa: E402
from scipy.linalg import cho_factor, cho_solve  # noqa: E402

from sixbirds_cosmo import manifest
from sixbirds_cosmo.cosmo_background import distance_modulus, transverse_comoving_distance
from sixbirds_cosmo.data_registry import fetch
from sixbirds_cosmo.datasets.des_sn5yr import (
    chi2_profiled_over_M,
    load_des_sn5yr_covariance,
    load_des_sn5yr_hubble_diagram,
    prepare_cov_solve,
)
from sixbirds_cosmo.datasets.des_y3_2pt import build_y3_block_index, load_des_y3_2pt_fits
from sixbirds_cosmo.datasets.des_y6_bao import load_alpha_likelihood_curve, load_bao_fiducial_info
from sixbirds_cosmo.lss.config import load_and_resolve_lss_config, load_yaml_config
from sixbirds_cosmo.lss.ppd_probe_split import get_probe_split_indices
from sixbirds_cosmo.reporting import write_run_markdown_stub
from sixbirds_cosmo.rewrite_background import (
    distance_modulus_rewrite,
    transverse_comoving_distance_rewrite,
    H_rewrite,
)
from sixbirds_cosmo.rewrite_effective_w import fit_effective_w0wa


def _zscore(x: np.ndarray) -> np.ndarray:
    finite = np.isfinite(x)
    if not np.any(finite):
        return np.zeros_like(x, dtype=float)
    xv = x[finite]
    mean = float(np.mean(xv))
    std = float(np.std(xv))
    if std == 0:
        return np.zeros_like(x, dtype=float)
    z = np.zeros_like(x, dtype=float)
    z[finite] = (xv - mean) / std
    return z


def _combine_alpha_curves(curves: list[dict]) -> dict:
    alpha_all = np.unique(np.concatenate([c["alpha_grid"] for c in curves]))
    alpha_all = np.sort(alpha_all)
    total = np.zeros_like(alpha_all)
    notes = []
    for c in curves:
        interp = np.interp(alpha_all, c["alpha_grid"], c["delta_chi2_grid"])
        total += interp
        notes.append({"filename": c["filename"], **c.get("notes", {})})
    return {"alpha_grid": alpha_all, "delta_chi2_grid": total, "notes": notes}


def _z_eff_from_bins(block_index: pd.DataFrame, z0: float, dz: float) -> np.ndarray:
    bin1 = block_index["bin1"].to_numpy(dtype=float)
    bin2 = block_index["bin2"].to_numpy(dtype=float)
    finite_bins = np.isfinite(bin1) & np.isfinite(bin2)
    if not np.any(finite_bins):
        return np.full(len(block_index), z0, dtype=float)

    max_bin = int(np.nanmax([bin1, bin2]))
    min_bin = int(np.nanmin([bin1, bin2]))
    if min_bin >= 1:
        offset = -1
        n_bins = max_bin
    else:
        offset = 0
        n_bins = max_bin + 1
    z_bins = z0 + dz * np.arange(n_bins)
    z_median = float(np.median(z_bins))

    def map_bin(b):
        if not np.isfinite(b):
            return np.nan
        idx = int(b) + offset
        if idx < 0 or idx >= len(z_bins):
            return np.nan
        return z_bins[idx]

    z1 = np.array([map_bin(b) for b in bin1], dtype=float)
    z2 = np.array([map_bin(b) for b in bin2], dtype=float)
    z_eff = 0.5 * (z1 + z2)
    z_eff[~np.isfinite(z_eff)] = z_median
    return z_eff


def _compute_sn_bundle(raw_dir: Path, cov_kind: str):
    cov_tmp = load_des_sn5yr_covariance(raw_dir, kind="stat+sys")
    hd = load_des_sn5yr_hubble_diagram(raw_dir, n_cov=cov_tmp["N"])
    mu_err = hd.get("mu_err")
    cov_info = load_des_sn5yr_covariance(raw_dir, kind="stat+sys", mu_err=mu_err)
    cov = cov_info["cov"]
    cho = prepare_cov_solve(cov)["cho"]
    return hd, cov, cho


def _chi2_sn_lcdm(z, mu_obs, cho, H0, om):
    mu_theory = distance_modulus(z, H0, om, 1.0 - om)
    resid0 = mu_obs - mu_theory
    chi2, _ = chi2_profiled_over_M(resid0, cho)
    return chi2


def _chi2_sn_rewrite(z, mu_obs, cho, H0, om, A, m):
    mu_theory = distance_modulus_rewrite(z, H0, om, A, m)
    resid0 = mu_obs - mu_theory
    chi2, _ = chi2_profiled_over_M(resid0, cho)
    return chi2


def _chi2_bao(alpha_grid, delta_chi2_grid, alpha_pred):
    alpha_min = float(alpha_grid.min())
    alpha_max = float(alpha_grid.max())
    alpha_eval = float(np.clip(alpha_pred, alpha_min, alpha_max))
    chi2 = float(np.interp(alpha_eval, alpha_grid, delta_chi2_grid))
    return chi2


def _chi2_from_resid(resid: np.ndarray, cho) -> float:
    inv = cho_solve(cho, resid)
    return float(resid @ inv)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/lss/y3_ppd_with_geometry.yaml")
    args = parser.parse_args()

    raw_cfg = load_yaml_config(Path(args.config))
    resolved = load_and_resolve_lss_config(Path(args.config))
    resolved["ppd"] = raw_cfg.get("ppd", {})
    resolved["surrogate_lss"] = raw_cfg.get("surrogate_lss", {})
    resolved["models"] = raw_cfg.get("models", {})
    resolved["anchors"] = raw_cfg.get("anchors", {})

    seed = resolved.get("seed", 0)
    exp_name = resolved.get("run", {}).get("exp_name", "y3_ppd_with_geometry")
    run_dir = manifest.create_run_dir(exp_name, seed=seed)

    manifest.write_config(run_dir, resolved)

    # LSS data
    fits_path = Path(resolved["dataset"]["resolved_path"])
    data = load_des_y3_2pt_fits(fits_path)
    block_index = build_y3_block_index(fits_path)
    y = data["data_vector"]
    cov = data["cov"]

    idx_A, idx_B = get_probe_split_indices(block_index)
    n_A = int(idx_A.size)
    n_B = int(idx_B.size)
    n_data = int(y.size)

    # surrogate injection
    sur = resolved["surrogate_lss"]
    frac_sigma = float(sur.get("frac_sigma", 0.1))
    beta_x = float(sur.get("beta_x", 0.0))
    kappa_probe = dict(sur.get("kappa_probe", {}))
    p_probe = dict(sur.get("p_probe", {}))
    z0 = float(sur.get("z0", 0.3))
    dz = float(sur.get("dz", 0.2))
    fid = sur.get("fiducial", {})
    H0_lss = float(fid.get("H0", 70.0))

    sigma = np.sqrt(np.diag(cov))
    x_z = _zscore(block_index["x"].to_numpy(dtype=float))
    kappa = np.array([kappa_probe.get(p, 0.0) for p in block_index["probe"].to_numpy()])
    r0 = frac_sigma * sigma * (1.0 + kappa + beta_x * x_z)

    z_eff = _z_eff_from_bins(block_index, z0=z0, dz=dz)

    # LSS fiducials
    om_fid_lcdm = float(fid.get("lcdm", {}).get("om", 0.3))
    om_fid_rw = float(fid.get("rewrite", {}).get("om", 0.3))
    A_fid_rw = float(fid.get("rewrite", {}).get("A", 0.7))
    m_fid_rw = float(fid.get("rewrite", {}).get("m", 0.0))

    dm_fid_lcdm = transverse_comoving_distance(z_eff, H0_lss, om_fid_lcdm, 1.0 - om_fid_lcdm)
    dm_fid_rw = transverse_comoving_distance_rewrite(z_eff, H0_lss, om_fid_rw, A_fid_rw, m_fid_rw)

    # Anchor datasets
    anchors = resolved["anchors"]
    sn_cfg = anchors.get("sn", {})
    bao_cfg = anchors.get("bao", {})

    sn_raw = Path(fetch(sn_cfg.get("dataset_key", "des_sn5yr_distances")))
    sn_hd, sn_cov, sn_cho = _compute_sn_bundle(sn_raw, sn_cfg.get("cov_kind", "total"))
    z_sn = sn_hd["z"]
    mu_sn = sn_hd["mu"]
    H0_sn = float(sn_cfg.get("H0", 70.0))

    bao_raw = Path(fetch(bao_cfg.get("dataset_key", "des_y6_bao_release")))
    curves = [
        load_alpha_likelihood_curve(bao_raw, "acf"),
        load_alpha_likelihood_curve(bao_raw, "aps"),
        load_alpha_likelihood_curve(bao_raw, "pcf"),
    ]
    combined = _combine_alpha_curves(curves)
    alpha_grid = combined["alpha_grid"]
    delta_chi2_grid = combined["delta_chi2_grid"]
    z_eff_bao = load_bao_fiducial_info(bao_raw)["z_eff"]
    H0_bao = float(bao_cfg.get("H0_fid", 67.4))
    om_fid_bao = float(bao_cfg.get("om_fid", 0.315))
    ol_fid_bao = float(bao_cfg.get("ol_fid", 0.685))
    dm_fid_bao = transverse_comoving_distance(z_eff_bao, H0_bao, om_fid_bao, ol_fid_bao)

    cho_A = cho_factor(cov[np.ix_(idx_A, idx_A)], lower=True)
    cho_B = cho_factor(cov[np.ix_(idx_B, idx_B)], lower=True)

    def lss_residual(theta, model):
        if model == "lcdm":
            om = float(theta[0])
            dm = transverse_comoving_distance(z_eff, H0_lss, om, 1.0 - om)
            dm_fid = dm_fid_lcdm
        else:
            om, A, m = map(float, theta)
            dm = transverse_comoving_distance_rewrite(z_eff, H0_lss, om, A, m)
            dm_fid = dm_fid_rw
        ratio = np.maximum(dm / dm_fid, 1e-6)
        p_vec = np.array([p_probe.get(p, 1.0) for p in block_index["probe"].to_numpy()])
        s = np.power(ratio, -p_vec)
        resid = -r0 * s
        return resid

    def chi2_lss(theta, model, idx_train, cho):
        resid = lss_residual(theta, model)
        resid_train = resid[idx_train]
        return _chi2_from_resid(resid_train, cho)

    def chi2_sn(theta, model):
        if model == "lcdm":
            om = float(theta[0])
            return _chi2_sn_lcdm(z_sn, mu_sn, sn_cho, H0_sn, om)
        om, A, m = map(float, theta)
        return _chi2_sn_rewrite(z_sn, mu_sn, sn_cho, H0_sn, om, A, m)

    def chi2_bao(theta, model):
        if model == "lcdm":
            om = float(theta[0])
            dm_model = transverse_comoving_distance(z_eff_bao, H0_bao, om, 1.0 - om)
        else:
            om, A, m = map(float, theta)
            dm_model = transverse_comoving_distance_rewrite(z_eff_bao, H0_bao, om, A, m)
        alpha_pred = float(dm_model / dm_fid_bao)
        return _chi2_bao(alpha_grid, delta_chi2_grid, alpha_pred)

    results = []

    directions = [("AtoB", idx_A, idx_B, cho_A, cho_B), ("BtoA", idx_B, idx_A, cho_B, cho_A)]
    anchor_modes = resolved.get("ppd", {}).get("anchors_modes", ["none", "sn+bao"])

    for direction, idx_train, idx_test, cho_train, cho_test in directions:
        for anchors_mode in anchor_modes:
            for model in ["lcdm", "rewrite"]:
                if model == "lcdm":
                    bounds = [tuple(resolved["models"]["lcdm"]["bounds"]["om"])]
                    theta0 = np.array([om_fid_lcdm], dtype=float)
                else:
                    bounds = [
                        tuple(resolved["models"]["rewrite"]["bounds"]["om"]),
                        tuple(resolved["models"]["rewrite"]["bounds"]["A"]),
                        tuple(resolved["models"]["rewrite"]["bounds"]["m"]),
                    ]
                    theta0 = np.array([om_fid_rw, A_fid_rw, m_fid_rw], dtype=float)

                def objective(theta):
                    try:
                        chi2 = chi2_lss(theta, model, idx_train, cho_train)
                        if anchors_mode == "sn+bao":
                            chi2 += chi2_sn(theta, model)
                            chi2 += chi2_bao(theta, model)
                        if not np.isfinite(chi2):
                            return 1e30
                        return float(chi2)
                    except Exception:
                        return 1e30

                res = minimize(objective, theta0, method="L-BFGS-B", bounds=bounds)
                theta_hat = res.x

                resid = lss_residual(theta_hat, model)
                resid_train = resid[idx_train]
                resid_test = resid[idx_test]
                chi2_train_lss = _chi2_from_resid(resid_train, cho_train)
                chi2_test_lss = _chi2_from_resid(resid_test, cho_test)
                chi2_test_over_n = float(chi2_test_lss / len(idx_test)) if len(idx_test) else np.nan
                rmse_test = float(np.sqrt(np.mean(resid_test**2)))
                rmse_weighted = float(np.sqrt(chi2_test_lss / len(idx_test))) if len(idx_test) else np.nan
                bias_test = float(np.mean(resid_test))

                chi2_sn_val = chi2_sn(theta_hat, model) if anchors_mode == "sn+bao" else np.nan
                chi2_bao_val = chi2_bao(theta_hat, model) if anchors_mode == "sn+bao" else np.nan

                w0_hat = np.nan
                wa_hat = np.nan
                w0wa_rms = np.nan
                if model == "rewrite":
                    z_grid = np.linspace(0.0, 2.0, 201)
                    H_rw = H_rewrite(z_grid, H0_lss, float(theta_hat[0]), float(theta_hat[1]), float(theta_hat[2]))
                    wfit = fit_effective_w0wa(z_grid, H_rw, H0=H0_lss, om=float(theta_hat[0]))
                    w0_hat = wfit["w0_hat"]
                    wa_hat = wfit["wa_hat"]
                    w0wa_rms = wfit["rms_frac_err"]

                results.append(
                    {
                        "direction": direction,
                        "anchors": anchors_mode,
                        "model": model,
                        "n_train": int(len(idx_train)),
                        "n_test": int(len(idx_test)),
                        "theta_hat": " ".join(f"{v:.6g}" for v in theta_hat),
                        "chi2_train_lss": float(chi2_train_lss),
                        "chi2_test_lss": float(chi2_test_lss),
                        "chi2_test_over_n": float(chi2_test_over_n),
                        "rmse_test": float(rmse_test),
                        "rmse_weighted_test": float(rmse_weighted),
                        "bias_test": float(bias_test),
                        "chi2_sn": float(chi2_sn_val) if np.isfinite(chi2_sn_val) else np.nan,
                        "chi2_bao": float(chi2_bao_val) if np.isfinite(chi2_bao_val) else np.nan,
                        "w0_hat": float(w0_hat) if np.isfinite(w0_hat) else np.nan,
                        "wa_hat": float(wa_hat) if np.isfinite(wa_hat) else np.nan,
                        "w0wa_rms_frac_err": float(w0wa_rms) if np.isfinite(w0wa_rms) else np.nan,
                    }
                )

    summary = pd.DataFrame(results)
    summary_path = run_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)

    # plots
    for direction in ["AtoB", "BtoA"]:
        subset = summary[summary["direction"] == direction]
        labels = []
        values = []
        for model in ["lcdm", "rewrite"]:
            for anchors_mode in ["none", "sn+bao"]:
                row = subset[(subset["model"] == model) & (subset["anchors"] == anchors_mode)]
                if not row.empty:
                    labels.append(f"{model}_{anchors_mode}")
                    values.append(row["chi2_test_lss"].iloc[0])
        fig, ax = plt.subplots()
        ax.bar(labels, values)
        ax.set_ylabel("chi2_test")
        ax.set_title(direction)
        plot_name = f"heldout_chi2_{direction}.png"
        manifest.save_fig(fig, run_dir / plot_name)
        plt.close(fig)

    metrics = {
        "n_A": n_A,
        "n_B": n_B,
        "n_data": n_data,
        "plot_files": ["heldout_chi2_AtoB.png", "heldout_chi2_BtoA.png"],
    }

    def _get(direction, model, anchors_mode):
        row = summary[
            (summary["direction"] == direction)
            & (summary["model"] == model)
            & (summary["anchors"] == anchors_mode)
        ]
        if row.empty:
            return np.nan
        return float(row["chi2_test_lss"].iloc[0])

    metrics["chi2_B_lcdm_noanchors"] = _get("AtoB", "lcdm", "none")
    metrics["chi2_B_lcdm_anchors"] = _get("AtoB", "lcdm", "sn+bao")
    metrics["chi2_B_rewrite_noanchors"] = _get("AtoB", "rewrite", "none")
    metrics["chi2_B_rewrite_anchors"] = _get("AtoB", "rewrite", "sn+bao")

    metrics["chi2_A_lcdm_noanchors"] = _get("BtoA", "lcdm", "none")
    metrics["chi2_A_lcdm_anchors"] = _get("BtoA", "lcdm", "sn+bao")
    metrics["chi2_A_rewrite_noanchors"] = _get("BtoA", "rewrite", "none")
    metrics["chi2_A_rewrite_anchors"] = _get("BtoA", "rewrite", "sn+bao")

    manifest.write_metrics(run_dir, metrics)
    manifest.write_provenance(run_dir, exp_name, seed=seed)
    write_run_markdown_stub(run_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
