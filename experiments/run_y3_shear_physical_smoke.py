import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.integrate import trapezoid
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.special import j0, jv  # noqa: E402

from sixbirds_cosmo import manifest
from sixbirds_cosmo.datasets.des_y3_2pt import build_y3_block_index, load_des_y3_2pt_fits
from sixbirds_cosmo.likelihoods.gaussian_vector import chi2_gaussian
from sixbirds_cosmo.lss.config import load_and_resolve_lss_config, load_yaml_config
from sixbirds_cosmo.lss.masks import mask_cosmic_shear
from sixbirds_cosmo.reporting import write_run_markdown_stub


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _write_backend_probe(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _theta_to_rad(theta: np.ndarray, unit: str | None) -> np.ndarray:
    theta = np.asarray(theta, dtype=float)
    if not np.all(np.isfinite(theta)) or np.any(theta < 0):
        raise ValueError("shear angles must be finite and nonnegative")
    if unit:
        u = unit.lower().strip()
        if u in {"arcmin", "amin", "arcminute", "arcminutes"}:
            return np.deg2rad(theta / 60.0)
        if u in {"arcsec", "asec", "arcsecond", "arcseconds"}:
            return np.deg2rad(theta / 3600.0)
        if u in {"deg", "degree", "degrees"}:
            return np.deg2rad(theta)
        if u in {"rad", "radian", "radians"}:
            return theta
        raise ValueError(f"unsupported shear angle unit: {unit}")
    # default assumption: arcmin
    return np.deg2rad(theta / 60.0)


def _load_nz_from_fits(path: Path) -> tuple[np.ndarray, list[np.ndarray], list[int]] | None:
    """Read the fiducial SOURCE distribution, not lens bins or realizations.

    Keep FITS BIN labels explicit; column position is not a tomography label.
    """
    from astropy.io import fits

    with fits.open(path, memmap=False) as hdul:
        nz_hdu = None
        for hdu in hdul:
            name = (hdu.name or "").lower()
            if name == "nz_source" and hasattr(hdu, "data") and hdu.data is not None:
                nz_hdu = hdu
                break
        if nz_hdu is None or not hasattr(nz_hdu, "columns"):
            return None
        colnames = [c.name for c in nz_hdu.columns]
        z_col = None
        for candidate in colnames:
            if candidate.lower() in {"z_mid", "z", "redshift"}:
                z_col = candidate
                break
        if z_col is None:
            raise ValueError("nz_source requires a redshift or Z_MID column")
        z = np.asarray(nz_hdu.data[z_col], dtype=float).reshape(-1)
        if len(z) < 2 or not np.all(np.isfinite(z)) or np.any(z < 0) or np.any(np.diff(z) <= 0):
            raise ValueError("source redshifts must be finite, nonnegative and increasing")
        bins = sorted((int(match.group(1)), c) for c in colnames
                      if (match := re.fullmatch(r"BIN(\d+)", c, re.IGNORECASE)))
        nz_list = []
        for _, c in bins:
            vals = np.asarray(nz_hdu.data[c], dtype=float).reshape(-1)
            # The public BIN1 kernel contains small signed calibration tails.
            # Preserve the released kernel rather than clipping it into a new
            # distribution. The smoke run reports the signed mass explicitly.
            if vals.shape != z.shape or not np.all(np.isfinite(vals)) or trapezoid(vals, z) <= 0:
                raise ValueError(f"source kernel {c} must be finite with positive integral")
            nz_list.append(vals)
        if not nz_list or len({b for b, _ in bins}) != len(bins):
            raise ValueError("nz_source must contain distinct BIN columns")
        return z, nz_list, [b for b, _ in bins]


def _toy_nz(n_bins: int, zmax: float, nz_points: int) -> tuple[np.ndarray, list[np.ndarray]]:
    z = np.linspace(0.0, zmax, nz_points)
    nz_list = []
    for b in range(n_bins):
        z0 = 0.5 + 0.2 * b
        nz = z**2 * np.exp(-(z / z0) ** 1.5)
        nz = nz / trapezoid(nz, z)
        nz_list.append(nz)
    return z, nz_list


def _build_theory_vector(block_df: pd.DataFrame, cosmo, tracers, ell: np.ndarray,
                         *, tracer_bin_ids: list[int]) -> np.ndarray:
    import pyccl as ccl  # type: ignore
    ell = np.asarray(ell, dtype=float)
    if ell.ndim != 1 or len(ell) < 2 or not np.all(np.isfinite(ell)) or np.any(ell <= 0) or np.any(np.diff(ell) <= 0):
        raise ValueError("multipoles must be a finite positive increasing vector")
    if len(tracer_bin_ids) != len(tracers) or len(set(tracer_bin_ids)) != len(tracers):
        raise ValueError("tracer labels must identify each tracer exactly once")
    tracer_index = {label: i for i, label in enumerate(tracer_bin_ids)}

    pair_cache: dict[tuple[int, int], np.ndarray] = {}

    def get_cl(i: int, j: int) -> np.ndarray:
        key = (i, j)
        if key in pair_cache:
            return pair_cache[key]
        cl = ccl.angular_cl(cosmo, tracers[i], tracers[j], ell)
        if np.shape(cl) != ell.shape or not np.all(np.isfinite(cl)):
            raise ValueError("angular power spectrum must be finite and match multipoles")
        pair_cache[key] = cl
        return cl

    theory = np.zeros(len(block_df), dtype=float)
    theta_vals = block_df["x"].to_numpy(dtype=float)
    units = block_df["x_unit"].tolist() if "x_unit" in block_df.columns else [None] * len(block_df)
    theta_rad = np.array([_theta_to_rad(np.array([val]), unit)[0]
                          for val, unit in zip(theta_vals, units)])

    bin1 = block_df["bin1"].to_numpy()
    bin2 = block_df["bin2"].to_numpy()
    stat = block_df["stat"].to_numpy()

    for idx in range(len(block_df)):
        if bin1[idx] not in tracer_index or bin2[idx] not in tracer_index:
            raise ValueError("shear bin label has no corresponding source distribution")
        b1 = tracer_index[bin1[idx]]
        b2 = tracer_index[bin2[idx]]
        cl = get_cl(b1, b2)
        t = theta_rad[idx]
        integrand = ell * cl / (2.0 * math.pi)
        if stat[idx] == "xip":
            kernel = j0(ell * t)
        elif stat[idx] == "xim":
            kernel = jv(4, ell * t)
        else:
            raise ValueError("shear statistic must be xip or xim")
        theory[idx] = trapezoid(integrand * kernel, ell)

    return theory


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/lss/y3_shear_physical_smoke.yaml")
    args = parser.parse_args()

    raw_cfg = load_yaml_config(Path(args.config))
    resolved = load_and_resolve_lss_config(Path(args.config))
    resolved["physical"] = raw_cfg.get("physical", {})

    seed = resolved.get("seed", 0)
    exp_name = resolved.get("run", {}).get("exp_name", "y3_shear_physical_smoke")
    run_dir = manifest.create_run_dir(exp_name, seed=seed)

    manifest.write_config(run_dir, resolved)

    fits_path = Path(resolved["dataset"]["resolved_path"])
    data = load_des_y3_2pt_fits(fits_path)
    block_index = build_y3_block_index(fits_path)

    idx_shear = mask_cosmic_shear(block_index)
    block_shear = block_index.loc[block_index["i"].isin(idx_shear)].copy()
    block_shear = block_shear[block_shear["stat"].isin(["xip", "xim"])].copy()
    block_shear = block_shear.sort_values("i")

    indices = block_shear["i"].to_numpy(dtype=int)
    y = data["data_vector"][indices]
    cov = data["cov"][np.ix_(indices, indices)]

    subset_path = run_dir / "shear_subset.csv"
    block_shear[["i", "stat", "bin1", "bin2", "x", "x_unit", "value"]].to_csv(
        subset_path, index=False
    )

    preview_path = run_dir / "shear_data_preview.png"
    fig, ax = plt.subplots()
    x_vals = block_shear["x"].to_numpy(dtype=float)
    v_vals = block_shear["value"].to_numpy(dtype=float)
    if np.all(np.isfinite(x_vals)):
        order = np.argsort(x_vals)
        ax.plot(x_vals[order], v_vals[order])
        ax.set_xlabel("x")
    else:
        ax.plot(np.arange(len(v_vals)), v_vals)
        ax.set_xlabel("index")
    ax.set_ylabel("value")
    manifest.save_fig(fig, preview_path)
    plt.close(fig)

    backend_probe = {
        "timestamp_utc": _utc_now_iso(),
        "backend_attempted": "pyccl",
        "backend_available": False,
        "backend_version": None,
        "success_stage": "none",
        "errors": [],
    }

    metrics = {
        "audit_kind": "finite_multipole_shear_smoke",
        "full_survey_likelihood": False,
        "backend_attempted": "pyccl",
        "backend_available": False,
        "n_used": int(len(indices)),
        "note": "To enable physical mode: python -m pip install -e '.[dev,lss]' then rerun.",
    }

    try:
        import pyccl as ccl  # type: ignore
    except Exception as exc:
        backend_probe["errors"].append(
            {
                "stage": "import",
                "type": type(exc).__name__,
                "message": str(exc),
            }
        )
        metrics["blocked_reason"] = f"{type(exc).__name__}: {exc}"
        _write_backend_probe(run_dir / "backend_probe.json", backend_probe)
        manifest.write_metrics(run_dir, metrics)
        manifest.write_provenance(run_dir, exp_name, seed=seed)
        write_run_markdown_stub(run_dir)
        return 0

    backend_probe["backend_available"] = True
    backend_probe["backend_version"] = getattr(ccl, "__version__", None)
    backend_probe["success_stage"] = "import"

    physical = resolved.get("physical", {})
    zmax = float(physical.get("zmax", 3.0))
    nz_points = int(physical.get("nz_points", 300))
    ell_min = float(physical.get("ell_min", 10))
    ell_max = float(physical.get("ell_max", 5000))
    ell_n = int(physical.get("ell_n", 200))
    fid = physical.get("fiducial", {})

    nz_data = _load_nz_from_fits(fits_path)
    if nz_data is None:
        if not physical.get("allow_toy_nz", False):
            raise ValueError("fiducial nz_source is absent; toy distributions require explicit allow_toy_nz")
        nz_source = "toy"
        bin_vals = block_shear[["bin1", "bin2"]].to_numpy(dtype=int)
        max_bin = int(np.nanmax(bin_vals)) if bin_vals.size else 0
        if max_bin >= 1 and np.nanmin(bin_vals) >= 1:
            n_bins = max_bin
            bin_offset = -1
        else:
            n_bins = max_bin + 1
            bin_offset = 0
        z, nz_list = _toy_nz(n_bins, zmax, nz_points)
        tracer_bin_ids = list(range(1, n_bins + 1)) if bin_offset == -1 else list(range(n_bins))
    else:
        nz_source = "fits"
        z, nz_list, tracer_bin_ids = nz_data

    cosmo = ccl.Cosmology(
        Omega_c=float(fid.get("Omega_c", 0.25)),
        Omega_b=float(fid.get("Omega_b", 0.05)),
        h=float(fid.get("h", 0.67)),
        sigma8=float(fid.get("sigma8", 0.8)),
        n_s=float(fid.get("n_s", 0.965)),
    )

    tracers = [ccl.WeakLensingTracer(cosmo, dndz=(z, nz)) for nz in nz_list]

    ell = np.logspace(np.log10(ell_min), np.log10(ell_max), ell_n)

    theory = _build_theory_vector(block_shear, cosmo, tracers, ell, tracer_bin_ids=tracer_bin_ids)
    resid = y - theory
    chi2, jitter_used = chi2_gaussian(resid, cov)

    metrics.update(
        {
            "backend_available": True,
            "chi2": float(chi2),
            "chi2_over_n": float(chi2 / len(indices)) if len(indices) else None,
            "n_used": int(len(indices)),
            "chol_jitter_used": float(jitter_used),
            "nz_source": nz_source,
            "source_bin_labels": tracer_bin_ids,
            "source_kernel_policy": "released_tabulation_unmodified",
            "source_negative_mass_fraction": [float(trapezoid(np.maximum(-nz, 0), z) / trapezoid(nz, z)) for nz in nz_list],
            "approximation": "flat_sky_J0_J4_with_finite_multipole_trapezoid",
            "ell_range": [ell_min, ell_max],
            "ell_n": ell_n,
        }
    )

    theory_path = run_dir / "shear_theory.csv"
    out_df = block_shear[["i", "stat", "bin1", "bin2", "x", "x_unit", "value"]].copy()
    out_df["theory"] = theory
    out_df["residual"] = resid
    out_df.to_csv(theory_path, index=False)

    residual_plot = run_dir / "shear_residuals.png"
    fig, ax = plt.subplots()
    x_vals = block_shear["x"].to_numpy(dtype=float)
    if np.all(np.isfinite(x_vals)):
        order = np.argsort(x_vals)
        ax.plot(x_vals[order], resid[order])
        ax.set_xlabel("x")
    else:
        ax.plot(np.arange(len(resid)), resid)
        ax.set_xlabel("index")
    ax.set_ylabel("residual")
    manifest.save_fig(fig, residual_plot)
    plt.close(fig)

    backend_probe["success_stage"] = "pk"  # indicates backend usable
    _write_backend_probe(run_dir / "backend_probe.json", backend_probe)

    manifest.write_metrics(run_dir, metrics)
    manifest.write_provenance(run_dir, exp_name, seed=seed)
    write_run_markdown_stub(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
