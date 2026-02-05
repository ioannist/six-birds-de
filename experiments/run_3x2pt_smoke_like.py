#!/usr/bin/env python3
"""Smoke Gaussian chi2 on DES Y3 2pt datavector using a fixed reference theory."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from sixbirds_cosmo import manifest
from sixbirds_cosmo.data_registry import fetch
from sixbirds_cosmo.datasets.des_y3_2pt import build_y3_block_index, load_des_y3_2pt_fits
from sixbirds_cosmo.lss.theory.reference import ReferenceVectorBackend
from sixbirds_cosmo.lss.theory.stub import StubBackend
from sixbirds_cosmo.likelihoods.gaussian_vector import apply_mask, chi2_gaussian
from sixbirds_cosmo.reporting import write_run_markdown_stub


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/3x2pt_y3_maglim_smoke.yaml")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    config_data = yaml.safe_load(cfg_path.read_text())

    seed = 123
    exp_name = config_data.get("exp_name", "3x2pt_smoke_like")
    run_dir = manifest.create_run_dir(exp_name, seed=seed)

    dataset_key = config_data["dataset_key"]
    fits_filename = config_data["fits_filename"]
    raw_dir = Path(fetch(dataset_key))
    fits_path = raw_dir / fits_filename
    data = load_des_y3_2pt_fits(fits_path)

    y = data["data_vector"]
    cov = data["cov"]
    sigma = np.sqrt(np.diag(cov))

    backend_cfg = config_data.get("theory_backend")
    if backend_cfg:
        backend_kind = backend_cfg.get("kind", "stub")
        frac_sigma = float(backend_cfg.get("frac_sigma", 0.1))
        ref_path = backend_cfg.get("ref_path")
        ref_format = backend_cfg.get("ref_format", "auto")
    else:
        backend_kind = "stub"
        frac_sigma = float(config_data.get("theory_stub", {}).get("frac_sigma", 0.1))
        ref_path = None
        ref_format = "auto"

    mask_cfg = config_data.get("mask", {})
    mask_kind = mask_cfg.get("kind", "all")
    use_first_n = mask_cfg.get("first_n")
    if mask_kind == "all" or use_first_n is None:
        mask = np.arange(len(y))
        mask_kind = "all"
    elif mask_kind == "first_n":
        mask = np.arange(min(use_first_n, len(y)))
        mask_kind = "first_n"
    else:
        raise ValueError("mask.kind must be 'all' or 'first_n'.")

    # Apply mask to data/cov
    y_masked, cov_masked = apply_mask(y, cov, mask)

    # Build block index (for backend alignment)
    if fits_filename.endswith(".fits") and "y3" in dataset_key:
        full_block_index = build_y3_block_index(fits_path)
        block_index = full_block_index.iloc[mask].reset_index(drop=True)
    else:
        block_index = pd.DataFrame({"i": mask})

    if backend_kind == "stub":
        backend = StubBackend(y_masked, cov_masked, frac_sigma=frac_sigma)
        t_pred = backend.predict(np.array([]), block_index)
    elif backend_kind == "reference":
        if ref_path is None:
            raise ValueError("theory_backend.ref_path must be set for reference backend.")
        ref_path = Path(ref_path)
        if not ref_path.is_absolute():
            ref_path = (Path.cwd() / ref_path).resolve()
        backend = ReferenceVectorBackend(ref_path=ref_path, ref_format=ref_format)
        t_pred = backend.predict(np.array([]), block_index)
    else:
        raise ValueError("theory_backend.kind must be 'stub' or 'reference'.")

    resid = y_masked - t_pred
    chi2, jitter_used = chi2_gaussian(resid, cov_masked)
    n_used = int(len(y_masked))
    chi2_over_n = float(chi2 / n_used)

    assert np.isfinite(chi2) and chi2 >= 0.0
    assert jitter_used <= 1e-6

    fig, ax = plt.subplots()
    ax.plot(np.arange(n_used), resid, linewidth=0.8)
    sigma_masked = np.sqrt(np.diag(cov_masked))
    ax.plot(np.arange(n_used), sigma_masked, linewidth=0.5)
    ax.plot(np.arange(n_used), -sigma_masked, linewidth=0.5)
    ax.set_xlabel("index")
    ax.set_ylabel("residual")
    ax.set_title("3x2pt smoke residuals")

    plot_path = run_dir / "3x2pt_smoke_residuals.png"
    manifest.save_fig(fig, plot_path)
    plt.close(fig)

    summary = pd.DataFrame(
        [
            {
                "n_used": n_used,
                "chi2": chi2,
                "chi2_over_n": chi2_over_n,
                "frac_sigma": frac_sigma,
                "jitter_used": jitter_used,
            }
        ]
    )
    manifest.save_table(summary, run_dir / "summary.csv")

    metrics = {
        "n_data": int(len(y)),
        "n_used": n_used,
        "frac_sigma": frac_sigma if backend_kind == "stub" else None,
        "chi2": chi2,
        "chi2_over_n": chi2_over_n,
        "jitter_used": jitter_used,
        "fits_path": str(fits_path),
        "mask_kind": mask_kind,
        "theory_kind": "t_ref = y + frac_sigma*sqrt(diag(cov))" if backend_kind == "stub" else "reference_vector",
        "theory_backend_kind": backend_kind,
        "ref_path": str(ref_path) if backend_kind == "reference" and ref_path is not None else None,
        "ref_format": ref_format if backend_kind == "reference" else None,
    }
    config = {
        "seed": seed,
        "config_path": str(cfg_path),
        "dataset_key": dataset_key,
        "fits_filename": fits_filename,
        "mask": mask_cfg,
        "frac_sigma": frac_sigma,
        "fits_path": str(fits_path),
        "theory_backend": backend_cfg or {"kind": "stub", "frac_sigma": frac_sigma},
    }

    backend_info = {
        "kind": backend_kind,
        "config": backend_cfg or {"kind": "stub", "frac_sigma": frac_sigma},
    }
    (run_dir / "backend_info.json").write_text(yaml.safe_dump(backend_info, sort_keys=False))

    manifest.write_config(run_dir, config)
    manifest.write_metrics(run_dir, metrics)
    manifest.write_provenance(run_dir, "3x2pt_smoke_like", seed)
    write_run_markdown_stub(run_dir)

    print(str(run_dir))


if __name__ == "__main__":
    main()
