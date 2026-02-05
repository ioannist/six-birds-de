#!/usr/bin/env python3
"""Sanity checks for DES Y3 2pt datavector FITS and chains."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import cho_factor
from scipy.sparse.linalg import LinearOperator, eigsh

from sixbirds_cosmo import manifest
from sixbirds_cosmo.data_registry import fetch
from sixbirds_cosmo.datasets.des_y3_2pt import load_des_y3_2pt_fits, load_des_y3_chain_txt
from sixbirds_cosmo.reporting import write_run_markdown_stub


def cholesky_with_jitter(cov: np.ndarray) -> dict:
    jitter_levels = [0.0, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4]
    for jitter in jitter_levels:
        try:
            cho_factor(cov + jitter * np.eye(cov.shape[0]), lower=True)
            return {"chol_success": True, "chol_jitter_used": jitter}
        except Exception:
            continue
    return {"chol_success": False, "chol_jitter_used": jitter_levels[-1]}


def eig_diagnostics(cov: np.ndarray) -> dict:
    n = cov.shape[0]
    if n <= 2000:
        evals = np.linalg.eigvalsh(cov)
        min_eig = float(evals[0])
        max_eig = float(evals[-1])
    else:
        def matvec(x):
            return cov @ x

        op = LinearOperator(cov.shape, matvec=matvec, dtype=cov.dtype)
        max_eig = float(eigsh(op, k=1, which="LA", return_eigenvectors=False)[0])
        min_eig = float(eigsh(op, k=1, which="SA", return_eigenvectors=False)[0])
    cond_est = float(max_eig / min_eig) if min_eig > 0 else float("inf")
    return {"min_eig_est": min_eig, "cond_est": cond_est}


def plot_datavector(run_dir: Path, data_vector: np.ndarray, slices: list[dict]) -> Path:
    fig, ax = plt.subplots()
    ax.plot(np.arange(data_vector.size), data_vector, linewidth=0.8)
    if slices:
        for sl in slices:
            ax.axvline(sl["end"], color="black", linewidth=0.3, alpha=0.4)
    ax.set_xlabel("index")
    ax.set_ylabel("value")
    ax.set_title("DES Y3 2pt datavector components")
    out = run_dir / "y3_2pt_datavector_components.png"
    manifest.save_fig(fig, out)
    plt.close(fig)
    return out


def main() -> None:
    seed = 123
    run_dir = manifest.create_run_dir("des_y3_vector_sanity", seed=seed)

    raw_vectors = Path(fetch("des_y3a2_datavectors_2pt"))
    raw_chains = Path(fetch("des_y3a2_chains_3x2pt"))

    maglim_path = raw_vectors / "2pt_maglim_covupdate.fits"
    redmagic_path = raw_vectors / "2pt_redmagic_covupdate.fits"

    maglim = load_des_y3_2pt_fits(maglim_path)
    redmagic = None
    if redmagic_path.exists():
        redmagic = load_des_y3_2pt_fits(redmagic_path)

    diag_maglim = {**cholesky_with_jitter(maglim["cov"]), **eig_diagnostics(maglim["cov"])}
    diag_redmagic = None
    if redmagic is not None:
        diag_redmagic = {**cholesky_with_jitter(redmagic["cov"]), **eig_diagnostics(redmagic["cov"])}

    plot_path = plot_datavector(run_dir, maglim["data_vector"], maglim.get("spectra_slices", []))

    chain_path = raw_chains / "chain_3x2pt_lcdm_maglim_opt.txt"
    chain_info = load_des_y3_chain_txt(chain_path)

    metrics = {
        "maglim_n_data": maglim["n_data"],
        "maglim_cov_shape": maglim["cov_shape"],
        "maglim_chol_success": diag_maglim["chol_success"],
        "maglim_chol_jitter_used": diag_maglim["chol_jitter_used"],
        "maglim_min_eig_est": diag_maglim["min_eig_est"],
        "maglim_cond_est": diag_maglim["cond_est"],
        "maglim_extraction_method": maglim["extraction_method"],
        "maglim_spectra_used": maglim.get("spectra_used", []),
        "chain_n_rows": chain_info["n_rows"],
        "chain_columns": chain_info["columns"],
        "chain_summary": chain_info["summary"],
    }

    if redmagic is not None and diag_redmagic is not None:
        metrics.update(
            {
                "redmagic_n_data": redmagic["n_data"],
                "redmagic_cov_shape": redmagic["cov_shape"],
                "redmagic_chol_success": diag_redmagic["chol_success"],
                "redmagic_chol_jitter_used": diag_redmagic["chol_jitter_used"],
                "redmagic_min_eig_est": diag_redmagic["min_eig_est"],
                "redmagic_cond_est": diag_redmagic["cond_est"],
                "redmagic_extraction_method": redmagic["extraction_method"],
                "redmagic_spectra_used": redmagic.get("spectra_used", []),
            }
        )

    config = {
        "seed": seed,
        "maglim_fits": str(maglim_path),
        "redmagic_fits": str(redmagic_path) if redmagic is not None else None,
        "chain_file": str(chain_path),
    }

    manifest.write_config(run_dir, config)
    manifest.write_metrics(run_dir, metrics)
    manifest.write_provenance(run_dir, "des_y3_vector_sanity", seed)
    write_run_markdown_stub(run_dir)

    print(str(plot_path))


if __name__ == "__main__":
    main()
