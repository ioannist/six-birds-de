#!/usr/bin/env python3
"""Validate rewrite -> effective w0-wa fit."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sixbirds_cosmo import manifest
from sixbirds_cosmo.reporting import write_run_markdown_stub
from sixbirds_cosmo.rewrite_background import H_rewrite
from sixbirds_cosmo.rewrite_effective_w import H_w0wa, fit_effective_w0wa, compare_H_models


def main() -> None:
    seed = 123
    run_dir = manifest.create_run_dir("rewrite_effective_w_validate", seed=seed)

    H0 = 67.4
    om = 0.317071
    A = 0.554624
    m = 0.0

    z_grid = np.linspace(0.0, 2.0, 201)
    H_rw = H_rewrite(z_grid, H0, om, A, m)

    fit = fit_effective_w0wa(z_grid, H_rw, H0=H0, om=om)
    w0_hat = fit["w0_hat"]
    wa_hat = fit["wa_hat"]

    H_w = H_w0wa(z_grid, H0, om, w0_hat, wa_hat)
    comp = compare_H_models(z_grid, H_rw, H_w)

    fig, ax = plt.subplots()
    frac = (H_w - H_rw) / H_rw
    ax.plot(z_grid, frac)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("z")
    ax.set_ylabel("fractional residual")
    ax.set_title("H_w0wa vs H_rewrite fractional residual")

    plot_path = run_dir / "H_frac_residuals.png"
    manifest.save_fig(fig, plot_path)
    plt.close(fig)

    metrics = {
        "om": om,
        "A": A,
        "m": m,
        "H0": H0,
        "w0_hat": w0_hat,
        "wa_hat": wa_hat,
        "rms_frac_err": comp["rms_frac_err"],
        "max_frac_err": comp["max_frac_err"],
        "z_max": float(z_grid.max()),
        "n_grid": int(z_grid.size),
    }
    config = {
        "seed": seed,
        "z_grid": [float(z_grid.min()), float(z_grid.max()), int(z_grid.size)],
        "rewrite_params": {"H0": H0, "om": om, "A": A, "m": m},
    }

    assert np.isfinite(w0_hat) and np.isfinite(wa_hat)
    assert comp["rms_frac_err"] < 0.005

    manifest.write_config(run_dir, config)
    manifest.write_metrics(run_dir, metrics)
    manifest.write_provenance(run_dir, "rewrite_effective_w_validate", seed)
    write_run_markdown_stub(run_dir)

    print(str(run_dir))


if __name__ == "__main__":
    main()
