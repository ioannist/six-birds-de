#!/usr/bin/env python3
"""Smoke test for MAP fitting harness."""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sixbirds_cosmo import manifest
from sixbirds_cosmo.likelihoods.base import Dataset
from sixbirds_cosmo.infer.fit import fit_map
from sixbirds_cosmo.reporting import write_run_markdown_stub


@dataclass
class QuadraticDataset(Dataset):
    mu: np.ndarray | None = None
    sigma: np.ndarray | None = None

    def loglike(self, theta: np.ndarray) -> float:
        theta = np.asarray(theta, dtype=float)
        diff = (theta - self.mu) / self.sigma
        return -0.5 * float(np.sum(diff**2))


def main() -> None:
    seed = 123
    rng = np.random.default_rng(seed)
    _ = rng  # deterministic placeholder for future extensions

    mu = np.array([1.0, -2.0])
    sigma = np.array([0.5, 2.0])
    theta0 = np.array([0.0, 0.0])
    bounds = [(-10.0, 10.0), (-10.0, 10.0)]
    param_names = ["theta0", "theta1"]

    dataset = QuadraticDataset(
        name="quadratic_smoke",
        n_data=2,
        param_names=param_names,
        mu=mu,
        sigma=sigma,
    )

    fit = fit_map(dataset, theta0, bounds=bounds)

    theta_hat = np.array(fit["theta_hat"], dtype=float)
    chi2_hat = fit["chi2_hat"]

    assert fit["success"] is True
    assert np.max(np.abs(theta_hat - mu)) < 1e-6
    assert chi2_hat is not None and chi2_hat < 1e-10

    run_dir = manifest.create_run_dir("smoke_fit", seed=seed)

    config = {
        "seed": seed,
        "theta0": theta0.tolist(),
        "mu": mu.tolist(),
        "sigma": sigma.tolist(),
        "bounds": bounds,
    }
    metrics = {
        "theta_hat": fit["theta_hat"],
        "loglike_hat": fit["loglike_hat"],
        "chi2_hat": fit["chi2_hat"],
        "aic": fit["aic"],
        "bic": fit["bic"],
        "k": fit["k"],
        "n_data": fit["n_data"],
        "success": fit["success"],
        "message": fit["message"],
    }

    manifest.write_config(run_dir, config)
    manifest.write_metrics(run_dir, metrics)
    manifest.write_provenance(run_dir, "smoke_fit", seed)

    t0_grid = np.linspace(-1.0, 3.0, 120)
    t1_grid = np.linspace(-6.0, 2.0, 120)
    T0, T1 = np.meshgrid(t0_grid, t1_grid)
    chi2_grid = ((T0 - mu[0]) / sigma[0]) ** 2 + ((T1 - mu[1]) / sigma[1]) ** 2

    fig, ax = plt.subplots()
    ax.contour(T0, T1, chi2_grid, levels=8)
    ax.plot(theta0[0], theta0[1], marker="o")
    ax.plot(theta_hat[0], theta_hat[1], marker="x")
    ax.set_xlabel(param_names[0])
    ax.set_ylabel(param_names[1])
    ax.set_title("Quadratic smoke fit")

    plot_path = run_dir / "smoke_fit_contour.png"
    manifest.save_fig(fig, plot_path)
    plt.close(fig)

    write_run_markdown_stub(run_dir)

    print(json.dumps({"run_dir": str(run_dir)}, indent=2))


if __name__ == "__main__":
    main()
