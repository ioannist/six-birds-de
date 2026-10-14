"""Build the composed manuscript figures from audited run bundles.

Each figure reads the run bundle's metrics/config. Where a bundle does not store
the plotted series, the series is recomputed with the library function and the
bundle's configuration, and the recomputation is checked against the bundle's
recorded metrics before anything is drawn. Output: vector PDFs in
docs/paper/figures/ plus an entry in docs/paper/figures/provenance_composed.yaml.

Run from the repository root with PYTHONPATH=src.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from sixbirds_cosmo.datasets.des_y6_bao import load_alpha_likelihood_curve
from sixbirds_cosmo.toy1 import simulate_micro
from sixbirds_cosmo.toy2_patch import simulate_patches

ROOT = Path(".")
OUT = ROOT / "docs" / "paper" / "figures"
SHA = "bfa9bdeba8652e7e08c666e1cf5d0a656985db77"
RUNS = {
    "toy1": "results/toy1/20261013T185542Z_" + SHA,
    "toy2": "results/toy2_patch/20261013T185543Z_" + SHA,
    "scale_sweep": "results/scale_sweep/20261013T185551Z_" + SHA,
    "rewrite_vs_lambda": "results/rewrite_vs_lambda/20261013T185546Z_" + SHA,
    "fit_des_y6_bao": "results/fit_des_y6_bao/20261014T120546Z_39346c3187a3de10dcd41edc828a511080220084",
    "ppd_background": "results/ppd_background/20261014T120547Z_39346c3187a3de10dcd41edc828a511080220084",
    "rewrite_background": "results/rewrite_background_vs_lcdm/20261014T120554Z_39346c3187a3de10dcd41edc828a511080220084",
    "des_y3_blockmap": "results/des_y3_blockmap/20261013T190117Z_" + SHA,
}

# Validated categorical slots 1-3 (light surface) plus neutral ink.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Latin Modern Roman", "CMU Serif", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 1.6,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    }
)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def check(name: str, value: float, expected: float, rtol: float = 1e-9) -> None:
    if not np.isclose(value, expected, rtol=rtol, atol=0.0):
        raise SystemExit(f"[make_paper_figures] {name}: recomputed {value!r} != recorded {expected!r}")


def panel_label(ax, text: str) -> None:
    ax.text(0.0, 1.03, text, transform=ax.transAxes, fontweight="bold", va="bottom", ha="left")


def save(fig, name: str, sources: list[str], provenance: dict) -> None:
    path = OUT / name
    fig.savefig(path)
    plt.close(fig)
    provenance[name] = {
        "sources": sources,
        "script": "scripts/make_paper_figures.py",
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }


def fig_mechanism(provenance: dict) -> None:
    """Toy 1 mismatch against the exact identity, and the staging sweep."""
    run = ROOT / RUNS["toy1"]
    cfg = load_yaml(run / "config.yaml")
    metrics = load_json(run / "metrics.json")
    rows = metrics["rm_vs_beta_table"]
    beta = np.array([r["beta"] for r in rows])
    rm_max = np.array([r["rm_max"] for r in rows])

    params = {k: cfg[k] for k in ("alpha", "dt", "x0_dist", "x0_low", "x0_high")}
    # The beta grid runs at n=1000 with the base seed (experiments/run_toy1.py).
    var0 = None
    for b, recorded in zip(beta, rm_max):
        sim = simulate_micro(n=1000, steps=cfg["steps"], params={**params, "beta": b}, seed=cfg["seed"])
        check(f"toy1 rm_max(beta={b})", float(np.max(sim["rm"])), recorded, rtol=1e-12)
        var0 = sim["x_var"][0]
    beta_line = np.linspace(0.0, beta.max(), 50)

    sweep = pd.read_csv(ROOT / RUNS["scale_sweep"] / "summary.csv")
    sweep_metrics = load_json(ROOT / RUNS["scale_sweep"] / "metrics.json")
    check("scale sweep rm(Lmax)", float(sweep["rm_mean_late"].iloc[-1]), sweep_metrics["rm_Lmax"])

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.5))
    ax = axes[0]
    ax.plot(beta_line, cfg["dt"] * beta_line * var0, color=MUTED, ls="--", lw=1.0,
            label=r"$\Delta t\,\beta\,v_0$ (exact)")
    ax.plot(beta, rm_max, "o", color=BLUE, ms=5, label="simulation")
    ax.set_xlabel(r"nonlinearity $\beta$")
    ax.set_ylabel(r"max. observable mismatch")
    ax.legend(loc="upper left")
    panel_label(ax, "(a) Toy 1")

    ax = axes[1]
    pos = sweep[sweep["rm_mean_late"] > 0]
    ax.loglog(pos["L"], pos["rm_mean_late"], "o-", color=ORANGE, ms=4)
    ax.set_xlabel(r"lens block size $L$ (patches per domain)")
    ax.set_ylabel(r"late-time mean mismatch")
    panel_label(ax, "(b) Staging sweep")
    fig.tight_layout(w_pad=2.0)
    save(fig, "fig_mechanism.pdf", [RUNS["toy1"], RUNS["scale_sweep"]], provenance)


def fig_toy2(provenance: dict) -> None:
    """Q, the dust term and the domain acceleration for the patch proxy."""
    run = ROOT / RUNS["toy2"]
    cfg = load_yaml(run / "config.yaml")
    metrics = load_json(run / "metrics.json")
    sims = {}
    for scenario in ("homogeneous", "heterogeneous"):
        sims[scenario] = simulate_patches(N=cfg["N"], steps=cfg["steps"], params=cfg[scenario], seed=cfg["seed"])
    het = sims["heterogeneous"]
    check("toy2 Q_peak", float(np.max(het["Q"])), metrics["hetero_Q_peak"])
    g4pi = cfg["heterogeneous"]["g4pi"]
    t = het["t"]
    dust = g4pi * het["rhoD"]

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.5))
    ax = axes[0]
    ax.plot(t, het["Q"] / dust, color=BLUE)
    ax.axhline(1.0, color=MUTED, ls="--", lw=1.0)
    ax.text(0.5 * t[-1], 0.985, r"$\ddot a_{\mathcal{D}}>0$ above this line", color=MUTED, ha="center", va="top", fontsize=7)
    ax.set_ylim(0.93, None)
    ax.set_xlabel(r"time $t$")
    ax.set_ylabel(r"$Q\,/\,(g\,\rho_{\mathcal{D}})$")
    panel_label(ax, "(a) Heterogeneous run")

    ax = axes[1]
    for scenario, color, ls in (("homogeneous", MUTED, "--"), ("heterogeneous", ORANGE, "-")):
        sim = sims[scenario]
        ax.plot(sim["t"], sim["ddot_aD"] / sim["aD"], color=color, ls=ls, label=scenario)
    ax.axhline(0.0, color=INK, lw=0.6)
    ax.set_xlabel(r"time $t$")
    ax.set_ylabel(r"$\ddot a_{\mathcal{D}}/a_{\mathcal{D}}$")
    ax.set_xlim(0, 6)
    ax.legend(loc="lower right")
    panel_label(ax, "(b) Domain acceleration")
    fig.tight_layout(w_pad=2.0)
    save(fig, "fig_toy2.pdf", [RUNS["toy2"]], provenance)


def fig_synthetic(provenance: dict) -> None:
    """Synthetic comoving-distance mock with three homogeneous macro fits."""
    spec = importlib.util.spec_from_file_location("rvl", ROOT / "experiments" / "run_rewrite_vs_lambda.py")
    rvl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rvl)
    from sixbirds_cosmo.infer_distance import D_lcdm, D_matter_only, fit_lcdm, fit_matter_only, make_redshift_from_a
    from sixbirds_cosmo.rewrite import D_rewrite, fit_rewrite, prepare_proxy_shape

    run = ROOT / RUNS["rewrite_vs_lambda"]
    metrics = load_json(run / "metrics.json")
    cfg = load_yaml(run / "config.yaml")
    seed = cfg["seed"]
    # Ensemble size and duration are fixed in experiments/run_rewrite_vs_lambda.py
    # (not written to config.yaml); the chi2 checks below confirm the match.
    sim = simulate_patches(N=500, steps=1200, params=cfg["base_params"], seed=seed)
    proxy = prepare_proxy_shape(z_proxy=make_redshift_from_a(sim["aD"], normalize_to_last=True), var_proxy=sim["var_H"])
    _, z, D_true, D_obs, sigma, _ = rvl._build_mock_dataset(sim, seed, cfg["noise_frac"], cfg["noise_floor"])
    fa, fb, fr = fit_matter_only(z, D_obs, sigma), fit_lcdm(z, D_obs, sigma), fit_rewrite(z, D_obs, sigma, proxy)
    check("synthetic chi2 matter-only", fa.chi2, metrics["chi2_matter_only"], rtol=1e-6)
    check("synthetic chi2 LCDM", fb.chi2, metrics["chi2_lcdm"], rtol=1e-6)
    check("synthetic chi2 rewrite", fr["chi2"], metrics["chi2_rewrite"], rtol=1e-6)
    models = [
        ("matter only", D_matter_only(z, fa.params["H0"]), MUTED, ":"),
        (r"flat $\Lambda$CDM", D_lcdm(z, fb.params["H0"], fb.params["omega_lambda"]), BLUE, "-"),
        ("heterogeneity rewrite", D_rewrite(z, fr["params"]["H0"], fr["params"]["A"], proxy["z_sorted"], proxy["g_sorted"]), ORANGE, "--"),
    ]

    fig, axes = plt.subplots(2, 1, figsize=(4.6, 4.2), sharex=True, gridspec_kw={"height_ratios": [1.6, 1]})
    ax = axes[0]
    ax.errorbar(z, D_obs, yerr=sigma, fmt="o", ms=2.5, color=INK, elinewidth=0.6, label="mock data", zorder=3)
    for label, D, color, ls in models:
        ax.plot(z, D, color=color, ls=ls, label=label)
    ax.set_ylabel(r"comoving distance $\chi(z)$ [$c=1$]")
    ax.legend(loc="lower right")
    ax = axes[1]
    for label, D, color, ls in models[1:]:
        ax.plot(z, (D_obs - D) / sigma, "o", ms=3, color=color, label=label, alpha=0.9)
    ax.axhline(0.0, color=INK, lw=0.6)
    ax.set_xlabel(r"redshift $z$")
    ax.set_ylabel(r"(data $-$ fit)$/\sigma$")
    fig.tight_layout(h_pad=0.6)
    save(fig, "fig_synthetic_fits.pdf", [RUNS["rewrite_vs_lambda"]], provenance)


def fig_background(provenance: dict) -> None:
    """BAO consensus curve with the alpha values implied by SN-based fits."""
    bao_metrics = load_json(ROOT / RUNS["fit_des_y6_bao"] / "metrics.json")
    ppd = load_json(ROOT / RUNS["ppd_background"] / "metrics.json")
    rw = pd.read_csv(ROOT / RUNS["rewrite_background"] / "summary.csv")
    curve = load_alpha_likelihood_curve(ROOT / "data" / "raw" / "des_y6_bao_release", "fid")
    alpha, dchi2 = np.asarray(curve["alpha_grid"]), np.asarray(curve["delta_chi2_grid"])
    check("BAO alpha minimum", float(alpha[np.argmin(dchi2)]), bao_metrics["alpha_hat_combined"])
    alpha_sn = ppd["alpha_pred_from_sn"]
    alpha_joint = float(rw.loc[(rw["dataset"] == "SN+BAO") & (rw["model"] == "lcdm"), "alpha_pred"].iloc[0])

    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    ax.plot(alpha, dchi2, color=INK, lw=1.2)
    ax.axhline(1.0, color=GRID, lw=0.8)
    ax.axvline(alpha_sn, color=BLUE, ls="--", lw=1.2)
    ax.axvline(alpha_joint, color=AQUA, ls=":", lw=1.4)
    ax.text(alpha_sn + 0.002, 8.6, "SN-only fit\n(flat $\\Lambda$CDM)", color=INK, ha="left", va="top", fontsize=7)
    ax.text(alpha_joint - 0.002, 8.6, "joint SN+BAO fit\n($\\Lambda$CDM, free $\\Omega_k$)", color=INK, ha="right", va="top", fontsize=7)
    ax.set_xlim(0.90, 1.03)
    ax.set_ylim(0, 9)
    ax.set_xlabel(r"BAO dilation $\alpha$ at $z_{\rm eff}=0.851$")
    ax.set_ylabel(r"$\Delta\chi^2(\alpha)$")
    fig.tight_layout()
    save(fig, "fig_background.pdf", [RUNS["fit_des_y6_bao"], RUNS["ppd_background"], RUNS["rewrite_background"]], provenance)


def fig_blockmap(provenance: dict) -> None:
    """Signal-to-noise of each DES Y3 element, with the declared block structure."""
    from sixbirds_cosmo.datasets.des_y3_2pt import load_des_y3_2pt_fits

    run = ROOT / RUNS["des_y3_blockmap"]
    metrics = load_json(run / "metrics.json")
    index = pd.read_csv(run / "block_index.csv")
    data = load_des_y3_2pt_fits(ROOT / metrics["fits_file"])
    y, cov = np.asarray(data["data_vector"]), np.asarray(data["cov"])
    if len(index) != y.size or not np.allclose(index["value"].to_numpy(), y, rtol=1e-11, atol=0):
        raise SystemExit("[make_paper_figures] block index does not reproduce the data vector")
    snr = y / np.sqrt(np.diag(cov))
    names = {"xip": r"$\xi_+$", "xim": r"$\xi_-$", "gammat": r"$\gamma_t$", "wtheta": r"$w(\theta)$"}
    probe_color = {"shear": BLUE, "ggl": ORANGE, "clustering": AQUA}

    fig, ax = plt.subplots(figsize=(6.4, 2.3))
    for stat, grp in index.groupby("stat", sort=False):
        lo, hi = grp["i"].min(), grp["i"].max()
        color = probe_color[grp["probe"].iloc[0]]
        ax.axvspan(lo - 0.5, hi + 0.5, color=color, alpha=0.10, lw=0)
        ax.plot(grp["i"], snr[grp["i"]], ".", ms=1.8, color=color)
        ax.text(0.5 * (lo + hi), 1.02, f"{names[stat]} ({hi - lo + 1})", transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=8)
    ax.axhline(0.0, color=INK, lw=0.5)
    ax.set_xlim(-5, y.size + 4)
    ax.set_xlabel("data-vector index (declared covariance ordering)")
    ax.set_ylabel(r"$y_i/\sqrt{C_{ii}}$")
    fig.tight_layout()
    save(fig, "fig_y3_blockmap.pdf", [RUNS["des_y3_blockmap"]], provenance)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    provenance: dict = {}
    fig_mechanism(provenance)
    fig_toy2(provenance)
    fig_synthetic(provenance)
    fig_background(provenance)
    fig_blockmap(provenance)
    with (OUT / "provenance_composed.yaml").open("w") as f:
        yaml.safe_dump(provenance, f, sort_keys=True)
    print(f"[make_paper_figures] wrote {len(provenance)} figures")


if __name__ == "__main__":
    main()
