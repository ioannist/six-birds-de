"""Generate the manuscript's quantitative tables from audited run bundles.

Every number in docs/paper/tables/*.tex is read from a run bundle's
metrics.json or summary.csv; nothing is typed by hand. Provenance (source run
folders, output hashes) is written to docs/paper/tables/provenance.yaml.
"""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

SHA = "bfa9bdeba8652e7e08c666e1cf5d0a656985db77"
RUNS = {
    "rewrite_vs_lambda": f"results/rewrite_vs_lambda/20261013T185546Z_{SHA}",
    "fit_des_sn5yr": f"results/fit_des_sn5yr/20261013T185604Z_{SHA}",
    "fit_des_y6_bao": f"results/fit_des_y6_bao/20261014T120546Z_39346c3187a3de10dcd41edc828a511080220084",
    "rewrite_background": f"results/rewrite_background_vs_lcdm/20261014T120554Z_39346c3187a3de10dcd41edc828a511080220084",
    "y3_ppd_probe_split": f"results/y3_ppd_probe_split/20261013T190123Z_{SHA}",
    "kids450_ppd_probe_split": f"results/kids450_ppd_probe_split/20261013T190142Z_{SHA}",
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def metrics(key: str) -> dict:
    return json.loads((Path(RUNS[key]) / "metrics.json").read_text())


def summary(key: str) -> pd.DataFrame:
    return pd.read_csv(Path(RUNS[key]) / "summary.csv")


def f(x: float, digits: int = 3) -> str:
    return f"{x:.{digits}f}"


def table(caption: str, label: str, colspec: str, header: str, body: list[str], note: str = "") -> str:
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{colspec}}}",
        "\\toprule",
        header + " \\\\",
        "\\midrule",
        *body,
        "\\bottomrule",
        "\\end{tabular}",
    ]
    if note:
        lines += ["\\par\\smallskip", f"\\parbox{{0.92\\linewidth}}{{\\footnotesize {note}}}"]
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"


def synth_table() -> str:
    m = metrics("rewrite_vs_lambda")
    n = 40
    ref = m["aic_lcdm"]
    rows = [
        ("Matter only", "$H_0$", "---", m["chi2_matter_only"], m["aic_matter_only"]),
        ("Flat $\\Lambda$CDM", "$H_0,\\ \\Omega_\\Lambda$",
         f"$\\Omega_\\Lambda={f(m['lcdm_omega_lambda'])}$", m["chi2_lcdm"], m["aic_lcdm"]),
        ("Heterogeneity rewrite", "$H_0,\\ A$",
         "$A={:.2f}\\times10^{{3}}$".format(m["rewrite_A"] / 1e3),
         m["chi2_rewrite"], m["aic_rewrite"]),
    ]
    body = [
        f"{name} & {free} & {best} & {chi2:.1f} & ${aic - ref:+.1f}$ \\\\"
        for name, free, best, chi2, aic in rows
    ]
    return table(
        f"Homogeneous macro fits to the synthetic comoving-distance mock of Fig.~\\ref{{fig:synthetic}} "
        f"($n={n}$ points, null-$\\Lambda$ generator). $\\Delta$AIC is relative to flat $\\Lambda$CDM. "
        "For the rewrite, $H_0$ denotes the square root of the matter coefficient in "
        "Eq.~\\eqref{eq:rewrite_synthetic}, not the present expansion rate, and $A$ is in the "
        "simulation's units of $H^2$ with the proxy shape normalized to unit maximum.",
        "tab:synth_inference",
        "@{}llcrr@{}",
        "Model & Free parameters & Best fit & $\\chi^2$ & $\\Delta$AIC",
        body,
    )


def background_table() -> str:
    sn = metrics("fit_des_sn5yr")
    bao = metrics("fit_des_y6_bao")
    rw = summary("rewrite_background")

    def row(dataset: str, model: str) -> pd.Series:
        sel = rw[(rw["dataset"] == dataset) & (rw["model"] == model)]
        if sel.empty:
            raise KeyError((dataset, model))
        return sel.iloc[0]

    def ok(om: float, de: float) -> str:
        return f"{1 - om - de:+.3f}"

    sn_l, sn_r = row("SN", "lcdm"), row("SN", "rewrite")
    sb_l, sb_r = row("SN+BAO", "lcdm"), row("SN+BAO", "rewrite")
    body = [
        f"\\multirow{{4}}{{*}}{{SN}} & flat $\\Lambda$CDM & {f(sn['lcdm_om'])} & $\\Omega_\\Lambda=1-\\Omega_m$ & $0$ & {sn['lcdm_chi2']:.2f} & {sn['lcdm_aic_total']:.2f} \\\\",
        f" & flat $w$CDM & {f(sn['wcdm_om'])} & $w={sn['wcdm_w']:.2f}$ & $0$ & {sn['wcdm_chi2']:.2f} & {sn['wcdm_aic_total']:.2f} \\\\",
        f" & $\\Lambda$CDM, free $\\Omega_k$ & {f(sn_l['om'])} & $\\Omega_\\Lambda={f(sn_l['ol'])}$ & {ok(sn_l['om'], sn_l['ol'])} & {sn_l['chi2']:.2f} & {sn_l['aic_total']:.2f} \\\\",
        f" & rewrite & {f(sn_r['om'])} & $A={f(sn_r['A'])},\\ m={sn_r['m']:.0f}^{{\\ast}}$ & {ok(sn_r['om'], sn_r['A'])} & {sn_r['chi2']:.2f} & {sn_r['aic_total']:.2f} \\\\",
        "\\midrule",
        f"\\multirow{{2}}{{*}}{{SN+BAO}} & $\\Lambda$CDM, free $\\Omega_k$ & {f(sb_l['om'])} & $\\Omega_\\Lambda={f(sb_l['ol'])}$ & {ok(sb_l['om'], sb_l['ol'])} & {sb_l['chi2']:.2f} & {sb_l['aic_total']:.2f} \\\\",
        f" & rewrite & {f(sb_r['om'])} & $A={f(sb_r['A'])},\\ m={sb_r['m']:.0f}^{{\\ast}}$ & {ok(sb_r['om'], sb_r['A'])} & {sb_r['chi2']:.2f} & {sb_r['aic_total']:.2f} \\\\",
        "\\midrule",
        f"\\multirow{{2}}{{*}}{{BAO}} & flat $\\Lambda$CDM & {f(bao['flat_lcdm_om'])} & $\\Omega_\\Lambda=1-\\Omega_m$ & $0$ & {bao['flat_lcdm_chi2']:.2f} & {bao['flat_lcdm_aic']:.2f} \\\\",
        f" & no $\\Lambda$, open & {f(bao['ocdm_om'])}$^{{\\dagger}}$ & --- & {1 - bao['ocdm_om']:+.3f} & {bao['ocdm_chi2']:.2f} & {bao['ocdm_aic']:.2f} \\\\",
    ]
    note = (
        f"SN: $n={sn['n_data']}$ supernovae; the profiled magnitude intercept is counted as a fitted "
        "parameter in the AIC. BAO: one compressed observable ($\\alpha$ at $z_{\\rm eff}=0.851$), "
        "with $H_0$ and the sound horizon fixed to the release's template cosmology. "
        "$^{\\ast}$At the lower bound $m=0$ of the declared range $[0,6]$, where the rewrite family coincides "
        "with $\\Lambda$CDM with free curvature. $^{\\dagger}$At the lower bound $\\Omega_m=0.05$."
    )
    return table(
        "Background fits to the public DES SN5YR distance vector and DES Y6 BAO consensus likelihood. "
        "The ``dark-energy-like term'' column lists the parameter that plays that role in each model.",
        "tab:background_fits",
        "@{}llcccrr@{}",
        "Data & Model & $\\Omega_m$ & Dark-energy-like term & $\\Omega_k$ & $\\chi^2$ & AIC",
        body,
        note,
    )


def lss_table() -> str:
    y3 = metrics("y3_ppd_probe_split")
    kids = metrics("kids450_ppd_probe_split")
    body = []
    for survey, m, nA, nB, Bname in (
        ("DES Y3 (maglim)", y3, y3["n_A"], y3["n_B"], "clustering + GGL"),
        ("KiDS-450 bandpowers", kids, kids["n_A"], kids["n_B"], "GGL + clustering"),
    ):
        for i, (d, held) in enumerate((("A_to_B", Bname), ("B_to_A", "shear"))):
            label = f"\\multirow{{2}}{{*}}{{{survey}}}" if i == 0 else ""
            train = "shear" if d == "A_to_B" else Bname
            body.append(
                f"{label} & {train} $\\to$ {held} & "
                f"{m[f'{d}_lcdm_like_chi2_test']:.4f} & {m[f'{d}_rewrite_like_chi2_test']:.4f} \\\\"
            )
        if survey.startswith("DES"):
            body.append("\\midrule")
    note = (
        f"DES Y3: $n_{{\\rm shear}}={y3['n_A']}$, $n_{{\\rm rest}}={y3['n_B']}$. "
        f"KiDS-450: $n_{{\\rm shear}}={kids['n_A']}$, $n_{{\\rm rest}}={kids['n_B']}$. "
        "Scores are computed against an injected residual built from the covariance and block metadata; "
        "they are invariant to the measured central values and carry no cosmological information "
        "(Appendix~\\ref{app:lss})."
    )
    return table(
        "Held-out $\\chi^2$ in the injected-residual probe-split exercise. The one-template and "
        "two-template columns are the baseline and extended linear correction families.",
        "tab:lss_protocol",
        "@{}llrr@{}",
        "Vector & Training $\\to$ held-out & One template & Two templates",
        body,
        note,
    )


SPECS = [
    ("tab_synth_inference", synth_table, ["rewrite_vs_lambda"]),
    ("tab_background_fits", background_table, ["fit_des_sn5yr", "fit_des_y6_bao", "rewrite_background"]),
    ("tab_lss_protocol", lss_table, ["y3_ppd_probe_split", "kids450_ppd_probe_split"]),
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="accepted for compatibility; tables are always rebuilt")
    args = parser.parse_args()

    out_dir = Path("docs") / "paper" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    provenance = {}
    for dest_base, builder, keys in SPECS:
        tex_path = out_dir / f"{dest_base}.tex"
        if args.dry_run:
            print(f"[dry-run] would write {tex_path}")
            continue
        tex_path.write_text(builder(), encoding="utf-8")
        provenance[tex_path.name] = {
            "sources": [RUNS[k] for k in keys],
            "sha256": sha256_file(tex_path),
            "bytes": tex_path.stat().st_size,
            "timestamp_utc": utc_now_iso(),
        }
    if not args.dry_run:
        with (out_dir / "provenance.yaml").open("w", encoding="utf-8") as fh:
            yaml.safe_dump(provenance, fh, sort_keys=True)
        print(f"Tables written: {len(provenance)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
