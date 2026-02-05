import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml
import pandas as pd


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_metrics(run_dir: Path) -> dict:
    path = run_dir / "metrics.json"
    return json.loads(path.read_text())


def pick(metrics: dict, candidates: list[str]) -> tuple[str, float]:
    for key in candidates:
        if key in metrics:
            return key, metrics[key]
    raise KeyError(
        f"Missing expected keys. Tried: {candidates}. Available: {sorted(metrics.keys())}"
    )


def fmt(val):
    if isinstance(val, (int, float)):
        return f"{val:.6g}"
    return str(val)


def write_csv(path: Path, headers: list[str], rows: list[list]):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for row in rows:
            writer.writerow(row)


def latex_escape(text: str) -> str:
    return text.replace("\\", "\\textbackslash{}").replace("_", "\\_")


def write_tex_table(path: Path, caption: str, label: str, headers: list[str], rows: list[list]):
    col_spec = "l" * len(headers)
    with path.open("w", encoding="utf-8") as f:
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write(f"\\caption{{{latex_escape(caption)}}}\n")
        f.write(f"\\label{{{label}}}\n")
        f.write(f"\\begin{{tabular}}{{{col_spec}}}\n")
        f.write("\\toprule\n")
        f.write(" " + " & ".join(latex_escape(h) for h in headers) + " \\\\\n")
        f.write("\\midrule\n")
        for row in rows:
            f.write(" " + " & ".join(latex_escape(str(x)) for x in row) + " \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = Path(".")
    out_dir = root / "docs" / "paper" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = [
        {
            "table_key": "toy1",
            "dest_base": "tab_toy1_summary",
            "run_dirs": ["results/toy1/20260202T175625Z_nogit"],
            "kind": "toy1",
        },
        {
            "table_key": "toy2",
            "dest_base": "tab_toy2_summary",
            "run_dirs": ["results/toy2_patch/20260202T182630Z_nogit"],
            "kind": "toy2",
        },
        {
            "table_key": "synth_inference",
            "dest_base": "tab_synth_inference_compare",
            "run_dirs": ["results/rewrite_vs_lambda/20260202T194715Z_nogit"],
            "kind": "synth_inference",
        },
        {
            "table_key": "background_fits",
            "dest_base": "tab_background_fits",
            "run_dirs": [
                "results/rewrite_background_vs_lcdm/20260203T101804Z_nogit",
                "results/fit_des_y6_bao/20260203T075814Z_nogit",
            ],
            "kind": "background_fits",
        },
    ]

    provenance = {}
    written = 0

    for spec in specs:
        dest_base = spec["dest_base"]
        csv_path = out_dir / f"{dest_base}.csv"
        tex_path = out_dir / f"{dest_base}.tex"

        if args.dry_run:
            print(f"[dry-run] would write {csv_path} and {tex_path}")
            continue

        if csv_path.exists() and tex_path.exists() and not args.overwrite:
            pass

        if spec["kind"] == "toy1":
            metrics = load_metrics(root / spec["run_dirs"][0])
            headers = ["metric", "value"]
            rows = [
                ["beta_linear_case_rm_max", fmt(metrics["beta_linear_case_rm_max"])],
                ["beta_small_rm_max", fmt(metrics["beta_small_rm_max"])],
                ["beta_large_rm_max", fmt(metrics["beta_large_rm_max"])],
                ["delta_max_overall", fmt(metrics["delta_max_overall"])],
            ]
            write_csv(csv_path, headers, rows)
            write_tex_table(
                tex_path,
                "Toy 1 mismatch summary.",
                "tab:toy1",
                headers,
                rows,
            )

        elif spec["kind"] == "toy2":
            metrics = load_metrics(root / spec["run_dirs"][0])
            headers = ["metric", "value"]
            rows = [
                ["homogeneous_Q_max_abs", fmt(metrics["homogeneous_Q_max_abs"])],
                ["hetero_Q_peak", fmt(metrics["hetero_Q_peak"])],
                ["hetero_Q_mean_late", fmt(metrics["hetero_Q_mean_late"])],
                ["hetero_frac_ddot_aD_pos_late", fmt(metrics["hetero_frac_ddot_aD_pos_late"])],
            ]
            write_csv(csv_path, headers, rows)
            write_tex_table(
                tex_path,
                "Toy 2 patch model summary.",
                "tab:toy2",
                headers,
                rows,
            )

        elif spec["kind"] == "synth_inference":
            metrics = load_metrics(root / spec["run_dirs"][0])
            # strict requirement for matter-only keys
            if "chi2_matter_only" not in metrics or "aic_matter_only" not in metrics:
                raise KeyError(
                    "Missing matter-only keys in rewrite_vs_lambda metrics. "
                    f"Available: {sorted(metrics.keys())}. "
                    "Regenerate rewrite_vs_lambda run if needed."
                )
            headers = ["model", "chi2", "aic", "param"]
            rows = [
                [
                    "matter_only",
                    fmt(metrics["chi2_matter_only"]),
                    fmt(metrics["aic_matter_only"]),
                    "",
                ],
                [
                    "lcdm",
                    fmt(metrics["chi2_lcdm"]),
                    fmt(metrics["aic_lcdm"]),
                    f"Omega_L={fmt(metrics.get('lcdm_omega_lambda', ''))}",
                ],
                [
                    "rewrite",
                    fmt(metrics["chi2_rewrite"]),
                    fmt(metrics["aic_rewrite"]),
                    f"A={fmt(metrics.get('rewrite_A', ''))}",
                ],
            ]
            write_csv(csv_path, headers, rows)
            write_tex_table(
                tex_path,
                "Synthetic inference comparison.",
                "tab:synth_inference",
                headers,
                rows,
            )

        elif spec["kind"] == "background_fits":
            run_rw = root / spec["run_dirs"][0]
            run_bao = root / spec["run_dirs"][1]
            summary_path = run_rw / "summary.csv"
            if not summary_path.exists():
                raise FileNotFoundError(f"Missing summary.csv: {summary_path}")
            df = pd.read_csv(summary_path)

            def get_row(dataset, model):
                row = df[(df["dataset"] == dataset) & (df["model"] == model)]
                if row.empty:
                    raise KeyError(f"Missing row for dataset={dataset}, model={model}")
                return row.iloc[0]

            sn_lcdm = get_row("SN", "lcdm")
            sn_rw = get_row("SN", "rewrite")
            snb_lcdm = get_row("SN+BAO", "lcdm")
            snb_rw = get_row("SN+BAO", "rewrite")

            metrics_bao = load_metrics(run_bao)
            _, chi2_bao = pick(metrics_bao, ["flat_lcdm_chi2", "flat_lcdm_chi2", "chi2_lcdm"])
            _, aic_bao = pick(metrics_bao, ["flat_lcdm_aic", "aic_lcdm", "flat_lcdm_aic"])

            headers = ["dataset", "model", "chi2", "aic"]
            rows = [
                ["SN", "lcdm", fmt(sn_lcdm["chi2"]), fmt(sn_lcdm["aic_total"])],
                ["SN", "rewrite", fmt(sn_rw["chi2"]), fmt(sn_rw["aic_total"])],
                ["SN+BAO", "lcdm", fmt(snb_lcdm["chi2"]), fmt(snb_lcdm["aic_total"])],
                ["SN+BAO", "rewrite", fmt(snb_rw["chi2"]), fmt(snb_rw["aic_total"])],
                ["BAO", "lcdm", fmt(chi2_bao), fmt(aic_bao)],
            ]

            write_csv(csv_path, headers, rows)
            write_tex_table(
                tex_path,
                "Background fits summary.",
                "tab:background_fits",
                headers,
                rows,
            )

        else:
            raise ValueError(f"Unknown table kind: {spec['kind']}")

        sha = sha256_file(tex_path)
        provenance[f"{dest_base}.tex"] = {
            "sources": spec["run_dirs"],
            "metrics_files": [
                str(Path(d) / "metrics.json") for d in spec["run_dirs"] if (Path(d) / "metrics.json").exists()
            ],
            "generated_files": [
                str(Path("docs/paper/tables") / f"{dest_base}.csv"),
                str(Path("docs/paper/tables") / f"{dest_base}.tex"),
            ],
            "sha256": sha,
            "bytes": tex_path.stat().st_size,
            "timestamp_utc": utc_now_iso(),
        }
        written += 1

    if not args.dry_run:
        provenance_path = out_dir / "provenance.yaml"
        with provenance_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(provenance, f, sort_keys=True)
        print(f"Tables written: {written}")
        print(f"Output dir: {out_dir}")
        print(f"Provenance: {provenance_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
