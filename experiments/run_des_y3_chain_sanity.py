#!/usr/bin/env python3
"""Sanity check for DES Y3 chain metadata and best-fit extraction."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from sixbirds_cosmo import manifest
from sixbirds_cosmo.data_registry import fetch
from sixbirds_cosmo.datasets.des_y3_chain_meta import (
    extract_bestfit_row,
    infer_loglike_columns,
    load_chain_with_names,
    parse_chain_header_comments,
    parse_paramnames_file,
    scan_chain_metadata_files,
)
from sixbirds_cosmo.reporting import write_run_markdown_stub


def main() -> None:
    seed = 123
    run_dir = manifest.create_run_dir("des_y3_chain_sanity", seed=seed)

    raw_dir = Path(fetch("des_y3a2_chains_3x2pt"))
    chain_path = raw_dir / "chain_3x2pt_lcdm_maglim_opt.txt"

    scan = scan_chain_metadata_files(raw_dir)
    paramnames_paths = [Path(p) for p in scan["paramnames_paths"]]

    names = None
    name_source = "none"
    blocked = False
    blocked_reason = ""
    if paramnames_paths:
        names = parse_paramnames_file(paramnames_paths[0])
        name_source = "paramnames_file"
    else:
        header = parse_chain_header_comments(chain_path)
        if header["parsed_names"]:
            names = header["parsed_names"]
            name_source = "comments"
        else:
            blocked = True
            blocked_reason = "No paramnames file or header names found."

    load_info = load_chain_with_names(chain_path, names, name_source=name_source)
    df = load_info["df"]
    n_rows = load_info["n_rows"]
    n_cols = load_info["n_cols"]
    column_names = load_info["column_names"]
    name_source = load_info["name_source"]
    n_named_params = sum(1 for c in column_names if not c.startswith("col"))

    info = infer_loglike_columns(column_names)
    bestfit = extract_bestfit_row(df, info=info)

    paramnames_report = {
        "status": "ok" if n_named_params >= 10 else "blocked",
        "source": name_source,
        "names": column_names if n_named_params >= 10 else [],
    }
    if n_named_params < 10:
        paramnames_report["reason"] = blocked_reason or "insufficient names recovered"
        paramnames_report["dir_listing_report"] = "docs/experiments/des_y3_chains_dir_listing.json"
        if not Path(paramnames_report["dir_listing_report"]).exists():
            try:
                import subprocess

                subprocess.run(["python", "scripts/list_des_y3_chain_dir.py"], check=True)
            except Exception:
                pass
        scan = scan_chain_metadata_files(raw_dir)
        paramnames_report["searched_paths"] = scan["candidates"]

    (run_dir / "chain_paramnames.json").write_text(json.dumps(paramnames_report, indent=2))
    if bestfit is not None:
        (run_dir / "bestfit_params.json").write_text(json.dumps(bestfit, indent=2))

    preview = df.iloc[:5, :10]
    preview.to_csv(run_dir / "chain_columns_preview.csv", index=False)

    metrics = {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "name_source": name_source,
        "n_named_params": n_named_params,
        "bestfit_available": bestfit is not None,
        "bestfit_criterion": bestfit["criterion"] if bestfit is not None else None,
        "blocked": n_named_params < 10,
        "blocked_reason": paramnames_report.get("reason", ""),
        "chain_file": str(chain_path),
    }
    config = {
        "seed": seed,
        "chain_file": str(chain_path),
    }

    manifest.write_config(run_dir, config)
    manifest.write_metrics(run_dir, metrics)
    manifest.write_provenance(run_dir, "des_y3_chain_sanity", seed)
    write_run_markdown_stub(run_dir)

    print(str(run_dir))


if __name__ == "__main__":
    main()
