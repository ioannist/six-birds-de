#!/usr/bin/env python3
"""Smoke test for LSS config resolver."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from sixbirds_cosmo import manifest
from sixbirds_cosmo.lss.config import load_and_resolve_lss_config
from sixbirds_cosmo.reporting import write_run_markdown_stub


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/lss/smoke_small.yaml")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    resolved = load_and_resolve_lss_config(cfg_path)

    exp_name = resolved["run"]["exp_name"]
    seed = resolved["seed"]
    run_dir = manifest.create_run_dir(exp_name, seed=seed)

    metrics = {
        "schema_version": resolved["schema_version"],
        "dataset_key": resolved["dataset"]["key"],
        "dataset_file": resolved["dataset"]["file"],
        "resolved_path_exists": True,
        "mask_kind": resolved["mask"]["kind"],
        "split_kind": resolved["split"]["kind"],
        "seed": seed,
    }

    manifest.write_config(run_dir, resolved)
    manifest.write_metrics(run_dir, metrics)
    manifest.write_provenance(run_dir, exp_name, seed)
    write_run_markdown_stub(run_dir)

    resolved_path_txt = run_dir / "resolved_dataset_path.txt"
    resolved_path_txt.write_text(str(resolved["dataset"]["resolved_path"]) + "\n")

    print(str(run_dir))


if __name__ == "__main__":
    main()
