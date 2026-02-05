#!/usr/bin/env python3
"""Wrapper for Y6 3x2pt smoke run with blocked behavior."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import yaml

from sixbirds_cosmo import manifest


def dataset_exists(dataset_key: str) -> bool:
    reg_path = Path("data/registry.yaml")
    if not reg_path.exists():
        return False
    reg = yaml.safe_load(reg_path.read_text())
    return dataset_key in reg.get("datasets", {})


def run_availability_check() -> Path:
    script = Path("scripts/check_des_y6_3x2pt_availability.py")
    subprocess.run(["python", str(script)], check=True)
    return Path("docs/experiments/y6_3x2pt_availability.json")


def main() -> None:
    cfg_path = Path("configs/3x2pt_y6_3x2pt_smoke.yaml")
    cfg = yaml.safe_load(cfg_path.read_text())
    dataset_key = cfg["dataset_key"]

    if not dataset_exists(dataset_key):
        report_path = run_availability_check()
        run_dir = manifest.create_run_dir("des_y6_3x2pt_smoke", seed=123)
        if "placeholder" in dataset_key:
            blocked_reason = "placeholder config; waiting for public Y6 3×2pt artifacts"
        else:
            blocked_reason = f"dataset_key {dataset_key} not in registry"
        metrics = {
            "available": False,
            "blocked_reason": blocked_reason,
            "availability_report": str(report_path),
        }
        config = {
            "config_path": str(cfg_path),
            "dataset_key": dataset_key,
        }
        manifest.write_config(run_dir, config)
        manifest.write_metrics(run_dir, metrics)
        manifest.write_provenance(run_dir, "des_y6_3x2pt_smoke", 123)
        from sixbirds_cosmo.reporting import write_run_markdown_stub

        write_run_markdown_stub(run_dir)
        print(str(run_dir))
        return

    subprocess.run(
        [
            "python",
            "experiments/run_3x2pt_smoke_like.py",
            "--config",
            str(cfg_path),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
