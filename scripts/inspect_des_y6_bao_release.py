#!/usr/bin/env python3
"""Inspect DES Y6 BAO release assets and emit an inventory."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

from sixbirds_cosmo.data_registry import fetch

KEYWORDS = ["alpha", "rd", "dm", "da", "dv", "distance", "compressed", "constraint", "cov"]


def extract_zip(zip_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)


def main() -> None:
    dataset_dir = fetch("des_y6_bao_release")
    dataset_dir = Path(dataset_dir)
    extract_root = dataset_dir / "extracted"
    extract_root.mkdir(parents=True, exist_ok=True)

    zip_files = sorted(dataset_dir.glob("*.zip"))
    inventory = {
        "dataset": "des_y6_bao_release",
        "files": [],
        "keyword_hits": [],
    }

    for zip_path in zip_files:
        out_dir = extract_root / zip_path.stem
        if not out_dir.exists() or not any(out_dir.iterdir()):
            extract_zip(zip_path, out_dir)

        for path in sorted(out_dir.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(extract_root).as_posix()
            size = path.stat().st_size
            hits = [kw for kw in KEYWORDS if kw in rel.lower()]
            record = {"path": rel, "bytes": size, "keyword_hits": hits}
            inventory["files"].append(record)
            if hits:
                inventory["keyword_hits"].append(record)

    inventory_path = extract_root / "INVENTORY.json"
    inventory_path.write_text(json.dumps(inventory, indent=2))

    hits = inventory["keyword_hits"]
    hits_sorted = sorted(hits, key=lambda x: x["path"])[:30]
    for entry in hits_sorted:
        print(entry["path"])


if __name__ == "__main__":
    main()
