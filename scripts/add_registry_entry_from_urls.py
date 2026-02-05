#!/usr/bin/env python3
"""Add a dataset entry to data/registry.yaml by computing sha256 for URLs."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from urllib.request import urlretrieve

import yaml


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--url", action="append", required=True)
    args = parser.parse_args()

    files = []
    for url in args.url:
        filename = url.split("/")[-1]
        tmp = Path("/tmp") / filename
        urlretrieve(url, tmp)
        sha = sha256_file(tmp)
        tmp.unlink(missing_ok=True)
        files.append({"url": url, "filename": filename, "sha256": sha})

    reg_path = Path("data/registry.yaml")
    reg = yaml.safe_load(reg_path.read_text()) if reg_path.exists() else {"datasets": {}}
    reg.setdefault("datasets", {})
    reg["datasets"][args.dataset] = {
        "description": f"{args.dataset} (auto-added)",
        "version": args.version,
        "files": files,
    }
    reg_path.write_text(yaml.safe_dump(reg, sort_keys=False))

    for f in files:
        print(f"{f['filename']}: {f['sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
