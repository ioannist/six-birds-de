#!/usr/bin/env python3
"""CLI for fetching datasets from the registry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sixbirds_cosmo.data_registry import fetch, find_repo_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch datasets listed in data/registry.yaml")
    parser.add_argument("--dataset", required=True, help="Dataset name in the registry")
    parser.add_argument("--force", action="store_true", help="Force re-fetch even if cached")
    args = parser.parse_args()

    dataset_dir = fetch(args.dataset, force=args.force)
    repo_root = find_repo_root()
    try:
        rel_dir = dataset_dir.relative_to(repo_root)
        output_dir = str(rel_dir)
    except ValueError:
        output_dir = str(dataset_dir)

    provenance_path = dataset_dir / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    files = provenance.get("files", [])
    filenames = [entry.get("filename", "") for entry in files]
    cache_hit = any(entry.get("cache_hit", False) for entry in files)

    print(f"Dataset: {args.dataset}")
    print(f"Output: {output_dir}")
    print(f"Files: {', '.join(filenames)}")
    print(f"Cache hit: {'yes' if cache_hit else 'no'}")


if __name__ == "__main__":
    main()
