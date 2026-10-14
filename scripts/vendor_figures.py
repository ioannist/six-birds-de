import argparse
import hashlib
import shutil
from datetime import datetime, timezone
from pathlib import Path

import yaml


# Raw run-bundle figures vendored as-is. The composed manuscript figures are
# built separately by scripts/make_paper_figures.py from the same run bundles.
# All manuscript figures are now composed by scripts/make_paper_figures.py from
# the audited run bundles; no raw run-bundle figure is vendored unchanged.
FIG_SPECS: list[dict] = [
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = Path(".")
    dest_dir = root / "docs" / "paper" / "figures"
    provenance_path = dest_dir / "provenance.yaml"

    copied = 0
    skipped = 0
    provenance = {}

    for spec in FIG_SPECS:
        src_rel = spec["src_rel"]
        dest_name = spec["dest_name"]
        src_path = root / src_rel
        dest_path = dest_dir / dest_name

        if not src_path.exists():
            raise FileNotFoundError(f"Missing source figure: {src_rel}")

        if args.dry_run:
            print(f"[dry-run] {src_rel} -> {dest_path}")
            continue

        dest_dir.mkdir(parents=True, exist_ok=True)

        if dest_path.exists() and not args.overwrite:
            src_hash = sha256_file(src_path)
            dest_hash = sha256_file(dest_path)
            if src_hash == dest_hash:
                skipped += 1
            else:
                shutil.copy2(src_path, dest_path)
                copied += 1
        else:
            shutil.copy2(src_path, dest_path)
            copied += 1

        sha = sha256_file(dest_path)
        provenance[dest_name] = {
            "run_dir": str(Path(src_rel).parent),
            "source_file": Path(src_rel).name,
            "source_relpath": src_rel,
            "dest_relpath": str(dest_path.relative_to(root)),
            "sha256": sha,
            "bytes": dest_path.stat().st_size,
            "timestamp_utc": utc_now_iso(),
        }

    if args.dry_run:
        print("[dry-run] no files copied; no provenance written")
        return 0

    with provenance_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(provenance, f, sort_keys=True)

    print(f"Copied: {copied}, Skipped: {skipped}")
    print(f"Wrote provenance: {provenance_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
