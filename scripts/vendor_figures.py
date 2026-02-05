import argparse
import hashlib
import shutil
from datetime import datetime, timezone
from pathlib import Path

import yaml


FIG_SPECS = [
    {
        "dest_name": "fig_toy1_rm_summary_vs_beta.png",
        "src_rel": "results/toy1/20260202T175625Z_nogit/rm_summary_vs_beta.png",
    },
    {
        "dest_name": "fig_toy2_Q_vs_time.png",
        "src_rel": "results/toy2_patch/20260202T182630Z_nogit/Q_vs_time.png",
    },
    {
        "dest_name": "fig_infer_distance_fit.png",
        "src_rel": "results/infer_distance/20260202T190430Z_nogit/distance_fit.png",
    },
    {
        "dest_name": "fig_rewrite_vs_lambda_distance_fit.png",
        "src_rel": "results/rewrite_vs_lambda/20260202T194715Z_nogit/rewrite_distance_fit.png",
    },
    {
        "dest_name": "fig_ppd_synth_rmse_across_seeds.png",
        "src_rel": "results/ppd_synthetic/20260202T201136Z_nogit/ppd_rmse_across_seeds.png",
    },
    {
        "dest_name": "fig_des_sn5yr_mu_residuals.png",
        "src_rel": "results/fit_des_sn5yr/20260205T121850Z_0e613291189050d851e90a930e75f83ee7085b9b/sn5yr_mu_residuals.png",
    },
    {
        "dest_name": "fig_des_y6_bao_alpha_likelihood.png",
        "src_rel": "results/fit_des_y6_bao/20260203T075814Z_nogit/alpha_likelihood.png",
    },
    {
        "dest_name": "fig_ppd_background_bao_alpha.png",
        "src_rel": "results/ppd_background/20260203T094137Z_nogit/ppd_bao_alpha.png",
    },
    {
        "dest_name": "fig_y3_block_boundaries.png",
        "src_rel": "results/des_y3_blockmap/20260204T045551Z_nogit/block_boundaries.png",
    },
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
