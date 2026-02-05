#!/usr/bin/env python3
"""Build DES Y3 2pt block index and diagnostics."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sixbirds_cosmo import manifest
from sixbirds_cosmo.data_registry import fetch
from sixbirds_cosmo.datasets.des_y3_2pt import build_y3_block_index, load_des_y3_2pt_fits
from sixbirds_cosmo.reporting import write_run_markdown_stub


def main() -> None:
    seed = 123
    run_dir = manifest.create_run_dir("des_y3_blockmap", seed=seed)

    raw_dir = Path(fetch("des_y3a2_datavectors_2pt"))
    fits_path = raw_dir / "2pt_maglim_covupdate.fits"

    data = load_des_y3_2pt_fits(fits_path)
    data_vector = data["data_vector"]
    cov = data["cov"]

    block_index = build_y3_block_index(fits_path)

    n_data = len(data_vector)
    if len(block_index) != n_data:
        raise ValueError("block_index length does not match data_vector length.")
    if not np.array_equal(block_index["i"].to_numpy(dtype=int), np.arange(n_data)):
        raise ValueError("block_index i column is not 0..n-1.")
    if np.max(np.abs(block_index["value"].to_numpy(dtype=float) - data_vector)) > 1e-12:
        raise ValueError("block_index values do not match data_vector ordering.")

    counts = block_index["probe"].value_counts().to_dict()
    n_shear = int(counts.get("shear", 0))
    n_clust = int(counts.get("clustering", 0))
    n_ggl = int(counts.get("ggl", 0))
    n_unknown = int(counts.get("unknown", 0))

    block_index_path = run_dir / "block_index.csv"
    block_index.to_csv(block_index_path, index=False)

    block_counts = pd.DataFrame(
        [
            {
                "n_data": n_data,
                "n_shear": n_shear,
                "n_clust": n_clust,
                "n_ggl": n_ggl,
                "n_unknown": n_unknown,
            }
        ]
    )
    block_counts_path = run_dir / "block_counts.csv"
    manifest.save_table(block_counts, block_counts_path)

    # Plot with boundaries where probe/stat or source_hdu changes
    fig, ax = plt.subplots()
    ax.plot(block_index["i"], block_index["value"])
    change = (block_index["probe"] + ":" + block_index["stat"] + ":" + block_index["source_hdu"]).to_numpy()
    boundaries = np.where(change[1:] != change[:-1])[0] + 1
    for b in boundaries:
        ax.axvline(b, linewidth=0.5)
    ax.set_xlabel("i")
    ax.set_ylabel("value")
    ax.set_title("DES Y3 block boundaries")
    plot_path = run_dir / "block_boundaries.png"
    manifest.save_fig(fig, plot_path)
    plt.close(fig)

    unique_stats = sorted(set(block_index["stat"].dropna().unique()))
    unique_probes = sorted(set(block_index["probe"].dropna().unique()))

    try:
        rel_fits = fits_path.relative_to(Path.cwd())
        fits_rel_str = str(rel_fits)
    except Exception:
        fits_rel_str = str(fits_path)

    metrics = {
        "n_data": n_data,
        "n_shear": n_shear,
        "n_clust": n_clust,
        "n_ggl": n_ggl,
        "n_unknown": n_unknown,
        "unique_stats": unique_stats,
        "unique_probes": unique_probes,
        "fits_file": fits_rel_str,
        "block_index_csv": "block_index.csv",
        "plot_file": "block_boundaries.png",
    }
    config = {
        "seed": seed,
        "fits_file": fits_rel_str,
        "n_data": n_data,
    }

    manifest.write_config(run_dir, config)
    manifest.write_metrics(run_dir, metrics)
    manifest.write_provenance(run_dir, "des_y3_blockmap", seed)
    write_run_markdown_stub(run_dir)

    print(str(run_dir))


if __name__ == "__main__":
    main()
