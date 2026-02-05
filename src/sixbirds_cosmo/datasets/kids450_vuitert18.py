"""KiDS-450 vUitert18 bandpowers loader."""

from __future__ import annotations

import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor


ELL_BINS = [200, 331, 548, 906, 1500]


def _extract_tar_member(tar_path: Path, member_name: str) -> Path:
    out_dir = tar_path.parent / "extracted"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / member_name
    if out_path.exists():
        out_path.unlink()
    with tarfile.open(tar_path, "r:gz") as tar:
        member = None
        for m in tar.getmembers():
            name = m.name.lstrip("./")
            if name == member_name or name.endswith("/" + member_name):
                if "/._" in name or name.startswith("._"):
                    continue
                member = m
                break
        if member is None:
            for m in tar.getmembers():
                if m.name.endswith(member_name):
                    if "/._" in m.name or m.name.startswith("._"):
                        continue
                    member = m
                    break
        if member is None:
            raise FileNotFoundError(f"Member not found in tar: {member_name}")
        tar.extract(member, out_dir)
        extracted_path = out_dir / member.name
        if extracted_path.exists() and extracted_path != out_path:
            extracted_path.rename(out_path)
    return out_path


def _read_datavector(path: Path) -> np.ndarray:
    values: List[float] = []
    for line in path.read_text(encoding="latin-1", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        floats = [float(p) for p in parts]
        values.append(floats[-1])
    data = np.array(values, dtype=float)
    if data.size != 100:
        raise ValueError(f"Expected 100 data points, got {data.size}.")
    return data


def _parse_triplet_cov(path: Path, n_expected: int) -> Tuple[np.ndarray, float]:
    triplets = []
    for line in path.read_text(encoding="latin-1", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        i, j, val = int(parts[0]), int(parts[1]), float(parts[2])
        triplets.append((i, j, val))

    if not triplets:
        raise ValueError("No covariance triplets parsed.")

    min_idx = min(min(i, j) for i, j, _ in triplets)
    max_idx = max(max(i, j) for i, j, _ in triplets)
    if min_idx == 1 and max_idx == n_expected:
        offset = -1
    elif min_idx == 0 and max_idx == n_expected - 1:
        offset = 0
    else:
        raise ValueError(
            f"Unrecognized triplet index base: min={min_idx}, max={max_idx}, N={n_expected}"
        )

    cov = np.zeros((n_expected, n_expected), dtype=float)
    for i, j, val in triplets:
        ii = i + offset
        jj = j + offset
        cov[ii, jj] = val
        cov[jj, ii] = val

    symmetry = float(np.max(np.abs(cov - cov.T)))
    if symmetry > 1e-12:
        raise ValueError(f"Covariance symmetry check failed: max|C-C^T|={symmetry}")

    jitter_used = 0.0
    for jitter in [0.0, 1e-12, 1e-10, 1e-8, 1e-6]:
        try:
            cho_factor(cov + jitter * np.eye(n_expected), lower=True)
            jitter_used = jitter
            break
        except Exception:
            continue
    return cov, jitter_used


def _build_block_index(data_vector: np.ndarray) -> pd.DataFrame:
    rows = []
    i = 0

    # shear: S1..S4, unique pairs
    for b1 in range(1, 5):
        for b2 in range(b1, 5):
            for ell in ELL_BINS:
                rows.append(
                    {
                        "i": i,
                        "probe": "shear",
                        "stat": "bandpower_E",
                        "bin1": b1,
                        "bin2": b2,
                        "x": float(ell),
                        "x_unit": "ell",
                        "value": float(data_vector[i]),
                        "source": "vUitert18_KiDS450_data",
                    }
                )
                i += 1

    # ggl: F1..F2 with S1..S4
    for f in range(1, 3):
        for s in range(1, 5):
            for ell in ELL_BINS:
                rows.append(
                    {
                        "i": i,
                        "probe": "ggl",
                        "stat": "bandpower_gm",
                        "bin1": f,
                        "bin2": s,
                        "x": float(ell),
                        "x_unit": "ell",
                        "value": float(data_vector[i]),
                        "source": "vUitert18_KiDS450_data",
                    }
                )
                i += 1

    # clustering: F1xF1, F2xF2
    for f in range(1, 3):
        for ell in ELL_BINS:
            rows.append(
                {
                    "i": i,
                    "probe": "clustering",
                    "stat": "bandpower_gg",
                    "bin1": f,
                    "bin2": f,
                    "x": float(ell),
                    "x_unit": "ell",
                    "value": float(data_vector[i]),
                    "source": "vUitert18_KiDS450_data",
                }
            )
            i += 1

    if i != 100:
        raise ValueError(f"Block index length mismatch: {i} rows created.")

    return pd.DataFrame(rows)


def load_kids450_vuitert18_bandpowers(raw_dir: Path) -> Dict[str, Any]:
    raw_dir = Path(raw_dir)
    tar_path = raw_dir / "vUitert18_KiDS450_data.tar.gz"
    if not tar_path.exists():
        raise FileNotFoundError(f"Missing tarball: {tar_path}")

    data_path = _extract_tar_member(tar_path, "Pkids_data_full_ibc2.dat")
    cov_path = _extract_tar_member(tar_path, "Pkidscov_data_PlanckIter2.dat")

    data_vector = _read_datavector(data_path)
    cov, jitter_used = _parse_triplet_cov(cov_path, data_vector.size)

    block_index = _build_block_index(data_vector)

    meta = {
        "ell_bins": ELL_BINS,
        "n_shear": 50,
        "n_ggl": 40,
        "n_clust": 10,
        "cov_chol_jitter_used": float(jitter_used),
        "data_file": str(data_path),
        "cov_file": str(cov_path),
    }

    return {
        "data_vector": data_vector,
        "cov": cov,
        "block_index": block_index,
        "meta": meta,
    }
