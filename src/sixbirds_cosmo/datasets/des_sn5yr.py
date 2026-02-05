"""DES-SN5YR distance modulus data loader and likelihood."""

from __future__ import annotations

import csv
import gzip
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve

from sixbirds_cosmo.cosmo_background import luminosity_distance, luminosity_distance_wcdm
from sixbirds_cosmo.likelihoods.base import Dataset


MU_ERR_CANDIDATES = ["MUERR", "muerr", "MU_ERR", "MUERR_FINAL", "SIGMU", "sigma_mu"]


def _extract_subdir(zip_path: Path, subdir: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.namelist():
            if subdir in member and not member.endswith("/"):
                zf.extract(member, out_dir)
    return out_dir


def _write_filelist(root: Path) -> None:
    filelist = root / "FILELIST.txt"
    with filelist.open("w") as handle:
        for path in sorted([p for p in root.rglob("*") if p.is_file()]):
            handle.write(f"{path.relative_to(root)}\n")


def extract_distances_covmat(raw_dir: Path) -> Path:
    raw_dir = Path(raw_dir)
    zip_path = raw_dir / "DES-SN5YR_1.3.zip"
    if not zip_path.exists():
        raise FileNotFoundError(f"Missing zip: {zip_path}")

    out_dir = raw_dir / "extracted" / "4_DISTANCES_COVMAT"
    if not out_dir.exists() or not any(out_dir.iterdir()):
        _extract_subdir(zip_path, "4_DISTANCES_COVMAT", out_dir)
        _write_filelist(out_dir)
    return out_dir


def _read_text_file(path: Path) -> str:
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as handle:
            return handle.read()
    return path.read_text()


def _find_mu_error_column(columns: list[str]) -> str | None:
    for candidate in MU_ERR_CANDIDATES:
        if candidate in columns:
            return candidate
    return None


def _read_hd_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_csv(path, delim_whitespace=True)


def select_hd_file_matching_cov(raw_dir: Path, n_cov: int) -> Path:
    extract_dir = extract_distances_covmat(raw_dir)
    candidates = sorted([p for p in extract_dir.rglob("*.csv")])
    candidates += sorted([p for p in extract_dir.rglob("*.txt")])

    viable = []
    for path in candidates:
        try:
            df = _read_hd_table(path)
        except Exception:
            continue
        columns = list(df.columns)
        if not any(col.lower() == "zhd" for col in columns):
            continue
        if not any(col in columns for col in ["MU", "mu", "mub"]):
            continue
        viable.append((path, len(df)))

    matches = [item for item in viable if item[1] == n_cov]
    if not matches:
        details = ", ".join([f"{p.name}:{n}" for p, n in viable])
        raise ValueError(f"No HD file matches N_cov={n_cov}. Candidates: {details}")

    def score(path: Path) -> int:
        name = path.name.lower()
        score_val = 0
        if "hd" in name:
            score_val += 2
        if "metadata" in name:
            score_val += 1
        return score_val

    matches.sort(key=lambda item: score(item[0]), reverse=True)
    return matches[0][0]


def load_des_sn5yr_hubble_diagram(raw_dir: Path, n_cov: int | None = None) -> Dict[str, Any]:
    extract_dir = extract_distances_covmat(raw_dir)
    if n_cov is None:
        candidates = sorted(extract_dir.rglob("*.csv"))
        if not candidates:
            raise FileNotFoundError("No CSV files found in 4_DISTANCES_COVMAT.")
        hd_path = candidates[0]
    else:
        hd_path = select_hd_file_matching_cov(raw_dir, n_cov)

    df = _read_hd_table(hd_path)
    columns = list(df.columns)

    z_col = None
    for col in columns:
        if col.lower() == "zhd":
            z_col = col
            break
    if z_col is None:
        raise ValueError("zHD column not found in hubble diagram file.")

    mu_col = None
    for candidate in ["MU", "mu", "mub"]:
        if candidate in columns:
            mu_col = candidate
            break
    if mu_col is None:
        raise ValueError("No MU/mu/mub column found in hubble diagram file.")

    mu_err_col = _find_mu_error_column(columns)

    z = df[z_col].to_numpy(dtype=float)
    mu = df[mu_col].to_numpy(dtype=float)
    mu_err = df[mu_err_col].to_numpy(dtype=float) if mu_err_col else None

    return {
        "z": z,
        "mu": mu,
        "mu_err": mu_err,
        "mu_err_col": mu_err_col,
        "n": int(len(z)),
        "columns": columns,
        "filename": str(hd_path),
    }


def _parse_cov_dense(text: str) -> np.ndarray:
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        rows.append([float(x) for x in parts])
    return np.array(rows, dtype=float)


def _parse_cov_vector(text: str) -> np.ndarray:
    values = [float(x) for x in text.split() if x.strip()]
    n = int(np.sqrt(len(values)))
    if n * n != len(values):
        raise ValueError("Flat covariance vector has unexpected length.")
    return np.array(values, dtype=float).reshape(n, n)


def _parse_cov_vector_with_header(lines: list[str]) -> np.ndarray:
    header = lines[0].strip()
    try:
        n_header = int(float(header))
    except ValueError as exc:
        raise ValueError("Header does not contain an integer size.") from exc
    values = [float(x) for line in lines[1:] for x in line.split() if x.strip()]
    if len(values) != n_header * n_header:
        raise ValueError("Flat covariance vector length does not match header size.")
    return np.array(values, dtype=float).reshape(n_header, n_header)


def parse_triplet_cov(path: Path, *, assume_symmetric: bool = True) -> Dict[str, Any]:
    text = _read_text_file(path)
    triplets = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        i, j, val = int(parts[0]), int(parts[1]), float(parts[2])
        triplets.append((i, j, val))

    if not triplets:
        raise ValueError("No triplets found in covariance file.")

    min_i = min(i for i, _, _ in triplets)
    max_i = max(i for i, _, _ in triplets)
    min_j = min(j for _, j, _ in triplets)
    max_j = max(j for _, j, _ in triplets)

    if any(i == 0 or j == 0 for i, j, _ in triplets):
        index_base = "0-based"
        norm_triplets = triplets
    elif min(min_i, min_j) == 1:
        index_base = "1-based"
        norm_triplets = [(i - 1, j - 1, v) for i, j, v in triplets]
    else:
        index_base = "0-based"
        norm_triplets = triplets

    max_i_n = max(i for i, _, _ in norm_triplets)
    max_j_n = max(j for _, j, _ in norm_triplets)
    n = max(max_i_n, max_j_n) + 1
    cov = np.zeros((n, n), dtype=float)

    for i, j, val in norm_triplets:
        existing = cov[i, j]
        if existing != 0.0 and abs(existing - val) > 1e-12:
            raise ValueError("Duplicate triplet entries are inconsistent.")
        cov[i, j] = val
        if assume_symmetric:
            existing_sym = cov[j, i]
            if existing_sym != 0.0 and abs(existing_sym - val) > 1e-12:
                raise ValueError("Duplicate symmetric triplets are inconsistent.")
            cov[j, i] = val

    return {
        "cov": cov,
        "N": n,
        "index_base_detected": index_base,
        "triplet_min_i": min_i,
        "triplet_max_i": max_i,
        "triplet_min_j": min_j,
        "triplet_max_j": max_j,
        "n_triplets": len(triplets),
    }


def load_des_sn5yr_covariance(
    raw_dir: Path, *, kind: str, mu_err: np.ndarray | None = None
) -> Dict[str, Any]:
    extract_dir = extract_distances_covmat(raw_dir)
    cov_files = [p for p in extract_dir.rglob("*") if p.is_file() and p.suffix in {".txt", ".dat", ".gz"}]

    def matches(path: Path) -> bool:
        name = path.name.lower()
        if kind == "stat+sys":
            return "stat" in name and "sys" in name
        if kind == "stat":
            return "stat" in name and "sys" not in name
        raise ValueError("kind must be 'stat' or 'stat+sys'.")

    selected = [p for p in cov_files if matches(p)]
    if not selected:
        raise FileNotFoundError(f"No covariance file found for kind={kind}.")

    cov_path = selected[0]
    text = _read_text_file(cov_path)
    lines = [line for line in text.splitlines() if line.strip() and not line.strip().startswith("#")]
    first_parts = lines[0].split() if lines else []

    format_detected = "dense"
    triplet_info = None

    if len(first_parts) == 3:
        triplet_info = parse_triplet_cov(cov_path)
        cov = triplet_info["cov"]
        format_detected = "triplet"
    elif len(first_parts) == 1:
        try:
            cov = _parse_cov_vector_with_header(lines)
            format_detected = "flat"
        except Exception:
            cov = _parse_cov_vector(text)
            format_detected = "flat"
    else:
        cov = _parse_cov_dense(text)
        if cov.shape[0] != cov.shape[1]:
            raise ValueError("Dense covariance is not square.")

    diag = np.diag(cov)
    matrix_kind = "covariance"
    if mu_err is not None and np.allclose(diag, 0.0):
        if len(mu_err) != cov.shape[0]:
            raise ValueError("mu_err length does not match covariance size for diag build.")
        cov = np.diag(mu_err**2)
        matrix_kind = "diag_from_muerr"
        diag = np.diag(cov)
    if mu_err is not None and np.median(diag) >= 0.5 and np.median(diag) <= 2.0:
        if len(mu_err) != cov.shape[0]:
            raise ValueError("mu_err length does not match covariance size for rescaling.")
        cov = cov * np.outer(mu_err, mu_err)
        matrix_kind = "correlation_rescaled"

    if kind == "stat+sys" and mu_err is not None and matrix_kind == "covariance":
        if len(mu_err) != cov.shape[0]:
            raise ValueError("mu_err length does not match covariance size for stat+sys add.")
        cov = cov + np.diag(mu_err**2)
        matrix_kind = "systematics_plus_stat"

    result = {
        "cov": cov,
        "N": cov.shape[0],
        "filename": str(cov_path),
        "format_detected": format_detected,
        "matrix_kind_detected": matrix_kind,
    }
    if triplet_info:
        result.update(triplet_info)
    return result


def prepare_cov_solve(cov: np.ndarray, *, jitter0: float = 0.0) -> Dict[str, Any]:
    jitter = jitter0
    for _ in range(10):
        try:
            cho = cho_factor(cov + jitter * np.eye(cov.shape[0]), lower=True)
            if jitter > 1e-6:
                raise RuntimeError("Cholesky required excessive jitter.")
            return {"cho": cho, "jitter": jitter, "psd_correction": False, "min_eig": None}
        except Exception:
            jitter = 1e-12 if jitter == 0.0 else jitter * 10.0
            if jitter > 1e-6:
                break

    evals, evecs = np.linalg.eigh(cov)
    min_eig = float(evals[0])
    evals_clipped = np.clip(evals, 1e-12, None)
    cov_psd = (evecs * evals_clipped) @ evecs.T
    cho = cho_factor(cov_psd, lower=True)
    return {"cho": cho, "jitter": 0.0, "psd_correction": True, "min_eig": min_eig}


def chi2_profiled_over_M(resid0: np.ndarray, cho: Tuple[np.ndarray, bool]) -> Tuple[float, float]:
    ones = np.ones_like(resid0)
    invC_resid = cho_solve(cho, resid0)
    invC_ones = cho_solve(cho, ones)
    denom = float(ones @ invC_ones)
    numer = float(ones @ invC_resid)
    M_hat = numer / denom
    chi2 = float(resid0 @ invC_resid) - numer**2 / denom
    return chi2, M_hat


@dataclass
class DESSN5YRDistanceDataset(Dataset):
    z: np.ndarray = field(default_factory=lambda: np.array([]))
    mu: np.ndarray = field(default_factory=lambda: np.array([]))
    cov: np.ndarray = field(default_factory=lambda: np.array([[]]))
    cov_kind: str = ""
    model_kind: str = "flat_lcdm"
    H0_fixed: float = 70.0
    last_M_hat: float | None = None
    cho: Tuple[np.ndarray, bool] | None = None

    def loglike(self, theta: np.ndarray) -> float:
        theta = np.asarray(theta, dtype=float)
        if self.model_kind == "flat_lcdm":
            om = float(theta[0])
            dl = luminosity_distance(self.z, self.H0_fixed, om, 1.0 - om)
        elif self.model_kind == "flat_wcdm":
            om, w = float(theta[0]), float(theta[1])
            dl = luminosity_distance_wcdm(self.z, self.H0_fixed, om, w)
        else:
            raise ValueError("model_kind must be 'flat_lcdm' or 'flat_wcdm'.")

        mu_theory = 5.0 * np.log10(np.asarray(dl, dtype=float)) + 25.0
        resid0 = self.mu - mu_theory

        if self.cho is None:
            self.cho = prepare_cov_solve(self.cov)["cho"]

        chi2, M_hat = chi2_profiled_over_M(resid0, self.cho)
        self.last_M_hat = M_hat
        return -0.5 * chi2
