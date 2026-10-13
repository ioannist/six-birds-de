"""DES Y6 BAO dataset utilities and alpha-likelihood."""

from __future__ import annotations

import csv
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from sixbirds_cosmo.cosmo_background import (
    H_lcdm,
    angular_diameter_distance,
    transverse_comoving_distance,
)
from sixbirds_cosmo.likelihoods.base import Dataset


def _extract_zip(zip_path: Path, extract_dir: Path) -> None:
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)


def _ensure_extracted(raw_dir: Path, zip_name: str, extract_subdir: str) -> Path:
    raw_dir = Path(raw_dir)
    zip_path = raw_dir / zip_name
    if not zip_path.exists():
        raise FileNotFoundError(f"Missing zip: {zip_path}")
    extract_root = raw_dir / "extracted"
    extract_root.mkdir(parents=True, exist_ok=True)
    extract_dir = extract_root / extract_subdir
    if not extract_dir.exists() or not any(extract_dir.iterdir()):
        _extract_zip(zip_path, extract_dir)
    return extract_dir


def _write_filelist(extract_dir: Path) -> list[Path]:
    files = sorted([p for p in extract_dir.rglob("*") if p.is_file()])
    list_path = extract_dir / "FILELIST.txt"
    with list_path.open("w") as handle:
        for path in files:
            handle.write(f"{path.relative_to(extract_dir)}\n")
    return files


def _looks_like_data_file(path: Path) -> bool:
    return path.suffix.lower() in {".txt", ".dat", ".csv"}


def _looks_like_cov_file(path: Path) -> bool:
    name = path.name.lower()
    return any(key in name for key in ["cov", "covmat", "covariance"]) and _looks_like_data_file(path)


def _parse_numeric_table(path: Path) -> np.ndarray:
    rows: list[list[float]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = re.split(r"[\s,]+", line)
        try:
            row = [float(val) for val in parts if val]
        except ValueError:
            continue
        if row:
            rows.append(row)
    return np.array(rows, dtype=float)


def _extract_statistic_and_bin(path: Path) -> tuple[str, str]:
    statistic = path.parent.name.lower()
    bin_match = re.search(r"bin(\d+)", path.name.lower())
    bin_label = f"bin{bin_match.group(1)}" if bin_match else "unknown"
    return statistic, bin_label


def load_des_y6_bao_dataset(raw_dir: Path) -> Dict[str, Any]:
    """Load clustering datavectors (ACF/APS/PCF) for transparency."""
    extract_dir = _ensure_extracted(
        raw_dir, "DESY6BAO_datavectors.zip", "DESY6BAO_datavectors"
    )

    files = _write_filelist(extract_dir)
    data_candidates = [p for p in files if _looks_like_data_file(p)]

    parsed_csv = Path(raw_dir) / "extracted" / "parsed_datavector.csv"
    with parsed_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["statistic", "bin", "x", "value", "sigma"]
        )
        writer.writeheader()
        for path in data_candidates:
            if _looks_like_cov_file(path):
                continue
            table = _parse_numeric_table(path)
            if table.ndim == 1:
                table = table.reshape(1, -1)
            if table.shape[1] < 3:
                continue
            statistic, bin_label = _extract_statistic_and_bin(path)
            for row in table:
                writer.writerow(
                    {
                        "statistic": statistic,
                        "bin": bin_label,
                        "x": float(row[0]),
                        "value": float(row[1]),
                        "sigma": float(row[2]),
                    }
                )

    return {
        "n": len(data_candidates),
        "notes": {
            "parsed_csv": str(parsed_csv.relative_to(raw_dir)),
            "data_files": [str(p.relative_to(raw_dir)) for p in data_candidates],
        },
    }


def load_alpha_likelihood_curve(raw_dir: Path, kind: str) -> Dict[str, Any]:
    if kind not in {"acf", "aps", "pcf", "fid"}:
        raise ValueError("kind must be one of: acf, aps, pcf, fid")

    extract_dir = _ensure_extracted(raw_dir, "DESY6BAO_likelihood.zip", "DESY6BAO_likelihood")
    filename = f"likelihood_{kind}.txt"
    path = extract_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Likelihood file not found: {path}")

    alpha_vals: list[float] = []
    col2_vals: list[float] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("%"):
            continue
        parts = re.split(r"[\s,]+", line)
        if len(parts) < 2:
            continue
        try:
            alpha = float(parts[0])
            val = float(parts[1])
        except ValueError:
            continue
        alpha_vals.append(alpha)
        col2_vals.append(val)

    alpha_grid = np.array(alpha_vals, dtype=float)
    col2 = np.array(col2_vals, dtype=float)
    if alpha_grid.size == 0:
        raise ValueError(f"No data parsed from {path}")

    sort_idx = np.argsort(alpha_grid)
    alpha_grid = alpha_grid[sort_idx]
    col2 = col2[sort_idx]

    # The release README declares this column to be Delta chi-square. Its
    # meaning is not inferred from its sign (log likelihoods can be positive).
    if not np.all(np.isfinite(alpha_grid)) or not np.all(np.isfinite(col2)):
        raise ValueError("Likelihood curve must be finite.")
    if np.any(np.diff(alpha_grid) <= 0):
        raise ValueError("Likelihood alpha grid must be strictly increasing.")
    delta_chi2 = col2.copy()
    delta_chi2 = delta_chi2 - np.min(delta_chi2)

    return {
        "alpha_grid": alpha_grid,
        "delta_chi2_grid": delta_chi2,
        "kind": kind,
        "filename": str(path),
        "notes": {"mode": "delta_chi2", "combination": "released_consensus" if kind == "fid" else "single_statistic"},
    }


def evaluate_alpha_chi2(alpha: float, alpha_grid: np.ndarray, delta_chi2: np.ndarray) -> float:
    """Interpolate within released support; never flatten tails by edge filling."""
    alpha_grid = np.asarray(alpha_grid, dtype=float)
    delta_chi2 = np.asarray(delta_chi2, dtype=float)
    if alpha_grid.ndim != 1 or len(alpha_grid) < 2 or delta_chi2.shape != alpha_grid.shape:
        raise ValueError("Likelihood grid and chi-square must be matching vectors.")
    if not np.all(np.isfinite(alpha_grid)) or not np.all(np.isfinite(delta_chi2)) or np.any(np.diff(alpha_grid) <= 0):
        raise ValueError("Likelihood grid must be finite and strictly increasing.")
    if not np.isfinite(alpha) or alpha < alpha_grid[0] or alpha > alpha_grid[-1]:
        return float("inf")
    return float(np.interp(alpha, alpha_grid, delta_chi2))


def load_bao_fiducial_info(raw_dir: Path) -> Dict[str, Any]:
    raw_dir = Path(raw_dir)
    z_eff = None
    source = "default"

    candidates = []
    readme = raw_dir / "README.md"
    if readme.exists():
        candidates.append(readme)
    extract_root = raw_dir / "extracted"
    if extract_root.exists():
        candidates.extend(sorted(extract_root.rglob("*.txt")))

    pattern = re.compile(r"z[_ ]?eff\s*=?\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)
    for path in candidates:
        try:
            text = path.read_text()
        except Exception:
            continue
        match = pattern.search(text)
        if match:
            z_eff = float(match.group(1))
            source = str(path)
            break

    if z_eff is None:
        # Released Likelihood/README.md specifies z_eff=0.851.
        z_eff = 0.851
        source = "DES Y6 BAO Likelihood/README.md (release convention)"

    return {"z_eff": z_eff, "z_eff_source": source, "fid_model_notes": {}}


def _validate_distance_like(z: float, alpha: float) -> None:
    if z < 0.0 or z > 3.0:
        raise ValueError("z_eff out of plausible range; check fiducial parsing.")
    if not (1e-2 <= abs(alpha) <= 1e2):
        raise ValueError("alpha_pred out of plausible range; check model mapping.")


@dataclass
class DESY6BAOAlphaDataset(Dataset):
    alpha_grid: np.ndarray = field(default_factory=lambda: np.array([]))
    delta_chi2_grid: np.ndarray = field(default_factory=lambda: np.array([]))
    z_eff: float = 0.851
    om_fid: float = 0.315
    ol_fid: float = 0.685
    H0_fid: float = 67.4
    model_kind: str = "flat_lcdm"
    notes: Dict[str, Any] = field(default_factory=dict)

    def alpha_pred(self, om: float) -> float | None:
        if self.model_kind == "ocdm":
            ol = 0.0
        elif self.model_kind == "flat_lcdm":
            ol = 1.0 - om
            if ol < 0.0 or ol > 1.5:
                return None
        else:
            raise ValueError("model_kind must be 'ocdm' or 'flat_lcdm'.")

        dm_pred = transverse_comoving_distance(
            self.z_eff, self.H0_fid, om, ol, c_km_s=299792.458
        )
        dm_fid = transverse_comoving_distance(
            self.z_eff, self.H0_fid, self.om_fid, self.ol_fid, c_km_s=299792.458
        )
        alpha = float(dm_pred / dm_fid)
        _validate_distance_like(self.z_eff, alpha)
        return alpha

    def loglike(self, theta: np.ndarray) -> float:
        theta = np.asarray(theta, dtype=float)
        if theta.shape != (1,) or not np.all(np.isfinite(theta)):
            raise ValueError("compressed BAO model requires exactly one finite parameter")
        om = float(theta[0])
        alpha = self.alpha_pred(om)
        if alpha is None:
            return -0.5 * 1e6

        chi2 = evaluate_alpha_chi2(alpha, self.alpha_grid, self.delta_chi2_grid)
        self.notes["alpha_outside_support"] = not np.isfinite(chi2)
        return -0.5 * chi2


@dataclass
class BAODataset(Dataset):
    z: np.ndarray | None = None
    obs_type: List[str] | None = None
    data: np.ndarray | None = None
    cov: np.ndarray | None = None
    rd_mpc: float = 147.05
    c_km_s: float = 299792.458
    model_kind: str = "flat_lcdm"
    fiducial: Dict[str, Any] | None = None

    def loglike(self, theta: np.ndarray) -> float:
        theta = np.asarray(theta, dtype=float)
        if theta.shape != (2,) or not np.all(np.isfinite(theta)):
            raise ValueError("BAO vector model requires exactly two finite parameters")
        H0, om = float(theta[0]), float(theta[1])

        if self.model_kind == "ocdm":
            ol = 0.0
        elif self.model_kind == "flat_lcdm":
            ol = 1.0 - om
        else:
            raise ValueError("model_kind must be 'ocdm' or 'flat_lcdm'.")

        z = np.asarray(self.z, dtype=float)
        data = np.asarray(self.data, dtype=float)
        if z.ndim != 1 or not len(z) or data.shape != z.shape or not np.all(np.isfinite(z)) or np.any(z < 0) or not np.all(np.isfinite(data)):
            raise ValueError("BAO data and nonnegative redshifts must be matching finite vectors")
        if np.shape(self.cov) != (len(z), len(z)):
            raise ValueError("BAO covariance must match the data vector")
        if self.obs_type is None or len(self.obs_type) != len(z):
            raise ValueError("one observable type is required for each BAO data point")
        if not np.isfinite(self.rd_mpc) or self.rd_mpc <= 0 or not np.isfinite(self.c_km_s) or self.c_km_s <= 0:
            raise ValueError("BAO sound horizon and light speed must be finite and positive")
        obs_type = self.obs_type
        predictions: list[float] = []

        for zi, obs in zip(z, obs_type):
            if obs == "DM_over_rd":
                dm = transverse_comoving_distance(zi, H0, om, ol, c_km_s=self.c_km_s)
                predictions.append(float(dm / self.rd_mpc))
            elif obs == "DA_over_rd":
                da = angular_diameter_distance(zi, H0, om, ol, c_km_s=self.c_km_s)
                predictions.append(float(da / self.rd_mpc))
            elif obs in {"DH_over_rd", "c_over_H_over_rd"}:
                hz = H_lcdm(zi, H0, om, ol)
                predictions.append(float((self.c_km_s / hz) / self.rd_mpc))
            elif obs == "DV_over_rd":
                dm = transverse_comoving_distance(zi, H0, om, ol, c_km_s=self.c_km_s)
                hz = H_lcdm(zi, H0, om, ol)
                dv = (dm**2 * (self.c_km_s * zi) / hz) ** (1.0 / 3.0)
                predictions.append(float(dv / self.rd_mpc))
            elif obs == "alpha":
                if not self.fiducial or "DM_over_rd_fid" not in self.fiducial:
                    raise ValueError("alpha requires fiducial DM_over_rd_fid in dataset fiducial.")
                dm = transverse_comoving_distance(zi, H0, om, ol, c_km_s=self.c_km_s)
                pred = float(dm / self.rd_mpc) / float(self.fiducial["DM_over_rd_fid"])
                predictions.append(pred)
            else:
                raise ValueError(f"Unknown obs_type: {obs}")

        residual = np.asarray(self.data, dtype=float) - np.asarray(predictions, dtype=float)
        from sixbirds_cosmo.likelihoods.gaussian_vector import chi2_gaussian
        chi2, _ = chi2_gaussian(residual, self.cov)
        return -0.5 * chi2
