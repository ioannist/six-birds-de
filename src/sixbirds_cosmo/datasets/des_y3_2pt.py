"""Loaders for DES Y3A2 2pt datavectors and chains."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
import re

import numpy as np
import pandas as pd
from astropy.io import fits


def _hdu_summary(hdu, idx: int) -> Dict[str, Any]:
    name = (hdu.name or "").strip()
    hdu_type = type(hdu).__name__
    shape = None
    cols = None
    if hasattr(hdu, "data") and hdu.data is not None:
        try:
            shape = hdu.data.shape
        except Exception:
            shape = None
    if hasattr(hdu, "columns") and hdu.columns is not None:
        cols = [col.name for col in hdu.columns]
    return {"index": idx, "name": name, "type": hdu_type, "shape": shape, "columns": cols}


def _is_numeric_array(arr: np.ndarray) -> bool:
    return np.issubdtype(arr.dtype, np.number)


def _extract_covariance(hdul) -> tuple[np.ndarray, str]:
    # Prefer ImageHDU with COV in EXTNAME
    for hdu in hdul:
        name = (hdu.name or "").lower()
        if "cov" in name and hasattr(hdu, "data") and hdu.data is not None:
            data = np.asarray(hdu.data)
            if data.ndim == 2 and data.shape[0] == data.shape[1]:
                return data.astype(float), "COV_IMAGE_HDU"

    # Fallback: search table columns for square vector
    for hdu in hdul:
        if "cov" not in (hdu.name or "").lower():
            continue
        if not hasattr(hdu, "data") or hdu.data is None:
            continue
        if not hasattr(hdu, "columns") or hdu.columns is None:
            continue
        for col in hdu.columns:
            try:
                col_data = np.asarray(hdu.data[col.name])
            except Exception:
                continue
            if not _is_numeric_array(col_data):
                continue
            flat = col_data.reshape(-1)
            n = int(np.sqrt(flat.size))
            if n * n == flat.size:
                cov = flat.reshape((n, n)).astype(float)
                return cov, f"COV_TABLE_COL:{col.name}"

    raise ValueError("No covariance HDU/column found.")


def _extract_vector_from_hdu(hdu) -> np.ndarray | None:
    if not hasattr(hdu, "data") or hdu.data is None:
        return None
    data = hdu.data
    if isinstance(data, np.ndarray) and data.dtype.names is None:
        if data.ndim == 1 and _is_numeric_array(data):
            return np.asarray(data, dtype=float)
        if data.ndim == 2 and 1 in data.shape and _is_numeric_array(data):
            return np.asarray(data).reshape(-1).astype(float)
        return None
    if hasattr(hdu, "columns") and hdu.columns is not None:
        numeric_cols = [col.name for col in hdu.columns if np.issubdtype(col.dtype, np.number)]
        if len(numeric_cols) == 1:
            return np.asarray(hdu.data[numeric_cols[0]], dtype=float).reshape(-1)
    return None


def _extract_vector_direct(hdul) -> tuple[np.ndarray | None, str]:
    for hdu in hdul:
        name = (hdu.name or "").lower()
        if "cov" not in name and ("data" in name or "vector" in name):
            vec = _extract_vector_from_hdu(hdu)
            if vec is not None:
                return vec, f"DATA_VECTOR_HDU:{hdu.name}"
    return None, "DATA_VECTOR_NOT_FOUND"


def _spectrum_names(hdul) -> tuple[list[str], list[int] | None]:
    """Read the declared covariance ordering, including repeated FITS A formats."""
    for hdu in hdul:
        if "cov" not in (hdu.name or "").lower():
            continue
        labels = sorted(int(m.group(1)) for key in hdu.header
                        if (m := re.fullmatch(r"NAME_(\d+)", key)))
        if labels:
            if labels != list(range(len(labels))):
                raise ValueError("covariance spectrum labels must be consecutive")
            names = [str(hdu.header[f"NAME_{i}"]).strip() for i in labels]
            starts = [hdu.header.get(f"STRT_{i}") for i in labels]
            if any(type(start) is not int or start < 0 for start in starts) or starts[0] != 0 or any(b <= a for a, b in zip(starts, starts[1:])):
                raise ValueError("covariance spectrum starts must be distinct increasing integers from zero")
            if len(set(name.lower() for name in names)) != len(names):
                raise ValueError("covariance spectrum names must be distinct")
            return names, starts
    for hdu in hdul:
        if "spectra" not in (hdu.name or "").lower() or not hasattr(hdu, "columns"):
            continue
        cols = [col.name for col in hdu.columns if re.fullmatch(r"\d*A", str(col.format))]
        if not cols:
            raise ValueError("declared spectra table has no string name column")
        preferred = [c for c in cols if c.lower() in {"name", "extname", "spectrum"}]
        if len(preferred) != 1 and len(cols) != 1:
            raise ValueError("declared spectra table has ambiguous name columns")
        column = preferred[0] if len(preferred) == 1 else cols[0]
        names = [(val.decode() if isinstance(val, bytes) else str(val)).strip()
                 for val in hdu.data[column]]
        if not names or any(not n for n in names) or len(set(n.lower() for n in names)) != len(names):
            raise ValueError("declared spectrum names must be nonempty and distinct")
        return names, None
    return [], None


def _extract_vector_from_spectra(hdul) -> tuple[np.ndarray | None, List[str], List[Dict[str, int]], str]:
    ext_names, expected_starts = _spectrum_names(hdul)
    if not ext_names:
        return None, [], [], "SPECTRA_NOT_FOUND"
    return _extract_declared_spectra(hdul, ext_names, expected_starts)


def _extract_declared_spectra(hdul, ext_names, expected_starts):
    """Do not silently replace a missing VALUE column by a bin-number column."""
    values, slices, spectra_used = [], [], []
    start = 0
    for j, ext in enumerate(ext_names):
        if expected_starts is not None and expected_starts[j] != start:
            raise ValueError("spectrum length does not agree with covariance start metadata")
        hdu = hdul[ext]
        if not hasattr(hdu, "columns") or "VALUE" not in hdu.columns.names:
            raise ValueError(f"spectrum {ext} requires an explicit VALUE column")
        vec = np.asarray(hdu.data["VALUE"], dtype=float).reshape(-1)
        if not np.all(np.isfinite(vec)):
            raise ValueError("spectrum values must be finite")
        end = start + len(vec)
        spectra_used.append(ext)
        slices.append({"name": ext, "start": start, "end": end})
        values.append(vec)
        start = end
    return np.concatenate(values), spectra_used, slices, "DECLARED_SPECTRA_CONCAT"




def _extract_vector_from_known(hdul) -> tuple[np.ndarray | None, List[str], List[Dict[str, int]], str]:
    order = ["xip", "xim", "gammat", "wtheta"]
    values = []
    spectra_used = []
    slices = []
    start = 0
    for name in order:
        for hdu in hdul:
            if (hdu.name or "").lower() != name:
                continue
            if not hasattr(hdu, "columns") or hdu.columns is None:
                continue
            if "VALUE" not in hdu.columns.names:
                continue
            vec = np.asarray(hdu.data["VALUE"], dtype=float).reshape(-1)
            end = start + len(vec)
            spectra_used.append(hdu.name)
            slices.append({"name": hdu.name, "start": start, "end": end})
            values.append(vec)
            start = end
    if not values:
        return None, [], [], "KNOWN_STATS_NOT_FOUND"
    return np.concatenate(values).astype(float), spectra_used, slices, "KNOWN_STATS_CONCAT"


def _stat_from_name(name: str) -> str:
    lname = name.lower()
    if "xip" in lname:
        return "xip"
    if "xim" in lname:
        return "xim"
    if "wtheta" in lname or ("w" in lname and "theta" in lname):
        return "wtheta"
    if "gammat" in lname or "gamma_t" in lname:
        return "gammat"
    return "unknown"


def _probe_from_stat(stat: str) -> str:
    if stat in {"xip", "xim"}:
        return "shear"
    if stat == "wtheta":
        return "clustering"
    if stat == "gammat":
        return "ggl"
    return "unknown"


def _find_column(columns: List[str], candidates: List[str]) -> str | None:
    lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def _first_float_column(columns: List[str], hdu, exclude: List[str]) -> str | None:
    exclude_lower = {c.lower() for c in exclude}
    for col in hdu.columns:
        if col.name.lower() in exclude_lower:
            continue
        if np.issubdtype(col.dtype, np.number):
            return col.name
    return None


def _column_unit(hdu, col_name: str) -> str | None:
    try:
        col = hdu.columns[col_name]
        if col.unit is not None:
            return str(col.unit)
    except Exception:
        pass
    return None


def _build_block_rows_from_hdu(hdu, start_idx: int) -> pd.DataFrame:
    cols = [c.name for c in hdu.columns]
    value_col = _find_column(cols, ["VALUE"])
    bin1_col = _find_column(cols, ["BIN1"])
    bin2_col = _find_column(cols, ["BIN2"])
    x_col = _find_column(cols, ["ANG", "THETA", "ELL", "L", "S", "R", "RADIUS"])

    exclude = [c for c in [value_col, bin1_col, bin2_col, x_col] if c]
    if value_col is None:
        raise ValueError(f"An explicit VALUE column is required in HDU {hdu.name}.")

    values = np.asarray(hdu.data[value_col], dtype=float).reshape(-1)
    x = np.asarray(hdu.data[x_col], dtype=float).reshape(-1) if x_col else np.full_like(values, np.nan)
    bin1 = np.asarray(hdu.data[bin1_col], dtype=int).reshape(-1) if bin1_col else np.full(values.size, np.nan)
    bin2 = np.asarray(hdu.data[bin2_col], dtype=int).reshape(-1) if bin2_col else np.full(values.size, np.nan)
    x_unit = _column_unit(hdu, x_col) if x_col else None

    stat = _stat_from_name(hdu.name or "")
    probe = _probe_from_stat(stat)
    n = values.size
    df = pd.DataFrame(
        {
            "i": np.arange(start_idx, start_idx + n, dtype=int),
            "probe": probe,
            "stat": stat,
            "bin1": bin1,
            "bin2": bin2,
            "x": x,
            "x_unit": [x_unit] * n,
            "value": values,
            "source_hdu": [hdu.name] * n,
        }
    )
    return df


def build_y3_block_index(path: Path, *, prefer_order: str = "spectra_table") -> pd.DataFrame:
    """Build block index mapping each data vector element to metadata."""
    path = Path(path)
    with fits.open(path, memmap=False) as hdul:
        hdu_summary = [_hdu_summary(hdu, idx) for idx, hdu in enumerate(hdul)]

        block_frames: List[pd.DataFrame] = []
        start_idx = 0

        spectra_hdu = None
        for hdu in hdul:
            name = (hdu.name or "").lower()
            if "spectra" in name and hasattr(hdu, "data") and hdu.data is not None:
                spectra_hdu = hdu
                break

        def add_hdu_by_name(ext_name: str) -> None:
            nonlocal start_idx
            try:
                hdu = hdul[ext_name]
            except Exception as exc:
                raise ValueError(f"Missing spectra extension {ext_name}") from exc
            if not hasattr(hdu, "data") or hdu.data is None:
                raise ValueError(f"Empty spectra extension {ext_name}")
            block_frames.append(_build_block_rows_from_hdu(hdu, start_idx))
            start_idx += len(block_frames[-1])

        ext_names, expected_starts = _spectrum_names(hdul)
        if ext_names:
            for j, ext in enumerate(ext_names):
                if expected_starts is not None and expected_starts[j] != start_idx:
                    raise ValueError("block lengths do not agree with covariance spectrum ordering")
                add_hdu_by_name(ext)

        if not block_frames:
            known_order = ["xip", "xim", "gammat", "wtheta"]
            for name in known_order:
                for hdu in hdul:
                    if (hdu.name or "").lower() == name:
                        block_frames.append(_build_block_rows_from_hdu(hdu, start_idx))
                        start_idx += len(block_frames[-1])

        if not block_frames:
            for hdu in hdul:
                if not hasattr(hdu, "data") or hdu.data is None:
                    continue
                if not hasattr(hdu, "columns") or hdu.columns is None:
                    continue
                try:
                    block_frames.append(_build_block_rows_from_hdu(hdu, start_idx))
                    start_idx += len(block_frames[-1])
                except Exception:
                    continue

        if not block_frames:
            raise ValueError(f"Could not build block index. HDU summary: {hdu_summary}")

        block_index = pd.concat(block_frames, ignore_index=True)
        block_index["i"] = block_index["i"].astype(int)
        return block_index


def load_des_y3_2pt_fits(path: Path) -> Dict[str, Any]:
    path = Path(path)
    with fits.open(path, memmap=False) as hdul:
        hdu_summary = [_hdu_summary(hdu, idx) for idx, hdu in enumerate(hdul)]
        cov, cov_method = _extract_covariance(hdul)

        vec_direct, direct_method = _extract_vector_direct(hdul)
        vec_spectra, spectra_used, spectra_slices, spectra_method = _extract_vector_from_spectra(hdul)
        vec_known, known_used, known_slices, known_method = _extract_vector_from_known(hdul)

        declared_names, _ = _spectrum_names(hdul)
        if declared_names and (vec_spectra is None or vec_spectra.size != cov.shape[0]):
            raise ValueError("declared spectra do not cover the covariance dimension")

        data_vector = None
        extraction_method = ""
        spectra_used_final: List[str] = []
        spectra_slices_final: List[Dict[str, int]] = []
        if declared_names:
            data_vector = vec_spectra
            extraction_method = spectra_method
            spectra_used_final = spectra_used
            spectra_slices_final = spectra_slices
            if vec_direct is not None and vec_direct.size == cov.shape[0] and not np.array_equal(vec_direct, vec_spectra):
                raise ValueError("direct vector disagrees with the declared covariance ordering")
        elif vec_direct is not None and vec_direct.size == cov.shape[0]:
            data_vector = vec_direct
            extraction_method = direct_method
        elif vec_spectra is not None and vec_spectra.size == cov.shape[0]:
            data_vector = vec_spectra
            extraction_method = spectra_method
            spectra_used_final = spectra_used
            spectra_slices_final = spectra_slices
        elif vec_known is not None and vec_known.size == cov.shape[0]:
            data_vector = vec_known
            extraction_method = known_method
            spectra_used_final = known_used
            spectra_slices_final = known_slices
        else:
            msg = (
                f"Data vector length mismatch. cov={cov.shape}, "
                f"direct_len={None if vec_direct is None else vec_direct.size}, "
                f"spectra_len={None if vec_spectra is None else vec_spectra.size}, "
                f"known_len={None if vec_known is None else vec_known.size}."
            )
            raise ValueError(msg + f" HDU summary: {hdu_summary}")

    return {
        "path": str(path),
        "hdu_summary": hdu_summary,
        "data_vector": data_vector,
        "cov": cov,
        "n_data": int(data_vector.size),
        "cov_shape": cov.shape,
        "spectra_used": spectra_used_final,
        "spectra_slices": spectra_slices_final,
        "extraction_method": extraction_method or cov_method,
    }


@dataclass
class ChainSummary:
    n_rows: int
    columns: List[str]
    summary: Dict[str, Any]
    data: pd.DataFrame | None = None


def load_des_y3_chain_txt(path: Path, *, max_rows: int | None = None) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        lines = [line for line in f if line.strip() and not line.lstrip().startswith("#")]

    header = None
    if lines:
        first = lines[0].strip().split()
        try:
            [float(x) for x in first]
        except Exception:
            header = 0

    df = pd.read_csv(
        path,
        sep=r"\s+",
        comment="#",
        header=header,
        nrows=max_rows,
    )

    if header is None:
        df.columns = [f"col{i}" for i in range(df.shape[1])]

    columns = list(df.columns)
    summary = {}
    target = None
    for col in columns:
        if "omegam" in col.lower():
            target = col
            break
    if target is not None:
        vals = np.asarray(df[target], dtype=float)
        summary = {
            "column": target,
            "p16": float(np.percentile(vals, 16)),
            "p50": float(np.percentile(vals, 50)),
            "p84": float(np.percentile(vals, 84)),
        }

    return {
        "n_rows": int(df.shape[0]),
        "columns": columns,
        "summary": summary,
        "data": df if max_rows is not None else None,
    }
