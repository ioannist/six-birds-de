"""Utilities for DES Y3 chain metadata recovery."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def scan_chain_metadata_files(raw_dir: Path) -> Dict[str, Any]:
    raw_dir = Path(raw_dir)
    candidates = []
    readme_paths = []
    paramnames_paths = []
    patterns = [
        "*param*name*",
        "*params*",
        "*labels*",
        "*header*",
        "*readme*",
        "*.yaml",
        "*.yml",
        "*.ini",
        "*.json",
        "*.tex",
        "*.md",
        "*.paramnames",
    ]
    for pat in patterns:
        for p in raw_dir.rglob(pat):
            if p.is_file():
                candidates.append(p)
                name = p.name.lower()
                if "readme" in name or name.endswith(".md"):
                    readme_paths.append(p)
                if "param" in name or name.endswith(".paramnames"):
                    paramnames_paths.append(p)
    return {
        "candidates": sorted({str(p) for p in candidates}),
        "readme_paths": sorted({str(p) for p in readme_paths}),
        "paramnames_paths": sorted({str(p) for p in paramnames_paths}),
        "notes": "",
    }


def parse_paramnames_file(path: Path) -> List[str]:
    lines = Path(path).read_text().splitlines()
    names: List[str] = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("%"):
            continue
        parts = line.split()
        if not parts:
            continue
        if parts[0].isdigit() and len(parts) > 1:
            names.append(parts[1])
        else:
            names.append(parts[0])
    return names


def parse_chain_header_comments(path: Path, *, max_lines: int = 200) -> Dict[str, Any]:
    header_lines = []
    parsed_names: List[str] = []
    with Path(path).open("r", encoding="utf-8", errors="ignore") as f:
        for _ in range(max_lines):
            line = f.readline()
            if not line:
                break
            if line.startswith("#") or line.startswith("%"):
                header_lines.append(line.rstrip())
            else:
                break
    for line in header_lines:
        stripped = line.lstrip("#%").strip()
        if stripped.lower().startswith("columns:"):
            stripped = stripped.split(":", 1)[1].strip()
        parts = stripped.split()
        if len(parts) >= 2 and all(p.isidentifier() for p in parts):
            parsed_names = parts
            break
    source = "comments" if parsed_names else "none"
    return {"header_lines": header_lines, "parsed_names": parsed_names, "source": source}


def load_chain_with_names(chain_path: Path, names: List[str] | None, *, name_source: str | None = None) -> Dict[str, Any]:
    df = pd.read_csv(chain_path, delim_whitespace=True, comment="#", header=None)
    n_cols = df.shape[1]
    name_source = name_source
    if names is not None and len(names) == n_cols:
        df.columns = names
        if name_source is None:
            name_source = "paramnames_file"
    else:
        df.columns = [f"col{i}" for i in range(n_cols)]
        if names and name_source is None:
            name_source = "none/mismatch"
        if name_source is None:
            name_source = "none"
    return {
        "df": df,
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "column_names": list(df.columns),
        "name_source": name_source,
    }


def infer_loglike_columns(columns: List[str]) -> Dict[str, str]:
    info: Dict[str, str] = {}
    lower = [c.lower() for c in columns]
    def find(keys: List[str]) -> str | None:
        for key in keys:
            for c in columns:
                if key == c.lower():
                    return c
        for key in keys:
            for c in columns:
                if key in c.lower():
                    return c
        return None

    logpost = find(["logpost", "lnpost", "post"])
    loglike = find(["loglike", "lnlike", "like"])
    chi2 = find(["chi2", "chisq"])
    weight = find(["weight", "w"])
    if logpost:
        info["logpost"] = logpost
    if loglike:
        info["loglike"] = loglike
    if chi2:
        info["chi2"] = chi2
    if weight:
        info["weight"] = weight
    return info


def extract_bestfit_row(df: pd.DataFrame, *, info: Dict[str, str]) -> Dict[str, Any] | None:
    if "logpost" in info:
        col = info["logpost"]
        idx = int(df[col].idxmax())
        score = float(df.loc[idx, col])
        criterion = "logpost"
    elif "loglike" in info:
        col = info["loglike"]
        idx = int(df[col].idxmax())
        score = float(df.loc[idx, col])
        criterion = "loglike"
    elif "chi2" in info:
        col = info["chi2"]
        idx = int(df[col].idxmin())
        score = float(df.loc[idx, col])
        criterion = "chi2"
    else:
        return None
    params = df.loc[idx].to_dict()
    return {
        "criterion": criterion,
        "row_index": idx,
        "params": params,
        "score": score,
    }
