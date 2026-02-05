"""LSS config loader and resolver (schema v0)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

from sixbirds_cosmo.data_registry import fetch, find_repo_root


def load_yaml_config(path: Path) -> Dict[str, Any]:
    """Load YAML config and return dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("Config YAML must parse to a dict.")
    return data


def _require_str(cfg: Dict[str, Any], path: str) -> str:
    cur = cfg
    parts = path.split(".")
    for part in parts[:-1]:
        cur = cur.get(part, {})
    val = cur.get(parts[-1])
    if not isinstance(val, str):
        raise ValueError(f"{path} must be a string.")
    return val


def resolve_lss_config(cfg: Dict[str, Any], *, repo_root: Path | None = None) -> Dict[str, Any]:
    """Apply defaults, validate, and resolve dataset paths."""
    if repo_root is None:
        repo_root = find_repo_root()
    repo_root = Path(repo_root)

    resolved: Dict[str, Any] = {}
    resolved["schema_version"] = int(cfg.get("schema_version", 0))
    resolved["seed"] = int(cfg.get("seed", 0))

    dataset = dict(cfg.get("dataset", {}))
    dataset_key = dataset.get("key")
    dataset_file = dataset.get("file")
    if not isinstance(dataset_key, str):
        raise ValueError("dataset.key must be a string.")
    if not isinstance(dataset_file, str):
        raise ValueError("dataset.file must be a string.")
    dataset_kind = dataset.get("kind", "generic")
    if not isinstance(dataset_kind, str):
        raise ValueError("dataset.kind must be a string.")

    mask = dict(cfg.get("mask", {}))
    mask_kind = mask.get("kind", "all")
    if mask_kind not in {"all", "first_n", "indices"}:
        raise ValueError("mask.kind must be one of: all, first_n, indices.")
    if mask_kind == "first_n":
        first_n = mask.get("first_n")
        if not isinstance(first_n, int) or first_n < 1:
            raise ValueError("mask.first_n must be an int >= 1 when mask.kind=first_n.")
    if mask_kind == "indices":
        indices = mask.get("indices")
        if not isinstance(indices, list) or not indices or not all(isinstance(i, int) for i in indices):
            raise ValueError("mask.indices must be a non-empty list of ints when mask.kind=indices.")

    split = dict(cfg.get("split", {}))
    split_kind = split.get("kind", "none")
    if split_kind not in {"none", "first_half", "alternating", "custom"}:
        raise ValueError("split.kind must be one of: none, first_half, alternating, custom.")
    if split_kind == "custom":
        A = split.get("A")
        B = split.get("B")
        if not isinstance(A, list) or not A or not all(isinstance(i, int) for i in A):
            raise ValueError("split.A must be a non-empty list of ints when split.kind=custom.")
        if not isinstance(B, list) or not B or not all(isinstance(i, int) for i in B):
            raise ValueError("split.B must be a non-empty list of ints when split.kind=custom.")
        if set(A) & set(B):
            raise ValueError("split.A and split.B must be disjoint.")

    theory_stub = dict(cfg.get("theory_stub", {}))
    frac_sigma = theory_stub.get("frac_sigma", 0.1)
    if not isinstance(frac_sigma, (int, float)) or not float(frac_sigma) > 0 or not float(frac_sigma) == float(frac_sigma):
        raise ValueError("theory_stub.frac_sigma must be a finite float > 0.")

    run = dict(cfg.get("run", {}))
    exp_name = run.get("exp_name", "lss_config_smoke")
    note = run.get("note", "")
    if not isinstance(exp_name, str):
        raise ValueError("run.exp_name must be a string.")
    if not isinstance(note, str):
        raise ValueError("run.note must be a string.")

    resolved["dataset"] = {
        "key": dataset_key,
        "file": dataset_file,
        "kind": dataset_kind,
    }
    resolved["mask"] = {
        "kind": mask_kind,
        "first_n": mask.get("first_n"),
        "indices": mask.get("indices"),
    }
    resolved["split"] = {
        "kind": split_kind,
        "A": split.get("A"),
        "B": split.get("B"),
    }
    resolved["theory_stub"] = {"frac_sigma": float(frac_sigma)}
    resolved["run"] = {"exp_name": exp_name, "note": note}

    raw_dir = Path(fetch(dataset_key))
    try:
        rel_dir = raw_dir.relative_to(repo_root)
        resolved_dir = str(rel_dir)
    except Exception:
        resolved_dir = str(raw_dir)

    resolved_path = raw_dir / dataset_file
    if not resolved_path.exists():
        raise FileNotFoundError(f"Resolved dataset path does not exist: {resolved_path}")
    try:
        rel_path = resolved_path.relative_to(repo_root)
        resolved_path_str = str(rel_path)
    except Exception:
        resolved_path_str = str(resolved_path)

    resolved["dataset"]["resolved_dir"] = resolved_dir
    resolved["dataset"]["resolved_path"] = resolved_path_str

    return resolved


def load_and_resolve_lss_config(path: Path, *, repo_root: Path | None = None) -> Dict[str, Any]:
    """Load YAML and resolve LSS config."""
    cfg = load_yaml_config(path)
    return resolve_lss_config(cfg, repo_root=repo_root)
