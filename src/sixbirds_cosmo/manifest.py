"""Run manifest utilities for experiments."""
from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - optional import
    np = None  # type: ignore[assignment]

try:
    from importlib import metadata as importlib_metadata
except ImportError:  # pragma: no cover
    import importlib_metadata  # type: ignore[assignment]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _timestamp_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _git_info(root: Path) -> tuple[str | None, bool | None]:
    try:
        sha = (
            subprocess.check_output(
                ["git", "-C", str(root), "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
                text=True,
            )
            .strip()
        )
    except Exception:
        return None, None

    try:
        status = subprocess.check_output(
            ["git", "-C", str(root), "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        dirty = bool(status.strip())
    except Exception:
        dirty = None
    return sha, dirty


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if np is not None and isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, set):
        return [_to_jsonable(v) for v in sorted(value)]
    return value


def _package_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def _package_versions() -> dict[str, str | None]:
    versions = {
        "sixbirds-cosmo": _package_version("sixbirds-cosmo"),
        "numpy": _package_version("numpy"),
        "scipy": _package_version("scipy"),
        "pandas": _package_version("pandas"),
        "matplotlib": _package_version("matplotlib"),
        "pyyaml": _package_version("PyYAML") or _package_version("pyyaml"),
    }
    if versions["sixbirds-cosmo"] is None:
        alt = _package_version("sixbirds_cosmo")
        if alt is not None:
            versions["sixbirds-cosmo"] = alt
    return versions


@dataclass(frozen=True)
class RunManifest:
    exp_name: str
    run_dir: Path
    seed: int | None


def create_run_dir(
    exp_name: str, base_dir: Path | str = "results", seed: int | None = None
) -> Path:
    root = _repo_root()
    base = root / base_dir
    timestamp = _timestamp_utc()
    git_sha, _ = _git_info(root)
    git_tag = git_sha if git_sha is not None else "nogit"
    run_dir = base / exp_name / f"{timestamp}_{git_tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_config(run_dir: Path, config: dict) -> Path:
    resolved = _to_jsonable(config)
    path = run_dir / "config.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(resolved, f, sort_keys=False)
    return path


def write_metrics(run_dir: Path, metrics: dict) -> Path:
    resolved = _to_jsonable(metrics)
    path = run_dir / "metrics.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(resolved, indent=2), encoding="utf-8")
    return path


def write_provenance(
    run_dir: Path, exp_name: str, seed: int | None, extra: dict | None = None
) -> Path:
    root = _repo_root()
    git_sha, git_dirty = _git_info(root)
    payload: dict[str, Any] = {
        "exp_name": exp_name,
        "run_dir": str(run_dir.relative_to(root)),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "git_sha": git_sha,
        "git_dirty": git_dirty,
        "random_seed": seed,
        "package_versions": _package_versions(),
    }
    if extra:
        payload.update(_to_jsonable(extra))
    path = run_dir / "provenance.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def save_fig(fig, path: Path | str) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, bbox_inches="tight")
    return target


def save_table(df, path: Path | str) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    suffix = target.suffix.lower()
    if suffix == ".csv":
        df.to_csv(target, index=False)
        return target
    raise ValueError(f"Unsupported table format: {suffix}")
