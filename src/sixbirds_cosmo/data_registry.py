"""Data registry and fetch utilities."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import tempfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

try:
    import yaml
except ImportError:  # pragma: no cover - pyyaml is a project dependency
    yaml = None


def find_repo_root(start: Path | None = None) -> Path:
    """Find the repository root by walking up to a pyproject.toml."""
    current = Path(start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise FileNotFoundError("Could not find repo root (pyproject.toml not found).")


def sha256_file(path: Path) -> str:
    """Compute SHA256 hash for a file."""
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_registry(path: Path) -> Dict[str, Any]:
    """Load dataset registry and validate minimal schema."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Registry file not found: {path}")

    if path.suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise ImportError("pyyaml is required to read YAML registries.")
        data = yaml.safe_load(path.read_text())
    elif path.suffix == ".json":
        data = json.loads(path.read_text())
    else:
        raise ValueError(f"Unsupported registry format: {path.suffix}")

    _validate_registry(data)
    return data


def fetch(
    dataset_name: str,
    *,
    force: bool = False,
    registry_path: Path | None = None,
    data_dir: Path | None = None,
) -> Path:
    """Fetch a dataset into the local data cache and write provenance."""
    repo_root = find_repo_root()
    registry_path = _resolve_registry_path(repo_root, registry_path)
    registry = load_registry(registry_path)

    datasets = registry.get("datasets", {})
    if dataset_name not in datasets:
        raise KeyError(f"Dataset not found in registry: {dataset_name}")

    dataset_spec = datasets[dataset_name]
    data_dir = _resolve_data_dir(repo_root, data_dir)
    dataset_dir = data_dir / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    file_entries = []
    for file_spec in dataset_spec.get("files", []):
        url = file_spec["url"]
        filename = file_spec["filename"]
        sha_expected = str(file_spec["sha256"]).lower()
        dest_path = dataset_dir / filename

        cache_hit = False
        if dest_path.exists() and not force:
            sha_actual = sha256_file(dest_path)
            if sha_actual == sha_expected:
                cache_hit = True
                file_entries.append(
                    {
                        "url": url,
                        "filename": filename,
                        "sha256_expected": sha_expected,
                        "sha256_actual": sha_actual,
                        "bytes": dest_path.stat().st_size,
                        "cache_hit": True,
                    }
                )
                continue

        tmp_path = _temp_path(dest_path)
        try:
            if _is_http_url(url):
                _download_to_tmp(url, tmp_path)
            else:
                src_path = repo_root / url
                if not src_path.exists():
                    raise FileNotFoundError(f"Source file not found: {src_path}")
                _copy_to_tmp(src_path, tmp_path)

            sha_actual = sha256_file(tmp_path)
            if sha_actual != sha_expected:
                tmp_path.unlink(missing_ok=True)
                raise ValueError(
                    f"Checksum mismatch for {filename}: expected {sha_expected} got {sha_actual}"
                )

            os.replace(tmp_path, dest_path)
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink()
            raise

        file_entries.append(
            {
                "url": url,
                "filename": filename,
                "sha256_expected": sha_expected,
                "sha256_actual": sha_actual,
                "bytes": dest_path.stat().st_size,
                "cache_hit": cache_hit,
            }
        )

    provenance = {
        "dataset_name": dataset_name,
        "timestamp_utc": _utc_now_iso(),
        "repo_root": ".",
        "files": file_entries,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }

    provenance_path = dataset_dir / "provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True))

    return dataset_dir


def _resolve_registry_path(repo_root: Path, registry_path: Path | None) -> Path:
    if registry_path is None:
        yaml_path = repo_root / "data" / "registry.yaml"
        if yaml_path.exists():
            return yaml_path
        json_path = repo_root / "data" / "registry.json"
        if json_path.exists():
            return json_path
        raise FileNotFoundError("No registry file found under data/.")

    registry_path = Path(registry_path)
    if not registry_path.is_absolute():
        registry_path = repo_root / registry_path
    return registry_path


def _resolve_data_dir(repo_root: Path, data_dir: Path | None) -> Path:
    if data_dir is None:
        return repo_root / "data" / "raw"
    data_dir = Path(data_dir)
    if not data_dir.is_absolute():
        data_dir = repo_root / data_dir
    return data_dir


def _validate_registry(data: Any) -> None:
    if not isinstance(data, dict):
        raise ValueError("Registry must be a mapping with a 'datasets' key.")
    datasets = data.get("datasets")
    if not isinstance(datasets, dict) or not datasets:
        raise ValueError("Registry 'datasets' must be a non-empty mapping.")

    for name, spec in datasets.items():
        if not isinstance(spec, dict):
            raise ValueError(f"Dataset entry must be a mapping: {name}")
        if "files" not in spec or not isinstance(spec["files"], list):
            raise ValueError(f"Dataset '{name}' must include a 'files' list.")
        for idx, file_spec in enumerate(spec["files"]):
            if not isinstance(file_spec, dict):
                raise ValueError(f"Dataset '{name}' file entry {idx} must be a mapping.")
            for key in ("url", "filename", "sha256"):
                if key not in file_spec:
                    raise ValueError(
                        f"Dataset '{name}' file entry {idx} missing required key '{key}'."
                    )


def _is_http_url(value: str) -> bool:
    return value.startswith("http://") or value.startswith("https://")


def _download_to_tmp(url: str, tmp_path: Path) -> None:
    with urllib.request.urlopen(url) as response, tmp_path.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def _copy_to_tmp(src_path: Path, tmp_path: Path) -> None:
    shutil.copy2(src_path, tmp_path)


def _temp_path(dest_path: Path) -> Path:
    dest_path = Path(dest_path)
    with tempfile.NamedTemporaryFile(
        delete=False, dir=dest_path.parent, prefix=f".{dest_path.name}.", suffix=".tmp"
    ) as handle:
        return Path(handle.name)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
