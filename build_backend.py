"""Local PEP 517 backend wrapper to expose build_editable support."""
from __future__ import annotations

import os
import shutil
import tempfile
try:
    import tomllib as _toml
except ModuleNotFoundError:  # Python < 3.11
    import tomli as _toml
from pathlib import Path

from setuptools import build_meta as _build_meta
from wheel.wheelfile import WheelFile


def _supported_features() -> list[str]:
    return ["build_editable"]


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    return _build_meta.build_wheel(
        wheel_directory, config_settings, metadata_directory
    )


def build_sdist(sdist_directory, config_settings=None):
    return _build_meta.build_sdist(sdist_directory, config_settings)


def _load_project_metadata() -> dict:
    pyproject = Path(__file__).resolve().parent / "pyproject.toml"
    data = _toml.loads(pyproject.read_text(encoding="utf-8"))
    project = data.get("project", {})
    return {
        "name": project.get("name", "UNKNOWN"),
        "version": project.get("version", "0.0.0"),
        "requires_python": project.get("requires-python"),
        "dependencies": list(project.get("dependencies", [])),
        "optional_dependencies": dict(project.get("optional-dependencies", {})),
    }


def _dist_info_name(project_name: str, version: str) -> str:
    normalized = project_name.replace("-", "_")
    return f"{normalized}-{version}.dist-info"


def _write_metadata(dist_info: Path, meta: dict) -> None:
    dist_info.mkdir(parents=True, exist_ok=True)
    lines = [
        "Metadata-Version: 2.1",
        f"Name: {meta['name']}",
        f"Version: {meta['version']}",
    ]
    if meta.get("requires_python"):
        lines.append(f"Requires-Python: {meta['requires_python']}")
    for dep in meta.get("dependencies", []):
        lines.append(f"Requires-Dist: {dep}")
    for extra, deps in meta.get("optional_dependencies", {}).items():
        lines.append(f"Provides-Extra: {extra}")
        for dep in deps:
            lines.append(f"Requires-Dist: {dep}; extra == \"{extra}\"")
    (dist_info / "METADATA").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (dist_info / "WHEEL").write_text(
        "Wheel-Version: 1.0\n"
        "Generator: build_backend\n"
        "Root-Is-Purelib: true\n"
        "Tag: py3-none-any\n",
        encoding="utf-8",
    )
    (dist_info / "top_level.txt").write_text("sixbirds_cosmo\n", encoding="utf-8")


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    root = Path(__file__).resolve().parent
    src_path = (root / "src").resolve()
    meta = _load_project_metadata()
    dist_info_dir_name = _dist_info_name(meta["name"], meta["version"])
    dist_name = dist_info_dir_name.removesuffix(".dist-info").rsplit("-", 1)[0]
    wheel_name = f"{dist_name}-{meta['version']}-py3-none-any.whl"
    wheel_path = Path(wheel_directory) / wheel_name

    with tempfile.TemporaryDirectory() as staging_dir:
        staging = Path(staging_dir)
        _write_metadata(staging / dist_info_dir_name, meta)
        (staging / "sixbirds_cosmo.pth").write_text(
            f"{src_path}\n", encoding="utf-8"
        )
        with WheelFile(wheel_path, "w") as wf:
            wf.write_files(staging)
    return os.path.basename(wheel_path)


def get_requires_for_build_wheel(config_settings=None):
    return _build_meta.get_requires_for_build_wheel(config_settings)


def get_requires_for_build_sdist(config_settings=None):
    return _build_meta.get_requires_for_build_sdist(config_settings)


def get_requires_for_build_editable(config_settings=None):
    return _build_meta.get_requires_for_build_wheel(config_settings)


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    return _build_meta.prepare_metadata_for_build_wheel(
        metadata_directory, config_settings
    )


def prepare_metadata_for_build_editable(metadata_directory, config_settings=None):
    meta = _load_project_metadata()
    dist_info_name = _dist_info_name(meta["name"], meta["version"])
    dist_info = Path(metadata_directory) / dist_info_name
    _write_metadata(dist_info, meta)
    return dist_info_name
