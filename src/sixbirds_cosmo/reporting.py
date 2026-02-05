"""Utilities for generating run log stubs from experiment outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _escape_string(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _render_value(value: Any) -> str:
    if isinstance(value, str):
        rendered = _escape_string(value)
    elif isinstance(value, (dict, list)):
        rendered = json.dumps(value, sort_keys=True, separators=(",", ":"))
    else:
        rendered = str(value)
    if len(rendered) > 160:
        rendered = rendered[:159] + "…"
    return rendered


def metrics_to_markdown_table(metrics: dict) -> str:
    """Return a two-column Markdown table from a metrics dict."""
    lines = ["| Metric | Value |", "|---|---|"]
    for key in sorted(metrics.keys()):
        value = _render_value(metrics[key])
        lines.append(f"| {key} | {value} |")
    return "\n".join(lines)


def list_plot_files(run_dir: Path) -> list[str]:
    """List plot files in the run directory (top-level only)."""
    exts = {".png", ".pdf", ".svg"}
    names = [
        path.name
        for path in run_dir.iterdir()
        if path.is_file() and path.suffix.lower() in exts
    ]
    return sorted(names)


def write_run_markdown_stub(
    run_dir: Path,
    *,
    template_path: Path | None = None,
    output_name: str = "RUN_LOG.md",
    metrics_name: str = "metrics.json",
) -> Path:
    """Generate a run-local markdown log stub from a template."""
    template = (
        template_path
        if template_path is not None
        else _repo_root() / "docs" / "experiments" / "TEMPLATE.md"
    )
    content = template.read_text(encoding="utf-8")

    metrics_path = run_dir / metrics_name
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    table = metrics_to_markdown_table(metrics)

    plots = list_plot_files(run_dir)
    if plots:
        plots_md = "\n".join(f"- {name}" for name in plots)
    else:
        plots_md = "- (none)"

    if "<!-- METRICS_TABLE -->" not in content or "<!-- PLOTS_LIST -->" not in content:
        raise ValueError("Template missing required markers.")

    content = content.replace("<!-- METRICS_TABLE -->", table)
    content = content.replace("<!-- PLOTS_LIST -->", plots_md)

    output_path = run_dir / output_name
    output_path.write_text(content, encoding="utf-8")
    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a run log stub.")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output", default="RUN_LOG.md")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    write_run_markdown_stub(args.run_dir, output_name=args.output)


if __name__ == "__main__":
    main()
