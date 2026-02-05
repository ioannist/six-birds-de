#!/usr/bin/env python3
"""Extract URLs from a TeX file and write a categorized markdown report."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


URL_PATTERN = re.compile(r"https?://[^\s\}\\]+")
URL_CMD_PATTERN = re.compile(r"\\url\{([^}]+)\}")
HREF_PATTERN = re.compile(r"\\href\{([^}]+)\}\{[^}]*\}")
COMMENT_PATTERN = re.compile(r"(?<!\\)%.*$")

CATEGORIES = [
    "SN",
    "BAO",
    "CMB",
    "LSS / WL",
    "Clusters",
    "Software / Code",
    "Other",
]


def strip_comments(line: str) -> str:
    return COMMENT_PATTERN.sub("", line)


def clean_url(url: str) -> str:
    url = url.strip()
    while url and url[-1] in ".,);]":
        url = url[:-1]
    return url


def extract_urls_from_line(line: str) -> list[str]:
    urls: list[str] = []
    urls.extend(HREF_PATTERN.findall(line))
    urls.extend(URL_CMD_PATTERN.findall(line))
    urls.extend(URL_PATTERN.findall(line))
    return [clean_url(url) for url in urls if url]


def categorize(url: str, context: str) -> str:
    text = context.lower()

    if re.search(r"\bsn\b", text) or any(
        key in text for key in ["supernova", "pantheon", "jla", "salt"]
    ):
        return "SN"
    if "bao" in text or "baryon" in text:
        return "BAO"
    if any(key in text for key in ["cmb", "planck", "wmap", "act", "spt"]):
        return "CMB"
    if any(
        key in text
        for key in [
            "lensing",
            "shear",
            "galaxy clustering",
            "des",
            "kids",
            "hsc",
            "euclid",
            "lsst",
            "eboss",
            "boss",
        ]
    ):
        return "LSS / WL"
    if any(key in text for key in ["cluster", "sz", "x-ray", "xray"]):
        return "Clusters"

    software_domains = ["github.com", "gitlab", "readthedocs", "pypi", "conda"]
    if any(domain in url for domain in software_domains) or any(
        key in text for key in ["code", "software", "package", "pipeline", "library"]
    ):
        return "Software / Code"

    return "Other"


def build_context(lines: list[str], idx: int, context_lines: int) -> list[str]:
    start = max(0, idx - context_lines)
    end = min(len(lines), idx + context_lines + 1)
    snippet = [lines[i].rstrip() for i in range(start, end)]
    snippet = [line for line in snippet if line.strip()]
    return snippet


def collect_source_files(input_path: Path) -> list[Path]:
    input_path = input_path.resolve()
    if input_path.suffix == ".tex":
        bib_files = sorted(input_path.parent.glob("*.bib"))
        return [input_path, *bib_files]
    return [input_path]


def extract_urls(paths: list[Path], context_lines: int) -> list[dict]:
    seen: set[str] = set()
    entries: list[dict] = []

    for path in paths:
        raw_lines = path.read_text().splitlines()
        lines = [strip_comments(line) for line in raw_lines]

        for idx, line in enumerate(lines):
            urls = extract_urls_from_line(line)
            if not urls:
                continue
            context_snippet = build_context(lines, idx, context_lines)
            context_text = "\n".join(context_snippet)
            for url in urls:
                if url in seen:
                    continue
                seen.add(url)
                entries.append(
                    {
                        "url": url,
                        "line": idx + 1,
                        "context": context_snippet,
                        "category": categorize(url, context_text),
                    }
                )

    return entries


def write_report(entries: list[dict], output_path: Path) -> None:
    grouped = {category: [] for category in CATEGORIES}
    for entry in entries:
        grouped[entry["category"]].append(entry)

    lines: list[str] = [
        "# Extracted Data Sources",
        "",
        f"Found {len(entries)} URLs (unique).",
        "",
    ]

    for category in CATEGORIES:
        items = grouped[category]
        lines.append(f"## {category} ({len(items)})")
        if not items:
            lines.append("- (none)")
            lines.append("")
            continue
        for entry in items:
            url = entry["url"]
            line_no = entry["line"]
            lines.append(f"- [{url}]({url}) (line {line_no})")
            for ctx_line in entry["context"][:3]:
                lines.append(f"  > {ctx_line.strip()}")
        lines.append("")

    output_path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract URLs from a TeX file.")
    parser.add_argument(
        "--input",
        default="docs/knowledge_base/dark_energy_survey.tex",
        help="Input TeX file",
    )
    parser.add_argument(
        "--output",
        default="docs/experiments/data_sources_extracted.md",
        help="Output markdown report",
    )
    parser.add_argument(
        "--context-lines",
        type=int,
        default=1,
        help="Number of context lines before and after each match",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    source_files = collect_source_files(input_path)
    entries = extract_urls(source_files, args.context_lines)
    write_report(entries, output_path)


if __name__ == "__main__":
    main()
