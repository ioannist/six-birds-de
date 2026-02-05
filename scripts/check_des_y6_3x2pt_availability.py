#!/usr/bin/env python3
"""Check availability of DES Y6 3x2pt public artifacts."""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


BASE_URL = "https://desdr-server.ncsa.illinois.edu/despublic/y6a2_files/"
PROBE_PATHS = [
    "y6_3x2pt/",
    "y6_2pt/",
    "y6_cosmology/",
    "y6_lss/",
]
KEYWORDS = [
    "3x2pt",
    "3×2pt",
    "2pt",
    "datavector",
    "cov",
    "chains",
    "likelihood",
    "cosmo",
]


def fetch_text(url: str) -> str:
    with urlopen(url, timeout=20) as resp:
        return resp.read().decode("utf-8", errors="ignore")


def probe_path(path: str) -> dict:
    url = BASE_URL + path
    try:
        with urlopen(url, timeout=20) as resp:
            status = resp.status
            text = resp.read().decode("utf-8", errors="ignore")
        return {"path": path, "status_code": status, "has_assets": bool(re.search(r"\\.(fits|zip)", text, re.I))}
    except HTTPError as exc:
        return {"path": path, "status_code": exc.code}
    except URLError as exc:
        return {"path": path, "error": str(exc)}


def main() -> int:
    timestamp = datetime.now(timezone.utc).isoformat()
    found_candidates = []
    probed = []
    available = False

    try:
        html = fetch_text(BASE_URL)
        for kw in KEYWORDS:
            if kw in html:
                found_candidates.append(kw)
    except Exception as exc:
        found_candidates.append(f"error:{exc}")

    for path in PROBE_PATHS:
        info = probe_path(path)
        probed.append(info)
        if info.get("status_code") == 200 and info.get("has_assets"):
            available = True

    report = {
        "timestamp_utc": timestamp,
        "base_url": BASE_URL,
        "found_candidates": found_candidates,
        "probed_paths": probed,
        "available": available,
    }

    out_path = Path("docs/experiments/y6_3x2pt_availability.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
