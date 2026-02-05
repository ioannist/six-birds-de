#!/usr/bin/env python3
"""List DES Y3 chain directory and extract metadata candidates."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

URL = "https://desdr-server.ncsa.illinois.edu/despublic/y3a2_files/chains/"
PATTERN = re.compile(r'href="([^"]+)"', re.IGNORECASE)
FILTER = re.compile(r"(param|name|readme|meta|header|yaml|yml|ini|json)", re.IGNORECASE)


def main() -> int:
    html = urlopen(URL, timeout=20).read().decode("utf-8", errors="ignore")
    links = PATTERN.findall(html)
    filenames = sorted({link for link in links if not link.endswith("/")})
    candidates = [name for name in filenames if FILTER.search(name)]
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "url": URL,
        "filenames": filenames,
        "candidates": candidates,
    }
    out_path = Path("docs/experiments/des_y3_chains_dir_listing.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
