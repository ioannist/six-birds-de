#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PAPER_DIR="$ROOT_DIR/docs/paper"
BUILD_DIR="$PAPER_DIR/build"
CACHE_DIR="$ROOT_DIR/.cache/tectonic"

mkdir -p "$BUILD_DIR"

# Always refresh vendored figures before building the PDF.
python "$ROOT_DIR/scripts/vendor_figures.py" --overwrite

run_latexmk() {
  (cd "$PAPER_DIR" && latexmk -pdf -interaction=nonstopmode -halt-on-error \
    -outdir="$BUILD_DIR" "main.tex")
}

run_pdflatex() {
  (cd "$PAPER_DIR" && pdflatex -interaction=nonstopmode -halt-on-error \
    -output-directory "$BUILD_DIR" "main.tex")
  (cd "$PAPER_DIR" && pdflatex -interaction=nonstopmode -halt-on-error \
    -output-directory "$BUILD_DIR" "main.tex")
}

run_tectonic() {
  # Tectonic V1 CLI: -o/--outdir places output in BUILD_DIR
  (cd "$PAPER_DIR" && tectonic -o "$BUILD_DIR" --keep-logs "main.tex")
}

ensure_tectonic_downloaded() {
  mkdir -p "$CACHE_DIR"
  if [[ -x "$CACHE_DIR/tectonic" ]]; then
    return 0
  fi

  echo "[build_paper] No LaTeX engine found; downloading tectonic into $CACHE_DIR ..."
  tmpdir="$(mktemp -d)"
  # Official installer from tectonic-typesetting.github.io/install.html
  (cd "$tmpdir" && curl --proto '=https' --tlsv1.2 -fsSL https://drop-sh.fullyjustified.net | sh)

  if [[ ! -x "$tmpdir/tectonic" ]]; then
    echo "[build_paper] ERROR: tectonic installer did not produce an executable." >&2
    exit 2
  fi

  mv "$tmpdir/tectonic" "$CACHE_DIR/tectonic"
  chmod +x "$CACHE_DIR/tectonic"
  rm -rf "$tmpdir"
}

if command -v latexmk >/dev/null 2>&1; then
  run_latexmk
elif command -v tectonic >/dev/null 2>&1; then
  run_tectonic
elif command -v pdflatex >/dev/null 2>&1; then
  run_pdflatex
else
  ensure_tectonic_downloaded
  (cd "$PAPER_DIR" && "$CACHE_DIR/tectonic" -o "$BUILD_DIR" --keep-logs "main.tex")
fi

# Normalize output name
if [[ -f "$BUILD_DIR/main.pdf" ]]; then
  cp -f "$BUILD_DIR/main.pdf" "$BUILD_DIR/sixbirds_dark_energy.pdf"
elif [[ -f "$PAPER_DIR/main.pdf" ]]; then
  mv "$PAPER_DIR/main.pdf" "$BUILD_DIR/sixbirds_dark_energy.pdf"
fi

if [[ ! -f "$BUILD_DIR/sixbirds_dark_energy.pdf" ]]; then
  echo "[build_paper] ERROR: expected PDF was not produced." >&2
  exit 3
fi

# Write HAL metadata sidecar (must match title/abstract/keywords).
ROOT_DIR="$ROOT_DIR" python - <<'PY'
import json
import re
from pathlib import Path
import os

root = Path(os.environ["ROOT_DIR"]).resolve()
tex = (root / "docs" / "paper" / "main.tex").read_text()

def _extract(pattern, name):
    m = re.search(pattern, tex, re.S)
    if not m:
        raise SystemExit(f"[build_paper] ERROR: could not extract {name} from main.tex")
    return m.group(1).strip()

title = _extract(r"\\title\{(.*?)\}", "title")
abstract = _extract(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", "abstract")
abstract = abstract.strip()
if "\n" in abstract or "\r" in abstract:
    raise SystemExit("[build_paper] ERROR: abstract must be a single paragraph with no line breaks")

kw_match = re.search(r"\\textbf\{keywords:\}\s*([^\n]+)", tex, re.IGNORECASE)
if not kw_match:
    raise SystemExit("[build_paper] ERROR: could not extract Keywords line from main.tex")
keywords_line = kw_match.group(1).strip()
keywords = [k.strip() for k in keywords_line.split(";") if k.strip()]
if any(k != k.lower() for k in keywords):
    raise SystemExit("[build_paper] ERROR: keywords must be lowercase and semicolon-separated")

meta = {
    "hal_metadata": {
        "title": title,
        "abstract": abstract,
        "keywords": keywords,
        "language": "en",
        "domain": "astro-ph.CO",
        "license": "CC-BY 4.0",
        "authors": [
            {
                "first_name": "Ioannis",
                "last_name": "Tsiokos",
                "email": "ioannis@automorph.io",
                "orcid": "0009-0009-7659-5964",
                "affiliation_structure": {
                    "name": "Automorph Inc.",
                    "type": "Entreprise",
                    "address": "1207 Delaware Ave #4131, Wilmington, DE 19806",
                    "country": "US"
                },
                "role": "author"
            }
        ]
    }
}

out = root / "docs" / "paper" / "build" / "metadata.json"
out.write_text(json.dumps(meta, indent=2))
PY

# HAL naming convention copy (no spaces/special characters).
# Use title prefix up to the first colon, per HAL guidance, and author last name.
HAL_NAME="$(python - <<'PY'
import re
from pathlib import Path

tex = Path("docs/paper/main.tex").read_text()
m = re.search(r"\\title\{(.*?)\}", tex, re.S)
if not m:
    raise SystemExit("missing title")
title = m.group(1).strip()
short = title.split(":", 1)[0].strip()
# Replace non-alphanumeric with underscores, collapse, and trim.
short = re.sub(r"[^A-Za-z0-9]+", "_", short).strip("_")
print(f"2026_Tsiokos_{short}_v1.pdf")
PY
)"
cp -f "$BUILD_DIR/sixbirds_dark_energy.pdf" "$BUILD_DIR/$HAL_NAME"

# Font embedding and Type 3 check (if pdffonts is available).
if command -v pdffonts >/dev/null 2>&1; then
  python - <<'PY'
import re
import subprocess
import sys

pdf = "docs/paper/build/sixbirds_dark_energy.pdf"
out = subprocess.check_output(["pdffonts", pdf], text=True)
lines = out.strip().splitlines()[2:]  # skip header
type3 = []
not_embedded = []

for line in lines:
    parts = line.split()
    if len(parts) < 5:
        continue
    # Type may be one token (TrueType) or two tokens ("Type 1", "Type 3").
    if parts[1] == "Type" and len(parts) >= 5:
        ftype = f"{parts[1]} {parts[2]}"
        emb = parts[4]
    else:
        ftype = parts[1]
        emb = parts[3]
    if ftype == "Type 3":
        type3.append(line)
    if emb == "no":
        not_embedded.append(line)

if not_embedded:
    print("[build_paper] ERROR: non-embedded fonts detected in PDF.", file=sys.stderr)
    sys.exit(4)
if type3:
    print("[build_paper] ERROR: Type 3 fonts detected in PDF.", file=sys.stderr)
    sys.exit(5)
PY
fi

# HAL QA checks (best-effort; fail if critical requirements are violated).
python - <<'PY'
import os
import re
import shutil
import subprocess
import sys

pdf = "docs/paper/build/sixbirds_dark_energy.pdf"
size = os.path.getsize(pdf)
if size > 50 * 1024 * 1024:
    print("[build_paper] ERROR: PDF exceeds 50MB size limit.", file=sys.stderr)
    sys.exit(6)

if shutil.which("pdfinfo"):
    info = subprocess.check_output(["pdfinfo", pdf], text=True, errors="ignore")
    m = re.search(r"PDF version:\s*([0-9.]+)", info)
    if m:
        try:
            if float(m.group(1)) < 1.4:
                print("[build_paper] ERROR: PDF version is < 1.4.", file=sys.stderr)
                sys.exit(7)
        except ValueError:
            pass
    if re.search(r"Encrypted:\s*yes", info, re.IGNORECASE):
        print("[build_paper] ERROR: PDF is encrypted.", file=sys.stderr)
        sys.exit(8)

if shutil.which("pdftotext"):
    # Draft/confidential check over full text.
    text = subprocess.check_output(["pdftotext", pdf, "-"], text=True, errors="ignore")
    if re.search(r"(confidential|draft|do not distribute)", text, re.IGNORECASE):
        print("[build_paper] ERROR: draft/confidential watermark text detected.", file=sys.stderr)
        sys.exit(9)
    # Affiliation check on page 1.
    page1 = subprocess.check_output(["pdftotext", "-f", "1", "-l", "1", pdf, "-"], text=True, errors="ignore")
    if "Automorph Inc." not in page1:
        print("[build_paper] ERROR: affiliation 'Automorph Inc.' not found on page 1.", file=sys.stderr)
        sys.exit(10)
PY

# Build HAL source upload zip (LaTeX + .bbl + figures/tables).
bash "$ROOT_DIR/scripts/make_hal_source_zip.sh"

echo "[build_paper] Wrote $BUILD_DIR/sixbirds_dark_energy.pdf"
