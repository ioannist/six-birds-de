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

echo "[build_paper] Wrote $BUILD_DIR/sixbirds_dark_energy.pdf"
