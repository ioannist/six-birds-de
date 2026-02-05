#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PAPER_DIR="$ROOT_DIR/docs/paper"
BUILD_DIR="$PAPER_DIR/build"
STAGE_DIR="$BUILD_DIR/arxiv_staging"
ZIP_PATH="$BUILD_DIR/sixbirds_dark_energy_arxiv.zip"

rm -rf "$STAGE_DIR"
mkdir -p "$STAGE_DIR/tables" "$STAGE_DIR/figures"

cp "$PAPER_DIR/main.tex" "$STAGE_DIR/main.tex"
cp "$PAPER_DIR/macros.tex" "$STAGE_DIR/macros.tex"
cp "$PAPER_DIR/refs.bib" "$STAGE_DIR/refs.bib"

# Copy only compilable table inputs (tex). CSV/YAML not needed for arXiv build.
cp "$PAPER_DIR/tables/"*.tex "$STAGE_DIR/tables/"

# Copy vendored figures used by paper build.
cp "$PAPER_DIR/figures/"*.png "$STAGE_DIR/figures/"

# Compile check inside staging (ensures zip is self-contained)
pushd "$STAGE_DIR" >/dev/null
if command -v latexmk >/dev/null 2>&1; then
  latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
elif command -v pdflatex >/dev/null 2>&1; then
  pdflatex -interaction=nonstopmode -halt-on-error main.tex
  bibtex main || true
  pdflatex -interaction=nonstopmode -halt-on-error main.tex
  pdflatex -interaction=nonstopmode -halt-on-error main.tex
else
  echo "[make_arxiv_bundle] ERROR: no LaTeX engine available to validate staging build." >&2
  exit 2
fi
popd >/dev/null

# Build zip (root contains main.tex, macros.tex, refs.bib, figures/, tables/)
rm -f "$ZIP_PATH"
pushd "$STAGE_DIR" >/dev/null
zip -r "$ZIP_PATH" main.tex macros.tex refs.bib figures tables >/dev/null
popd >/dev/null

echo "[make_arxiv_bundle] Wrote $ZIP_PATH"
