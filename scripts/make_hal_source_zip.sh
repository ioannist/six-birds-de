#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PAPER_DIR="$ROOT_DIR/docs/paper"
BUILD_DIR="$PAPER_DIR/build"
STAGE_DIR="$BUILD_DIR/hal_source_staging"
ZIP_PATH="$BUILD_DIR/hal_source_upload.zip"

rm -rf "$STAGE_DIR"
mkdir -p "$STAGE_DIR/tables" "$STAGE_DIR/figures"

# Sanity: no absolute paths in includegraphics
if rg -n "\\\\includegraphics\\{/" "$PAPER_DIR/main.tex" >/dev/null 2>&1; then
  echo "[make_hal_source_zip] ERROR: absolute path found in \\includegraphics{}." >&2
  exit 2
fi

# Ensure .bbl exists (arXiv does not run BibTeX).
if ! [[ -f "$BUILD_DIR/main.bbl" ]]; then
  echo "[make_hal_source_zip] main.bbl missing; compiling to generate it..." >&2
  (cd "$PAPER_DIR" && pdflatex -interaction=nonstopmode -halt-on-error -output-directory "$BUILD_DIR" main.tex)
  (cd "$BUILD_DIR" && bibtex main)
fi

if ! [[ -f "$BUILD_DIR/main.bbl" ]]; then
  echo "[make_hal_source_zip] ERROR: main.bbl not generated." >&2
  exit 3
fi

# Copy core sources
cp "$PAPER_DIR/main.tex" "$STAGE_DIR/main.tex"
cp "$PAPER_DIR/macros.tex" "$STAGE_DIR/macros.tex"
cp "$PAPER_DIR/refs.bib" "$STAGE_DIR/refs.bib"
cp "$BUILD_DIR/main.bbl" "$STAGE_DIR/main.bbl"

# Copy tables (tex only)
cp "$PAPER_DIR/tables/"*.tex "$STAGE_DIR/tables/"

# Copy figures (png/jpg/pdf only)
for ext in png jpg pdf; do
  if compgen -G "$PAPER_DIR/figures/*.$ext" > /dev/null; then
    cp "$PAPER_DIR/figures/"*.$ext "$STAGE_DIR/figures/"
  fi
done

# Reject eps figures if present
if compgen -G "$PAPER_DIR/figures/*.eps" > /dev/null; then
  echo "[make_hal_source_zip] ERROR: EPS figures detected; convert to PDF/PNG." >&2
  exit 4
fi

# Remove any LaTeX build artifacts if present in staging
find "$STAGE_DIR" -name "*.aux" -o -name "*.log" -o -name "*.out" -o -name "*.toc" | xargs -r rm -f

# Build zip (flat/semi-flat)
rm -f "$ZIP_PATH"
(cd "$STAGE_DIR" && zip -r "$ZIP_PATH" . >/dev/null)

echo "[make_hal_source_zip] Wrote $ZIP_PATH"
