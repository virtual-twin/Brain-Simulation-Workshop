#!/usr/bin/env bash
# Live-preview the combined slide deck (slides.qmd) with working chapter reload.
#
# Why this exists:
#   `quarto preview` does NOT watch files pulled in via {{< include >}}
#   (quarto-cli #6130 / #6413, closed "not planned"). So editing a chapter
#   partial (slides/_NN-*.qmd) on its own would not refresh the deck.
#   This script touches slides.qmd whenever any chapter changes, which makes
#   `quarto preview` re-render the whole deck — and because the chapters stay
#   `_`-prefixed, the preview never switches to a standalone sub-slide.
#
# Usage:
#   make preview                          # preferred entry point
#   ./scripts/slides-preview.sh           # opens the deck in your browser
#   ./scripts/slides-preview.sh --no-browser   # extra args pass through to quarto
#
set -euo pipefail
# Run from the repo root (this script lives in scripts/).
cd "$(dirname "$0")/.."

uv run quarto preview slides.qmd "$@" &
preview_pid=$!
trap 'kill "$preview_pid" 2>/dev/null || true' EXIT INT TERM

echo "[slides-preview] watching slides/_*.qmd — edit a chapter and the deck re-renders."
while kill -0 "$preview_pid" 2>/dev/null; do
  if [ -n "$(find slides -name '_*.qmd' -newer slides.qmd 2>/dev/null)" ]; then
    touch slides.qmd
  fi
  sleep 1
done
