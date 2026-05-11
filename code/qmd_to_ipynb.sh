#!/usr/bin/env bash
# Convert a list of .qmd notebooks to .ipynb using `quarto convert`.
# Run from the repo root: bash code/qmd_to_ipynb.sh

set -euo pipefail

FILES=(
  "notebooks/4_I_fMRI_BOLD_FC.qmd"
  "notebooks/4_II_MEG_Peak_Frequency.qmd"
  "notebooks/4_inference_demos.qmd"
  "notebooks/5_stimulation_with_bayesian_inference.qmd"
)

for qmd in "${FILES[@]}"; do
  if [[ ! -f "$qmd" ]]; then
    echo "skip (not found): $qmd"
    continue
  fi
  echo "convert: $qmd"
  quarto convert "$qmd"
done
