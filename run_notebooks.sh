#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="outputs"
mkdir -p "$OUT_DIR"

NOTEBOOK_LIST="notebooks_to_run.txt"
if [ ! -f "$NOTEBOOK_LIST" ]; then
  echo "Notebook list $NOTEBOOK_LIST not found"
  exit 1
fi

while IFS= read -r nb || [ -n "$nb" ]; do
  nb_trimmed="$(echo "$nb" | xargs)"
  [ -z "$nb_trimmed" ] && continue
  echo "Running $nb_trimmed"
  papermill "$nb_trimmed" "$OUT_DIR/$(basename "$nb_trimmed" .ipynb)-executed.ipynb"
done < "$NOTEBOOK_LIST"

echo "All notebooks executed. Outputs in $OUT_DIR"
