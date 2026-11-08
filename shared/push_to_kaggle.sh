#!/usr/bin/env bash
# Push a notebook to Kaggle (T4), check status, or download outputs. Run from anywhere.
#   NB=prosqa_monitor/notebooks/prosqa_monitor SLUG=prosqa-monitorability-controlled TITLE='prosqa monitorability controlled' shared/push_to_kaggle.sh [push|status|output]
# NB   = path to the notebook without .ipynb, relative to the repo root (default: the ProsQA experiment notebook)
# OUT  = where `output` downloads to (default: <repo>/kaggle_output/<slug>, a git-ignored scratch folder; curate into <experiment>/runs/<run>/results afterwards)
# NOTE: do NOT export KAGGLE_USERNAME: it switches the CLI to legacy auth and the push fails with "Authentication required".
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
KAGGLE="$ROOT/.venv/bin/kaggle"
NB=${NB:-prosqa_monitor/notebooks/prosqa_monitor}
SLUG=${SLUG:-prosqa-monitorability-controlled}
OUT=${OUT:-$ROOT/kaggle_output/$SLUG}

USER_NAME=$($KAGGLE config view 2>/dev/null | awk -F': ' '/username/{print $2}' | tr -d ' ')
[ -n "$USER_NAME" ] || { echo "No Kaggle username found. Run: .venv/bin/kaggle auth login (see shared/README_kaggle.md)"; exit 1; }

case "${1:-push}" in
  push)
    rm -rf .kaggle_push && mkdir .kaggle_push
    cp "$NB.ipynb" .kaggle_push/
    CODE_FILE="$(basename "$NB").ipynb"
    cat > .kaggle_push/kernel-metadata.json <<JSON
{
  "id": "$USER_NAME/$SLUG",
  "title": "${TITLE:-$SLUG}",
  "code_file": "$CODE_FILE",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": true,
  "dataset_sources": [${DATASETS:+\"$DATASETS\"}],
  "competition_sources": [],
  "kernel_sources": []
}
JSON
    echo "notebook sha1: $(shasum "$NB.ipynb" | cut -c1-10)  (record this in the run's RUN.md)"
    $KAGGLE kernels push -p .kaggle_push --accelerator NvidiaTeslaT4
    ;;
  status) $KAGGLE kernels status "$USER_NAME/$SLUG" ;;
  output) mkdir -p "$OUT" && $KAGGLE kernels output "$USER_NAME/$SLUG" -p "$OUT" ;;
esac
