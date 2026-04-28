#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR="$SCRIPT_DIR"

echo "Generating plots..."

if [ ! -f "$ROOT_DIR/results/history.jsonl" ]; then
  echo "No local history found in results/history.jsonl. Submit a job first!"
  exit 1
fi

uv run --project "$ROOT_DIR" --group analysis "$ROOT_DIR/analysis/plot_all.py"

echo "Job's done. Plots at results/plots/"
