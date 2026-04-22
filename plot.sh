#!/bin/bash

echo "Generating plots..."

if [ ! -f "results/history.jsonl" ]; then
  echo "No local history found in results/history.jsonl. Submit a job first!"
  exit 1
fi

cd analysis
uv run plot_performance.py

echo "Job's done. Plots at results/plots/"
