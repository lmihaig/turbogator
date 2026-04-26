#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR="$SCRIPT_DIR"

command -v jq >/dev/null 2>&1 || { echo "Error: jq is required but not installed."; exit 1; }

USER=$(whoami)
AUTH="turbogator:TurboGator2026"
URL="https://aos.licu.dev"

TMP_LOG=$(mktemp /tmp/aos_live_log_XXXXXX.log)
TMP_ARCHIVE=$(mktemp /tmp/aos_results_XXXXXX.tar.gz)
trap 'rm -f "$ROOT_DIR/workspace.tar.gz" "$TMP_LOG" "$TMP_ARCHIVE" 2>/dev/null' EXIT

echo "Packing workspace..."
if [ ! -f "$ROOT_DIR/config.py" ]; then
    echo "Error: config.py is required in workspace root."
    exit 1
fi
tar -czf "$ROOT_DIR/workspace.tar.gz" \
    --exclude='reference/ezgatr/.venv' \
    --exclude='reference/ezgatr/.pytest_cache' \
    --exclude='reference/ezgatr/.mypy_cache' \
    --exclude='reference/ezgatr/.ruff_cache' \
    --exclude='reference/ezgatr/**/__pycache__' \
    --exclude='reference/ezgatr/**/*.pyc' \
    --exclude='reference/ezgatr/.ipynb_checkpoints' \
    -C "$ROOT_DIR" reference config.py

echo "Submitting to Queue..."
RESPONSE=$(curl -s -u "$AUTH" -X POST "$URL/submit" \
    -F "user=${USER}" \
    -F "description=ezgatr" \
    -F "target=reference" \
    -F "workspace=@$ROOT_DIR/workspace.tar.gz")

JOB_ID=$(echo "$RESPONSE" | jq -r '.job_id // empty')
if [ -z "$JOB_ID" ]; then
    echo "Failed to submit. Server response: $RESPONSE"
    exit 1
fi

echo "Queued Job ID: $JOB_ID"
echo "Waiting for job completion..."
echo "--------------------------------------------------"

LAST_LINE=0
LAST_STATUS=""

while true; do
    STATUS=$(curl -s -u "$AUTH" "$URL/status/${JOB_ID}" | jq -r '.status // "unknown"')

    if [ "$STATUS" != "$LAST_STATUS" ] && [ "$STATUS" != "unknown" ]; then
        echo "Status: $STATUS"
        LAST_STATUS="$STATUS"
    fi

    curl -s -u "$AUTH" "$URL/logs/${JOB_ID}" > "$TMP_LOG"
    NEW_LINES=$(wc -l < "$TMP_LOG")

    if [ "$NEW_LINES" -gt "$LAST_LINE" ]; then
        tail -n +$((LAST_LINE + 1)) "$TMP_LOG"
        LAST_LINE=$NEW_LINES
    fi

    case "$STATUS" in
        completed) break ;;
        failed|error|cancelled)
            echo "Job ended with status: $STATUS"
            exit 1
            ;;
    esac
    sleep 2
done

echo "--------------------------------------------------"

mkdir -p "$ROOT_DIR/results/reference"
curl -s -u "$AUTH" -o "$TMP_ARCHIVE" "$URL/download/${JOB_ID}"
tar -xzf "$TMP_ARCHIVE" -C "$ROOT_DIR/results/reference"

echo "Results saved to results/reference/"

if [ -f "$ROOT_DIR/results/reference/run.log" ]; then
    echo "---- Final server log tail ----"
    tail -n 120 "$ROOT_DIR/results/reference/run.log"
    echo "-------------------------------"
fi

if grep -q '"error"' "$ROOT_DIR/results/reference/metrics.json"; then
    echo "Reference run failed!"
    cat "$ROOT_DIR/results/reference/metrics.json"
    exit 1
fi

echo "Update plots: ./plot.sh"
