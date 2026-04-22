#!/bin/bash
set -uo pipefail

command -v jq >/dev/null 2>&1 || { echo "Error: jq is required but not installed."; exit 1; }

DESC=${1:-}
if [ -z "$DESC" ]; then
  echo "Usage: ./submit.sh <description>"
  exit 1
fi

USER=$(whoami)
# password in the clear lul cybersecurity masters
AUTH="turbogator:TurboGator2026"
URL="https://aos.licu.dev"

TMP_LOG=$(mktemp /tmp/aos_live_log_XXXXXX.log)
TMP_ARCHIVE=$(mktemp /tmp/aos_results_XXXXXX.tar.gz)
trap 'rm -f workspace.tar.gz "$TMP_LOG" "$TMP_ARCHIVE" 2>/dev/null' EXIT

echo "Packing workspace..."
tar -czf workspace.tar.gz --exclude='build' turbogator/ config.py

echo "Submitting to Queue..."
RESPONSE=$(curl -s -u "$AUTH" -X POST "$URL/submit" -F "user=${USER}" -F "description=${DESC}" -F "workspace=@workspace.tar.gz")

JOB_ID=$(echo "$RESPONSE" | jq -r '.job_id // empty')

if [ -z "$JOB_ID" ]; then
  echo "Failed to submit. Server response: $RESPONSE"
  exit 1
fi

echo "Queued Job ID: $JOB_ID"

RAW_DIR="results/raw/${JOB_ID}"
mkdir -p "$RAW_DIR"
tar -xzf workspace.tar.gz -C "$RAW_DIR"

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

curl -s -u "$AUTH" -o "$TMP_ARCHIVE" "$URL/download/${JOB_ID}"
tar -xzf "$TMP_ARCHIVE" -C "$RAW_DIR"

echo "Results saved to $RAW_DIR/"

# worker cleanup races around completion.
ARTIFACT_RUN_LOG="$RAW_DIR/run.log"
if [ -f "$ARTIFACT_RUN_LOG" ]; then
  echo "---- Final server log tail ----"
  tail -n 120 "$ARTIFACT_RUN_LOG"
  echo "-------------------------------"
fi

METRICS_FILE="$RAW_DIR/metrics.json"

if grep -q '"error"' "$METRICS_FILE"; then
  echo "Benchmark failed! Check the logs:"
  cat "$METRICS_FILE"
  exit 1
fi

echo "Appending to local history..."
cat "$METRICS_FILE" >> results/history.jsonl
echo "" >> results/history.jsonl

echo "Job Complete! Data saved to history."
echo "Job info at results/raw/${JOB_ID}"
echo "Update plots: ./plot.sh"