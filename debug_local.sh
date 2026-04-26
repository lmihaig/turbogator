#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR="$SCRIPT_DIR"

FORCE_REGENERATE=0
case "${1:-}" in
    --force) FORCE_REGENERATE=1 ;;
    "") ;;
    *)
        echo "Usage: ./debug_local.sh [--force]"
        exit 1
        ;;
esac

echo "Local validation"

VALIDATION_DIR="$ROOT_DIR/results/reference"
VALIDATION_FILES=(
    "$VALIDATION_DIR/input.bin"
    "$VALIDATION_DIR/expected.bin"
    "$VALIDATION_DIR/validation_config.json"
)

needs_regeneration=0
if [ "$FORCE_REGENERATE" -eq 1 ]; then
    needs_regeneration=1
else
    for file_path in "${VALIDATION_FILES[@]}"; do
        if [ ! -f "$file_path" ]; then
            needs_regeneration=1
            break
        fi
    done
fi

if [ "$needs_regeneration" -eq 1 ]; then
    echo "Generating validation data..."
    uv run --project "$ROOT_DIR" "$ROOT_DIR/reference/generate_validation_data.py"
else
    echo "Using existing validation data."
fi

cd "$ROOT_DIR/turbogator"

if [ "$FORCE_REGENERATE" -eq 1 ]; then
    echo "Cleaning build..."
    make clean
fi

make validate

echo "Running local validation..."
echo "--------------------------------------------------"
./build/validate_kernel
EXIT_CODE=$?
echo "--------------------------------------------------"

if [ $EXIT_CODE -ne 0 ]; then
    echo "Validation failed (Exit Code: $EXIT_CODE)."
    exit "$EXIT_CODE"
fi

echo "Validation passed."
