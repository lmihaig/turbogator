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

cd "$ROOT_DIR"

echo "Building C++ extension..."
UV_PYTHON=$(uv run --no-project --with torch --with numpy --with einops \
    python -c "import sys; print(sys.executable)")
rm -f "$ROOT_DIR/turbogator"/turbogator_ext*.so
rm -rf "$ROOT_DIR/build/pybind"
cmake -S "$ROOT_DIR" -B "$ROOT_DIR/build/pybind" \
    -DCMAKE_BUILD_TYPE=Release \
    -DPython_EXECUTABLE="$UV_PYTHON"
cmake --build "$ROOT_DIR/build/pybind"

echo "Running Python validation..."
echo "--------------------------------------------------"
PYTHONPATH="$ROOT_DIR:$ROOT_DIR/turbogator:$ROOT_DIR/turbogator/ezgatr/src" \
TURBOGATOR_IMPL_GEOMETRIC_PRODUCT=baseline \
TURBOGATOR_IMPL_EQUI_JOIN=baseline \
TURBOGATOR_IMPL_EQUI_GEOMETRIC_ATTENTION=baseline \
TURBOGATOR_IMPL_SCALER_GATED_GELU=baseline \
uv run --no-project \
    --with torch \
    --with numpy \
    --with einops \
    "$ROOT_DIR/turbogator/validate.py"
EXIT_CODE=$?
echo "--------------------------------------------------"

if [ $EXIT_CODE -ne 0 ]; then
    echo "Validation failed (Exit Code: $EXIT_CODE)."
    exit "$EXIT_CODE"
fi

echo "Validation passed."
