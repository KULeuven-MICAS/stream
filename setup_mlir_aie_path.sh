#!/bin/bash

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "⚠️ Warning: Not in a Python virtual environment (.venv). Using system Python."
fi

# Get the site-packages directory of the active Python
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")

MLIR_AIE_BIN_DIR="$SITE_PACKAGES/mlir_aie/bin"

if [[ ! -d "$MLIR_AIE_BIN_DIR" ]]; then
  echo "❌ Error: mlir_aie/bin directory not found in $SITE_PACKAGES"
  echo "💡 Ensure mlir_aie is installed:"
  echo "    pip install mlir_aie -f https://github.com/Xilinx/mlir-aie/releases/expanded_assets/latest-wheels"
  exit 1
fi

export PATH="$MLIR_AIE_BIN_DIR:${PATH:-}"
echo "✅ PATH updated to include: $MLIR_AIE_BIN_DIR"
