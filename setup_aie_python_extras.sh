#!/bin/bash
# Installs aie-python-extras (provides aie.extras.context, used by AIE tracing codegen).
#
# This is the one AIE dependency that cannot live in pyproject's [aie] extra: it must be
# built with HOST_MLIR_PYTHON_PACKAGE_PREFIX=aie, a build-time env var that pip dependency
# specifications cannot express. Everything else (mlir_aie + llvm_aie, pinned) is installed
# by the [aie] extra.
#
# Run AFTER `pip install -e ".[aie]"`. Then `source setup_mlir_aie_pythonpath.sh` to put the
# mlir_aie bindings on PYTHONPATH.
set -euo pipefail

HOST_MLIR_PYTHON_PACKAGE_PREFIX=aie python3 -m pip install "git+https://github.com/makslevental/mlir-python-extras@f08db06"
