# Install stream requirements
pip install -r requirements.txt

# Install mlir-aie nightly
python3 -m pip install --upgrade pip

# Install IRON library and mlir-aie from a wheel
python3 -m pip install mlir_aie -f https://github.com/Xilinx/mlir-aie/releases/expanded_assets/latest-wheels

# Install Peano from llvm-aie wheel
python3 -m pip install "llvm-aie==19.0.0.2025063001" -f https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly

# Install aie-python-extras (provides aie.extras.context for tracing; not on PyPI)
HOST_MLIR_PYTHON_PACKAGE_PREFIX=aie python3 -m pip install "git+https://github.com/makslevental/mlir-python-extras@f08db06"

