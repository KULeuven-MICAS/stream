# Install stream requirements
pip install -r requirements.txt

# Install mlir-aie nightly
python3 -m pip install --upgrade pip

# Install IRON library and mlir-aie from a wheel
python3 -m pip install mlir_aie -f https://github.com/Xilinx/mlir-aie/releases/expanded_assets/latest-wheels

# Install Peano from llvm-aie wheel
python3 -m pip install llvm-aie -f https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly

# Install basic Python requirements (still needed for release v1.0, but is no longer needed for latest wheels)
python3 -m pip install -r mlir-aie/python/requirements.txt

# Install MLIR Python Extras
HOST_MLIR_PYTHON_PACKAGE_PREFIX=aie python3 -m pip install -r mlir-aie/python/requirements_extras.txt

# Source mlir-aie environment
# source mlir-aie/utils/env_setup.sh
