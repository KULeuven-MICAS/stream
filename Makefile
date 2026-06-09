.PHONY: all single_gemm uniform_gemm run_uniform_gemms

# GEMM -> AIE MLIR codegen via scripts/main_gemm_codegen.py.
# Size constraints asserted by the script: M % 128 == 0, N % 256 == 0, K % 32 == 0.
# For uniform M=N=K runs the size must therefore be a multiple of 256.
SIZES := 256 512

all: run_uniform_gemms

single_gemm:
	python3 scripts/main_gemm_codegen.py --M 128 --N 256 --K 64

uniform_gemm:
	@if [ -z "$(filter-out $@,$(MAKECMDGOALS))" ]; then \
		echo "Error: No size value provided. Please provide a size (e.g., 'make uniform_gemm 256')."; \
		exit 1; \
	fi
	python3 scripts/main_gemm_codegen.py --M $(filter-out $@,$(MAKECMDGOALS)) --N $(filter-out $@,$(MAKECMDGOALS)) --K $(filter-out $@,$(MAKECMDGOALS))

run_uniform_gemms:
	$(foreach s,$(SIZES), python3 scripts/main_gemm_codegen.py --M $(s) --N $(s) --K $(s);)

%:
	@:
