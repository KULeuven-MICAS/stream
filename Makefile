.PHONY: all run_all_heights run_height

HEIGHTS := 2 4 8 16 32 64 128 256 512 1024

all: run_all_heights

run_all_heights:
	$(foreach h,$(HEIGHTS), python3 main_aie_codegen_conv2d.py --height $(h);)

run_height:
	@if [ -z "$(filter-out $@,$(MAKECMDGOALS))" ]; then \
		echo "Error: No height value provided. Please provide a height value."; \
		exit 1; \
	fi
	python3 main_aie_codegen_conv2d.py --height $(filter-out $@,$(MAKECMDGOALS))

%:
	@:

SIZES := 64 128 256

single_gemm:
	python3 main_aie_codegen_gemm.py --M 64 --N 64 --K 64

uniform_gemm:
	@if [ -z "$(filter-out $@,$(MAKECMDGOALS))" ]; then \
		echo "Error: No size value provided. Please provide a size (e.g., 'make uniform_gemm 128')."; \
		exit 1; \
	fi
	python3 main_aie_codegen_gemm.py --M $(filter-out $@,$(MAKECMDGOALS)) --N $(filter-out $@,$(MAKECMDGOALS)) --K $(filter-out $@,$(MAKECMDGOALS))

run_uniform_gemms:
	$(foreach s,$(SIZES), python3 main_aie_codegen_gemm.py --M $(s) --N $(s) --K $(s);)

# ENTIRE COLUMN

single_gemm_col:
	python3 main_aie_codegen_gemm_col.py --M 64 --N 64 --K 64
