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

single_gemm:
	python3 main_aie_codegen_gemm.py --M 64 --N 64 --K 64
