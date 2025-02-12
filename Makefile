.PHONY: all

HEIGHTS := 2 4 8 16 32 64 128 256 512 1024

all: $(foreach h,$(HEIGHTS),conv2d_run_$(h))

$(foreach h,$(HEIGHTS),conv2d_run_$(h)):
	python3 main_aie_codegen_conv2d.py --height $(@:conv2d_run_%=%)