# Stream task runner. List recipes with `just --list`.
# Recipes use bare `python` — activate the venv first (`source .venv/bin/activate`)
# or ensure the project venv is on PATH.

# Show available recipes.
default:
    @just --list

# ----------------------------------------------------------------------------
# Non-AIE generic CO matrix (scripts/main_stream_co.py + the pytest test matrix)
# ----------------------------------------------------------------------------

# Regenerate the committed CI workload fixtures (2-conv + swiglu ONNX, weight values cleared) in
# stream/inputs/testing/workload/. The ONNX are committed to the repo — run this only to refresh them.
gen-workloads:
    python -c "from stream.inputs.testing.workload.make_2_conv import TwoConvWorkloadConfig, make_2_conv_workload; make_2_conv_workload(TwoConvWorkloadConfig(batch_size=1, in_channels=8, height=32, width=32, out_channels_1=16, out_channels_2=32, kernel_size=3, in_dtype='bf16', weight_dtype='bf16'))"
    python -c "from stream.inputs.testing.workload.make_swiglu import make_small_swiglu_workload; make_small_swiglu_workload()"

# Run the 2-conv workload on one hardware (default tpu_like_quad_core) via the generic CO pipeline.
# `hw` is a stem from stream/inputs/examples/hardware/ (e.g. fusemax, simba_small, eyeriss_like_dual_core).
co-2conv hw="tpu_like_quad_core":
    python scripts/main_stream_co.py \
      --hardware stream/inputs/examples/hardware/{{hw}}.yaml \
      --workload stream/inputs/testing/workload/2conv_1_8_32_32_16_32_3.onnx

# Run the swiglu workload on one hardware (default tpu_like_quad_core) via the generic CO pipeline.
co-swiglu hw="tpu_like_quad_core":
    python scripts/main_stream_co.py \
      --hardware stream/inputs/examples/hardware/{{hw}}.yaml \
      --workload stream/inputs/testing/workload/swiglu_1_16_32.onnx

# Run the full hardware x workload test matrix (parse + 2-conv + swiglu over all 8 non-AIE architectures).
matrix:
    python -m pytest tests/test_hardware_combinations.py

# ----------------------------------------------------------------------------
# AIE codegen (AMD Strix NPU — requires the [aie] extra and NPU hardware)
# ----------------------------------------------------------------------------

swiglu:
    python scripts/main_swiglu.py --embedding_dim 32 --hidden_dim 32 --seq_len 32 --m 1 --k 32 --n 32 --in_dtype bf16 --out_dtype bf16 --trace_size 1048576 --rows 4 --cols 1 --npu 2 --line_size 32

swiglu2:
    python scripts/main_swiglu.py --seq_len 32 --embedding_dim 32 --hidden_dim 32 --line_size 32 --m 1 --k 32 --n 32 --in_dtype bf16 --out_dtype bf16 --trace_size 1048576 --rows 4 --cols 4 --npu npu2 --mapping_version 2

swiglu3:
    python scripts/main_swiglu.py --seq_len 256 --embedding_dim 512 --hidden_dim 2048 --in_dtype bf16 --out_dtype bf16 --trace_size 1048576 --rows 4 --cols 8 --npu npu2 --embedding_tile_size 32 --hidden_tile_size 64

swiglu4:
    python scripts/main_swiglu.py --seq_len 256 --embedding_dim 2048 --hidden_dim 8192 --in_dtype bf16 --out_dtype bf16 --trace_size 1048576 --rows 4 --cols 8 --npu npu2 --embedding_tile_size 128 --hidden_tile_size 32 --seq_len_tile_size 16

gemm:
    python scripts/main_gemm.py --M 512 --N 512 --K 512 --m 16 --k 128 --n 32 --in_dtype bf16 --out_dtype bf16 --rows 4 --cols 8 --npu npu2 --trace_size 1048576
