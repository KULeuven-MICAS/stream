# Getting Started

This page runs Stream end-to-end twice. **Part 1** runs the constraint-optimization (CO) pipeline on a small workload - no code generation, only the base install needed. **Part 2** adds AIE code generation: it maps a SwiGLU block onto an AMD Ryzen AI NPU and emits the MLIR that AMD's toolchain compiles for the device.

Both assume you have [installed](installation.md) Stream and are in the repository root, so the relative `stream/inputs/...` paths resolve.

---

## Part 1 - A first run: 2-conv on a TPU-like accelerator

This needs only the base install (`pip install -e .`). We map a tiny two-layer convolution onto a multi-core accelerator and let Stream's MILP solver place every tensor and choose every transfer path.

### The inputs

`scripts/main_stream_co.py` takes two required inputs and one optional one:

- **`--hardware`** - a hardware description. `tpu_like_quad_core.yaml` is a system of **four TPU-like compute cores** plus a pooling engine, a SIMD unit, and an off-chip DRAM controller, wired together by an on-chip interconnect.
- **`--workload`** - an ONNX graph. `2conv_1_8_32_32_16_32_3.onnx` is **two chained `Conv` layers** (a committed test fixture; only the tensor shapes matter for cost estimation, so the weights are cleared and the file stays tiny).
- **`--mapping`** *(optional)* - a hand-written mapping YAML. **Omit it** and Stream auto-generates one: which cores each layer may run on, and how layers are tiled across cores.

### Run it

```bash
python scripts/main_stream_co.py \
  --hardware stream/inputs/examples/hardware/tpu_like_quad_core.yaml \
  --workload stream/inputs/testing/workload/2conv_1_8_32_32_16_32_3.onnx \
  --experiment-id first-run
```

Stream runs the full CO pipeline - **parse** hardware/workload/mapping → **generate tilings** → **estimate per-core cost** → **MILP allocation** (the `TransferAndTensorAllocator`) → **memory estimation**. It finishes in a few seconds. Amid the pipeline logs you will see the headline result:

```
Total latency: 14344.0
  Group 0: 14344 (100.0%, wall=9.3s)
```

`14344` is the steady-state latency in cycles; `wall` is how long the solver took.

> Omitting `--experiment-id` is fine - Stream derives one from the file names. We pass `first-run` here only so the output path below is short.

### What you get

Everything lands under `outputs/<experiment-id>/`:

```
outputs/first-run/
├── summary.yaml                         # headline latency result
├── workload_graph.png                   # the ONNX workload as a DAG
└── group_0/                             # one fused group of layers
    ├── mapping.yaml                     # the auto-generated mapping that was used
    ├── tiled_workload.png               # the workload after inter-core tiling
    ├── core_cost_lut.yaml               # per-node, per-core cost estimates
    └── tetra/                           # the MILP allocation result
        ├── optimization_metrics.yaml    # objective, solve time, gap, ...
        ├── slot_latency_breakdown.yaml  # where the latency is spent
        ├── steady_state_trace.json      # schedule trace (open in Perfetto)
        └── steady_state_workload_final.png
```

`summary.yaml` is the headline result:

```yaml
total_latency: 14344.0
total_wall_time_s: 9.34
groups:
  group_0:
    latency: 14344
    percentage: 100.0
    wall_time_s: 9.34
```

The PNGs are the quickest way to see what happened: `workload_graph.png` (the layers), `group_0/tiled_workload.png` (how they were split across cores), and `group_0/tetra/steady_state_workload_final.png` (the resulting steady-state schedule). `steady_state_trace.json` opens in [Perfetto](https://ui.perfetto.dev) for a timeline view. See [Outputs](outputs.md) for the full reference.

### The same run from Python

The CLI is a thin wrapper over `stream/api.py`:

```python
import tempfile
from stream.api import configure_logging, optimize_allocation_co_generic

configure_logging()

with tempfile.TemporaryDirectory() as tmp:
    ctx = optimize_allocation_co_generic(
        hardware="stream/inputs/examples/hardware/tpu_like_quad_core.yaml",
        workload="stream/inputs/testing/workload/2conv_1_8_32_32_16_32_3.onnx",
        experiment_id="first-run",
        output_path=tmp,
    )
    print("total_latency:", ctx.get("total_latency"))   # -> 14344.0
```

Every API entry point returns a `StageContext`; useful keys include `total_latency`, `group_latencies`, `scheduler`, `workload`, and `accelerator`. The companions are `optimize_allocation_co_with_mapping(...)` (run CO with a hand-written mapping) and `optimize_mapping(...)` (a DSE sweep over mapping variants). All accept `backend=` (default `"ortools_gscip"`; also `"ortools_highs"` and `"gurobi"`).

You can run the same command against any of the bundled example architectures or the swiglu workload - see the [User Guide](user-guide.md) for the input formats.

---

## Part 2 - AIE code generation: SwiGLU on the AMD Strix NPU

The same CO pipeline can additionally **generate MLIR** for AMD's AI Engine (AIE) array. Here we map a **SwiGLU** block onto the **AMD Strix** NPU and emit the MLIR that AMD's toolchain turns into an NPU binary.

### Prerequisites

Code generation needs the AIE toolchain, which is **not** part of the base install (the wheels are platform-specific and git-hosted, so they cannot live in PyPI metadata). Install it once with the console script:

```bash
stream-setup-aie        # add --dry-run to preview the steps first
```

This requires **Linux x86_64** and **CPython 3.12 or 3.13**. See [Installation](installation.md#install) for details.

### The inputs

The AIE entry points hard-wire their hardware and build the workload and mapping for you, so you only choose the problem size:

- **Hardware** - `stream/inputs/aie/hardware/whole_array_strix.yaml`: the AIE array of the **AMD Strix** NPU. It has eight columns, each with a shim-DMA tile, a 256 KB memory tile, and four AIE compute tiles - a 4×8 grid of compute tiles.
- **Workload** - a **SwiGLU** block: two projection `Gemm`s, a `SiLU` activation, an elementwise `Mul`, and a down-projection `Gemm`. It is built from the `--seq_len` / `--embedding_dim` / `--hidden_dim` you pass.
- **Mapping** - generated automatically from the tile-size flags (`--embedding_tile_size`, `--hidden_tile_size`, ...).

### Run it

```bash
python scripts/main_swiglu.py \
  --seq_len 256 --embedding_dim 512 --hidden_dim 2048 \
  --in_dtype bf16 --out_dtype bf16 \
  --rows 4 --cols 8 --npu npu2 \
  --embedding_tile_size 32 --hidden_tile_size 64
```

`--rows 4 --cols 8` uses the full 4×8 compute-tile array, and `--npu npu2` targets the Strix (XDNA2) NPU. This runs the CO pipeline **and** the AIE code-generation stage; the MILP allocation over the whole array takes a minute or two. The generated module is written into the run's experiment folder under `outputs/` (the same place Part 1's artifacts went):

```
Saved generated module to outputs/whole_array_strix-swiglu_256_512_2048-4_row_8_col/output.mlir
```

### The generated MLIR

The output is an MLIR module in AMD's `aie` / `aiex` dialects - tile placement, compute cores, and the object-FIFO data movement for the whole SwiGLU block:

```
builtin.module {
  aie.device(npu2) {
    %0 = aie.tile(0, 0)
    %1 = aie.tile(1, 0)
    ...
    %8 = aie.tile(1, 1)
    ...
  }
}
```

For this example that is roughly 2,400 lines: one `aie.device(npu2)`, 48 `aie.tile`s, 32 `aie.core`s, and the `aie.objectfifo`s that route activations and weights between them.

### From MLIR to a running NPU binary

This `.mlir` is the **hand-off point** to AMD's AIE toolchain. The `aie` / `aiex` dialects it uses are exactly those of [**mlir-aie**](https://github.com/Xilinx/mlir-aie) and its **IRON** programming framework. mlir-aie lowers and compiles the module - placing the cores, building the object-FIFOs, and generating the host control program - into an NPU binary (an `xclbin` plus an instruction sequence) that **runs on AMD Ryzen AI NPUs** (the `npu2` target here is the XDNA2 NPU in AMD Strix).

In short: Stream decides *what* runs *where* and emits the MLIR; **mlir-aie** and **IRON** build that MLIR and deploy it on the device.

---

## Where to go next

- [User Guide](user-guide.md) - the workload, hardware, and mapping input formats in detail.
- [Stages](stages.md) - what each pipeline stage does and how to extend the pipeline.
- [Using Stream with an AI agent](ai-agents.md) - the skills, MCP server, and IR models.
