# Getting Started

This page runs Stream end-to-end twice. **Part 1** prices a small workload on a multi-core accelerator - no code generation, only the base install needed. **Part 2** adds AIE code generation: it maps a SwiGLU block onto an AMD Ryzen AI NPU and emits the MLIR that AMD's toolchain compiles for the device.

Both assume you have [installed](installation.md) Stream and are in the repository root, so the relative `stream/inputs/...` paths resolve.

---

## Part 1 - A first run: 2-conv on a TPU-like accelerator

This needs only the base install (`pip install -e .`). We map a tiny two-layer convolution onto a multi-core accelerator and let Stream's MILP solver place every tensor and choose every transfer path.

### The inputs

- **Hardware** - `tpu_like_quad_core.yaml` is a system of **four TPU-like compute cores** plus a pooling engine, a SIMD unit, and an off-chip DRAM controller, wired together by an on-chip interconnect.
- **Workload** - `2conv_1_8_32_32_16_32_3.onnx` is **two chained `Conv` layers** (a committed test fixture; only the tensor shapes matter for cost estimation, so the weights are cleared and the file stays tiny).
- **Mapping** *(optional)* - a hand-written mapping YAML. **Omit it** and the mapping generator that claims the hardware proposes one: which cores each layer may run on, and how layers are tiled across cores.

### Run it

```python
from stream.api import evaluate_mapping

estimate = evaluate_mapping(
    "stream/inputs/examples/hardware/tpu_like_quad_core.yaml",
    "stream/inputs/testing/workload/2conv_1_8_32_32_16_32_3.onnx",
    "outputs/first-run",
)
print("cycles:", estimate.cycles)
print("per group:", estimate.group_cycles)
```

Stream parses the hardware and workload, proposes a mapping, and runs the allocation of each fused group - **generate tilings** → **estimate per-core cost** → **MILP allocation** (the `AllocationModel`) → **memory estimation**. It finishes in a few seconds. `cycles` is the steady-state estimate summed over the fused groups, plus whatever reconfiguring the array between them costs on hardware that declares it. The solved `estimate.context` holds the `allocation`, `workload`, `accelerator` and `group_latencies`.

`evaluate_mapping` takes the mapping YAML as its fourth argument, `select_mapping` picks the cheapest of several candidate mappings, and `SolveOptions` sets the solver backend (default `"ortools_gscip"`; also `"ortools_highs"` and `"gurobi"`), the columns, the [constraint families](stages.md#constraint-families) of the allocation model, the solve's time limit (`time_limit_s`, 300 s) and whether the solver prints its log (`solver_log`).

### What you get

Everything lands under the output directory, one folder per fused group. The `allocation/` folder describes the allocation solve; `SolveOptions(artifacts=False)` writes none of its reports, traces and figures, as a sweep that only needs the estimates does, and a solve without a solution writes its model there for diagnosis (see [Outputs](outputs.md#allocation-artifacts)):

```
outputs/first-run/
└── group_0/                                 # one fused group of layers
    ├── mapping.yaml                         # the generated mapping that was used
    ├── tiled_workload.svg                   # the workload after inter-core tiling
    ├── core_cost_lut.yaml                   # per-node, per-core cost estimates
    └── allocation/                          # the MILP allocation result
        ├── reports/
        │   ├── optimization_metrics.yaml    # objective, solve time, gap, model size
        │   ├── optimization_trace.yaml      # the solver's progress (Gurobi)
        │   └── slot_latency_breakdown.yaml  # where the latency is spent
        ├── traces/
        │   ├── steady_state_trace.json      # schedule trace (open in Perfetto)
        │   └── steady_state_trace_compact.json
        └── figures/
            ├── optimization_progress.png    # the solver's progress (Gurobi)
            └── steady_state_workload_final.svg
```

The pictures are the quickest way to see what happened: `group_0/tiled_workload.svg` (how the layers were split across cores) and `group_0/allocation/figures/steady_state_workload_final.svg` (the resulting steady-state schedule). `steady_state_trace.json` opens in [Perfetto](https://ui.perfetto.dev) for a timeline view. See [Outputs](outputs.md) for the full reference.

You can run the same call against any of the bundled example architectures or the swiglu workload - see the [User Guide](user-guide.md) for the input formats.

---

## Part 2 - AIE code generation: SwiGLU on the AMD Strix NPU

`generate_code` runs the same solve and then lowers it through the code generation backend that claims the hardware. Here we map a **SwiGLU** block onto the **AMD Strix** NPU and emit the MLIR that AMD's toolchain turns into an NPU binary.

### Prerequisites

Code generation needs the AIE toolchain, which is **not** part of the base install (the wheels are platform-specific and git-hosted, so they cannot live in PyPI metadata). Install it once with the console script:

```bash
stream-setup-aie        # add --dry-run to preview the steps first
```

This requires **Linux x86_64** and **CPython 3.12 or 3.13**. See [Installation](installation.md#install) for details.

### The inputs

- **Hardware** - `stream/inputs/aie/hardware/whole_array_strix.yaml`: the AIE array of the **AMD Strix** NPU. It has eight columns, each with a shim-DMA tile, a 512 KB memory tile, and four AIE compute tiles - a 4×8 grid of compute tiles.
- **Workload** - a **SwiGLU** block: two projection `Gemm`s, a `SiLU` activation, an elementwise `Mul`, and a down-projection `Gemm`, built for a problem size by `make_swiglu_workload`.
- **Mapping** - built from tile sizes by `make_swiglu_mapping`.

### Run it

```python
from stream.api import SolveOptions, generate_code
from stream.inputs.aie.mapping.make_swiglu_mapping import make_swiglu_mapping
from stream.inputs.aie.workload.make_onnx_swiglu import make_swiglu_workload

workload = make_swiglu_workload(256, 512, 2048, "bf16", "bf16", last_gemm_down=True)
mapping = make_swiglu_mapping(256, 512, 2048, True, 32, 32, 64)
estimate = generate_code(
    "stream/inputs/aie/hardware/whole_array_strix.yaml",
    workload,
    "outputs/swiglu",
    mapping,
    SolveOptions(nb_cols_to_use=8, stage_options={"npu": "npu2"}),
)
print(estimate.context.get("module"))
```

`nb_cols_to_use=8` uses the full 4×8 compute-tile array, and `npu` targets the Strix (XDNA2) NPU. The MILP allocation over the whole array takes a minute or two; each fused group's design is written under `outputs/swiglu/group_<index>/codegen/`.

### The generated MLIR

The output is an MLIR module in AMD's `aie` / `aiex` dialects - tile placement, compute cores, and the object-FIFO data movement for the whole SwiGLU block:

```
builtin.module {
  aie.device(npu2) {
    %0 = aie.tile(0, 0)
    %1 = aie.tile(1, 0)
    ...
  }
}
```

### From MLIR to a running NPU binary

This `.mlir` is the **hand-off point** to AMD's AIE toolchain. The `aie` / `aiex` dialects it uses are exactly those of [**mlir-aie**](https://github.com/Xilinx/mlir-aie) and its **IRON** programming framework. mlir-aie lowers and compiles the module - placing the cores, building the object-FIFOs, and generating the host control program - into an NPU binary (an `xclbin` plus an instruction sequence) that **runs on AMD Ryzen AI NPUs** (the `npu2` target here is the XDNA2 NPU in AMD Strix).

In short: Stream decides *what* runs *where* and emits the MLIR; **mlir-aie** and **IRON** build that MLIR and deploy it on the device.

---

## Where to go next

- [User Guide](user-guide.md) - the workload, hardware, and mapping input formats in detail.
- [Stages](stages.md) - what each pipeline stage does and how to extend the pipeline.
- [Using Stream with an AI agent](ai-agents.md) - the MCP server and IR models.
