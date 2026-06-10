# Getting Started

This page runs Stream end-to-end on a bundled workload. It assumes you have [installed](installation.md) the package (`pip install -e .`) and are in the repo root.

## Run from the command line

`scripts/main_stream_co.py` is the general-purpose entry point. Point it at a hardware YAML and an ONNX workload; omit `--mapping` to let the pipeline generate one automatically.

```bash
python scripts/main_stream_co.py \
  --hardware stream/inputs/examples/hardware/tpu_like_quad_core.yaml \
  --workload stream/inputs/testing/workload/2conv_1_8_32_32_16_32_3.onnx
```

This runs the full constraint-optimization (CO) pipeline — parse → tile → estimate cost → MILP allocation → memory estimation — on a small two-Conv workload over a TPU-like quad-core system. It finishes in a few seconds and prints the total latency:

```
Total latency: 14344.0
  Group 0: 14344 (100.0%, wall=9.4s)
```

A YAML summary is written under `outputs/<experiment-id>/summary.yaml`, alongside workload, tiling, and schedule PNG visualizations. See [Outputs](outputs.md).

### CLI options

```bash
python scripts/main_stream_co.py \
  --hardware PATH_TO_HW_YAML \
  --workload PATH_TO_ONNX \
  [--mapping PATH_TO_MAPPING_YAML]  # omit for an auto-generated mapping
  [--output OUTPUT_DIR]             # default: "outputs"
  [--experiment-id ID]
  [--skip-if-exists]
```

If this repo uses [`just`](https://github.com/casey/just), `just co-2conv <hw>` and `just co-swiglu <hw>` wrap the same command across the example architectures.

## Run from Python

The public API lives in `stream/api.py`. The primary entry point auto-generates the mapping:

```python
import tempfile
from stream.api import configure_logging, optimize_allocation_co_generic

configure_logging()

with tempfile.TemporaryDirectory() as tmp:
    ctx = optimize_allocation_co_generic(
        hardware="stream/inputs/examples/hardware/tpu_like_quad_core.yaml",
        workload="stream/inputs/testing/workload/2conv_1_8_32_32_16_32_3.onnx",
        experiment_id="my-first-run",
        output_path=tmp,
    )
    print("total_latency:", ctx.get("total_latency"))      # -> 14344.0
    print("group_latencies:", ctx.get("group_latencies"))
```

All API entry points return a `StageContext`; useful keys include `total_latency`, `group_latencies`, `scheduler`, `workload`, and `accelerator`.

The other public functions:

- `optimize_allocation_co_with_mapping(hardware, workload, mapping, experiment_id, output_path, ...)` — run CO with a hand-written mapping YAML. `optimize_allocation_co` is a backward-compatible alias.
- `optimize_mapping(hardware, workload, experiment_id, output_path, max_nb_mappings=20, ...)` — a DSE sweep that enumerates mapping variants and runs CO for each.

All of them accept `backend=` (default `"ortools_gscip"`; also `"ortools_highs"`, `"gurobi"`).

## Where to go next

- [User Guide](user-guide.md) — the workload, hardware, and mapping input formats in detail.
- [Stages](stages.md) — what each pipeline stage does and how to extend the pipeline.
- [Using Stream with an AI agent](ai-agents.md) — the skills, MCP server, and IR models.
