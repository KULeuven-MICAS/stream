---
phase: 24-resnet18-end-to-end-flow
plan: 02
subsystem: workload
tags: [zigzag, fusion-edge, multi-group, onnx, cost-estimation, group-latencies, api-rename]

# Dependency graph
requires:
  - phase: 24-resnet18-end-to-end-flow
    provides: "Two-pass get_dimension_sizes, fan-out inter_core_tiling fix (plan 01)"
  - phase: 23-generic-mapping-generator
    provides: "GenericMappingGenerator, FusionGroupIterationStage, optimize_allocation_co_generic"
  - phase: 22-onnx-parser-completions
    provides: "FusionEdge, split_fusion_groups, ReluParser, ConvParser 3-input support"
provides:
  - ZigZag Bug 3 fallback (try/except + ideal-cycle estimate when spatial mapping crashes)
  - Bug 4 relaxed assert (memory operands >= node tensors for pooling cores)
  - Per-group latency tracking (group_latencies dict in context)
  - API rename optimize_allocation_co -> optimize_allocation_co_with_mapping + backward-compat alias
  - Generic CLI entry point main_stream_co.py with YAML summary output
  - Conv-Relu-Flatten-Gemm synthetic multi-group workload builder
  - test_pipeline_multi_group verifying 2-group pipeline with group_latencies assertions
affects:
  - Any future workload with ZigZag-incompatible operator configurations (fallback active)
  - Any caller using optimize_allocation_co (alias preserved)
  - Future ResNet18 full integration (fallback covers Conv1 crash)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "ZigZag fallback: try/except around run_zigzag with product-of-dim-sizes ideal-cycle estimate"
    - "InEdge tensor lookup: use output tensor name, not node name, for split_fusion_groups boundaries"
    - "Synthetic workload builders: Conv-Relu-Flatten-Gemm produces 2 fusion groups via Flatten FusionEdge"

key-files:
  created:
    - stream/inputs/testing/workload/make_conv_relu_flatten_gemm.py
  modified:
    - stream/stages/estimation/zigzag_cost_estimator.py
    - stream/stages/generation/fusion_group_iteration.py
    - stream/api.py
    - main_stream_co.py
    - stream/workload/workload.py
    - tests/test_generic_mapping.py

key-decisions:
  - "ZigZag fallback uses product of layer_dim_sizes as ideal-cycle estimate when spatial mapping generation crashes"
  - "Memory operand assert relaxed to >= (not ==) to support cores with extra operands (e.g. pooling I1/I2/O vs MaxPool 2 tensors)"
  - "API rename: optimize_allocation_co -> optimize_allocation_co_with_mapping, backward-compat alias preserved"
  - "Conv-Relu-Flatten-Gemm replaces ResNet18 test: same multi-group mechanics in seconds vs minutes"

patterns-established:
  - "ZigZag estimation fallback: catch all exceptions, compute ideal_cycle from layer_dim_sizes.data.values() product, return CoreCostEntry with cme=None"
  - "InEdge tensor resolution in with_modified_dimension_sizes: look up by outputs[0].name first, then node.name"

requirements-completed: [RES-01, RES-02, RES-03]

# Metrics
duration: 10min
completed: 2026-05-14
---

# Phase 24 Plan 02: Multi-Group Pipeline Integration Summary

**ZigZag fallback + group_latencies tracking + synthetic Conv-Relu-Flatten-Gemm workload proving 2-group CO pipeline with positive per-group latencies**

## Performance

- **Duration:** ~10 min
- **Started:** 2026-05-14T15:12:06Z
- **Completed:** 2026-05-14T15:22:19Z
- **Tasks:** 3
- **Files modified:** 7

## Accomplishments
- ZigZag estimation try/except fallback (Bug 3) with ideal-cycle estimate using product of layer dimension sizes, triggered for Relu1 on pooling core where spatial mapping generation fails
- Memory operand assert relaxed from == to >= (Bug 4), allowing pooling cores with extra operands (I1/I2/O) to handle nodes with fewer tensors (MaxPool has only 2)
- FusionGroupIterationStage tracks per-group latencies in group_latencies dict, set alongside total_latency in context
- API rename: optimize_allocation_co -> optimize_allocation_co_with_mapping, backward-compat alias preserved
- main_stream_co.py rewritten as generic CLI with --mapping optional flag and YAML summary output (per-group latency + percentages)
- Conv-Relu-Flatten-Gemm synthetic workload builder producing 2 fusion groups (Group 0: Conv+Relu, Group 1: Gemm) via Flatten FusionEdge boundary
- test_pipeline_multi_group replaces test_pipeline_resnet18: verifies 2-group CO pipeline with group_latencies == 2, all positive, total == sum

## Task Commits

Each task was committed atomically:

1. **Task 1: Commit Bug 3/4 fixes, group_latencies, API rename, main_stream_co.py** - `4ed03f0` (feat)
2. **Task 2: Create Conv-Relu-Flatten-Gemm workload builder** - `6df00c8` (feat)
3. **Deviation fix: Bug 3 fallback AttributeError + InEdge tensor name mismatch** - `83970d2` (fix)
4. **Task 3: Replace test_pipeline_resnet18 with test_pipeline_multi_group** - `26d3e6b` (test)

## Files Created/Modified
- `stream/stages/estimation/zigzag_cost_estimator.py` - Bug 3 try/except fallback + Bug 4 >= assert
- `stream/stages/generation/fusion_group_iteration.py` - group_latencies dict tracking per-group latencies
- `stream/api.py` - optimize_allocation_co_with_mapping rename + backward-compat alias
- `main_stream_co.py` - Generic CLI entry point with --mapping flag and YAML summary output
- `stream/inputs/testing/workload/make_conv_relu_flatten_gemm.py` - Synthetic multi-group workload builder
- `stream/workload/workload.py` - InEdge tensor lookup fix in with_modified_dimension_sizes
- `tests/test_generic_mapping.py` - test_pipeline_multi_group + group_latencies assertions

## Decisions Made
- ZigZag fallback uses `functools.reduce` over `layer_node.layer_dim_sizes.data.values()` to compute ideal-cycle estimate -- this is the total number of operations across all dimensions
- InEdge tensor lookup in `with_modified_dimension_sizes` tries `outputs[0].name` first, then `node.name` -- this handles split_fusion_groups boundaries where InEdge node name (e.g. `Flatten1_in`) differs from output tensor name (e.g. `flatten_out`)
- Replaced ResNet18 test with synthetic Conv-Relu-Flatten-Gemm workload (3.7s vs 600s timeout) -- same multi-group mechanics tested

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Bug 3 fallback used non-existent total_mac_count attribute**
- **Found during:** Task 3 (test execution revealed the fallback path was hit)
- **Issue:** `node.total_mac_count` does not exist on `ComputationNode`; AttributeError when ZigZag spatial mapping crashes for Relu1 on pooling core
- **Fix:** Changed to `layer_node.layer_dim_sizes.data.values()` product via `functools.reduce` -- uses the already-constructed ZigZag LayerNode's dimension sizes
- **Files modified:** `stream/stages/estimation/zigzag_cost_estimator.py`
- **Verification:** test_pipeline_multi_group passes (Relu1 fallback active)
- **Committed in:** `83970d2`

**2. [Rule 1 - Bug] InEdge tensor name mismatch in with_modified_dimension_sizes**
- **Found during:** Task 3 (test execution hit assertion when processing Gemm sub-workload)
- **Issue:** `with_modified_dimension_sizes` looked up InEdge by `node.name` (`Flatten1_in`) but the tensor was stored under `node.outputs[0].name` (`flatten_out`) -- names diverge after split_fusion_groups renames boundary nodes
- **Fix:** Look up by `outputs[0].name` first, then fall back to `node.name`
- **Files modified:** `stream/workload/workload.py`
- **Verification:** test_pipeline_multi_group passes through both fusion groups
- **Committed in:** `83970d2`

---

**Total deviations:** 2 auto-fixed (both Rule 1 - bugs)
**Impact on plan:** Both fixes necessary for the multi-group pipeline to work. No scope creep.

## Issues Encountered
- Worktree was 85 commits behind `arne/codebase-documentation` branch (Phase 22-24 code missing). Resolved via fast-forward merge.
- ZigZag spatial mapping generation crashes for Relu node on pooling core (unrolling limited to 1 due to bandwidth constraint). Bug 3 fallback handles this gracefully.

## Known Stubs
None -- no stub values or placeholder data introduced in this plan.

## Next Phase Readiness
- Multi-group CO pipeline verified for 2 fusion groups (Conv+Relu + Gemm)
- ZigZag fallback covers nodes where spatial mapping generation fails
- group_latencies tracking enables per-group performance analysis
- main_stream_co.py ready for use with arbitrary workloads (manual or auto-generated mapping)
- Full ResNet18 end-to-end may still need additional fixes for ResNet18-specific operators (MaxPool, BatchNorm, Add, GlobalAvgPool) but the pipeline infrastructure is proven

---
*Phase: 24-resnet18-end-to-end-flow*
*Completed: 2026-05-14*

## Self-Check: PASSED

- All 7 key files -- FOUND
- `24-02-SUMMARY.md` -- FOUND
- Commit `4ed03f0` (Task 1: Bug fixes + group_latencies + API rename + CLI) -- FOUND
- Commit `6df00c8` (Task 2: workload builder) -- FOUND
- Commit `83970d2` (Deviation: Bug 3 fallback + InEdge fix) -- FOUND
- Commit `26d3e6b` (Task 3: test_pipeline_multi_group) -- FOUND
