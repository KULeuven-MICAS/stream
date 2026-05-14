# Phase 25: ResNet18 Sub-Graph Patterns - Context

**Gathered:** 2026-05-14
**Status:** Ready for planning

<domain>
## Phase Boundary

Verify that key ResNet18 structural patterns can each run through the full CO pipeline successfully as isolated sub-graphs. Build synthetic ONNX workloads for each pattern, run them through `optimize_allocation_co_generic` (single-group patterns) or `optimize_allocation_co_with_mapping` (multi-group with fixed mapping), and assert correctness.

This phase proves the pipeline handles real ResNet18 complexity — stride-2 convolutions, residual skip connections with fan-out, pooling core allocation, and multi-group residual blocks — without the cost of running the full 49-node model.

</domain>

<decisions>
## Implementation Decisions

### Sub-Graph Selection
- **D-01:** Test 4 distinct ResNet18 structural patterns as isolated sub-graphs:
  1. **Basic residual block:** Conv→Relu→Conv→Add with skip connection (fan-out from input to both main path and Add). Core ResNet building block (8 instances in ResNet18).
  2. **Stride-2 residual with downsample:** Conv(stride=2)→Relu→Conv + downsample Conv(1x1, stride=2)→Add. Tests dimension-changing skip connection (3 instances in ResNet18: layer2.0, layer3.0, layer4.0).
  3. **Front-end path:** Conv(7x7, stride=2)→Relu→MaxPool(stride=2). Tests large kernel + pooling core allocation + stride-2. Unique pattern in ResNet18.
  4. **Two back-to-back residual blocks:** Two complete basic residual blocks chained together, split into 2 fusion groups (one per block). The split is after the Add operator of the first block. Tests multi-group residual pipeline.

### Fusion Group Split for Dual-Residual
- **D-02:** Use a fixed mapping YAML for the dual-residual test. The builder generates both the ONNX model and a companion mapping YAML that defines the two fusion groups explicitly. Auto-detection of good cut points is deferred to Phase 26.

### Workload Builder Organization
- **D-03:** Single parametric builder: `make_resnet_subgraph()` with a config enum/dataclass that selects the pattern (BASIC_RESIDUAL, STRIDE2_DOWNSAMPLE, FRONTEND, DUAL_RESIDUAL). Conv/Relu/Add node creation logic is shared across patterns. Lives in `stream/inputs/testing/workload/`.
- **D-04:** For the dual-residual pattern, the builder returns `(onnx_path, mapping_path)` — both the ONNX model and the companion mapping YAML. Other patterns return just `onnx_path` (auto-generated mapping via GenericMappingGenerator).

### Test Structure
- **D-05:** New dedicated test file: `tests/test_resnet_patterns.py`. Keeps sub-graph pattern tests separate from the generic mapping tests in `test_generic_mapping.py`.
- **D-06:** Test assertions (beyond positive latency):
  1. Positive per-group latency + total (basic: total_latency > 0, all group_latencies positive)
  2. Correct group count (dual-residual: 2 groups, single-group patterns: 1 group)
  3. Core type allocation — verify pooling ops land on pooling cores (operator_types constraint respected)

### Failure Handling
- **D-07:** Fix all failures in Phase 25. The goal is to prove these patterns work. If a pattern fails (ZigZag crash, solver timeout, missing parser support), debug and fix it in this phase rather than deferring.

### Claude's Discretion
- Internal builder implementation details (helper functions, node creation patterns)
- Exact dimension values for each sub-graph (should be small enough for fast tests)
- How to inspect core type allocation in test assertions (mapping object or scheduler output)
- Whether the front-end path test uses `optimize_allocation_co_generic` or needs a fixed mapping (MaxPool may need pooling core in the mapping)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Workload Builders (reference pattern)
- `stream/inputs/testing/workload/make_2_conv.py` — TwoConvWorkloadConfig + builder pattern to follow
- `stream/inputs/testing/workload/make_conv_relu_flatten_gemm.py` — Multi-group builder with FusionEdge, most recent pattern

### Pipeline API
- `stream/api.py` — `optimize_allocation_co_generic()` (auto mapping) and `optimize_allocation_co_with_mapping()` (fixed mapping) entry points
- `stream/stages/generation/fusion_group_iteration.py` — FusionGroupIterationStage sets `total_latency` and `group_latencies` in context

### ONNX Parsers
- `stream/parser/onnx/model.py` — FUSION_EDGE_OPS set, registered parsers, shape inference call
- `stream/parser/onnx/conv.py` — ConvParser (handles stride, kernel_size, padding)
- `stream/parser/onnx/max_pool.py` — MaxPoolParser (stride, kernel indexing)
- `stream/parser/onnx/relu.py` — ReluParser (4D AffineMaps)
- `stream/parser/onnx/add.py` — AddParser (4D element-wise)

### Hardware
- `stream/inputs/examples/hardware/tpu_like_quad_core.yaml` — TPU hardware (cores 0-3 compute, 4 pooling, 5 simd, 6 offchip)
- `stream/inputs/examples/hardware/cores/pooling.yaml` — Pooling core with operator_types: [MaxPool, AveragePool, GlobalAveragePool, GlobalMaxPool]

### Mapping
- `stream/mapping/generic_generator.py` — GenericMappingGenerator (operator_types-aware core selection)
- `stream/parser/mapping_validator.py` — MappingValidator for fixed YAML validation
- `stream/inputs/examples/mapping/tpu_like_quad_core.yaml` — Reference fixed mapping (nested-list format)

### Existing Tests
- `tests/test_generic_mapping.py` — test_pipeline_multi_group (Phase 24 reference pattern)
- `tests/test_co.py` — test_co_tpu_two_conv (output_dir fixture pattern)

### ResNet18 Model (reference only)
- `stream/inputs/examples/workload/resnet18.onnx` — Full ResNet18 ONNX (20 Conv, 17 Relu, 8 Add, 1 MaxPool, 1 GlobalAveragePool, 1 Flatten, 1 Gemm = 49 nodes)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `make_conv_relu_flatten_gemm.py`: Most recent builder — configurable dataclass, ONNX helper functions, shape inference, save to file
- `TwoConvWorkloadConfig` + `make_2_conv_workload`: Original builder pattern with dtype mapping, weight initializers, `ClearField("float_data")`
- `GenericMappingGenerator`: Auto-generates mapping YAML for single-group workloads — can be used for basic residual, stride-2, and front-end patterns
- `MappingValidator`: Validates fixed YAML mappings before pipeline execution

### Established Patterns
- Builders save ONNX to `os.path.dirname(__file__)` with dimensions in filename
- Tests use `tempfile.TemporaryDirectory()` for output paths
- `optimize_allocation_co_generic` returns `StageContext` with `total_latency`, `group_latencies`, `scheduler`
- `@pytest.mark.timeout(120)` for tests that run the full pipeline

### Integration Points
- Builder lives in `stream/inputs/testing/workload/`
- Test imports builder + API functions
- For dual-residual: `optimize_allocation_co_with_mapping` takes explicit mapping path
- For single-group patterns: `optimize_allocation_co_generic` auto-generates mapping

</code_context>

<specifics>
## Specific Ideas

- The dual-residual split should be after the `Add` operator of the first block — this is a natural "synchronization point" where both paths have converged
- The user explicitly wants this to explore how fusion group boundaries can be detected at `Add` nodes, feeding into Phase 26's automatic fusion strategy
- Core type allocation assertion: verify that MaxPool in the front-end path lands on the pooling core (core 4), not a compute core
- Each sub-graph should use small dimensions (8-16 channels, 16x16 spatial) for fast test execution

</specifics>

<deferred>
## Deferred Ideas

- **Automatic cut-point detection at Add nodes** — Phase 26 scope. Phase 25 uses fixed mapping YAML instead.
- **max_group_depth parameter** — Phase 26 scope. Controls maximum nodes per fusion group.
- **Full ResNet18 E2E test** — Phase 27 scope. Requires Phase 26's fusion strategy to produce manageable groups.

</deferred>

---

*Phase: 25-resnet18-sub-graph-patterns*
*Context gathered: 2026-05-14*
