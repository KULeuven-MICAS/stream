---
gsd_state_version: 1.0
milestone: v1.6
milestone_name: ResNet18 Full Workload
status: executing
stopped_at: Completed 24-02-PLAN.md
last_updated: "2026-05-14T18:23:04.564Z"
last_activity: 2026-05-14 -- Phase 25 execution started
progress:
  total_phases: 6
  completed_phases: 3
  total_plans: 10
  completed_plans: 8
  percent: 0
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-05-14)

**Core value:** Enable users to explore the TETRA design space efficiently — selecting solver backends, toggling constraint groups, and understanding the impact of hardware constraints on schedule optimality
**Current focus:** Phase 25 — resnet18-sub-graph-patterns

## Current Position

Milestone: v1.6 (ResNet18 Full Workload) — PLANNING
Phase: 25 (resnet18-sub-graph-patterns) — EXECUTING
Plan: 2 of 2
Status: Executing Phase 25
Last activity: 2026-05-14 -- Completed 25-01-PLAN.md (FusionEdgeParser fix + ResNet18 sub-graph builder)

Progress: [░░░░░░░░░░] 0%

## Performance Metrics

**Velocity (from v1.0–v1.4):**

- Total plans completed: 33 (across 21 phases)
- Phases completed: 21

## Accumulated Context

### Decisions

Key decisions carried forward:

- [Phase 21]: Memory-less hardware (no on-chip memory tiles) falls back to single direct transfer; COMPUTE_TO_MEM uses offchip core as destination
- [Phase 21]: `"offchip"` core type treated like `"shim"` for transfer classification in determine_transfer_type
- [v1.4]: FMT-05 deferred — TPU mapping YAML validation assigned to Phase 23 (generic mapping generator)
- [v1.4]: determine_fusion_splits single-group assert must be satisfied; GenericMappingGenerator must emit exactly one FusedGroup
- [Phase 22]: FusionEdge(HasInputs, HasOutputs) not HasIterationSpace — dimension_relations auto-excludes FusionEdge edges
- [Phase 22]: split_fusion_groups() consumes FusionEdge nodes, producing OutEdge/InEdge boundary pairs in adjacent sub-workloads
- [Phase 22]: Drop optional bias silently via all_inputs[:2] — bias is not modeled in cost model; assert >= 2 guards minimum required inputs
- [Phase 22]: Shape inference runs in-memory via onnx.shape_inference.infer_shapes() in ONNXModelParser.run() before parse_workload() — populates intermediate tensor value_info
- [Phase 23-generic-mapping-generator]: Strip operator_types from core data before ZigZag validation, re-inject after normalization
- [Phase 23-generic-mapping-generator]: Specialized cores (non-None operator_types) take priority over generic compute cores in core selection
- [Phase 23]: File reads inside TemporaryDirectory context block — avoid FileNotFoundError after cleanup
- [Phase 23]: tpu_like_quad_core.yaml updated to nested-list format (FMT-05 validated)
- [Phase 24]: ZigZag fallback uses product of layer_dim_sizes as ideal-cycle estimate when spatial mapping generation crashes
- [Phase 24]: Memory operand assert relaxed to >= for cores with extra operands (pooling I1/I2/O vs MaxPool 2 tensors)
- [Phase 24]: API rename: optimize_allocation_co -> optimize_allocation_co_with_mapping, backward-compat alias preserved
- [Phase 24]: Conv-Relu-Flatten-Gemm synthetic workload replaces ResNet18 test: same multi-group mechanics in seconds

### Pending Todos

None.

### Blockers/Concerns

- ConvParser crashes on 3-input Conv nodes (bias) — blocks Phase 22; first fix in Phase 22
- PoolingParser broken at import — full rewrite required in Phase 22
- Fan-out dsts[0] assumption in SteadyStateScheduler — 8 fan-out points in ResNet18; fix in Phase 24

## Session Continuity

Last session: 2026-05-14T15:24:09.242Z
Stopped at: Completed 24-02-PLAN.md
Resume file: None
