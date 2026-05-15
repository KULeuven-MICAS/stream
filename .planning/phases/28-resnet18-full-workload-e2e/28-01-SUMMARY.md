---
phase: 28-resnet18-full-workload-e2e
plan: 01
subsystem: pipeline
tags: [onnx, visualization, sram, milp, timing, yaml]

# Dependency graph
requires:
  - phase: 24-multi-group-co-pipeline
    provides: FusionGroupIterationStage with group_latencies tracking
  - phase: 27-resnet18-bounded-fusion
    provides: ResNet18 fusion group splitting with cut points
provides:
  - Visualization guard for large ONNX workloads (>30 nodes)
  - 4MB SRAM for SIMD and pooling cores (MILP feasibility for ResNet18 Group 0)
  - Per-group wall-clock timing in FusionGroupIterationStage
  - Extended YAML summary with wall_time_s per group and total_wall_time_s
affects: [28-02-PLAN, resnet18-e2e-run]

# Tech tracking
tech-stack:
  added: []
  patterns: [node-count-guard-for-expensive-ops, wall-clock-instrumentation-around-pipeline-stages]

key-files:
  created: []
  modified:
    - stream/stages/parsing/onnx_model_parser.py
    - stream/inputs/examples/hardware/cores/simd.yaml
    - stream/inputs/examples/hardware/cores/pooling.yaml
    - stream/stages/generation/fusion_group_iteration.py
    - main_stream_co.py

key-decisions:
  - "VIZ_NODE_LIMIT=30: skip visualization for workloads exceeding 30 nodes to prevent hang on 93-node ResNet18"
  - "SRAM 4MB (33554432 bits): covers Group 0 reuse-factored tensor (3.07MB) with margin"
  - "Wall-clock timing uses time.time() around outer pipeline call, not pipeline internals"

patterns-established:
  - "Node-count guard: guard expensive visualization/IO with node count threshold and log WARNING when skipped"
  - "Pipeline timing: time.time() around sub_stage.run() + store in context dict for downstream consumers"

requirements-completed: [RNET-06]

# Metrics
duration: 7min
completed: 2026-05-15
---

# Phase 28 Plan 01: Fix Pipeline Blockers and Add Wall-Clock Timing Summary

**Visualization guard for large workloads, SRAM increase to 4MB for MILP feasibility, and per-group wall-clock timing with YAML summary output**

## Performance

- **Duration:** 7 min
- **Started:** 2026-05-15T09:19:25Z
- **Completed:** 2026-05-15T09:26:22Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments
- Added _VIZ_NODE_LIMIT=30 guard in ONNXModelParserStage to skip visualization for large workloads (prevents 480s hang on 93-node ResNet18)
- Increased SIMD and pooling core SRAM from 128KB to 4MB (33554432 bits) to accommodate reuse-factored tensors for MILP feasibility
- Added time.time() instrumentation around inner pipeline in FusionGroupIterationStage with group_wall_times stored in StageContext
- Extended _write_yaml_summary() in main_stream_co.py with wall_time_s per group and total_wall_time_s

## Task Commits

Each task was committed atomically:

1. **Task 1: Fix visualization hang and SRAM infeasibility blockers** - `2bbf655` (fix)
2. **Task 2: Add wall-clock timing to FusionGroupIterationStage and extend YAML summary** - `45507fb` (feat)

## Files Created/Modified
- `stream/stages/parsing/onnx_model_parser.py` - Added _VIZ_NODE_LIMIT=30 guard: skip workload.visualize() when node count exceeds threshold
- `stream/inputs/examples/hardware/cores/simd.yaml` - SRAM size increased from 1048576 to 33554432 (128KB to 4MB), key renamed to sram_4MB_2rw
- `stream/inputs/examples/hardware/cores/pooling.yaml` - SRAM size increased from 1048576 to 33554432 (128KB to 4MB), key renamed to sram_4MB
- `stream/stages/generation/fusion_group_iteration.py` - Added import time, group_wall_times dict, time.time() around inner pipeline, and per-group wall time logging
- `main_stream_co.py` - Extended _write_yaml_summary() with group_wall_times, wall_time_s per group, total_wall_time_s, and wall time in log output

## Decisions Made
- VIZ_NODE_LIMIT=30: threshold chosen because ResNet18 has 93 nodes (well above), while per-group sub-workloads have 3-6 nodes (well below)
- SRAM 4MB (33554432 bits): chosen to cover Group 0's 3.07MB reuse-factored tensor requirement with margin; Groups 1-10 require less
- Wall-clock timing uses time.time() around outer pipeline call for simplicity and accuracy (captures full group processing including setup)

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
- Test runner requires `python -m pytest` instead of `pytest` in this worktree to resolve module imports; pre-existing issue unrelated to changes

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Both pipeline blockers (visualization hang and MILP infeasibility) are resolved
- Wall-clock timing is wired through StageContext to YAML summary
- Ready for Phase 28 Plan 02: full ResNet18 E2E pipeline run
- All 194 existing tests pass without regression

## Self-Check: PASSED

All 6 files exist. Both commit hashes (2bbf655, 45507fb) verified. All 6 content checks passed.

---
*Phase: 28-resnet18-full-workload-e2e*
*Completed: 2026-05-15*
