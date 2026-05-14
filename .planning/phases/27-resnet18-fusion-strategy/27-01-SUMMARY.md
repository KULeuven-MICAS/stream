---
phase: 27-resnet18-fusion-strategy
plan: 01
subsystem: workload
tags: [fusion-groups, resnet18, graph-splitting, cut-points, residual-boundary]

requires:
  - phase: 22-onnx-parser-completions
    provides: FusionEdge splitting, OutEdge/InEdge boundary pair creation
  - phase: 23-generic-mapping-generator
    provides: GenericMappingGenerator with per-group YAML generation
provides:
  - "determine_fusion_cut_points(workload) -> list[str] module-level function"
  - "split_fusion_groups(cut_points=None) extended with optional cut-point parameter"
  - "GenericMappingGenerationStage.run() auto-detects cut points before generating groups"
  - "ResNet18 splits into 11 manageable groups (3-6 ComputationNodes each)"
affects: [27-02, resnet18-end-to-end, fusion-group-iteration]

tech-stack:
  added: []
  patterns:
    - "Cut-point heuristic: module-level analyzer function separated from splitting mechanism (policy vs mechanism)"
    - "Backward-compatible optional parameter: cut_points=None preserves existing FusionEdge-only behavior"

key-files:
  created:
    - tests/test_resnet_patterns.py (2 new tests added)
  modified:
    - stream/workload/workload.py
    - stream/mapping/generic_generator.py
    - stream/stages/generation/generic_mapping_generation.py

key-decisions:
  - "Removed fan-out guard from determine_fusion_cut_points: ResNet Relu nodes naturally fan out to 2 successors (next conv + skip Add); all 8 Add+Relu pairs are valid cut points"
  - "Added last-ComputationNode guard: prevent cut point at the last ComputationNode in a workload (would create empty trailing group)"

patterns-established:
  - "Cut-point boundary pairs: OutEdge/InEdge created at non-FusionEdge ComputationNodes using the node's output tensor"
  - "Graph analyzer as module-level function: determine_fusion_cut_points() is a standalone function, not a Workload method"

requirements-completed: [RNET-04]

duration: 11min
completed: 2026-05-14
---

# Phase 27 Plan 01: Bounded Fusion Group Splitting Summary

**Add+Relu residual boundary heuristic splits ResNet18 into 11 groups via determine_fusion_cut_points() and extended split_fusion_groups(cut_points=)**

## Performance

- **Duration:** 11 min
- **Started:** 2026-05-14T21:23:25Z
- **Completed:** 2026-05-14T21:35:11Z
- **Tasks:** 3
- **Files modified:** 4

## Accomplishments

- Implemented `determine_fusion_cut_points(workload)` module-level function that identifies 9 cut points for ResNet18 (1 MaxPool front-end boundary + 8 Add+Relu residual boundaries)
- Extended `split_fusion_groups()` with optional `cut_points` parameter that creates OutEdge/InEdge boundary pairs at cut-point nodes (same pattern as FusionEdge boundaries)
- Threaded cut_points through `GenericMappingGenerator.generate_all_groups()` and `GenericMappingGenerationStage.run()`
- Added 2 unit tests verifying 9 cut points and 11-group split
- All 6 test_resnet_patterns tests pass, all unit/test_onnx_parser tests pass (backward compatible)

## Files Created/Modified

- `stream/workload/workload.py` -- Added `determine_fusion_cut_points()` function and extended `split_fusion_groups(cut_points=None)`
- `stream/mapping/generic_generator.py` -- Extended `generate_all_groups(cut_points=None)` to pass cut_points through
- `stream/stages/generation/generic_mapping_generation.py` -- Added `determine_fusion_cut_points()` call in `run()`
- `tests/test_resnet_patterns.py` -- Added `test_fusion_cut_points_heuristic` and `test_resnet18_split_with_cut_points`

## Decisions Made

- **Removed fan-out guard:** The plan specified a fan-out guard (skip if Relu has >1 ComputationNode successor). However, in ResNet, 7 of 8 Relu_1 nodes have exactly 2 ComputationNode successors (next block's conv1 and skip connection's Add/downsample). Applying the guard would produce only 2 cut points instead of the required 9. Since all Relu successors go into the SAME next group (the InEdge boundary pair feeds them both), the fan-out is safe. Removed the guard to match the must_have truth of 9 cut points.
- **Added last-ComputationNode guard:** When a cut point is the last ComputationNode in a workload (e.g., MaxPool in the 3-node frontend subgraph), splitting after it creates an empty group with zero ComputationNodes, causing downstream sympy matrix errors. Added a guard that removes such trailing cut points.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Removed incorrect fan-out guard from determine_fusion_cut_points()**
- **Found during:** Task 1
- **Issue:** Plan specified `len(relu_comp_succs) <= 1` guard, but ResNet Relu nodes have 2 ComputationNode successors due to skip connections. Only 1 of 8 Relu_1 nodes would pass the guard, producing 2 cut points instead of the required 9.
- **Fix:** Removed the fan-out guard. The cut-point splitting mechanism correctly handles fan-out because all successors go into the same next group via the InEdge boundary pair.
- **Files modified:** stream/workload/workload.py
- **Committed in:** fe3da60

**2. [Rule 1 - Bug] Added last-ComputationNode guard to prevent empty trailing groups**
- **Found during:** Task 3 (running existing test_frontend_path)
- **Issue:** For the frontend subgraph (Conv+Relu+MaxPool), MaxPool is both a cut point AND the last ComputationNode. Splitting after it creates an empty group, causing sympy matrix shape mismatch in unique_dimensions().
- **Fix:** Added guard at end of determine_fusion_cut_points() that removes the last cut point if it matches the last ComputationNode name.
- **Files modified:** stream/workload/workload.py
- **Committed in:** 83e2984

---

**Total deviations:** 2 auto-fixed (both bug fixes)
**Impact on plan:** Both fixes were required for correctness. No scope creep.

## Issues Encountered

- None beyond the deviations documented above.

## Next Phase Readiness

- Cut-point infrastructure ready for Plan 02 (full ResNet18 E2E CO pipeline integration)
- All 11 groups produce valid sub-workloads with proper InEdge/OutEdge boundaries
- GenericMappingGenerator scales to 11 groups with existing per-group YAML generation

---
*Phase: 27-resnet18-fusion-strategy*
*Plan: 01*
*Completed: 2026-05-14*
