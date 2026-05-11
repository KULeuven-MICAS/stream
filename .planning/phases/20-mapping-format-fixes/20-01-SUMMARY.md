---
phase: 20-mapping-format-fixes
plan: 01
subsystem: mapping
tags: [mapping, yaml, workload, cost-model, testing]

# Dependency graph
requires:
  - phase: 19-ga-removal
    provides: clean api.py without GA imports, all 176 tests passing
provides:
  - Correct nested-list core_allocation/inter_core_tiling format in make_2_conv_mapping.py
  - NameError-safe with_updated_workload (FusedGroup constructed outside inner loop)
  - Empty inter_core_tiling guards in get_unique_dims_inter_core_tiling, get_tensor_single_core, _get_possible_memory_core_allocations
  - rsplit-based intra_core_tiling dim parser for dotted ONNX node names
  - Dead mapping files removed (testing_mapping, simple_example_mapping, tpu_like_quad_core in testing/mapping/)
  - All 194 tests passing (including test_core_cost_lut_caching)
affects: [21-tpu-e2e-test, mapping-layer, cost-model, workload-processing]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Guard inter_core_tiling empty-tuple before [0] access (three callsites)"
    - "Use rsplit('.', 1) instead of split('.') for ONNX node names with dots"
    - "Wrap mapping generator lists in extra list layer to match nested-list schema"
    - "Construct FusedGroup once after inner loop completes, not inside it"

key-files:
  created: []
  modified:
    - stream/inputs/testing/mapping/make_2_conv_mapping.py
    - stream/mapping/mapping.py
    - stream/parser/mapping_factory.py
    - stream/workload/workload.py
    - stream/cost_model/steady_state_scheduler.py
    - stream/inputs/testing/mapping/2conv_1_32_32_8_16_32_3.yaml
    - tests/test_core_cost_lut_caching.py

key-decisions:
  - "FMT-05 (tpu_like_quad_core.yaml in examples/ validates) deferred to v1.5 per D-01/D-02"
  - "Dead mapping files deleted rather than fixed (no references from tests or scripts)"
  - "No new unit tests for individual bug fixes — Phase 21 adds E2E TPU test (D-04)"

patterns-established:
  - "Guard pattern: if not node_mapping.inter_core_tiling: return () before any [0] access"
  - "Mapping generator format: core_allocation and inter_core_tiling must be list[list[...]]"

requirements-completed: [FMT-01, FMT-02, FMT-03, FMT-04, FMT-06]

# Metrics
duration: 18min
completed: 2026-05-11
---

# Phase 20 Plan 01: Mapping Format Fixes Summary

**Four independent mapping-layer bugs patched (NameError, IndexError x3, split, nested-list format) plus dead-file cleanup, unblocking test_core_cost_lut_caching and Phase 21 TPU E2E test**

## Performance

- **Duration:** ~18 min
- **Started:** 2026-05-11T11:24:00Z
- **Completed:** 2026-05-11T11:42:00Z
- **Tasks:** 2
- **Files modified:** 7 (+ 5 deleted)

## Accomplishments
- Fixed nested-list format in make_2_conv_mapping.py (core_allocation and inter_core_tiling now list[list[...]])
- Patched NameError in with_updated_workload by moving FusedGroup construction outside inner loop
- Added empty inter_core_tiling guards in get_unique_dims_inter_core_tiling, get_tensor_single_core, and _get_possible_memory_core_allocations
- Fixed rsplit(".", 1) in _convert_intra_core_tiling_entry for ONNX node names with dots
- Deleted 5 dead mapping files, regenerated 2conv YAML, removed vestigial layer_stacks test arg
- All 194 tests pass including test_core_cost_lut_caching

## Task Commits

Each task was committed atomically:

1. **Task 1: Fix make_2_conv_mapping format and patch three code-level bugs** - `e3656ce` (fix)
2. **Task 2: Delete dead mapping files, regenerate YAML, remove vestigial test arg** - `e45af2e` (chore)

**Plan metadata:** (docs commit — see below)

## Files Created/Modified
- `stream/inputs/testing/mapping/make_2_conv_mapping.py` - FMT-01: wrap core_allocation/inter_core_tiling in extra list layer
- `stream/mapping/mapping.py` - FMT-02: move FusedGroup construction outside inner loop in with_updated_workload
- `stream/parser/mapping_factory.py` - FMT-04: use rsplit(".", 1) for dotted ONNX names
- `stream/workload/workload.py` - FMT-03: guard empty inter_core_tiling in get_unique_dims_inter_core_tiling and get_tensor_single_core
- `stream/cost_model/steady_state_scheduler.py` - FMT-03: guard empty inter_core_tiling in _get_possible_memory_core_allocations
- `stream/inputs/testing/mapping/2conv_1_32_32_8_16_32_3.yaml` - Regenerated with nested format
- `tests/test_core_cost_lut_caching.py` - D-04: remove vestigial layer_stacks argument

Deleted (unreferenced dead files):
- `stream/inputs/testing/mapping/testing_mapping.yaml`
- `stream/inputs/testing/mapping/testing_mapping.py`
- `stream/inputs/testing/mapping/simple_example_mapping.yaml`
- `stream/inputs/testing/mapping/simple_example_mapping.py`
- `stream/inputs/testing/mapping/tpu_like_quad_core.yaml`

## Decisions Made
- Dead mapping files deleted (not fixed): all 5 had zero references outside their own directory
- FMT-05 (tpu_like_quad_core.yaml in examples/ validates) deferred to v1.5 per prior decisions D-01/D-02
- No new unit tests for individual bug fixes per D-04; Phase 21 will add E2E TPU test

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
- TwoConvWorkloadConfig requires all 9 positional arguments (no defaults) — used full config in verification command. Not a code change, just verification approach.

## Known Stubs

None — no stub values or placeholder text introduced.

## Next Phase Readiness
- All 194 tests pass; mapping layer bugs are fixed
- Phase 21 (TPU E2E test) is unblocked — make_2_conv_mapping now generates valid nested-list format that passes MappingValidator
- with_updated_workload handles empty intra_core_tiling, enabling non-AIE workloads through the CO pipeline

---
*Phase: 20-mapping-format-fixes*
*Completed: 2026-05-11*
