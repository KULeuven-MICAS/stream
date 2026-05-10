---
phase: 15-pre-flight-cleanup
plan: "02"
subsystem: mapping-ir-serialization
tags: [serialization, ir, mapping, scheduler, json, tdd]
dependency_graph:
  requires: []
  provides: [Mapping.get_ir(), SteadyStateScheduler.get_ir()]
  affects: [Phase 16 IR Models, Phase 18 MCP Tools]
tech_stack:
  added: []
  patterns: [get_ir() dict serialization pattern, isinstance-based resource dispatch, TDD red-green]
key_files:
  created:
    - tests/unit/test_get_ir.py
  modified:
    - stream/mapping/mapping.py
    - stream/cost_model/steady_state_scheduler.py
decisions:
  - "Use isinstance checks (Core vs MulticastPathPlan) to serialize resource_allocation entries as typed dicts"
  - "Exclude kernel field from Mapping.get_ir() — AIEKernel is compiler-internal, handled separately in Phase 16"
  - "Tests use real Core and LayerDim objects (not pure mocks) because isinstance checks require real types"
  - "SteadyStateScheduler.get_ir() delegates to Mapping.get_ir() rather than re-implementing node serialization"
metrics:
  duration_seconds: ~300
  completed_date: "2026-05-10"
  tasks_completed: 2
  files_changed: 3
---

# Phase 15 Plan 02: get_ir() Serialization Boundary Summary

**One-liner:** `Mapping.get_ir()` and `SteadyStateScheduler.get_ir()` return fully JSON-serializable nested dicts via isinstance dispatch on Core/MulticastPathPlan resource types.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| RED | Failing tests for both get_ir() methods | fece966 | tests/unit/test_get_ir.py (created) |
| 1 GREEN | Mapping.get_ir() implementation | e708643 | stream/mapping/mapping.py, tests/unit/test_get_ir.py |
| 2 GREEN | SteadyStateScheduler.get_ir() implementation | f15b835 | stream/cost_model/steady_state_scheduler.py |

## What Was Built

### Mapping.get_ir()

Returns a dict with:
- `"nodes"`: keyed by node.name, each entry has:
  - `"resource_allocation"`: list of slots, each slot is a list of typed dicts (`{"type": "core", "id": int}` or `{"type": "path", "sources": [int], "targets": [int], "hops": int}`)
  - `"inter_core_tiling"`: list of slots, each slot is a list of `[dim_str, factor]` pairs
  - `"memory_allocation"`: list of slots, each slot is a list of core IDs (ints)
- `"fused_groups"`: list of dicts with `name`, `layers`, `intra_core_tiling`
- `"runtime_args"`: dict of string key-value pairs

### SteadyStateScheduler.get_ir()

Returns a dict with:
- `"latency"`: `{"total": int, "per_iteration": int, "overlap_between_iterations": int}` (sentinel -1 pre-solve)
- `"backend"`: str (e.g., `"ORTOOLS_GSCIP"`)
- `"constraint_selection"`: dict of 4 bool fields or `null`
- `"fusion_splits"`: `{dim_str: int}` — LayerDim keys converted via `str()`
- `"mapping"`: output of `self.mapping.get_ir()`

## Tests

14 tests in `tests/unit/test_get_ir.py` covering:
- Empty Mapping, Mapping with cores, fused groups, MulticastPathPlan, runtime_args
- Pre-solve sentinel values, post-solve latency values, backend/constraint_selection, mapping delegation
- JSON round-trip for both get_ir() methods (the primary acceptance criterion)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Test helpers updated to use real Core/LayerDim objects**

- **Found during:** Task 1 GREEN implementation
- **Issue:** Tests used plain strings for `intra_core_tiling` dims and `MagicMock()` for Core objects. `isinstance(resource, Core)` checks in the implementation require real types; mock objects with no spec fail these checks.
- **Fix:** Changed `make_core()` to instantiate real `Core` objects; added `make_dim()` helper creating real `LayerDim` instances; updated all test fixtures accordingly.
- **Files modified:** tests/unit/test_get_ir.py
- **Commit:** e708643

## Known Stubs

None. Both `get_ir()` methods return fully wired data from real object attributes — no hardcoded placeholders or empty stubs.

## Self-Check: PASSED

- [x] `stream/mapping/mapping.py` contains `def get_ir` (1 occurrence)
- [x] `stream/cost_model/steady_state_scheduler.py` contains `def get_ir` (1 occurrence)
- [x] `tests/unit/test_get_ir.py` exists and contains 14 passing tests
- [x] Commits fece966, e708643, f15b835 exist in git log
- [x] JSON round-trip verified: `python -c "import json; from stream.mapping.mapping import Mapping; m = Mapping(); json.dumps(m.get_ir())"` succeeds
- [x] All 89 unit tests pass (no regressions)
- [x] `ruff check` passes on both modified files
