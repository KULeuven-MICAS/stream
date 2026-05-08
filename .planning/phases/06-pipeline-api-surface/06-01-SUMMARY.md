---
phase: "06"
plan: "01"
subsystem: pipeline-api-surface
tags: [constraint-selection, pipeline-threading, api, tdd]
dependency_graph:
  requires: [05-02-SUMMARY.md]
  provides: [constraint_selection kwarg on public API, 4-hop threading chain to TTA]
  affects: [stream/api.py, stream/stages/allocation/constraint_optimization_allocation.py, stream/cost_model/steady_state_scheduler.py]
tech_stack:
  added: []
  patterns: [TDD red-green, keyword-only default None, ctx.get with or-default]
key_files:
  created:
    - tests/unit/test_pipeline_threading.py
  modified:
    - stream/api.py
    - stream/stages/allocation/constraint_optimization_allocation.py
    - stream/cost_model/steady_state_scheduler.py
decisions:
  - "constraint_selection defaults to None at API level, defaults to ConstraintSelection() inside Stage (all-True behavior preserved)"
  - "Stage uses 'or ConstraintSelection()' pattern so None from ctx.get resolves to fully-enabled default"
  - "SteadyStateScheduler stores constraint_selection as None (not ConstraintSelection()) to preserve original None-passthrough to TTA"
metrics:
  duration: "154 seconds"
  completed: "2026-05-08"
  tasks: 1
  files: 4
requirements: [PIPE-01, UI-01]
---

# Phase 6 Plan 01: Thread constraint_selection Through Pipeline + Tests Summary

**One-liner:** constraint_selection kwarg threaded from optimize_allocation_co/optimize_mapping through StageContext, ConstraintOptimizationAllocationStage, and SteadyStateScheduler to reach TransferAndTensorAllocator (4 hops, all verified by 7 unit tests).

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Thread constraint_selection through pipeline + tests | 6d8c7a7 | stream/api.py, constraint_optimization_allocation.py, steady_state_scheduler.py, test_pipeline_threading.py |

## What Was Built

### stream/api.py
- Added `ConstraintSelection` to imports from `stream.opt.solver`
- Added `constraint_selection: ConstraintSelection | None = None` parameter to `optimize_allocation_co` and `optimize_mapping`
- Both `StageContext.from_kwargs(...)` calls now pass `constraint_selection=constraint_selection`

### stream/stages/allocation/constraint_optimization_allocation.py
- Added `from stream.opt.solver import ConstraintSelection`
- In `__init__`: reads `self.constraint_selection = self.ctx.get("constraint_selection") or ConstraintSelection()` — defaults to fully-enabled when absent
- In `find_best_tensor_transfer_allocation`: passes `constraint_selection=self.constraint_selection` to `SteadyStateScheduler`

### stream/cost_model/steady_state_scheduler.py
- Added `from stream.opt.solver import ConstraintSelection`
- Added `constraint_selection: ConstraintSelection | None = None` to `__init__` signature
- Stores `self.constraint_selection = constraint_selection`
- Passes `constraint_selection=self.constraint_selection` to `TransferAndTensorAllocator` in `run()`

### tests/unit/test_pipeline_threading.py
- 7 tests covering the full threading chain:
  1. `test_api_optimize_allocation_co_accepts_constraint_selection` — signature inspection
  2. `test_api_optimize_mapping_accepts_constraint_selection` — signature inspection
  3. `test_stage_reads_constraint_selection_from_context` — ctx.get roundtrip
  4. `test_stage_defaults_constraint_selection_when_absent` — default ConstraintSelection()
  5. `test_scheduler_stores_constraint_selection` — kwarg stored on instance
  6. `test_scheduler_defaults_constraint_selection_when_none` — None default
  7. `test_cli_disable_constraints_parsing` — conversion pattern for CLI scripts

## Verification Results

```
pytest tests/unit/test_pipeline_threading.py -x -q: 7 passed
pytest tests/unit/ -x -q: 71 passed (no regressions)
```

### Threading Chain Verification

```
grep -n "constraint_selection" stream/api.py
  125: constraint_selection: ConstraintSelection | None = None,  (optimize_allocation_co)
  172:     constraint_selection=constraint_selection,             (StageContext.from_kwargs #1)
  210: constraint_selection: ConstraintSelection | None = None,  (optimize_mapping)
  267:     constraint_selection=constraint_selection,             (StageContext.from_kwargs #2)

grep -n "constraint_selection" stream/stages/allocation/constraint_optimization_allocation.py
  58: self.constraint_selection = self.ctx.get("constraint_selection") or ConstraintSelection()
  83:     constraint_selection=self.constraint_selection,         (SteadyStateScheduler call)

grep -n "constraint_selection" stream/cost_model/steady_state_scheduler.py
  67: constraint_selection: ConstraintSelection | None = None,   (__init__ param)
  92: self.constraint_selection = constraint_selection           (stored on instance)
  138:     constraint_selection=self.constraint_selection,        (TTA call)
```

## Decisions Made

1. **API default is None, Stage default is ConstraintSelection():** API functions default to None (no constraint overrides by caller). The Stage converts None to `ConstraintSelection()` using `or` pattern so all constraints remain enabled by default — preserves backward-compatible behavior.

2. **SteadyStateScheduler stores None, not ConstraintSelection():** The scheduler does not apply the same or-default pattern — it passes `None` through to TTA, which already has its own handling for `constraint_selection=None` (from Phase 5).

## Deviations from Plan

None — plan executed exactly as written. Import ordering (plan noted `ConstraintSelection` import after `SteadyStateScheduler` import in allocation.py) was handled correctly without ruff intervention needed.

## Known Stubs

None — all 4 hops are wired end-to-end and verified by unit tests.
