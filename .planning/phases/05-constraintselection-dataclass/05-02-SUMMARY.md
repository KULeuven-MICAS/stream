---
phase: 05-constraintselection-dataclass
plan: "02"
subsystem: constraint-optimization
tags: [constraint-selection, if-guards, tda, objective, logging, tdd]
dependency_graph:
  requires: [05-01 (ConstraintSelection dataclass)]
  provides: [TTA with constraint_selection parameter, if-guards for all 4 constraint groups, conditional DMA objective]
  affects:
    - stream/opt/allocation/constraint_optimization/transfer_and_tensor_allocation.py
    - tests/unit/test_constraint_selection.py
tech_stack:
  added: []
  patterns: [call-site guards (not method-body guards), MagicMock spec binding for real method dispatch testing]
key_files:
  created: []
  modified:
    - stream/opt/allocation/constraint_optimization/transfer_and_tensor_allocation.py
    - tests/unit/test_constraint_selection.py
decisions:
  - "Guards placed at call site in _create_constraints() and _overlap_and_objective(), not inside the constraint methods (per D-01)"
  - "constraint_selection parameter added as keyword-only arg after backend= with None defaulting to ConstraintSelection()"
  - "_make_tta_stub uses bind_objective=False by default; only test_dma_objective_no_dma_terms binds the real objective method"
metrics:
  duration_s: 420
  completed_date: "2026-05-07"
  tasks_completed: 2
  files_modified: 2
---

# Phase 05 Plan 02: TTA Constraint Selection Guards Summary

**One-liner:** TTA accepts a ConstraintSelection parameter and skips disabled constraint groups via call-site if-guards in _create_constraints(), _overlap_and_objective(), and _set_total_latency_and_objective(), with a WARNING log per skipped group.

## What Was Built

Modified `TransferAndTensorAllocator` in `transfer_and_tensor_allocation.py` to:

1. Import `ConstraintSelection` from `stream.opt.solver` and add `import logging` with a module-level `_logger`.
2. Accept `constraint_selection: ConstraintSelection | None = None` as a keyword-only parameter. Stored as `self.constraint_selection = constraint_selection or ConstraintSelection()`.
3. Guard `_create_constraints()`: three call sites for `_memory_capacity_constraints()`, `_object_fifo_depth_constraints()`, `_buffer_descriptor_constraints()` are each wrapped in `if self.constraint_selection.<field>:` with `else: _logger.warning(...)`.
4. Guard `_overlap_and_objective()`: `_add_dma_usage_constraints()` call is wrapped with `if self.constraint_selection.dma_channels:`.
5. Guard `_set_total_latency_and_objective()`: objective expression is conditional — DMA terms (`max_core_dma_in._raw + max_core_dma_out._raw`) are only included when `dma_channels=True`.

Added 9 guard verification tests to `tests/unit/test_constraint_selection.py` covering all four guard points, the DMA-free objective, WARNING emission for all disabled groups, and the all-enabled baseline.

## Tasks

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Add constraint_selection parameter and if-guards to TTA | b17f074 | stream/opt/allocation/constraint_optimization/transfer_and_tensor_allocation.py |
| 2 | Add guard verification tests | 749f50d | tests/unit/test_constraint_selection.py |

## Decisions Made

- Guards placed at call site (not inside the constraint method bodies), per D-01: each constraint method retains its original body unchanged.
- `constraint_selection` parameter is keyword-only (after `*`) to avoid breaking positional callers.
- `_make_tta_stub` helper uses `bind_objective=False` by default; the real `_set_total_latency_and_objective` is only bound for the objective test, preventing interference in DMA guard tests.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Test stub binding of _set_total_latency_and_objective caused AttributeError**
- **Found during:** Task 2 (first test run)
- **Issue:** The plan's `_make_tta_stub` example bound `_set_total_latency_and_objective` as a real method unconditionally, causing AttributeError when `_overlap_and_objective()` tried to call it (the real method needed `self.model`, which is a MagicMock without that attribute on spec).
- **Fix:** Added `bind_objective=False` parameter to `_make_tta_stub`. Only `test_dma_objective_no_dma_terms` passes `bind_objective=True` and sets up the required model mocks.
- **Files modified:** tests/unit/test_constraint_selection.py
- **Commit:** 749f50d

## Known Stubs

None — all constraint guards are real conditional dispatches with no placeholder values.

## Self-Check: PASSED

- FOUND: stream/opt/allocation/constraint_optimization/transfer_and_tensor_allocation.py
- FOUND: tests/unit/test_constraint_selection.py
- FOUND commit: b17f074 (Task 1 — TTA guards)
- FOUND commit: 749f50d (Task 2 — guard tests)
- All 16 tests in test_constraint_selection.py pass
- All 64 unit tests pass (no regressions)
