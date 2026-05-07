---
phase: 05-constraintselection-dataclass
plan: "01"
subsystem: solver
tags: [dataclass, constraint-selection, frozen, tdd]
dependency_graph:
  requires: []
  provides: [ConstraintSelection frozen dataclass, stream.opt.solver public export]
  affects: [stream/opt/solver/solver.py, stream/opt/solver/__init__.py]
tech_stack:
  added: []
  patterns: [frozen dataclass with __post_init__ validation, TDD red-green cycle]
key_files:
  created:
    - tests/unit/test_constraint_selection.py
  modified:
    - stream/opt/solver/solver.py
    - stream/opt/solver/__init__.py
decisions:
  - "ConstraintSelection placed after SolveStats in solver.py, sharing the same _logger at module level"
  - "ConstraintSelection re-exported alphabetically first in __init__.py __all__ list"
metrics:
  duration_s: 61
  completed_date: "2026-05-07"
  tasks_completed: 1
  files_modified: 3
---

# Phase 05 Plan 01: ConstraintSelection Dataclass Summary

**One-liner:** Frozen ConstraintSelection dataclass with 4 bool toggles defaulting True and a __post_init__ WARNING for memory_capacity=False + object_fifo_depth=True.

## What Was Built

Added `ConstraintSelection` as a `@dataclass(frozen=True)` in `stream/opt/solver/solver.py` (after the existing `SolveStats` frozen dataclass). The class has four boolean fields — `memory_capacity`, `object_fifo_depth`, `buffer_descriptors`, `dma_channels` — all defaulting to `True`. Its `__post_init__` method emits a WARNING via the module-level `_logger` when the logically inconsistent combination `memory_capacity=False` + `object_fifo_depth=True` is used. The class is re-exported from `stream.opt.solver` (`__init__.py`) in both the import statement and `__all__` list.

7 unit tests cover all required behaviors; all 55 unit tests pass with no regressions.

## Tasks

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 (RED) | Add failing tests for ConstraintSelection | 8833ce0 | tests/unit/test_constraint_selection.py |
| 1 (GREEN) | Implement ConstraintSelection dataclass | ba5acfa | stream/opt/solver/solver.py, stream/opt/solver/__init__.py |

## Decisions Made

- `ConstraintSelection` placed after `SolveStats` in `solver.py`, before the Helper section — follows existing frozen dataclass pattern and shares the module-level `_logger`.
- `ConstraintSelection` added as the first entry in `__init__.py` `__all__` (alphabetical order).

## Deviations from Plan

None — plan executed exactly as written.

## Known Stubs

None — all fields are real defaults, no placeholder data flows to UI or downstream consumers.

## Self-Check: PASSED

- FOUND: stream/opt/solver/solver.py
- FOUND: stream/opt/solver/__init__.py
- FOUND: tests/unit/test_constraint_selection.py
- FOUND commit: 8833ce0 (RED — failing tests)
- FOUND commit: ba5acfa (GREEN — implementation)
