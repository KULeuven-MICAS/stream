---
phase: 07-end-to-end-validation
plan: "01"
subsystem: integration-tests
tags: [testing, constraint-selection, infeasibility-flip, cross-backend-parity]
dependency_graph:
  requires:
    - stream.api.optimize_allocation_co (constraint_selection kwarg, Phase 6)
    - stream.opt.solver.ConstraintSelection (frozen dataclass, Phase 5)
    - stream.hardware.architecture.core.Core (get_memory_capacity, max_object_fifo_depth)
    - stream.opt.allocation.constraint_optimization.transfer_and_tensor_allocation (_create_constraints guards)
    - stream.opt.allocation.constraint_optimization.context (build_transfer_context)
  provides:
    - TEST-01: Infeasibility-flip proof for all 4 constraint groups
    - TEST-02: Cross-backend parity for 7 constraint combinations
  affects:
    - tests/integration/test_constraint_toggles.py
tech_stack:
  added: []
  patterns:
    - patch.object(Core, "__init__") wrapper to override instance attribute set in __init__
    - patch.object(Core, "get_memory_capacity") to return tight memory limit
    - build_transfer_context wrapper patched in TTA namespace for DMA channel tightening
    - Dynamic Gurobi reference for parity tests (not hardcoded baseline) per Pitfall 3
key_files:
  created:
    - tests/integration/test_constraint_toggles.py
  modified: []
decisions:
  - Used __init__ wrapper (not class attribute patch) for Core.max_object_fifo_depth because __init__ sets self.max_object_fifo_depth directly from constructor args, which shadows class-level patches
  - Patched build_transfer_context in TTA's own namespace (_BUILD_TRANSFER_CONTEXT target) since TTA imports it directly
  - All flip tests disable all non-tested constraints to isolate proof (only one group enabled at a time)
  - buffer_descriptor flip reuses same max_object_fifo_depth=1 mechanism as FIFO flip (BD constraints share same RHS per research pitfall 6)
  - object_fifo_depth flip disables memory_capacity to avoid nonsensical SEL-05 combination warning
  - parity test "memory_off" disables both memory_capacity and object_fifo_depth per SEL-05 rule
metrics:
  duration_minutes: 13
  completed_date: "2026-05-08"
  tasks_completed: 2
  files_changed: 1
---

# Phase 7 Plan 01: Integration Tests for Constraint Toggle Feature Summary

End-to-end integration tests proving each constraint guard is structurally effective (infeasibility-flip on toggle) and that Gurobi/OR-Tools agree within 1% for 7 selective-constraint combinations.

## What Was Built

`tests/integration/test_constraint_toggles.py` with 11 test cases:

### TEST-01: Infeasibility-Flip Tests (4 tests)

| Test | Tight Limit | Mechanism | Enabled | Disabled |
|------|-------------|-----------|---------|---------|
| test_memory_capacity_flip | 1 bit memory | patch.object(Core, "get_memory_capacity", return_value=1) | RuntimeError | success |
| test_object_fifo_depth_flip | max_fifo=1 | Core.__init__ wrapper sets max_object_fifo_depth=1 | RuntimeError | success |
| test_buffer_descriptor_flip | max_fifo=1 | Core.__init__ wrapper sets max_object_fifo_depth=1 | RuntimeError | success |
| test_dma_channels_flip | DMA channels=1 | build_transfer_context wrapper (TTA namespace) | RuntimeError | success |

### TEST-02: Cross-Backend Parity Tests (7 parametrized cases)

| ID | ConstraintSelection | Gurobi vs OR-Tools |
|----|--------------------|--------------------|
| memory_off | memory_capacity=False, object_fifo_depth=False | within 1% |
| fifo_off | object_fifo_depth=False | within 1% |
| bd_off | buffer_descriptors=False | within 1% |
| dma_off | dma_channels=False | within 1% |
| memory_and_dma_off | memory_capacity=False, object_fifo_depth=False, dma_channels=False | within 1% |
| fifo_and_bd_off | object_fifo_depth=False, buffer_descriptors=False | within 1% |
| all_off | all four fields False | within 1% |

## Verification Results

- All 4 flip tests: PASSED (95.96s)
- All 7 parity tests: PASSED (193.61s)
- Unit tests (76): PASSED — no regressions
- Existing integration tests (test_cross_backend.py, 6 tests): PASSED — no interference

## Commits

| Task | Commit | Description |
|------|--------|-------------|
| Task 1 + Task 2 | 235c99e | feat(07-01): add infeasibility-flip tests for all four constraint groups (TEST-01) |

Note: Both tasks written in one file creation (full test module), committed together as Task 1. Task 2 parity tests verified separately with all 7 passing before completion.

## Deviations from Plan

### Implementation Note: Single Write for Both Tasks

Tasks 1 and 2 were implemented in a single file write since both tests belong to the same module (`test_constraint_toggles.py`). Writing incrementally would have required partial file creation followed by appending. The tests were written together, then each task's tests were verified independently before committing:
- Flip tests run and verified FIRST
- Parity tests run and verified SECOND
- Single commit captures both completed and verified tasks

This does not violate the task commit protocol since the acceptance criteria were fully verified for each task in sequence.

No other deviations from the plan.

## Known Stubs

None — all tests are fully wired with real TETRA workloads and hardware configuration.

## Self-Check: PASSED

- tests/integration/test_constraint_toggles.py: FOUND
- Commit 235c99e: FOUND
- 4 flip test functions: CONFIRMED (grep -c "def test_.*_flip" = 4)
- 7 parity parametrize cases: CONFIRMED (grep -c "pytest.param" = 7)
- All 11 tests passed on run
