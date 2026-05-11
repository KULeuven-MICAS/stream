---
phase: 21-tpu-end-to-end-test
plan: 01
subsystem: testing
tags: [pytest, tpu, co-pipeline, steady-state-scheduler, transfer-graph, memory-tiles]

requires:
  - phase: 20-mapping-format-fixes
    provides: correct mapping format (nested-list core_allocation, fixed FMT-01..04)

provides:
  - End-to-end pytest test for two-conv TPU CO pipeline in tests/test_co.py
  - Bug fix: offchip core type handled in determine_transfer_type
  - Bug fix: single-transfer fallback for input constants on memory-less hardware
  - Bug fix: single-transfer fallback for output constants on memory-less hardware
  - Bug fix: COMPUTE_TO_MEM fallback to offchip when no memory tiles

affects:
  - future-phases-using-non-aie-hardware
  - tests/

tech-stack:
  added: []
  patterns:
    - "TPU-like hardware with no memory tiles now supported in CO pipeline"
    - "Memory-tile existence check before two-node transfer pattern"

key-files:
  created:
    - tests/test_co.py
  modified:
    - stream/cost_model/steady_state_scheduler.py
  deleted:
    - test_co.py

key-decisions:
  - "Four bugs fixed in steady_state_scheduler.py beyond the planned offchip type fix (all Rule 1 auto-fixes)"
  - "Memory-less hardware (no on-chip memory tiles) now falls back to single direct transfer instead of two-node pattern"
  - "COMPUTE_TO_MEM with no memory tiles uses offchip core as fallback destination"

patterns-established:
  - "Non-AIE hardware without memory tiles: _get_accelerator_memory_cores() returns empty set → fallback paths triggered"

requirements-completed:
  - TPU-01
  - TPU-02

duration: 38min
completed: 2026-05-11
---

# Phase 21 Plan 01: TPU End-to-End Test Summary

**End-to-end pytest test for two-conv TPU CO pipeline (tests/test_co.py), fixing four scheduler bugs that blocked execution on memory-less TPU hardware**

## Performance

- **Duration:** ~38 min
- **Started:** 2026-05-11T14:20:00Z
- **Completed:** 2026-05-11T14:58:03Z
- **Tasks:** 2
- **Files modified:** 3 (2 modified, 1 created, 1 deleted)

## Accomplishments

- Fixed `determine_transfer_type` to handle `"offchip"` core type (4 `in` checks)
- Fixed `add_two_transfer_nodes_for_constant_input_transfer` to use single transfer when no memory tiles
- Fixed `add_two_transfer_nodes_for_constant_output_transfer` to use single transfer when no memory tiles
- Fixed `determine_possible_memory_allocations` COMPUTE_TO_MEM to fall back to offchip core
- Created `tests/test_co.py` with `test_co_tpu_two_conv` — asserts positive latency, positive iterations, 2 computation nodes, non-empty resource_allocation per node
- Deleted `test_co.py` from repo root (was running CO pipeline at import time, breaking pytest collection)
- Full suite: 195 tests pass (178 unit + 17 integration)

## Task Commits

1. **Task 1: Fix offchip transfer type bug in SteadyStateScheduler** - `9c40538` (fix)
2. **Task 2: Create tests/test_co.py, fix scheduler for memory-less hardware, delete root test_co.py** - `a50b93b` (feat)

**Plan metadata:** (docs commit, TBD)

## Files Created/Modified

- `stream/cost_model/steady_state_scheduler.py` - 4 bug fixes for TPU/memory-less hardware support
- `tests/test_co.py` - New pytest E2E test for two-conv TPU CO pipeline
- `test_co.py` - Deleted from repo root

## Decisions Made

- Four bugs required auto-fixing beyond the planned single offchip type fix (all Rule 1 — broken behavior)
- The two-node transfer pattern (offchip → mem tile → compute) hardcoded memory tile existence; TPU hardware has none
- Fallback: single direct transfer for input/output constants when no memory tiles
- Fallback: offchip core as COMPUTE_TO_MEM destination when no memory tiles

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] add_two_transfer_nodes_for_constant_input_transfer fails when no memory tiles**
- **Found during:** Task 2 (running test_co_tpu_two_conv)
- **Issue:** The two-node constant input transfer pattern forces `dst_type="memory"` for the first transfer, requiring memory tiles. TPU hardware has no memory tiles, so `_get_accelerator_memory_cores()` returns empty set, making `memory_allocation = ()` for the intermediate transfer node, which caused `ValueError: Tensor allocation options cannot be empty` in TTA.
- **Fix:** Added guard at start of method: if `_get_accelerator_memory_cores()` is empty, fall back to a single direct transfer using the actual `determine_transfer_type(src, dsts)` result (produces `MEM_TO_COMPUTE` for offchip→compute).
- **Files modified:** `stream/cost_model/steady_state_scheduler.py`
- **Verification:** Test passes, debug trace shows Transfer(input/weights_1/weights_2) now get compute core allocations
- **Committed in:** a50b93b (Task 2 commit)

**2. [Rule 1 - Bug] add_two_transfer_nodes_for_constant_output_transfer fails when no memory tiles**
- **Found during:** Task 2 (same test run, same root cause)
- **Issue:** Same problem for output constants: COMPUTE_TO_MEM intermediate transfer has empty `memory_allocation` when no memory tiles.
- **Fix:** Same fallback pattern as for input constants: single direct transfer when `_get_accelerator_memory_cores()` is empty.
- **Files modified:** `stream/cost_model/steady_state_scheduler.py`
- **Verification:** Part of same test pass
- **Committed in:** a50b93b (Task 2 commit)

**3. [Rule 1 - Bug] determine_possible_memory_allocations returns empty for COMPUTE_TO_MEM on memory-less hardware**
- **Found during:** Task 2 (trace after fix 1+2 still showed Transfer(output_1) with empty alloc)
- **Issue:** `determine_possible_memory_allocations` for COMPUTE_TO_MEM calls `_get_possible_memory_core_allocations` which returns `()` when no memory tiles, leaving memory_allocation empty for the single output transfer node created by fix 2.
- **Fix:** After calling `_get_possible_memory_core_allocations`, if result is empty, fall back to the offchip core as the destination (fetched via `self.accelerator.get_core(self.accelerator.offchip_core_id)`).
- **Files modified:** `stream/cost_model/steady_state_scheduler.py`
- **Verification:** Test passes end-to-end, scheduler.latency_total > 0
- **Committed in:** a50b93b (Task 2 commit)

---

**Total deviations:** 3 auto-fixed (all Rule 1 — broken behavior for valid hardware configuration)
**Impact on plan:** All three fixes are necessary for the TPU CO pipeline to run on memory-less hardware. The planned `determine_transfer_type` fix was insufficient alone; the deeper assumption of memory tile existence permeated the transfer graph building and memory allocation logic.

## Issues Encountered

- The RESEARCH.md identified only the `determine_transfer_type` bug. Live execution revealed three additional interconnected bugs in the scheduler's transfer graph building for memory-less hardware. All were auto-fixed under Rule 1.

## Known Stubs

None — tests/test_co.py asserts structural properties that are fully wired and verified by the actual CO solver.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- TPU CO pipeline runs end-to-end (195 tests green)
- ResNet18 TPU CO flow (if planned) can now proceed with the same memory-less hardware fixes in place

## Self-Check: PASSED

- `tests/test_co.py` — FOUND
- `stream/cost_model/steady_state_scheduler.py` — FOUND
- `.planning/phases/21-tpu-end-to-end-test/21-01-SUMMARY.md` — FOUND
- commit `9c40538` — FOUND
- commit `a50b93b` — FOUND

---
*Phase: 21-tpu-end-to-end-test*
*Completed: 2026-05-11*
