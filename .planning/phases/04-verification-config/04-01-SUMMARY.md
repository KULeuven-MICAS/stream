---
phase: 04-verification-config
plan: 01
subsystem: solver
tags: [ortools, gurobi, milp, solver-abstraction, backend-selection, pipeline-wiring]

# Dependency graph
requires:
  - phase: 01-solver-facade
    provides: SolverModel ABC, GurobiBackend, create_solver factory
  - phase: 02-ortoolsbackend-linear
    provides: ORToolsBackend with full MILP support

provides:
  - SolveStats frozen dataclass with 8 structured fields
  - solve_stats() abstract method on SolverModel ABC implemented by both backends
  - backend parameter on optimize_allocation_co() and optimize_mapping() (default "ortools")
  - Conditional Gurobi license check (only when backend == GUROBI)
  - Backend propagation: api.py -> StageContext -> ConstraintOptAllocationStage -> SteadyStateScheduler -> TransferAndTensorAllocator
  - Backend wired into ComputeAllocator via _create_basic_sets and get_optimal_allocations()
  - Hardcoded SolverBackend.GUROBI removed from both allocators

affects: [05-integration-tests, any caller of optimize_allocation_co or optimize_mapping]

# Tech tracking
tech-stack:
  added: [dataclasses.dataclass (frozen=True) for SolveStats]
  patterns:
    - "Backend string propagated through pipeline as str, converted to SolverBackend enum at point of use"
    - "Conditional license check pattern: only check Gurobi license when backend == GUROBI"
    - "Default backend is OR-Tools everywhere (intentional breaking change per D-03)"

key-files:
  created: []
  modified:
    - stream/opt/solver/solver.py
    - stream/opt/solver/__init__.py
    - stream/api.py
    - stream/stages/allocation/constraint_optimization_allocation.py
    - stream/cost_model/steady_state_scheduler.py
    - stream/opt/allocation/constraint_optimization/transfer_and_tensor_allocation.py
    - stream/opt/allocation/constraint_optimization/allocation.py

key-decisions:
  - "SolveStats is a frozen dataclass — immutable value object, safe to pass around and inspect"
  - "ORToolsBackend.solve_stats() returns None for mip_gap, node_count, iteration_count (MathOpt does not expose these directly)"
  - "Backend propagated as str through pipeline, converted to SolverBackend enum only at create_solver() call site"
  - "Default backend set to 'ORTOOLS' everywhere per D-03; OR-Tools is the new default"
  - "Gurobi license check skipped when backend is OR-Tools per D-04 — avoids Gurobi dependency on non-Gurobi runs"

patterns-established:
  - "Pipeline threading: pass string value through ctx, convert to enum at leaf"
  - "Conditional external-service check: gate on backend enum before checking credentials"

requirements-completed: [VER-02, VER-03]

# Metrics
duration: 12min
completed: 2026-05-07
---

# Phase 4 Plan 1: Backend Pipeline Wiring Summary

**SolveStats frozen dataclass added to solver module; backend selection threaded from api.py through StageContext, SteadyStateScheduler, and into both allocators, replacing hardcoded SolverBackend.GUROBI**

## Performance

- **Duration:** 12 min
- **Started:** 2026-05-07T11:44:00Z
- **Completed:** 2026-05-07T11:56:00Z
- **Tasks:** 2
- **Files modified:** 7

## Accomplishments

- Added `SolveStats` frozen dataclass with 8 fields to `stream/opt/solver/solver.py` and exported from `__init__.py`
- Implemented `solve_stats()` on `GurobiBackend` (all fields) and `ORToolsBackend` (mip_gap/node_count/iteration_count are None)
- Wired `backend: str = "ortools"` parameter through entire pipeline from both API entry points to both MILP allocators
- Removed hardcoded `SolverBackend.GUROBI` from `TransferAndTensorAllocator` and `ComputeAllocator`
- Made Gurobi license check conditional on backend selection (per D-04)

## Task Commits

Each task was committed atomically:

1. **Task 1: Add SolveStats dataclass and solve_stats() to solver module** - `7a06493` (feat)
2. **Task 2: Wire backend selection through pipeline** - `0808b31` (feat)

**Plan metadata:** _(docs commit to follow)_

## Files Created/Modified

- `stream/opt/solver/solver.py` - Added SolveStats dataclass, solve_stats() ABC method, GurobiBackend.solve_stats(), ORToolsBackend.solve_stats()
- `stream/opt/solver/__init__.py` - Added SolveStats to imports and __all__
- `stream/api.py` - Added SolverBackend import, backend param to optimize_allocation_co/optimize_mapping, conditional license check, backend in StageContext
- `stream/stages/allocation/constraint_optimization_allocation.py` - Reads backend from ctx, passes to SteadyStateScheduler
- `stream/cost_model/steady_state_scheduler.py` - Accepts backend kwarg, stores self.backend, passes to TransferAndTensorAllocator
- `stream/opt/allocation/constraint_optimization/transfer_and_tensor_allocation.py` - Accepts backend kwarg, uses SolverBackend[self.backend_str] in create_solver
- `stream/opt/allocation/constraint_optimization/allocation.py` - Accepts backend kwarg in ComputeAllocator and get_optimal_allocations(), uses SolverBackend[self.backend_str]

## Decisions Made

- Backend propagated as a string value (e.g., "ORTOOLS", "GUROBI") through the context and constructor chain, converted to `SolverBackend` enum only at the `create_solver()` call site. This avoids importing the enum in intermediate modules.
- `SolveStats` uses `frozen=True` to make it an immutable value object safe for caching and comparison.
- `ORToolsBackend.solve_stats()` returns `None` for `mip_gap`, `node_count`, and `iteration_count` since MathOpt does not expose these directly; callers should handle `None`.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Backend selection pipeline is fully wired; both allocators now use `SolverBackend[self.backend_str]`
- Phase 5 integration tests can now pass `backend="ortools"` to any API entry point to exercise the OR-Tools path end-to-end
- Known blocker (pre-existing, from STATE.md): division constraint in TTA uses `addGenConstrNL` — requires linearization before OR-Tools can solve instances that exercise it

---
*Phase: 04-verification-config*
*Completed: 2026-05-07*
