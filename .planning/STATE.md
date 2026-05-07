---
gsd_state_version: 1.0
milestone: v1.1
milestone_name: Selective Constraints
status: verifying
stopped_at: Completed 05-02-PLAN.md
last_updated: "2026-05-07T21:43:12.610Z"
last_activity: 2026-05-07
progress:
  total_phases: 3
  completed_phases: 1
  total_plans: 2
  completed_plans: 3
  percent: 0
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-05-07)

**Core value:** Explore TETRA design space — solver backends, constraint toggling, optimality impact
**Current focus:** Phase 05 — constraintselection-dataclass

## Current Position

Phase: 05 (constraintselection-dataclass) — EXECUTING
Plan: 2 of 2
Status: Phase complete — ready for verification
Last activity: 2026-05-07

Progress: [░░░░░░░░░░] 0%

## Performance Metrics

**Velocity (from v1.0):**

- Total plans completed: 8
- Phases completed: 4

**By Phase (v1.0):**

| Phase | Plans | Duration | Files |
|-------|-------|----------|-------|
| 01-solver-facade P01 | 2 tasks | 3min | 5 files |
| 01-solver-facade P02 | 2 tasks | 3min | 1 file |
| 01-solver-facade P03 | 2 tasks | 9min | 3 files |
| 01-solver-facade P04 | 2 tasks | 20min | 4 files |
| 02-ortoolsbackend P01 | 2 tasks | 5min | 6 files |
| 04-verification-config P01 | 2 tasks | 12min | 7 files |
| Phase 05-constraintselection-dataclass P01 | 61 | 1 tasks | 3 files |
| Phase 05-constraintselection-dataclass P02 | 420 | 2 tasks | 2 files |

## Accumulated Context

### Decisions

Carried from v1.0:

- Solver abstraction in stream/opt/solver/solver.py (SolverModel ABC, GurobiBackend, ORToolsBackend)
- SolverBackend enum: GUROBI, ORTOOLS_GSCIP, ORTOOLS_HIGHS, ORTOOLS_GUROBI
- Backend propagated as str through pipeline via StageContext
- Default backend is ORTOOLS_GSCIP (license-free)
- NamespaceConstraints pattern for per-core-type hardware constraints (AIE2Constraints)

v1.1 decisions:

- ConstraintSelection is a frozen dataclass (4 bool fields, all default True)
- DMA toggle skips context.add_dma_usage_constraints() only — accounting variables preserved for objective
- Structural constraints (link contention, reuse, slot latency) are never toggleable
- [Phase 05-constraintselection-dataclass]: ConstraintSelection placed after SolveStats in solver.py, sharing module-level _logger; re-exported first alphabetically in __init__.py __all__
- [Phase 05-constraintselection-dataclass]: Guards placed at call site in _create_constraints() not inside constraint methods (D-01); constraint_selection keyword-only after backend=
- [Phase 05-constraintselection-dataclass]: _make_tta_stub uses bind_objective=False default; only objective test binds real _set_total_latency_and_objective

### Pending Todos

None.

### Blockers/Concerns

- DMA split: toggle must not skip variable creation, only constraint dispatch — verify this in Phase 5
- Whole-method guards: ensure auxiliary variables inside constraint methods are also guarded, not just add_constr calls

## Session Continuity

Last session: 2026-05-07T21:43:12.607Z
Stopped at: Completed 05-02-PLAN.md
Resume file: None
