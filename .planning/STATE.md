---
gsd_state_version: 1.0
milestone: v1.1
milestone_name: Selective Constraints
status: verifying
stopped_at: Completed 07-01-PLAN.md
last_updated: "2026-05-08T12:39:44.423Z"
last_activity: 2026-05-08
progress:
  total_phases: 4
  completed_phases: 3
  total_plans: 5
  completed_plans: 6
  percent: 0
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-05-07)

**Core value:** Explore TETRA design space — solver backends, constraint toggling, optimality impact
**Current focus:** Phase 07 — end-to-end-validation

## Current Position

Phase: 07 (end-to-end-validation) — EXECUTING
Plan: 1 of 1
Status: Phase complete — ready for verification
Last activity: 2026-05-08

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
| Phase 06 P01 | 154 | 1 tasks | 4 files |
| Phase 06 P02 | 3 | 1 tasks | 5 files |
| Phase 07-end-to-end-validation P01 | 13 | 2 tasks | 1 files |

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
- [Phase 06]: constraint_selection defaults to None at API level, defaults to ConstraintSelection() inside Stage (all-True behavior preserved via or-default pattern)
- [Phase 06]: SteadyStateScheduler stores constraint_selection as None to preserve TTA's own None-handling from Phase 5
- [Phase 06]: ConstraintSelection constructed only when _disabled is non-empty; None otherwise preserves API all-True default
- [Phase 06]: main_swiglu_dse.py computes ConstraintSelection once at top of sweep_tile_size_combinations from args.disable_constraints
- [Phase 07]: Used Core.__init__ wrapper (not class attribute patch) for max_object_fifo_depth because __init__ sets instance attribute directly, shadowing class-level patches
- [Phase 07]: parity test 'memory_off' disables both memory_capacity and object_fifo_depth per SEL-05 nonsensical-combination rule

### Pending Todos

None.

### Blockers/Concerns

- DMA split: toggle must not skip variable creation, only constraint dispatch — verify this in Phase 5
- Whole-method guards: ensure auxiliary variables inside constraint methods are also guarded, not just add_constr calls

## Session Continuity

Last session: 2026-05-08T12:39:44.421Z
Stopped at: Completed 07-01-PLAN.md
Resume file: None
