# Roadmap: Stream AIE — OR-Tools TETRA Backend

## Overview

The milestone introduces OR-Tools as an alternative MILP solver backend for TETRA by building a solver abstraction layer, implementing ORToolsBackend on top of it, handling the one non-linear constraint through linearization, and finally wiring in a verification runner and configuration API that lets both backends run side-by-side. Every phase delivers a coherent, independently verifiable capability and the overall sequence preserves full backward compatibility with gurobipy throughout.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 1: Solver Facade** - SolverModel ABC + GurobiBackend refactor with zero behavioral change
- [ ] **Phase 2: ORToolsBackend (linear)** - OR-Tools MathOpt implementation of all standard MILP operations
- [ ] **Phase 3: Linearized Division** - Piecewise enumeration replaces the non-linear division constraint
- [ ] **Phase 4: Verification & Config** - Cross-backend runner, backend selection API, and SolveStats

## Phase Details

### Phase 1: Solver Facade
**Goal**: TETRA can run through a clean abstraction layer with gurobipy as the sole backend, with zero change in optimization results
**Depends on**: Nothing (first phase)
**Requirements**: ABS-01, ABS-02, ABS-03, ABS-04, ABS-05
**Success Criteria** (what must be TRUE):
  1. SolverModel ABC exists and defines all required interface methods (variable creation, constraint addition, objective setting, solve)
  2. GurobiBackend implements SolverModel and TETRA produces identical solutions to the pre-refactor baseline
  3. Solution values are accessible via the `.X` property on SolverVar across any backend
  4. LinExpr wrapper supports `+=`, `__add__`, and defaultdict accumulation patterns used throughout TETRA
  5. Calling the backend factory with GUROBI, SCIP, or ORTOOLS_GUROBI enum names returns the correct backend or raises a clear error
**Plans:** 4/4 plans executed

Plans:
- [x] 01-01-PLAN.md — Create solver facade (ABC, GurobiBackend, wrappers, enums, factory) + test scaffold
- [x] 01-02-PLAN.md — Refactor ComputeAllocator (tupledict, max_, min_, multidict replacement)
- [x] 01-03-PLAN.md — Refactor TTA + context.py + linearize division constraint
- [x] 01-04-PLAN.md — Refactor api.py + import leakage test + integration verification

### Phase 2: ORToolsBackend (linear)
**Goal**: OR-Tools MathOpt backend can execute the linear subset of the TETRA model and produce a valid feasible solution equivalent to Gurobi
**Depends on**: Phase 1
**Requirements**: ORT-01, ORT-03
**Success Criteria** (what must be TRUE):
  1. ORToolsBackend implements all SolverModel interface methods using OR-Tools MathOpt API
  2. Binary and continuous variable creation, linear constraints, quicksum equivalents, and objective minimization all work correctly through ORToolsBackend
  3. When OR-Tools reports infeasibility, the model is exported as an MPS file instead of attempting a Gurobi IIS
**Plans:** 1/2 plans executed

Plans:
- [x] 02-01-PLAN.md — Implement ORToolsBackend (MathOpt API) + unit tests
- [ ] 02-02-PLAN.md — Cross-backend integration tests on real TETRA instances (gemm + swiglu)

### Phase 3: Linearized Division
**Goal**: The division constraint in TETRA is encoded without any non-linear solver extension, making it solvable by OR-Tools
**Depends on**: Phase 2
**Requirements**: ORT-02
**Success Criteria** (what must be TRUE):
  1. The division constraint previously using `addGenConstrNL` is replaced by a piecewise-linear enumeration over discrete denominator values
  2. OR-Tools produces a feasible solution on a TETRA instance that requires the division constraint
  3. The reformulated constraint yields the same optimal objective value as the gurobipy path (within solver tolerance)
**Plans**: TBD

### Phase 4: Verification & Config
**Goal**: Users can select a backend via configuration and optionally run both backends on the same input to confirm result equivalence
**Depends on**: Phase 3
**Requirements**: VER-01, VER-02, VER-03
**Success Criteria** (what must be TRUE):
  1. The `optimize_allocation_co` API and main scripts accept a backend selection argument (GUROBI, ORTOOLS, or ORTOOLS_GUROBI)
  2. Cross-backend verification mode runs both backends on the same input and reports whether objective values agree within solver tolerance
  3. SolveStats dataclass captures solve time, objective value, and status for every backend run, with None for OR-Tools-unavailable fields such as MIP gap
**Plans:** 2 plans

Plans:
- [x] 04-01-PLAN.md — SolveStats dataclass + backend selection wiring through pipeline
- [ ] 04-02-PLAN.md — CLI --backend arguments + cross-backend verification runner

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Solver Facade | 4/4 | Complete |  |
| 2. ORToolsBackend (linear) | 1/2 | In Progress|  |
| 3. Linearized Division | 0/TBD | Not started | - |
| 4. Verification & Config | 0/2 | Not started | - |
