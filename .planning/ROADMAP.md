# Roadmap: Stream AIE — TETRA Constraint Optimization

## Milestones

- ✅ **v1.0 OR-Tools TETRA Backend** - Phases 1-4 (shipped 2026-05-07)
- 🚧 **v1.1 Selective Constraints** - Phases 5-7 (in progress)

## Phases

<details>
<summary>✅ v1.0 OR-Tools TETRA Backend (Phases 1-4) - SHIPPED 2026-05-07</summary>

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

</details>

### 🚧 v1.1 Selective Constraints (In Progress)

**Milestone Goal:** Allow users to selectively enable/disable hardware resource constraint groups in TETRA to study their impact on optimality.

#### Phase 5: ConstraintSelection Dataclass
**Goal**: A validated configuration object for constraint toggling exists and TTA correctly guards all four hardware constraint groups based on it
**Depends on**: Phase 4
**Requirements**: SEL-01, SEL-02, SEL-03, SEL-04, SEL-05
**Success Criteria** (what must be TRUE):
  1. `ConstraintSelection(memory_capacity=False)` instantiates with all other fields defaulting to True, and the object is immutable (frozen dataclass)
  2. Creating a `ConstraintSelection` with `memory_capacity=False, object_fifo_depth=True` emits a WARNING log about a nonsensical combination
  3. When `memory_capacity=False`, TransferAndTensorAllocator skips the memory capacity constraints block and the model solves without them
  4. When `dma_channels=False`, DMA is treated uniformly with other groups: the entire method is skipped (no variables, no constraints), and the objective excludes DMA terms
  5. Each disabled constraint group produces a distinct WARNING log entry naming the skipped group
**Plans:** 1/2 plans executed

Plans:
- [x] 05-01-PLAN.md — ConstraintSelection frozen dataclass + unit tests
- [ ] 05-02-PLAN.md — TTA if-guards for all four constraint groups + guard verification tests

#### Phase 6: Pipeline & API Surface
**Goal**: Users can pass a `ConstraintSelection` through every public entry point — programmatic API and CLI — and it reaches both allocators unchanged
**Depends on**: Phase 5
**Requirements**: PIPE-01, UI-01, UI-02
**Success Criteria** (what must be TRUE):
  1. `optimize_allocation_co()` and `optimize_mapping()` accept a `constraint_selection` keyword argument and pass it through to both allocators
  2. Running a main script with `--disable-constraints memory_capacity dma_channels` produces a `ConstraintSelection` with those two fields set to False and the others True
  3. Passing `constraint_selection` via the API and via the equivalent CLI flags produces identical solver behavior on the same input
**Plans**: TBD
**UI hint**: yes

#### Phase 7: End-to-End Validation
**Goal**: The constraint toggle feature is verified correct per group and cross-backend parity is confirmed with selective constraints active
**Depends on**: Phase 6
**Requirements**: TEST-01, TEST-02
**Success Criteria** (what must be TRUE):
  1. For each of the four constraint groups, a tight test instance flips from feasible to infeasible (or vice versa) when the toggle changes, confirming the guard is structurally effective
  2. Gurobi and OR-Tools produce objective values within solver tolerance for every enabled constraint combination tested
  3. All existing tests continue to pass (no regression from new guards or threading)
**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 5 → 6 → 7

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1. Solver Facade | v1.0 | 4/4 | Complete | 2026-05-07 |
| 2. ORToolsBackend (linear) | v1.0 | 1/2 | Complete | 2026-05-07 |
| 3. Linearized Division | v1.0 | TBD | Complete | 2026-05-07 |
| 4. Verification & Config | v1.0 | 2/2 | Complete | 2026-05-07 |
| 5. ConstraintSelection Dataclass | v1.1 | 1/2 | In Progress|  |
| 6. Pipeline & API Surface | v1.1 | 0/TBD | Not started | - |
| 7. End-to-End Validation | v1.1 | 0/TBD | Not started | - |
