# Roadmap: Stream AIE — TETRA Constraint Optimization

## Milestones

- ✅ **v1.0 OR-Tools TETRA Backend** - Phases 1-4 (shipped 2026-05-07)
- ✅ **v1.1 Selective Constraints** - Phases 5-8 (shipped 2026-05-08)
- 🚧 **v1.2 Codebase Documentation** - Phases 9-14 (in progress)

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

<details>
<summary>✅ v1.1 Selective Constraints (Phases 5-8) - SHIPPED 2026-05-08</summary>

### Phase 5: ConstraintSelection Dataclass
**Goal**: A validated configuration object for constraint toggling exists and TTA correctly guards all four hardware constraint groups based on it
**Depends on**: Phase 4
**Requirements**: SEL-01, SEL-02, SEL-03, SEL-04, SEL-05
**Success Criteria** (what must be TRUE):
  1. `ConstraintSelection(memory_capacity=False)` instantiates with all other fields defaulting to True, and the object is immutable (frozen dataclass)
  2. Creating a `ConstraintSelection` with `memory_capacity=False, object_fifo_depth=True` emits a WARNING log about a nonsensical combination
  3. When `memory_capacity=False`, TransferAndTensorAllocator skips the memory capacity constraints block and the model solves without them
  4. When `dma_channels=False`, DMA is treated uniformly with other groups: the entire method is skipped (no variables, no constraints), and the objective excludes DMA terms
  5. Each disabled constraint group produces a distinct WARNING log entry naming the skipped group
**Plans:** 2/2 plans executed

Plans:
- [x] 05-01-PLAN.md — ConstraintSelection frozen dataclass + unit tests
- [x] 05-02-PLAN.md — TTA if-guards for all four constraint groups + guard verification tests

### Phase 6: Pipeline & API Surface
**Goal**: Users can pass a `ConstraintSelection` through every public entry point — programmatic API and CLI — and it reaches both allocators unchanged
**Depends on**: Phase 5
**Requirements**: PIPE-01, UI-01, UI-02
**Success Criteria** (what must be TRUE):
  1. `optimize_allocation_co()` and `optimize_mapping()` accept a `constraint_selection` keyword argument and pass it through to both allocators
  2. Running a main script with `--disable-constraints memory_capacity dma_channels` produces a `ConstraintSelection` with those two fields set to False and the others True
  3. Passing `constraint_selection` via the API and via the equivalent CLI flags produces identical solver behavior on the same input
**Plans:** 2/2 plans complete

Plans:
- [x] 06-01-PLAN.md — Thread constraint_selection through pipeline (api.py, Stage, Scheduler) + tests
- [x] 06-02-PLAN.md — Add --disable-constraints CLI flag to all 4 main scripts + CLI tests

### Phase 7: End-to-End Validation
**Goal**: The constraint toggle feature is verified correct per group and cross-backend parity is confirmed with selective constraints active
**Depends on**: Phase 6
**Requirements**: TEST-01, TEST-02
**Success Criteria** (what must be TRUE):
  1. For each of the four constraint groups, a tight test instance flips from feasible to infeasible (or vice versa) when the toggle changes, confirming the guard is structurally effective
  2. Gurobi and OR-Tools produce objective values within solver tolerance for every enabled constraint combination tested
  3. All existing tests continue to pass (no regression from new guards or threading)
**Plans:** 1/1 plans complete

Plans:
- [x] 07-01-PLAN.md — Infeasibility-flip tests (TEST-01) + cross-backend parity tests (TEST-02)

### Phase 8: Constraint Toggle Study Script
**Goal**: A standalone script that runs all constraint toggle combinations on a TETRA workload, compares objective values and solve times, and produces a clear visualization (table + plots) showing the performance impact of disabling each constraint group
**Depends on**: Phase 7
**Requirements**: STUDY-01, STUDY-02
**Success Criteria** (what must be TRUE):
  1. Running the script on a GEMM workload produces a comparison table showing objective value, solve time, and relative delta for each of the 16 constraint combinations
  2. The script generates matplotlib bar/heatmap plots saved to disk showing which constraints have the largest impact on optimality
  3. Each combination is clearly labeled with human-readable constraint group names (not field names)
**Plans:** 1/1 plans complete

Plans:
- [x] 08-01-PLAN.md — Study script: enumerate 16 combinations, run pipeline, print table, generate plots

</details>

### 🚧 v1.2 Codebase Documentation (In Progress)

**Milestone Goal:** Comprehensive documentation of stream_aie components and architecture, structured as Claude Code skills (`.claude/skills/`) for AI agent auto-discovery, while remaining readable for human developers.

**Cross-cutting quality requirements:** SKILL-01 (trigger descriptions) and SKILL-02 (self-contained skills) are established in Phase 10 and enforced in every skill file produced in Phases 11-14.

#### Phase 9: Dead Code Cleanup
**Goal**: The codebase contains only active, referenced code -- unused stage files and their dead imports are removed before documentation begins
**Depends on**: Nothing (independent cleanup)
**Requirements**: CLEAN-01
**Success Criteria** (what must be TRUE):
  1. The files for StreamCostModelEvaluationStage, SetFixedAllocationStage, and UserDefinedModelParserStage no longer exist in the source tree
  2. No remaining import statements or registry entries reference the removed stage classes
  3. All existing tests pass after the removal (no regressions from dead code cleanup)
**Plans:** 1/1 plans complete

Plans:
- [x] 09-01-PLAN.md — Safety audit, delete three dead stage files, lint + regression verification

#### Phase 10: CLAUDE.md & Skill Scaffolding
**Goal**: Developers and AI agents landing in the repo can immediately orient themselves via a top-level CLAUDE.md and know where to find deeper topic documentation in a well-structured skills directory
**Depends on**: Phase 9
**Requirements**: NAV-01, NAV-02, SKILL-01, SKILL-02
**Success Criteria** (what must be TRUE):
  1. A CLAUDE.md file exists at the repo root containing a codebase overview, directory structure map, key entry points (main_gemm.py, main_swiglu.py, optimize_allocation_co, optimize_mapping), and coding conventions
  2. CLAUDE.md contains a "Skills" section that lists each `.claude/skills/` topic with a one-line description, so readers know exactly where to go for deep dives
  3. The `.claude/skills/` directory exists with a consistent structure: each skill is a subdirectory containing a SKILL.md (trigger description for AI agent auto-discovery) plus the skill content file
  4. A skill template/pattern is documented (naming, SKILL.md format, self-containment rule) so that Phases 11-14 produce uniform, independently readable skill files
**Plans:** 2/2 plans complete

Plans:
- [x] 10-01-PLAN.md — Fix .gitignore for .claude/skills/ + create four SKILL.md stubs
- [x] 10-02-PLAN.md — Create CLAUDE.md navigation hub

#### Phase 11: Solver System Skills
**Goal**: Anyone working on solver backends or constraint configuration can read a single skill file and understand the full solver abstraction layer or the constraint selection system without consulting source code
**Depends on**: Phase 10
**Requirements**: SOLVER-01, SOLVER-02
**Success Criteria** (what must be TRUE):
  1. A solver-facade skill file documents SolverModel ABC methods, GurobiBackend vs ORToolsBackend differences (nonlinear dispatch, infeasibility handling, MPS export), the factory pattern, SolverBackend enum values, and guidance on when to use each backend
  2. A constraint-selection skill file documents the ConstraintSelection dataclass (4 fields, frozen, defaults), its relationship to NamespaceConstraints and AIE2Constraints, which constraint groups map to which hardware resources, and the nonsensical-combination warning logic
  3. Each skill file has a SKILL.md with accurate trigger phrases and is readable independently without requiring the other skill or CLAUDE.md
**Plans:** 1/1 plans complete

Plans:
- [x] 11-01-PLAN.md — Write solver-backends.md and constraint-selection.md skill files, update SKILL.md

#### Phase 12: Pipeline Skills
**Goal**: Anyone debugging or extending the TETRA pipeline can trace data flow from input parsing through cost estimation to allocation output, understanding each stage's responsibility and the execution model
**Depends on**: Phase 10
**Requirements**: STAGE-01, STAGE-02
**Success Criteria** (what must be TRUE):
  1. A pipeline-stages skill file documents each active stage (AcceleratorParser, ONNXModelParser, MappingParser, TilingGeneration, CoreCostEstimation, ConstraintOptimizationAllocation, MemoryAccessesEstimation, MappingGeneration) with its responsibility, inputs/outputs, and position in the flow
  2. A stage-execution skill file documents StageContext (what data it holds, how stages access it) and the MainStage/LeafStage execution model (how stages compose, how the scheduler invokes them)
  3. Each skill file has a SKILL.md with accurate trigger phrases and is readable independently
**Plans:** 1/1 plans complete

Plans:
- [x] 12-01-PLAN.md — Write pipeline-stages.md and stage-execution.md skill files, update SKILL.md

#### Phase 13: MILP & Constraint Skills
**Goal**: Anyone modifying or debugging TETRA constraints can understand the full MILP formulation and how hardware-specific constraints are dispatched through the NamespaceConstraints pattern
**Depends on**: Phase 10
**Requirements**: MILP-01, MILP-02
**Success Criteria** (what must be TRUE):
  1. A MILP-formulation skill file documents the TransferAndTensorAllocator model structure: decision variables (binary placement, transfer selection), constraint groups (memory, FIFO, buffer descriptors, DMA), objective function (latency minimization), and how ConstraintSelection guards interact with constraint dispatch
  2. A namespace-constraints skill file documents the NamespaceConstraints base class, AIE2Constraints implementation, how hardware-specific constraints are dispatched based on the target architecture, and the relationship between constraint namespaces and the solver facade
  3. Each skill file has a SKILL.md with accurate trigger phrases and is readable independently
**Plans:** 1/1 plans complete

Plans:
- [x] 13-01-PLAN.md -- Write milp-formulation.md and namespace-constraints.md skill files, update SKILL.md

#### Phase 14: API & Testing Skills
**Goal**: Anyone using stream_aie programmatically or writing tests can find complete reference documentation for the public API surface and testing conventions
**Depends on**: Phase 10
**Requirements**: API-01, API-02
**Success Criteria** (what must be TRUE):
  1. An API-reference skill file documents optimize_allocation_co() and optimize_mapping() signatures, all CLI flags (--backend, --disable-constraints), common usage patterns, and return types (SolveStats fields)
  2. A testing skill file documents the test organization (unit vs integration), backend patching patterns, study scripts (constraint_toggle_study.py, cross-backend verification), and how to add new tests for constraint groups or backends
  3. Each skill file has a SKILL.md with accurate trigger phrases and is readable independently
**Plans:** 1/1 plans complete

Plans:
- [x] 14-01-PLAN.md — Write api-reference.md and testing-patterns.md skill files, update SKILL.md

## Progress

**Execution Order:**
Phase 9 is independent. Phase 10 depends on Phase 9. Phases 11-14 depend on Phase 10 and can run in any order: 9 -> 10 -> 11 -> 12 -> 13 -> 14

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1. Solver Facade | v1.0 | 4/4 | Complete | 2026-05-07 |
| 2. ORToolsBackend (linear) | v1.0 | 1/2 | Complete | 2026-05-07 |
| 3. Linearized Division | v1.0 | TBD | Complete | 2026-05-07 |
| 4. Verification & Config | v1.0 | 2/2 | Complete | 2026-05-07 |
| 5. ConstraintSelection Dataclass | v1.1 | 2/2 | Complete | 2026-05-08 |
| 6. Pipeline & API Surface | v1.1 | 2/2 | Complete | 2026-05-08 |
| 7. End-to-End Validation | v1.1 | 1/1 | Complete | 2026-05-08 |
| 8. Constraint Toggle Study Script | v1.1 | 1/1 | Complete | 2026-05-08 |
| 9. Dead Code Cleanup | v1.2 | 1/1 | Complete   | 2026-05-09 |
| 10. CLAUDE.md & Skill Scaffolding | v1.2 | 2/2 | Complete    | 2026-05-09 |
| 11. Solver System Skills | v1.2 | 1/1 | Complete    | 2026-05-09 |
| 12. Pipeline Skills | v1.2 | 1/1 | Complete    | 2026-05-09 |
| 13. MILP & Constraint Skills | v1.2 | 1/1 | Complete    | 2026-05-10 |
| 14. API & Testing Skills | v1.2 | 1/1 | Complete   | 2026-05-10 |
