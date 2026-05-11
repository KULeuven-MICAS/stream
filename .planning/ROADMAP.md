# Roadmap: Stream AIE — TETRA Constraint Optimization

## Milestones

- ✅ **v1.0 OR-Tools TETRA Backend** - Phases 1-4 (shipped 2026-05-07)
- ✅ **v1.1 Selective Constraints** - Phases 5-8 (shipped 2026-05-08)
- ✅ **v1.2 Codebase Documentation** - Phases 9-14 (shipped 2026-05-10)
- ✅ **v1.3 MCP Server & Intermediate Representations** - Phases 15-18 (shipped 2026-05-11)
- [ ] **v1.4 Robust Non-AIE Support** - Phases 19-21

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

<details>
<summary>✅ v1.2 Codebase Documentation (Phases 9-14) - SHIPPED 2026-05-10</summary>

### Phase 9: Dead Code Cleanup
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

</details>

<details>
<summary>✅ v1.3 MCP Server & Intermediate Representations (Phases 15-18) - SHIPPED 2026-05-11</summary>

### Phase 15: Pre-flight Cleanup
**Goal**: The MILP solve path emits no stdout, SteadyStateScheduler and Mapping expose serializable get_ir() methods, and the MCP server can configure logging independently
**Depends on**: Phase 14
**Requirements**: CLEAN-02, CLEAN-03, CLEAN-04
**Success Criteria** (what must be TRUE):
  1. Running a full TETRA solve with OR-Tools produces zero lines on stdout — all diagnostic output goes through the logger
  2. `SteadyStateScheduler.get_ir()` returns a dict with allocation results that round-trips cleanly through `json.dumps()` (no non-serializable objects)
  3. `Mapping.get_ir()` returns a dict representation of the mapping that round-trips cleanly through `json.dumps()`
  4. An MCP server process can call a helper to configure logging without triggering any module-level `basicConfig()` side effects from api.py
**Plans:** 2/2 plans complete

Plans:
- [x] 15-01-PLAN.md — Replace TTA print() calls with logger + move basicConfig to configure_logging helper
- [x] 15-02-PLAN.md — Add get_ir() methods to Mapping and SteadyStateScheduler + round-trip tests

### Phase 16: IR Models
**Goal**: Structured, versioned Pydantic IR classes exist for workloads, allocations, and hardware, with per-persona views that surface the right information for each user type
**Depends on**: Phase 15
**Requirements**: IR-01, IR-02
**Success Criteria** (what must be TRUE):
  1. `WorkloadIR`, `AllocationIR`, and `AcceleratorIR` Pydantic models exist, each with a `schema_version` field, and `Model.model_json_schema()` returns a valid JSON Schema dict
  2. An algorithmic engineer calling the IR can access latency and objective-function breakdown without digging through raw solver output
  3. A hardware engineer calling the IR can see per-core resource utilization derived from the allocation
  4. A compiler engineer calling the IR can read node-to-core mapping and transfer routing in a structured, navigable format
**Plans:** 2/2 plans complete

Plans:
- [x] 16-01-PLAN.md — WorkloadIR + AcceleratorIR Pydantic models with from_internal, views, and tests
- [x] 16-02-PLAN.md — AllocationIR Pydantic model with three persona views + IR skill documentation

**UI hint**: no

### Phase 17: MCP Server Skeleton
**Goal**: A FastMCP server boots cleanly as a Claude Code subprocess, registers tools, manages state across calls via lifespan, and handles long-running solves without timeout via an async job pattern
**Depends on**: Phase 15
**Requirements**: MCP-01, MCP-02, MCP-03
**Success Criteria** (what must be TRUE):
  1. Running the MCP server as a subprocess and connecting Claude Code to it takes under 1.5 seconds from process start to tools becoming discoverable
  2. Submitting an optimization job returns a job ID immediately — the caller does not block waiting for the MILP solve to complete
  3. Polling for job results returns the correct status (pending / complete / failed) and, once complete, the full result without re-running the solve
  4. Submitting the same hardware + workload + backend + constraints combination twice reuses the cached result — the second call returns instantly with the same job ID
**Plans:** 2/2 plans complete

Plans:
- [x] 17-01-PLAN.md — Create stream/mcp/ package with jobs.py (ServerState + experiment ID) + add fastmcp dep
- [x] 17-02-PLAN.md — FastMCP server with lifespan, 6 tool stubs, async job pattern

**UI hint**: no

### Phase 18: MCP Tools
**Goal**: AI agents can drive the full TETRA design space exploration workflow — launching solves, inspecting workloads and hardware, and retrieving allocation results — entirely through MCP tool calls returning structured IR JSON
**Depends on**: Phase 16, Phase 17
**Requirements**: TOOL-01, TOOL-02, TOOL-03
**Success Criteria** (what must be TRUE):
  1. An agent calling `run_optimization` with a workload path, hardware YAML, backend choice, and constraint selection receives a job ID and can subsequently retrieve the result via a polling tool
  2. An agent calling `get_workload_ir` or `get_accelerator_ir` receives a JSON object that validates against the Pydantic IR schema for that type
  3. An agent calling `get_allocation_ir` receives tensor placements and per-layer latencies as structured JSON; calling `get_solve_stats` receives solve time, objective value, and solver status
**Plans:** 2/2 plans complete

Plans:
- [x] 18-01-PLAN.md — Wire async job dispatch in run_optimization + implement get_allocation_ir and get_solve_stats
- [x] 18-02-PLAN.md — Implement get_workload_ir and get_accelerator_ir with dual-parameter pattern

**UI hint**: no

</details>

## Phase Details

### Phase 19: GA Removal
**Goal**: Users can import and use stream.api with no DEAP dependency and no genetic algorithm code in the source tree
**Depends on**: Phase 18
**Requirements**: CLEAN-01, CLEAN-02, CLEAN-03
**Success Criteria** (what must be TRUE):
  1. `import stream.api` succeeds in an environment without DEAP installed
  2. No files from the GA path (`main_aie_ga.py`, `main_stream_ga.py`, `genetic_algorithm/` package, GA stage, GA mapping YAML) remain in the repository
  3. `deap` is absent from `pyproject.toml` dependencies
  4. All 176 currently-passing tests continue to pass after removal
**Plans:** 1/1 plans complete

Plans:
- [x] 19-01-PLAN.md — Delete GA files, remove GA code from api.py/pyproject.toml, clean dead variables from scripts, update CLAUDE.md

### Phase 20: Mapping Format Fixes
**Goal**: The mapping layer correctly handles the nested-list format required by the validator, and the TPU mapping YAML validates against the current schema
**Depends on**: Phase 19
**Requirements**: FMT-01, FMT-02, FMT-03, FMT-04, FMT-05, FMT-06
**Success Criteria** (what must be TRUE):
  1. `make_2_conv_mapping` generates `core_allocation` as `list[list[int]]` and `inter_core_tiling` as `list[list[dict]]`, matching the MappingValidator schema
  2. `Mapping.with_updated_workload` completes without NameError when `intra_core_tiling` is empty
  3. `get_unique_dims_inter_core_tiling` returns an empty tuple rather than raising IndexError when a node has no inter-core tiling entries
  4. `_convert_intra_core_tiling_entry` parses dotted ONNX node names correctly (last dot separator only)
  5. `tpu_like_quad_core.yaml` passes `MappingValidator` validation without errors
  6. `test_core_cost_lut_caching` passes (all 177 tests green)
**Plans:** 1/1 plans complete

Plans:
- [x] 20-01-PLAN.md — Fix mapping format bugs (nested lists, NameError, IndexError, rsplit), delete dead files, verify 177 tests pass

### Phase 21: TPU End-to-End Test
**Goal**: Users can run the two-conv TPU CO pipeline as a standard pytest test and verify it produces a valid scheduler result
**Depends on**: Phase 20
**Requirements**: TPU-01, TPU-02
**Success Criteria** (what must be TRUE):
  1. `test_co.py` exists under `tests/` and is discovered and executed by `pytest tests/`
  2. The two-conv TPU CO pipeline runs without error from parsing through MILP allocation to memory estimation
  3. The scheduler result contains non-empty allocation data for both conv layers
**Plans**: TBD

## Progress

**Execution Order:**
Phases 1-18 complete. v1.4 order: 19 (blocker) -> 20 -> 21

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
| 9. Dead Code Cleanup | v1.2 | 1/1 | Complete | 2026-05-09 |
| 10. CLAUDE.md & Skill Scaffolding | v1.2 | 2/2 | Complete | 2026-05-09 |
| 11. Solver System Skills | v1.2 | 1/1 | Complete | 2026-05-09 |
| 12. Pipeline Skills | v1.2 | 1/1 | Complete | 2026-05-09 |
| 13. MILP & Constraint Skills | v1.2 | 1/1 | Complete | 2026-05-10 |
| 14. API & Testing Skills | v1.2 | 1/1 | Complete | 2026-05-10 |
| 15. Pre-flight Cleanup | v1.3 | 2/2 | Complete | 2026-05-10 |
| 16. IR Models | v1.3 | 2/2 | Complete | 2026-05-10 |
| 17. MCP Server Skeleton | v1.3 | 2/2 | Complete | 2026-05-10 |
| 18. MCP Tools | v1.3 | 2/2 | Complete | 2026-05-10 |
| 19. GA Removal | v1.4 | 1/1 | Complete    | 2026-05-11 |
| 20. Mapping Format Fixes | v1.4 | 1/1 | Complete   | 2026-05-11 |
| 21. TPU End-to-End Test | v1.4 | 0/TBD | Not started | - |
