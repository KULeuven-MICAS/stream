# Requirements: Stream AIE — OR-Tools TETRA Backend

**Defined:** 2026-05-07
**Core Value:** Identical optimization results regardless of solver backend

## v1.0 Requirements

Requirements for initial release. Each maps to roadmap phases.

### Solver Abstraction

- [x] **ABS-01**: SolverModel ABC defines interface for variable creation, constraint addition, objective setting, and solving
- [x] **ABS-02**: GurobiBackend implements SolverModel by delegating to gurobipy with zero behavioral change
- [x] **ABS-03**: SolverVar wrapper exposes `.X` property for solution extraction across backends
- [x] **ABS-04**: LinExpr wrapper supports `+=`, `__add__`, and `defaultdict` accumulation patterns
- [x] **ABS-05**: Backend factory instantiates solver by enum name (GUROBI, SCIP, ORTOOLS_GUROBI) with clear error on failure

### OR-Tools Implementation

- [x] **ORT-01**: ORToolsBackend implements all linear MILP operations (binary/continuous vars, linear constraints, quicksum, objective)
- [ ] **ORT-02**: Linearized division constraint replaces `addGenConstrNL` via piecewise enumeration over discrete denominator values
- [x] **ORT-03**: Infeasibility handler exports model as MPS file when OR-Tools reports infeasible (replacing Gurobi IIS)

### Verification & Configuration

- [ ] **VER-01**: Cross-backend verification runner compares objective values within solver tolerance on same input
- [x] **VER-02**: Backend selection configurable via `optimize_allocation_co` API and main scripts
- [x] **VER-03**: SolveStats dataclass captures solve time, objective, status, with None for OR-Tools-unavailable fields

## v2.0 Requirements

Deferred to future release. Tracked but not in current roadmap.

### Migration

- **MIG-01**: Full removal of gurobipy as required dependency
- **MIG-02**: MathOpt backend (OR-Tools 10.x successor to MPSolver)

### Diagnostics

- **DIAG-01**: MIP progress callback equivalent for OR-Tools (pending upstream support)
- **DIAG-02**: OR-Tools-native IIS computation (pending upstream OR-Tools issue #3019)

## Out of Scope

| Feature | Reason |
|---------|--------|
| CP-SAT solver support | TETRA is MILP; CP-SAT requires all-integer formulation |
| Changes to TETRA math formulation | Only API layer changes; exception: linearization of division |
| Full gurobipy removal | Coexistence first, migration later |
| MIP progress callbacks for OR-Tools | No upstream API exists (GitHub #1902) |
| timeslot_allocation.py changes | No gurobipy coupling found in this file |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| ABS-01 | Phase 1 | Complete |
| ABS-02 | Phase 1 | Complete |
| ABS-03 | Phase 1 | Complete |
| ABS-04 | Phase 1 | Complete |
| ABS-05 | Phase 1 | Complete |
| ORT-01 | Phase 2 | Complete |
| ORT-02 | Phase 3 | Pending |
| ORT-03 | Phase 2 | Complete |
| VER-01 | Phase 4 | Pending |
| VER-02 | Phase 4 | Complete |
| VER-03 | Phase 4 | Complete |

**Coverage:**
- v1.0 requirements: 11 total
- Mapped to phases: 11
- Unmapped: 0

---
*Requirements defined: 2026-05-07*
*Last updated: 2026-05-07 after roadmap creation*
