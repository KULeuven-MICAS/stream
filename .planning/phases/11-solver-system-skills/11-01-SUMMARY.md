---
phase: 11-solver-system-skills
plan: 01
subsystem: documentation
tags: [skills, solver, constraint-selection, documentation]
dependency_graph:
  requires: []
  provides: [solver-backends.md, constraint-selection.md, optimization-SKILL.md-updated]
  affects: [.claude/skills/optimization/]
tech_stack:
  added: []
  patterns: [conceptual-guide, no-code-examples, self-contained-skill-files]
key_files:
  created:
    - .claude/skills/optimization/solver-backends.md
    - .claude/skills/optimization/constraint-selection.md
  modified:
    - .claude/skills/optimization/SKILL.md
decisions:
  - "D-01 applied: Conceptual guide style — method signatures only for SolverModel ABC public interface; internal helpers (_GurobiVar, _ORToolsVar, etc.) described by role only"
  - "D-02 applied: No code examples in either file — pure prose with tables"
  - "D-03 applied: GurobiBackend vs ORToolsBackend presented as comparison table with 8 capability dimensions"
  - "D-04 applied: ConstraintSelection fields documented as hardware-mapping table with effect-when-disabled column"
  - "D-05 applied: Content covers how/why only; CLAUDE.md what/where not duplicated"
  - "D-06 applied: Both files self-contained and readable without following any link"
  - "D-07 applied: See also lines at end of each file; files complete without them"
metrics:
  duration: 260
  completed_date: "2026-05-09"
  tasks_completed: 2
  files_created: 3
---

# Phase 11 Plan 01: Solver System Skills Summary

## One-Liner

Conceptual skill files for the solver abstraction layer (SolverModel ABC, GurobiBackend vs ORToolsBackend comparison, factory pattern) and constraint selection system (ConstraintSelection 4-field dataclass, NamespaceConstraints strategy pattern, AIE2Constraints DMA table).

## What Was Built

### Task 1: solver-backends.md (173 lines)

A complete conceptual guide to the solver abstraction layer in `stream/opt/solver/solver.py`. Sections cover:

- SolverBackend enum (4 values with backend, underlying solver, license)
- SolverModel ABC (method table with purpose and key notes for all 12 abstract methods, plus non-abstract methods)
- SolverVar and LinExpr described by role (`.X` for solution extraction, `._raw` for expression building, accumulation pattern)
- Backend comparison table (8 dimensions: nonlinear, infeasibility, MPS export, parameters, license, name uniqueness, quicksum, callbacks)
- Factory pattern (`create_solver()` mapping, ValueError for unknown backends)
- SolveStats frozen dataclass (all 8 fields)
- SolverParams enum (5 members with Gurobi and OR-Tools mappings)
- SolverVarType enum (3 members)
- When-to-use guidance (default ORTOOLS_GSCIP, use GUROBI for nonlinear/IIS)

### Task 2: constraint-selection.md (108 lines) + SKILL.md updated

A complete conceptual guide to the constraint selection system. Sections cover:

- ConstraintSelection frozen dataclass (4 bool fields, all default True, immutable)
- Constraint-to-hardware mapping table (field, hardware resource, effect when disabled, warning)
- Nonsensical combination warning (memory_capacity=False + object_fifo_depth=True)
- Pipeline threading (api.py -> Stage None-to-default -> allocator checks)
- NamespaceConstraints strategy pattern (NAMESPACE, applies_to, 3 overridable no-op methods)
- AIE2Constraints (NAMESPACE="aie2", FIFO depth, buffer descriptors, DMA channel table)
- TransferAndTensorContext (namespace_constraints tuple, 3 dispatch methods, build_transfer_context factory)
- Two-layer interaction (ConstraintSelection coarse toggle + NamespaceConstraints fine-grained dispatch)

SKILL.md updated: removed `*Content files will be added by Phase 11.*` line while preserving Contents table and See also reference.

## Commits

| Task | Hash | Description |
|------|------|-------------|
| Task 1 | d886395 | feat(11-01): add solver-backends.md skill file |
| Task 2 | 7ebf027 | feat(11-01): add constraint-selection.md and update SKILL.md |

## Deviations from Plan

### Force-add to bypass .gitignore

The worktree's `.gitignore` still contained the old blanket `.claude/` exclusion (the worktree was checked out at `dcdab87`, before the Phase 10 gitignore fix). The main `arne/codebase-documentation` branch already has the updated `.gitignore`. Used `git add -f` to force-track the skill files — consistent with the project decision from Phase 10 that `.claude/skills/` should be committable.

All other plan instructions executed exactly as written.

## Known Stubs

None. Both skill files are complete and fully wired — no placeholder content, no TODO items, no empty sections.

## Self-Check: PASSED

Files exist:
- FOUND: .claude/skills/optimization/solver-backends.md (173 lines)
- FOUND: .claude/skills/optimization/constraint-selection.md (108 lines)
- FOUND: .claude/skills/optimization/SKILL.md (Phase 11 stub removed)

Commits exist:
- FOUND: d886395 (feat(11-01): add solver-backends.md skill file)
- FOUND: 7ebf027 (feat(11-01): add constraint-selection.md and update SKILL.md)

Acceptance criteria verified: all grep checks pass, no code blocks, line counts exceed minimums.
