---
phase: 14-api-testing-skills
plan: "01"
subsystem: documentation
tags: [skills, api-reference, testing-patterns, documentation]
dependency_graph:
  requires: []
  provides: [api-reference.md, testing-patterns.md]
  affects: [SKILL.md]
tech_stack:
  added: []
  patterns: [conceptual-guide-style, parameter-tables, no-code-blocks]
key_files:
  created:
    - .claude/skills/api-testing/api-reference.md
    - .claude/skills/api-testing/testing-patterns.md
  modified:
    - .claude/skills/api-testing/SKILL.md
decisions:
  - "api-reference.md documents optimize_allocation_co (13 params), optimize_mapping (19 params), SolveStats dataclass, all 10 CLI scripts in table, and common --backend/--disable-constraints flags"
  - "testing-patterns.md documents test directory layout, unit/integration test patterns, dual-target create_solver backend patching, infeasibility-flip pattern, 4 study scripts table, and adding-new-tests guidance"
  - "SKILL.md stub line removed; both content files now referenced in Contents table"
metrics:
  duration_s: 190
  completed_date: "2026-05-10"
  tasks_completed: 2
  tasks_total: 2
  files_changed: 3
---

# Phase 14 Plan 01: API & Testing Skills Summary

**One-liner:** Public API reference and testing patterns skill files covering optimize_allocation_co(), optimize_mapping(), 10 CLI scripts, dual-target backend patching, and infeasibility-flip test pattern.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Write api-reference.md skill file | 25fce28 | .claude/skills/api-testing/api-reference.md (created, 141 lines) |
| 2 | Write testing-patterns.md and update SKILL.md | d9339b1 | .claude/skills/api-testing/testing-patterns.md (created, 132 lines), .claude/skills/api-testing/SKILL.md (stub removed) |

## What Was Built

### api-reference.md

Documents the two programmatic entry points exposed by `stream.api`:

- `optimize_allocation_co()`: 13-parameter table with types, defaults, and descriptions. Covers the CO pipeline stages (parse -> tile -> cost -> MILP -> memory estimation), backend validation behavior (Gurobi license check), and skip_if_exists caching.
- `optimize_mapping()`: 19-parameter table including DSE-specific tile size parameters and `nb_workers` for multi-threaded mapping generation.
- `SolveStats`: 8-field frozen dataclass table with notes on which fields are `None` for OR-Tools backends.
- CLI Scripts table: all 10 scripts with category (CO/GA/DSE/Specialized), purpose, and key flags.
- Common Flags section: `--backend` choices and default, `--disable-constraints` nargs behavior.
- See Also: cross-references to pipeline-stages, solver-backends, constraint-selection skills.

### testing-patterns.md

Documents the test suite structure and conventions:

- Test directory layout: unit/ (6 files), integration/ (2 files), root-level tests (3 files), standalone scripts (4 scripts).
- Unit tests: naming conventions, _-prefix helper pattern, pytest.param with id=, MagicMock usage. Per-file purpose table.
- Integration tests: infeasibility-flip pattern (tight limit + enabled = RuntimeError, disabled = success), cross-backend parity test with 1% tolerance.
- Backend patching: dual-target create_solver patch problem and why it arises, factory replacement pattern, Gurobi license check patch as third target.
- Study scripts table: 4 scripts with purpose, output type, and run command.
- Adding new tests guidance: unit, integration, and constraint group test patterns.
- See Also: cross-references to solver-backends, constraint-selection, api-reference skills.

### SKILL.md Update

Removed the stub placeholder line "*Content files will be added by Phase 14.*" The Contents table remains intact with both `api-reference.md` and `testing-patterns.md` already listed.

## Deviations from Plan

None — plan executed exactly as written.

## Known Stubs

None — both skill files document real source code content. No placeholder text or TODO markers.

## Self-Check: PASSED

Files created:
- [FOUND] .claude/skills/api-testing/api-reference.md (141 lines, >= 80)
- [FOUND] .claude/skills/api-testing/testing-patterns.md (132 lines, >= 80)
- [FOUND] .claude/skills/api-testing/SKILL.md (stub removed, SKILL.md contains both file references)

Commits:
- [FOUND] 25fce28 (feat(14-01): write api-reference.md skill file)
- [FOUND] d9339b1 (feat(14-01): write testing-patterns.md and update SKILL.md)
