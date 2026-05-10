---
phase: 13-milp-constraint-skills
plan: 01
subsystem: documentation
tags: [milp, constraint-optimization, skill-files, tetra, aie2]

requires:
  - phase: 12-pipeline-skills
    provides: established skill-file style, format, and self-containment conventions (D-05 through D-08)
  - phase: 11-solver-system-skills
    provides: constraint-selection.md (ConstraintSelection dataclass, toggle-to-hardware mapping) referenced by milp-formulation.md

provides:
  - ".claude/skills/constraints/milp-formulation.md: conceptual guide to both ComputeAllocator and TransferAndTensorAllocator MILP structure"
  - ".claude/skills/constraints/namespace-constraints.md: NamespaceConstraints strategy pattern and AIE2Constraints hardware-specific constraint dispatch"
  - "Updated SKILL.md removing Phase 13 stub note"

affects: [any-phase-modifying-constraint-optimization, future-hardware-namespace-additions]

tech-stack:
  added: []
  patterns:
    - "Strategy pattern for hardware-specific MILP constraint dispatch (NamespaceConstraints base + AIE2Constraints subclass)"
    - "Three-level constraint filtering: ConstraintSelection toggle -> TransferAndTensorContext dispatch -> applies_to() core filtering"
    - "Two-stage MILP pipeline: ComputeAllocator (node-to-core) then TransferAndTensorAllocator (tensor placement + transfer routing)"

key-files:
  created:
    - ".claude/skills/constraints/milp-formulation.md"
    - ".claude/skills/constraints/namespace-constraints.md"
  modified:
    - ".claude/skills/constraints/SKILL.md"

key-decisions:
  - "milp-formulation.md covers both ComputeAllocator and TransferAndTensorAllocator as two MILP stages per D-04"
  - "Variable family table uses prefix notation (x_, y_, z_, L_, fires_, reuse_factor_) per D-02"
  - "ConstraintSelection-guarded method table placed inline in milp-formulation.md per D-03 with cross-reference to constraint-selection.md"
  - "DMA constraint guard noted as placed in _overlap_and_objective() not _create_constraints() with explanation of why"
  - "Three-level filtering interaction documented in namespace-constraints.md with dispatch flow table"
  - "No code examples in either file per D-06; indented code-like strings (e.g., formula lines) use plain text"

requirements-completed: [MILP-01, MILP-02]

duration: ~15min
completed: 2026-05-10
---

# Phase 13 Plan 01: MILP Constraint Skills Summary

**Self-contained conceptual guides to the TETRA two-stage MILP pipeline (ComputeAllocator + TransferAndTensorAllocator) and the NamespaceConstraints hardware dispatch strategy, covering all decision variables, constraint groups, ConstraintSelection guards, and AIE2-specific resource limits**

## Performance

- **Duration:** ~15 min
- **Started:** 2026-05-10T08:15:00Z
- **Completed:** 2026-05-10T08:30:00Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments

- Created `milp-formulation.md` (188 lines): covers ComputeAllocator (build pipeline, variable table with 14 variable families, 8 constraint groups, objective formula) and TransferAndTensorAllocator (build pipeline, 6-prefix variable family table, 10 always-active constraint groups, 4 ConstraintSelection-guarded constraint groups with guard table, overlap/objective computation, 3 linearization helpers)
- Created `namespace-constraints.md` (147 lines): covers NamespaceConstraints base class (NAMESPACE, applies_to, 3 no-op methods), AIE2Constraints (FIFO depth, buffer descriptors, DMA channels with per-tile-type table), TransferAndTensorContext dispatch pattern, build_transfer_context factory with extensibility guidance, ConstraintContext for ComputeAllocator, and three-level filtering interaction with dispatch flow table
- Updated `SKILL.md` to remove the Phase 13 placeholder stub line, leaving the Contents table and See also intact

## Task Commits

1. **Task 1: Write milp-formulation.md skill file** - `05dd538` (feat)
2. **Task 2: Write namespace-constraints.md and update SKILL.md** - `e89d8a3` (feat)

**Plan metadata:** (final docs commit — see below)

## Files Created/Modified

- `.claude/skills/constraints/milp-formulation.md` - Conceptual guide to both MILP stages: ComputeAllocator variables/constraints/objective, TTA build pipeline, variable families table, constraint groups with ConstraintSelection guard mapping, overlap/objective, linearization helpers
- `.claude/skills/constraints/namespace-constraints.md` - NamespaceConstraints strategy pattern, AIE2Constraints implementation, TransferAndTensorContext dispatch, build_transfer_context factory, three-level filtering
- `.claude/skills/constraints/SKILL.md` - Removed Phase 13 stub note

## Decisions Made

- milp-formulation.md covers both ComputeAllocator and TransferAndTensorAllocator (per D-04) in separate sections to give developers the full two-stage picture
- The DMA constraint guard is noted as living in `_overlap_and_objective()` (not `_create_constraints()`) with an explanation of why, since this is a common source of confusion
- namespace-constraints.md includes a three-level filtering interaction table showing which dispatch level applies to each enabled constraint group
- Both files follow D-05 through D-08: conceptual guide style, no code examples, self-contained without requiring cross-file reading

## Deviations from Plan

None - plan executed exactly as written.

The worktree did not have the `.claude/skills/` directory structure from the `arne/codebase-documentation` branch (worktree was based on main), so the prerequisite files (SKILL.md, constraint-selection.md) were checked out from the codebase-documentation branch before writing the new skill files. This is a worktree initialization concern, not a deviation from the plan's content.

## Issues Encountered

None.

## Next Phase Readiness

- Phase 13 plan 01 complete; both MILP-01 and MILP-02 requirements satisfied
- Developers and AI agents can now navigate the TETRA MILP formulation and hardware-specific constraint dispatch without opening the 2800+ line source files
- No blockers for next phases

---

## Self-Check: PASSED

Files verified to exist:
- `.claude/skills/constraints/milp-formulation.md` — FOUND (188 lines, >= 150)
- `.claude/skills/constraints/namespace-constraints.md` — FOUND (147 lines, >= 80)
- `.claude/skills/constraints/SKILL.md` — FOUND (Phase 13 stub removed)

Commits verified:
- 05dd538 — FOUND (feat(13-01): write milp-formulation.md skill file)
- e89d8a3 — FOUND (feat(13-01): write namespace-constraints.md and update SKILL.md)

---
*Phase: 13-milp-constraint-skills*
*Completed: 2026-05-10*
