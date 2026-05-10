---
phase: 16-ir-models
plan: 02
subsystem: ir
tags: [pydantic, json-schema, ir-models, allocation, scheduler, mcp, skills]

# Dependency graph
requires:
  - phase: 16-01
    provides: WorkloadIR, AcceleratorIR, stream/ir/ package, from_internal() pattern, test file structure
  - phase: 15-pre-flight-cleanup
    provides: SteadyStateScheduler.get_ir() returning JSON-safe dict with latency, backend, mapping embedded
provides:
  - stream/ir/allocation.py with AllocationIR Pydantic model, LatencyInfo, ConstraintSelectionIR, NodeAllocationIR, FusedGroupIR sub-models
  - Three per-persona view methods: algorithmic_view(), hardware_view(), compiler_view()
  - from_internal() with pre-solve guard (ValueError on latency_total == -1)
  - Updated stream/ir/__init__.py re-exporting AllocationIR and all sub-models
  - .claude/skills/ir/ skill group with SKILL.md trigger and ir-models.md conceptual guide
  - 9 additional unit tests (total 25 in test_ir_models.py)
affects: [17-mcp-server]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - AllocationIR.from_internal() with pre-solve sentinel guard (latency_total == -1 -> ValueError)
    - View methods project from already-validated Pydantic fields (no extra get_ir() calls)
    - NodeAllocationIR captures all three slot-indexed allocation fields for both hardware and compiler views
    - TDD: failing tests committed (RED), implementation committed (GREEN), ruff line-length fix (REFACTOR)

key-files:
  created:
    - stream/ir/allocation.py
    - .claude/skills/ir/SKILL.md
    - .claude/skills/ir/ir-models.md
  modified:
    - stream/ir/__init__.py
    - tests/unit/test_ir_models.py

key-decisions:
  - "AllocationIR.from_internal() raises ValueError on pre-solve scheduler (latency_total == -1 sentinel) to prevent confusing MCP consumers with invalid -1 latency values"
  - "All three view methods (algorithmic, hardware, compiler) share the same mapping_nodes dict[str, NodeAllocationIR] — NodeAllocationIR carries all three slot-indexed fields so both hardware and compiler views can expose the relevant subset"
  - "IR skill documentation uses conceptual-guide style with no code examples (Phase 11 decision), covering all three IR classes and persona views in under 80 lines"

patterns-established:
  - "Pre-solve guard: from_internal() checks scheduler.latency_total == -1 and raises ValueError before calling get_ir()"
  - "Shared sub-model: NodeAllocationIR serves both hardware_view (resource_allocation + memory_allocation) and compiler_view (inter_core_tiling + resource_allocation)"

requirements-completed: [IR-01, IR-02]

# Metrics
duration: 15min
completed: 2026-05-10
---

# Phase 16 Plan 02: ir-models Summary

**AllocationIR Pydantic model with LatencyInfo, ConstraintSelectionIR, NodeAllocationIR, FusedGroupIR sub-models, pre-solve guard, three persona views, and IR skill documentation covering all three IR classes**

## Performance

- **Duration:** ~15 min
- **Started:** 2026-05-10T20:22:29Z
- **Completed:** 2026-05-10
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments

- Created `stream/ir/allocation.py` with AllocationIR and sub-models: LatencyInfo, ConstraintSelectionIR, NodeAllocationIR, FusedGroupIR, and three view models
- `from_internal(scheduler)` raises ValueError if `scheduler.latency_total == -1` (pre-solve sentinel guard per RESEARCH.md Pitfall 2)
- Three view methods return Pydantic instances: `algorithmic_view()` (latency + backend + constraints), `hardware_view()` (resource/memory allocation per node), `compiler_view()` (inter-core tiling + fused groups + runtime args)
- Updated `stream/ir/__init__.py` to re-export all 8 AllocationIR-related names
- Added 9 TestAllocationIR tests (25 total in test_ir_models.py), all passing
- Created `.claude/skills/ir/SKILL.md` and `.claude/skills/ir/ir-models.md` conceptual guide (78 lines, no code examples)

## Task Commits

Each task was committed atomically:

1. **Task 1 RED: Add failing AllocationIR tests** - `14cdcb3` (test)
2. **Task 1 GREEN: Implement AllocationIR with three persona views** - `652e665` (feat)
3. **Task 2: Create IR skill documentation** - `1450332` (docs)

**Plan metadata:** (committed with docs commit)

_Note: TDD task — failing tests committed first (RED), implementation added (GREEN), ruff line-length fix inline (REFACTOR)_

## Files Created/Modified

- `stream/ir/allocation.py` - AllocationIR and sub-models (LatencyInfo, ConstraintSelectionIR, NodeAllocationIR, FusedGroupIR) with from_internal() and three view methods
- `stream/ir/__init__.py` - Extended to re-export AllocationIR, AllocationAlgorithmicView, AllocationHardwareView, AllocationCompilerView, LatencyInfo, ConstraintSelectionIR, NodeAllocationIR, FusedGroupIR
- `tests/unit/test_ir_models.py` - Added ALLOCATION_RAW fixture and TestAllocationIR with 9 tests
- `.claude/skills/ir/SKILL.md` - Skill trigger file for IR models (stream-aie-ir)
- `.claude/skills/ir/ir-models.md` - Conceptual guide: three IR classes, construction pattern, schema versioning, per-persona views, anti-patterns

## Decisions Made

- `AllocationIR.from_internal()` raises `ValueError` on pre-solve scheduler rather than accepting -1 sentinel silently. MCP tool consumers receiving latency=-1 would be confused — fail early with a clear message.
- `NodeAllocationIR` carries all three slot-indexed fields (resource_allocation, inter_core_tiling, memory_allocation) so a single sub-model serves both `hardware_view` and `compiler_view`, avoiding duplicate model definitions.
- IR skill files use conceptual-guide style (no code examples) per Phase 11 decision — avoids staleness, keeps files short enough to load without context cost.

## Deviations from Plan

None - plan executed exactly as written. One ruff line-length fix (E501, line 22 in allocation.py) applied during REFACTOR phase — safe, no behavior change.

## Issues Encountered

None. All 25 tests passed on first GREEN run. Single ruff E501 (line too long in LatencyInfo.overlap_between_iterations Field description) fixed by wrapping to two lines.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- `stream/ir/` package complete: `from stream.ir import WorkloadIR, AcceleratorIR, AllocationIR` works
- All three IR classes importable, all view methods available
- 114 unit tests pass, ruff clean
- IR skill documentation ready for agent auto-discovery
- Phase 17 (MCP server) can consume all three IR classes directly via `from_internal()` classmethods

## Self-Check: PASSED

Files verified to exist:
- FOUND: stream/ir/allocation.py
- FOUND: stream/ir/__init__.py
- FOUND: tests/unit/test_ir_models.py
- FOUND: .claude/skills/ir/SKILL.md
- FOUND: .claude/skills/ir/ir-models.md

Task commits verified present:
- FOUND: 14cdcb3 (test RED)
- FOUND: 652e665 (feat GREEN)
- FOUND: 1450332 (docs skill)

---
*Phase: 16-ir-models*
*Completed: 2026-05-10*
