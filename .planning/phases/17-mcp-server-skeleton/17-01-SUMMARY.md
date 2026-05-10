---
phase: 17-mcp-server-skeleton
plan: 01
subsystem: api
tags: [mcp, fastmcp, hashlib, sha256, dataclass, asyncio]

# Dependency graph
requires:
  - phase: 16-ir-models
    provides: WorkloadIR, AcceleratorIR, AllocationIR Pydantic models that MCP server will return
provides:
  - ServerState mutable dataclass with jobs dict for in-process job registry
  - make_experiment_id SHA-256 content-addressed 12-hex-char experiment ID helper
  - stream/mcp package init with re-exports
  - fastmcp>=3.2.4 declared as dependency in pyproject.toml
affects: [17-mcp-server-skeleton-plan-02]

# Tech tracking
tech-stack:
  added: [fastmcp>=3.2.4]
  patterns: [content-addressed IDs via SHA-256 file contents + backend + sorted constraints, mutable ServerState dataclass for job registry, stream/mcp/ package separating jobs.py from server.py for testability]

key-files:
  created:
    - stream/mcp/__init__.py
    - stream/mcp/jobs.py
    - tests/unit/test_mcp_jobs.py
  modified:
    - pyproject.toml

key-decisions:
  - "make_experiment_id is a public function (no underscore prefix) since it is re-exported in __init__ and tested directly — not private despite being a helper"
  - "ServerState is NOT frozen (mutable by design — holds the live job registry)"
  - "jobs.py isolated from FastMCP imports — all 8 unit tests run without importing fastmcp, keeping test footprint minimal"

patterns-established:
  - "Pattern: content-addressed experiment IDs via SHA-256 of file CONTENTS (not paths) + backend + sorted JSON of constraint dict, truncated to 12 hex chars"
  - "Pattern: ServerState mutable dataclass with jobs dict[str, dict[str, Any]] owned via FastMCP lifespan in Plan 02"

requirements-completed: [MCP-03]

# Metrics
duration: 2min
completed: 2026-05-10
---

# Phase 17 Plan 01: MCP Server Skeleton - Jobs Package Summary

**stream/mcp/ package created with SHA-256 content-addressed experiment IDs, mutable ServerState job registry dataclass, 8 passing unit tests, and fastmcp>=3.2.4 declared in pyproject.toml**

## Performance

- **Duration:** 2 min
- **Started:** 2026-05-10T21:10:31Z
- **Completed:** 2026-05-10T21:12:31Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- Created stream/mcp/jobs.py with ServerState (mutable dataclass, jobs dict) and make_experiment_id (SHA-256 content-addressed 12-hex-char ID using file contents + backend + sorted constraint JSON)
- Created stream/mcp/__init__.py re-exporting ServerState and make_experiment_id with __all__
- Wrote 8 unit tests covering all plan behavior specs: ID determinism, backend differentiation, constraint differentiation, content differentiation, ID length/format, job lifecycle (pending/running/complete), job failure
- Added fastmcp>=3.2.4 to pyproject.toml dependencies; installed and importable as 3.2.4

## Task Commits

Each task was committed atomically:

1. **Task 1: Create stream/mcp/ package with jobs.py (ServerState + experiment ID)** - `848ea56` (feat)
2. **Task 2: Add fastmcp dependency to pyproject.toml** - `cf82f93` (chore)

**Plan metadata:** (docs commit follows)

_Note: Task 1 used TDD — tests written first (RED/import error), implementation written second (GREEN/8 passed)_

## Files Created/Modified
- `stream/mcp/jobs.py` - ServerState dataclass and make_experiment_id SHA-256 helper
- `stream/mcp/__init__.py` - Package init re-exporting both symbols with __all__
- `tests/unit/test_mcp_jobs.py` - 8 unit tests for MCP-03 and job registry (8 passed)
- `pyproject.toml` - Added "fastmcp>=3.2.4" after pydantic dependency

## Decisions Made
- make_experiment_id named without leading underscore since it is a public API (re-exported in __init__ and directly tested)
- ServerState is mutable (NOT frozen) since it holds the live in-process job registry that Plan 02's lifespan context manager will mutate
- jobs.py deliberately avoids importing fastmcp, stream.api, or any solver — enables unit tests to run without FastMCP installed and keeps cold-start import chain minimal

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
- fastmcp not yet installed in environment (expected) — ran `pip install "fastmcp>=3.2.4"` explicitly as directed in plan action

## Known Stubs
None. This plan creates foundational data structures (no UI rendering, no placeholder data flows).

## Next Phase Readiness
- stream/mcp/jobs.py is ready for Plan 02 to import ServerState into the FastMCP lifespan
- make_experiment_id is importable by server.py for run_optimization job ID generation
- fastmcp 3.2.4 is installed and available for stream/mcp/server.py creation in Plan 02
- No blockers for Plan 02

---
*Phase: 17-mcp-server-skeleton*
*Completed: 2026-05-10*
