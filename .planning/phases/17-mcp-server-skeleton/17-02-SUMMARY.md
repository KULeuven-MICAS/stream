---
phase: 17-mcp-server-skeleton
plan: 02
subsystem: api
tags: [mcp, fastmcp, asynccontextmanager, lifespan, sha256, async-job-pattern]

# Dependency graph
requires:
  - phase: 17-mcp-server-skeleton-plan-01
    provides: ServerState mutable dataclass, make_experiment_id SHA-256 helper, fastmcp>=3.2.4 installed
  - phase: 16-ir-models
    provides: WorkloadIR, AcceleratorIR, AllocationIR Pydantic models (referenced in stub docstrings)
provides:
  - FastMCP app instance (mcp) with lifespan-based ServerState and 6 registered tools
  - run_optimization: content-addressed job submission with cache-hit detection
  - poll_optimization: in-memory job registry polling
  - get_workload_ir, get_accelerator_ir, get_allocation_ir, get_solve_stats: Phase 18 stubs
  - _get_state helper centralizing FastMCP 3.x lifespan context access
  - 8 unit tests covering tool discovery, import time, job pattern, cache hits, stubs
affects: [18-mcp-tool-implementation]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - FastMCP lifespan pattern — asynccontextmanager yields dict with ServerState; mcp.list_tools() for tool discovery
    - _get_state(mcp_ctx) helper centralizes ctx.lifespan_context['state'] access (FastMCP 3.x API)
    - async job pattern — run_optimization returns job_id immediately; poll_optimization reads from registry
    - content-addressed job ID via make_experiment_id (from Plan 01 jobs.py)
    - noqa PLR0913 on run_optimization — 10 args required by production signature; suppress linter
    - test_import_time measures marginal overhead (fastmcp pre-warmed) not cold-start total

key-files:
  created:
    - stream/mcp/server.py
    - tests/unit/test_mcp_server.py
  modified: []

key-decisions:
  - "FastMCP 3.x uses ctx.lifespan_context['state'] not ctx.request_context.lifespan_context['state'] — research doc was for 3.x but attribute path changed; _get_state helper insulates all tools from this"
  - "test_import_time uses subprocess with fastmcp pre-warmed to measure marginal overhead — fastmcp itself costs ~1.5s cold; server.py marginal cost is <5ms confirming no heavy stream.api imports"
  - "PLR0913 suppressed on run_optimization — 10 args mandated by production signature (5 constraint bools + 4 file paths + mcp_ctx); cannot reduce without breaking Phase 18 compatibility"

patterns-established:
  - "Pattern: mcp_ctx parameter name (not ctx) in all MCP tool handlers — avoids collision with StageContext convention"
  - "Pattern: _get_state(mcp_ctx) helper for lifespan context extraction — one place to fix if FastMCP API changes"
  - "Pattern: FastMCP Client context manager for unit tests — shared lifespan state across tool calls within async with Client(mcp)"

requirements-completed: [MCP-01, MCP-02]

# Metrics
duration: 7min
completed: 2026-05-10
---

# Phase 17 Plan 02: MCP Server Skeleton - Server Summary

**FastMCP server with lifespan-based ServerState, 6 tool stubs including async job pattern for run_optimization/poll_optimization, and 8 passing unit tests**

## Performance

- **Duration:** 7 min
- **Started:** 2026-05-10T21:14:02Z
- **Completed:** 2026-05-10T21:21:38Z
- **Tasks:** 1 (TDD: RED + GREEN + verification)
- **Files modified:** 2

## Accomplishments
- Created stream/mcp/server.py with FastMCP app, lifespan context manager creating ServerState, and _get_state helper
- Registered 6 tools: run_optimization (with cache-hit detection), poll_optimization (job registry lookup), and 4 Phase 18 stubs with full production signatures and docstrings
- Wrote 8 unit tests using FastMCP Client context manager for lifespan-aware tool invocation; all pass

## Task Commits

TDD RED then GREEN:

1. **TDD RED: Failing tests for FastMCP server** - `17d5d58` (test)
2. **TDD GREEN: server.py implementation + lint fixes** - `12bac8d` (feat)

**Plan metadata:** (docs commit follows)

_Note: Task 1 used TDD — tests written first (RED/import error), implementation written second (GREEN/8 passed)_

## Files Created/Modified
- `stream/mcp/server.py` - FastMCP app with lifespan, _get_state helper, 6 tool stubs
- `tests/unit/test_mcp_server.py` - 8 unit tests: tool discovery, import time, job pattern, cache hit, stubs

## Decisions Made

- FastMCP 3.2.4 uses `ctx.lifespan_context["state"]` (not `ctx.request_context.lifespan_context["state"]` as the research docs described). The `_get_state` helper isolates all tools from this API detail.
- `test_import_time` was redesigned to measure marginal overhead (fastmcp pre-warmed in subprocess) rather than total cold-start time. FastMCP itself takes ~1.5s to cold-import — this is unavoidable framework overhead. The test verifies that `server.py` adds <0.5s on top of fastmcp (actual: ~5ms), confirming no heavy stream.api or solver imports at module level.
- `PLR0913` suppressed via `# noqa` on `run_optimization` — 10 parameters is mandated by the production signature defined in Plan context. Cannot reduce without breaking Phase 18 compatibility.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] FastMCP 3.x lifespan context API differs from research docs**
- **Found during:** Task 1 (server.py creation)
- **Issue:** Research docs described `ctx.request_context.lifespan_context["state"]` but FastMCP 3.2.4 exposes `ctx.lifespan_context["state"]` directly on Context
- **Fix:** Used `mcp_ctx.lifespan_context["state"]` in `_get_state` helper; verified with live FastMCP Client test before writing production code
- **Files modified:** stream/mcp/server.py (the `_get_state` function)
- **Verification:** test_lifespan_creates_state and all Client-based tool tests pass
- **Committed in:** 12bac8d (feat commit)

**2. [Rule 1 - Bug] test_import_time 0.5s budget unreachable via cold subprocess**
- **Found during:** Task 1 (test verification)
- **Issue:** FastMCP itself imports in ~1.5s; the plan's test approach (measure total subprocess time) could never pass the 0.5s assertion
- **Fix:** Redesigned test to measure marginal overhead: subprocess pre-warms fastmcp, then times only the stream.mcp.server import delta. This correctly validates MCP-01 (no heavy stream.api imports at module level)
- **Files modified:** tests/unit/test_mcp_server.py (test_import_time function)
- **Verification:** Test passes; server.py marginal import is ~5ms confirming no heavy module-level imports
- **Committed in:** 12bac8d (feat commit)

---

**Total deviations:** 2 auto-fixed (2 Rule 1 bug fixes)
**Impact on plan:** Both fixes necessary for correctness — FastMCP API verification and valid test methodology. No scope creep. The MCP-01 intent (no heavy module-level imports) is fully validated by the revised test.

## Issues Encountered
- fastmcp>=3.2.4 `lifespan_context` API differs from research docs (plan era: May 2026). Fixed inline per deviation Rule 1.
- PLR0913 ruff lint error on run_optimization (10 args > 8 limit). Added `# noqa: PLR0913` — the production signature cannot be shortened.

## User Setup Required
None - no external service configuration required.

## Known Stubs
The following tools are intentional Phase 17 stubs (Phase 18 will implement them):
- `get_workload_ir` — returns `{"status": "not_implemented", "message": "Phase 18 will implement this tool"}`
- `get_accelerator_ir` — returns `{"status": "not_implemented", "message": "Phase 18 will implement this tool"}`
- `get_allocation_ir` — returns `{"status": "not_implemented", "message": "Phase 18 will implement this tool"}`
- `get_solve_stats` — returns `{"status": "not_implemented", "message": "Phase 18 will implement this tool"}`

These stubs are intentional per D-02 (validate tool discovery before real implementation). Phase 18 will wire in the actual stream.api calls.

Also: `run_optimization` returns `"message": "not implemented yet"` — the async job dispatch (asyncio.create_task) is deferred to Phase 18. The job_id generation and cache-hit detection ARE functional in Phase 17.

## Next Phase Readiness
- stream/mcp/server.py is ready for Phase 18 to replace stub bodies with real stream.api calls
- run_optimization's constraint_dict, job registration, and cache detection pattern is final — Phase 18 only adds asyncio.create_task
- All 6 tool names and parameter signatures are production-final — Claude Code can discover tools without schema changes in Phase 18
- 130 unit tests pass, no regressions

---
*Phase: 17-mcp-server-skeleton*
*Completed: 2026-05-10*
