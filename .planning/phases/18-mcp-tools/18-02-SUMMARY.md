---
phase: 18-mcp-tools
plan: 02
subsystem: api
tags: [fastmcp, mcp, pydantic, workload-ir, accelerator-ir, inspection-tools]

# Dependency graph
requires:
  - phase: 18-01
    provides: run_optimization, poll_optimization, get_allocation_ir, get_solve_stats with async job pattern
  - phase: 16-ir-models
    provides: WorkloadIR.from_internal(), AcceleratorIR.from_internal(), Pydantic IR models

provides:
  - get_workload_ir MCP tool with D-01 dual-parameter pattern (file path or experiment_id)
  - get_accelerator_ir MCP tool with D-01 dual-parameter pattern (file path or experiment_id)
  - Zero stub tools in server.py (all 6 tools fully implemented)
  - 6 unit tests covering inspection tool error paths

affects: [19-mcp-integration, mcp-server-consumers, agent-workflows]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "D-01 dual-parameter pattern: tools accept workload/hardware path OR experiment_id, experiment_id takes precedence"
    - "D-03 structured error dicts: {status, error_type, message} with types invalid_input/not_found/not_ready"
    - "D-04 lazy imports inside tool body with noqa: PLC0415"
    - "noqa: PLR0911 for tool handlers with >6 return paths (dual-parameter + error branching is expected)"

key-files:
  created: []
  modified:
    - stream/mcp/server.py
    - tests/unit/test_mcp_server.py

key-decisions:
  - "D-01 dual-parameter pattern: both workload/hardware and experiment_id are optional; experiment_id takes precedence when both provided"
  - "Lazy imports with noqa: PLR0911 accepted for inspection tools — many return branches are inherent to dual-parameter + D-03 error handling"
  - "Stub test (test_stub_tools_return_not_implemented) removed after Plan 02 — no stubs remain in server.py"

patterns-established:
  - "Inspection tool pattern: validate inputs -> check experiment_id path -> check file path -> parse through pipeline -> return IR.model_dump()"
  - "Error hierarchy for MCP tools: invalid_input (bad params/missing file) > not_found (unknown experiment) > not_ready (job not complete)"

requirements-completed: [TOOL-02]

# Metrics
duration: 2min
completed: 2026-05-10
---

# Phase 18 Plan 02: Inspection Tools Summary

**get_workload_ir and get_accelerator_ir implemented with D-01 dual-parameter pattern (experiment_id or file path), returning Pydantic IR JSON with D-03 structured error handling; all 6 MCP tools fully implemented with zero stubs remaining**

## Performance

- **Duration:** 2 min
- **Started:** 2026-05-10T21:53:26Z
- **Completed:** 2026-05-10T21:55:06Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments

- Replaced both not_implemented stubs with full tool implementations supporting D-01 dual-parameter pattern
- Both tools parse through the stream pipeline stage chain using lazy imports (D-04) for file-path path
- Both tools extract parsed objects from StageContext in completed jobs for experiment_id path
- All error paths return structured D-03 error dicts (invalid_input, not_found, not_ready)
- Removed stub test (test_stub_tools_return_not_implemented) and added 6 new error-path unit tests; all 18 tests pass

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement get_workload_ir and get_accelerator_ir with dual-parameter pattern** - `39893e5` (feat)
2. **Task 2: Add tests for inspection tools and update remaining stub test** - `795beee` (test)

**Plan metadata:** (docs commit follows)

## Files Created/Modified

- `/home/micas/stream_aie/stream/mcp/server.py` - get_workload_ir and get_accelerator_ir implementations replacing not_implemented stubs; dual-parameter pattern per D-01, lazy imports per D-04, structured error handling per D-03
- `/home/micas/stream_aie/tests/unit/test_mcp_server.py` - removed test_stub_tools_return_not_implemented; added 6 error-path tests for inspection tools (no_params, invalid_path, not_found_experiment for each tool)

## Decisions Made

- `noqa: PLR0911` accepted for both inspection tools — the dual-parameter pattern plus 3-tier error handling (invalid_input, not_found, not_ready) genuinely requires >6 return statements; suppression is intentional and documented
- Stub test removed immediately: plan stated no stubs remain after Plan 02, so the stub test is dead code and was replaced with meaningful error-path coverage

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

**Minor linting fix during Task 1:** ruff flagged `PLR0911` (too many return statements > 6) and a `PLC0415` noqa comment on the wrong line of a multi-line import. Both resolved inline:
- Added `# noqa: PLR0911` to both tool function signatures
- Moved `# noqa: PLC0415` to the first line of the parenthesized import

These are expected patterns for dual-parameter tools with multi-tier error handling, not bugs.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- All 6 MCP tools fully implemented and tested (run_optimization, poll_optimization, get_workload_ir, get_accelerator_ir, get_allocation_ir, get_solve_stats)
- Integration tests with real ONNX/hardware fixtures (testing successful parsing returning actual WorkloadIR/AcceleratorIR JSON) are deferred to phase verification
- MCP server is ready for end-to-end integration testing with real workloads

---
*Phase: 18-mcp-tools*
*Completed: 2026-05-10*
