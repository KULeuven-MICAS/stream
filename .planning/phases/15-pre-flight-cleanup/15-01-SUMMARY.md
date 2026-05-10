---
phase: 15-pre-flight-cleanup
plan: 01
subsystem: api
tags: [logging, mcp, stdio, python-logging, basicconfig]

# Dependency graph
requires: []
provides:
  - configure_logging() helper in stream/api.py (no module-level basicConfig side effect)
  - Zero bare print() calls in the MILP solve path (TTA file)
affects: [16-mcp-server, 17-mcp-tools, 18-mcp-advanced]

# Tech tracking
tech-stack:
  added: []
  patterns: [configure_logging() called explicitly by CLI scripts; MCP server manages its own logging]

key-files:
  created: []
  modified:
    - stream/api.py
    - stream/opt/allocation/constraint_optimization/transfer_and_tensor_allocation.py
    - main_gemm.py
    - main_swiglu.py
    - main_swiglu_dse.py
    - main_swiglu_dse_single.py
    - main_aie_co.py
    - main_aie_ga.py
    - main_aie_codegen_conv2d.py
    - main_stream_co.py
    - main_stream_ga.py
    - main_gemm_manual.py
    - tests/study_constraint_toggles.py
    - tests/study_swiglu_backends.py
    - tests/verify_backends.py
    - tests/study_constraint_toggles_cross_backend.py

key-decisions:
  - "configure_logging() uses _logging.basicConfig() with idempotency guarantee (subsequent calls no-op once handlers exist)"
  - "Scripts without if __name__ == '__main__' guard call configure_logging() at module top-level execution path"
  - "study_constraint_toggles_cross_backend.py adds stream.api import for configure_logging even though it uses no other stream.api functions"

patterns-established:
  - "CLI pattern: import configure_logging from stream.api, call as first line of __main__ block"
  - "Logger pattern: use %-formatting not f-strings in _logger.info/warning calls"

requirements-completed: [CLEAN-02, CLEAN-04]

# Metrics
duration: 15min
completed: 2026-05-10
---

# Phase 15 Plan 01: Silence stdout and decouple root logging Summary

**Zero print() calls in the MILP solve path and stream.api import no longer triggers basicConfig() side effect, enabling clean MCP stdio transport**

## Performance

- **Duration:** ~15 min
- **Started:** 2026-05-10T16:04:00Z
- **Completed:** 2026-05-10T16:09:38Z
- **Tasks:** 2
- **Files modified:** 16

## Accomplishments

- Replaced all 5 bare print() calls in transfer_and_tensor_allocation.py with _logger.info()/_logger.warning() using %-formatting
- Moved module-level `_logging.basicConfig()` from stream/api.py into a `configure_logging()` helper function
- Updated all 14 CLI scripts and study/verify scripts to call configure_logging() explicitly at startup

## Task Commits

Each task was committed atomically:

1. **Task 1: Replace print() calls in TTA with logger calls** - `f3c475a` (fix)
2. **Task 2: Move module-level basicConfig to configure_logging helper** - `1307aa4` (feat)

**Plan metadata:** (see below)

## Files Created/Modified

- `stream/api.py` - Added configure_logging() function; removed module-level basicConfig call
- `stream/opt/allocation/constraint_optimization/transfer_and_tensor_allocation.py` - 5 print() calls replaced with _logger.info()/_logger.warning()
- `main_gemm.py` - Added configure_logging import and call in __main__
- `main_swiglu.py` - Added configure_logging import and call in __main__
- `main_swiglu_dse.py` - Added configure_logging import and call in __main__
- `main_swiglu_dse_single.py` - Added configure_logging import and call in __main__
- `main_aie_co.py` - Replaced module-level basicConfig with configure_logging() call
- `main_aie_ga.py` - Replaced module-level basicConfig with configure_logging() call
- `main_aie_codegen_conv2d.py` - Added configure_logging import and call in __main__
- `main_stream_co.py` - Replaced module-level basicConfig with configure_logging() call
- `main_stream_ga.py` - Replaced module-level basicConfig with configure_logging() call
- `main_gemm_manual.py` - Added configure_logging import and call in __main__
- `tests/study_constraint_toggles.py` - Added configure_logging import and call in __main__
- `tests/study_swiglu_backends.py` - Added configure_logging import and call in __main__
- `tests/verify_backends.py` - Added configure_logging import and call in __main__
- `tests/study_constraint_toggles_cross_backend.py` - Added configure_logging import and call in main()

## Decisions Made

- Scripts without an `if __name__ == "__main__"` guard (main_aie_co.py, main_aie_ga.py, main_stream_co.py, main_stream_ga.py) call configure_logging() at their top-level execution path using explicit level/fmt kwargs matching their local format string
- configure_logging() uses _logging.basicConfig() which Python guarantees is idempotent (no-op if root handlers already exist), so calling it in scripts that set up custom handlers inside functions is safe
- All _logger calls in TTA use %-formatting (not f-strings) per Python logging best practices

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- MCP stdio transport is now clean: no print() calls on the MILP solve path
- Importing stream.api no longer configures root logging — MCP server can control its own logging
- All CLI scripts retain their logging output via explicit configure_logging() calls
- 75 unit tests pass, no regression

---
*Phase: 15-pre-flight-cleanup*
*Completed: 2026-05-10*

## Self-Check: PASSED

- `f3c475a` exists: FOUND
- `1307aa4` exists: FOUND
- `stream/api.py` modified: FOUND (configure_logging at line 29)
- `transfer_and_tensor_allocation.py` modified: FOUND (no print() calls)
- Unit tests: 75 passed
