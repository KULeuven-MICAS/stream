---
phase: 28-resnet18-full-workload-e2e
plan: 02
subsystem: testing
tags: [resnet18, onnx, milp, e2e-test, integration, pytest-slow]

# Dependency graph
requires:
  - phase: 28-resnet18-full-workload-e2e
    plan: 01
    provides: Visualization guard, 4MB SRAM (simd/pooling), wall-clock timing
  - phase: 27-resnet18-fusion-strategy
    provides: determine_fusion_cut_points(), 11 fusion groups for ResNet18
provides:
  - test_resnet18_full_e2e() capstone integration test
  - 8MB SRAM across all core types for full-scale ResNet18 MILP feasibility
  - ZigZag fallback covering get_layer_node failures (GlobalAveragePool)
  - Guard for empty links_used in transfer latency path computation
affects: [v1.6-milestone-complete]

# Tech tracking
tech-stack:
  added: []
  patterns: [zigzag-fallback-covers-node-construction, empty-path-guard-for-direct-transfers]

key-files:
  created: []
  modified:
    - tests/test_resnet_patterns.py
    - stream/inputs/examples/hardware/cores/tpu_like.yaml
    - stream/inputs/examples/hardware/cores/simd.yaml
    - stream/inputs/examples/hardware/cores/pooling.yaml
    - stream/opt/allocation/constraint_optimization/transfer_and_tensor_allocation.py
    - stream/stages/estimation/zigzag_cost_estimator.py

key-decisions:
  - "SRAM 8MB for all cores: Group 0 SIMD core must hold Conv output + Relu output simultaneously (2x 3.07MB = 6.14MB), 4MB was insufficient"
  - "Empty links_used returns 0 latency: direct/same-core transfers have no communication links, min() on empty iterable was crashing"
  - "ZigZag fallback expanded to cover get_layer_node: GlobalAveragePool has AffineConstantExpr in affine mapping, triggers NotImplementedError before run_zigzag"

patterns-established:
  - "ZigZag fallback scope: estimate() try-block now covers both get_layer_node() and run_zigzag(), using workload dimension sizes when layer_node unavailable"
  - "Transfer latency guard: _transfer_latency_for_path checks path.links_used emptiness before min() call"

requirements-completed: [RNET-06]

# Metrics
duration: 35min
completed: 2026-05-15
---

# Phase 28 Plan 02: Full ResNet18 E2E Integration Test Summary

**Capstone test_resnet18_full_e2e passes: 11 groups, all positive latency, total=sum, 254s runtime on full 49-node ResNet18**

## Performance

- **Duration:** 35 min
- **Started:** 2026-05-15T09:31:42Z
- **Completed:** 2026-05-15T10:06:52Z
- **Tasks:** 2
- **Files modified:** 6

## Accomplishments
- Added test_resnet18_full_e2e() with @pytest.mark.slow + @pytest.mark.timeout(900)
- Fixed 3 pipeline bugs blocking full-scale ResNet18 E2E: SRAM sizing, empty transfer paths, ZigZag node construction
- Full pipeline completes in 254s for all 11 fusion groups with valid latency values
- All 194 non-slow tests pass without regression

## Task Commits

Each task was committed atomically:

1. **Task 1: Write test_resnet18_full_e2e integration test** - `9f9c839` (test)
2. **Pipeline fixes (deviation)** - `03b6c2f` (fix): 8MB SRAM, empty links guard, ZigZag fallback expansion

## Files Created/Modified
- `tests/test_resnet_patterns.py` - Added test_resnet18_full_e2e() with @pytest.mark.slow, 11-group assertion, total=sum check
- `stream/inputs/examples/hardware/cores/tpu_like.yaml` - SRAM increased from 2MB to 8MB (sram_2MB -> sram_8MB)
- `stream/inputs/examples/hardware/cores/simd.yaml` - SRAM increased from 4MB to 8MB (sram_4MB_2rw -> sram_8MB_2rw)
- `stream/inputs/examples/hardware/cores/pooling.yaml` - SRAM increased from 4MB to 8MB (sram_4MB -> sram_8MB)
- `stream/opt/allocation/constraint_optimization/transfer_and_tensor_allocation.py` - Guard for empty links_used in _transfer_latency_for_path
- `stream/stages/estimation/zigzag_cost_estimator.py` - Expanded ZigZag fallback try-block to cover get_layer_node failures

## Decisions Made
- SRAM 8MB for all core types: Plan 01 set simd/pooling to 4MB but Group 0 needs 2 tensors simultaneously on SIMD core (Conv output + Relu output = 2 x 3.07MB = 6.14MB). tpu_like compute cores also increased to 8MB for headroom.
- Empty transfer path returns 0 latency (not crash): direct/same-core transfers in Groups 1+ have no communication links.
- ZigZag fallback expansion: layer_node initialization moved inside try-block; fallback uses workload.get_dims() when layer_node unavailable.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] MILP infeasibility for Group 0: SRAM insufficient on all core types**
- **Found during:** Task 2 (running slow test)
- **Issue:** Plan 01 increased simd/pooling SRAM to 4MB, but Group 0 SIMD core (Core 5) must hold Conv output AND Relu output simultaneously (2 x 3.07MB = 6.14MB > 4MB). Also compute cores (tpu_like) needed increase for the same Conv output tensor with reuse factor.
- **Fix:** Increased all core SRAM to 8MB (67108864 bits): tpu_like, simd, pooling
- **Files modified:** stream/inputs/examples/hardware/cores/tpu_like.yaml, simd.yaml, pooling.yaml
- **Verification:** Group 0 MILP solves optimally with Primal Bound +1.21e+08
- **Committed in:** 03b6c2f

**2. [Rule 1 - Bug] ValueError: min() iterable argument is empty in _transfer_latency_for_path**
- **Found during:** Task 2 (pipeline progressed to Group 1 after SRAM fix)
- **Issue:** path.links_used is empty for direct/same-core transfers (no communication links), causing min() to crash
- **Fix:** Added guard `if not path.links_used: return 0` before min() call
- **Files modified:** stream/opt/allocation/constraint_optimization/transfer_and_tensor_allocation.py
- **Verification:** Groups 1-8 complete successfully
- **Committed in:** 03b6c2f

**3. [Rule 1 - Bug] NotImplementedError: Unsupported affine expr type AffineConstantExpr in GlobalAveragePool**
- **Found during:** Task 2 (pipeline progressed to Group 9 after prior fixes)
- **Issue:** ZigZag cost estimator's get_layer_node() fails for GlobalAveragePool due to AffineConstantExpr in affine mapping. The existing fallback only covered run_zigzag() failures, not get_layer_node() failures.
- **Fix:** Moved layer_node construction inside try-block; fallback uses workload.get_dims() when layer_node is None
- **Files modified:** stream/stages/estimation/zigzag_cost_estimator.py
- **Verification:** Group 9 falls back to ideal-cycle estimate (31488), pipeline continues to Group 10
- **Committed in:** 03b6c2f

---

**Total deviations:** 3 auto-fixed (2 bugs, 1 blocking)
**Impact on plan:** All 3 fixes necessary for the E2E pipeline to complete. Without them, the test cannot pass. No scope creep.

## Test Results

```
pytest tests/test_resnet_patterns.py::test_resnet18_full_e2e -v -s --timeout=900
PASSED in 254.56s (0:04:14)

total_latency: 1,820,754,448.0
group_latencies:
  Group 0 (frontend):          121,114,672
  Group 1 (basic residual):    231,901,182
  Group 2 (basic residual):    231,901,182
  Group 3 (stride-2):          180,187,918
  Group 4 (basic residual):    231,588,350
  Group 5 (stride-2):          180,072,094
  Group 6 (basic residual):    231,527,806
  Group 7 (stride-2):          180,216,318
  Group 8 (basic residual):    231,885,502
  Group 9 (GlobalAvgPool):          31,488
  Group 10 (Gemm):                 327,936

total = sum(group_latencies): MATCH (diff = 0.0)
All 11 groups: positive latency: PASS
Non-slow tests: 194 passed, 18 deselected
```

## Issues Encountered
- Plan 01's SRAM analysis underestimated requirements: it only considered single-tensor sizing (3.07MB) but the MILP constraint sums ALL tensors on a core simultaneously. SIMD core holds both input and output tensors of Relu, requiring 2x the single-tensor size.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Full ResNet18 E2E pipeline verified at 224x224 scale
- v1.6 milestone capstone test passes
- All 194 + 1 slow tests pass

## Self-Check: PASSED

All 7 files exist. Both commit hashes (9f9c839, 03b6c2f) verified. All 6 content checks passed.

---
*Phase: 28-resnet18-full-workload-e2e*
*Completed: 2026-05-15*
