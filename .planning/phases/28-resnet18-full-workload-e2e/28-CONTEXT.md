# Phase 28: ResNet18 Full Workload E2E - Context

**Gathered:** 2026-05-15
**Status:** In progress — fixed mapping YAML approach decided, implementation pending

<domain>
## Phase Boundary

Run the complete ResNet18 ONNX model end-to-end through the CO pipeline on TPU hardware. The Phase 27 fusion strategy splits it into 11 groups — this phase proves the entire pipeline works at full scale. Add a timing breakdown to the YAML summary using existing log timestamps for bottleneck analysis.

</domain>

<decisions>
## Implementation Decisions

### Test Strategy
- **D-01:** Single `test_resnet18_full_e2e()` integration test. Uses `optimize_allocation_co_with_mapping` with a **fixed hand-crafted mapping YAML** (NOT auto-generated). The fixed mapping specifies proper spatial tiling (tile=1 on one spatial dim per group) so activation tiles fit in 128KB SRAM. Asserts: positive `total_latency`, exactly 11 `group_latencies` entries, all positive, `total_latency == sum(group_latencies)`.
- **D-01b (REVISED):** The `GenericMappingGenerator` stays simple (Phase 27 version, no memory-aware tiling logic). It doesn't know how to tile for memory — that's the MILP's job given a proper mapping. The E2E test uses a fixed mapping that encodes the correct spatial tiling per group.

### Timeout & Performance
- **D-02:** Mark with `@pytest.mark.slow` and `@pytest.mark.timeout(900)`. Default test runs skip it (`pytest -m "not slow"`). Explicit opt-in: `pytest -m slow`. Update `conftest.py` to register the `slow` marker. Keeps CI fast while the test exists for manual verification.

### Failure Handling
- **D-03:** The 11 fusion groups should match the patterns tested in Phase 25. If any group fails, fix holistically in this phase — it indicates a real integration bug, not a known limitation.

### Timing Analysis
- **D-04:** Parse existing log timestamps to extract per-group and per-stage durations. No `time.time()` calls in pipeline internals. The pipeline already emits INFO logs with timestamps for each group and key stages.
- **D-05:** Include the timing breakdown in the YAML summary output from `main_stream_co.py`. Extends the existing per-group latency breakdown with wall-clock timing. Enables bottleneck analysis (Phase 24 D-04 intent).

### Fusion Groups
- **D-06:** The full ResNet18 workload should produce exactly 11 groups via `determine_fusion_cut_points()` (from Phase 27). The test asserts this count. The groups should correspond to: 1 front-end (Conv1+Relu+MaxPool), 8 residual blocks (each Conv+Relu+Conv+Add+Relu), 1 tail (GlobalAveragePool), and 1 post-Flatten (Gemm).

### Claude's Discretion
- Exact dimension indices for the fixed mapping YAML per group (requires inspecting each group's unique_dimensions)
- Whether the empty `links_used` crash (from 28-02 worktree) needs fixing
- Exact YAML summary format for timing data
- Test file location (extend test_resnet_patterns.py or new file)

### Implementation Progress (as of 2026-05-15)

**COMPLETED:**
- `e51daa2`: GlobalAveragePool parser fixed — proper 6D iteration space `(b, c, oh, ow, ih, iw)` instead of AffineConstantExpr(0) hack
- `c1b0247`: SRAM reverted to original 128KB on simd/pooling cores (hardware must not change)
- `f73fb25`: GenericMappingGenerator reverted to clean Phase 27 version (no intra_core_tiling spaghetti)
- `2bbf655`: Visualization guard for large workloads (skip >30 nodes)
- `45507fb`: Wall-clock timing per fusion group (group_wall_times in FusionGroupIterationStage)
- All 194 existing tests pass

**REMAINING:**
1. Create `stream/inputs/examples/mapping/resnet18_tpu_quad_core.yaml` — a fixed mapping for all 11 ResNet18 fusion groups with proper spatial tiling. Each group tiles one spatial dim to tile=1 so that per-iteration activation tiles fit in 128KB. The mapping uses `optimize_allocation_co_with_mapping` (takes explicit mapping path).
2. Investigate each group's dimension structure to determine which local dim index corresponds to the spatial height for that group's reference node.
3. Fix the empty `links_used` crash if it surfaces (transfer paths with no communication links → min() on empty iterable in `_transfer_latency_for_path`).
4. Write `test_resnet18_full_e2e()` in `tests/test_resnet_patterns.py` with @pytest.mark.slow.
5. Verify the full pipeline completes with positive latency for all 11 groups.

**KEY INSIGHT (from debugging session):**
- The MILP becomes infeasible when activations don't fit in 128KB SRAM
- The fix is NOT to increase SRAM, but to tile one spatial dim to 1 in the intra_core_tiling
- For Group 0: Conv output is (1, 64, 112, 112). Tile the last spatial dim (W=112) to 1 → per-iteration tile becomes (1, 64, 112, 1) = 14 KB. Fits easily.
- The inter_core_tiling already splits channels (D1 split=4 for Conv on 4 cores)
- Each group needs: identify the z-variable for height/W, find which node+local_dim it maps to, set tile=1

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Pipeline API
- `stream/api.py` — `optimize_allocation_co_generic()` entry point
- `stream/stages/generation/fusion_group_iteration.py` — FusionGroupIterationStage (sets `total_latency`, `group_latencies`)
- `stream/stages/generation/generic_mapping_generation.py` — calls `determine_fusion_cut_points()`

### Fusion Strategy (Phase 27)
- `stream/workload/workload.py` §`determine_fusion_cut_points()` — 9 cut points for ResNet18
- `stream/workload/workload.py` §`split_fusion_groups(cut_points=)` — extended split mechanism

### CLI & Output
- `main_stream_co.py` — YAML summary with per-group latency + percentages (Phase 24 D-04)

### Test Infrastructure
- `tests/test_resnet_patterns.py` — Phase 25-27 pattern and integration tests
- `tests/conftest.py` — pytest configuration (--keep-output flag)

### ResNet18 Model
- `stream/inputs/examples/workload/resnet18.onnx` — Full ResNet18 ONNX (49 nodes)
- `stream/inputs/examples/hardware/tpu_like_quad_core.yaml` — TPU hardware

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `optimize_allocation_co_generic()` handles the full pipeline including fusion group splitting
- `FusionGroupIterationStage` already logs `f"Group {i} latency: {group_latency}"` per group with timestamps
- `main_stream_co.py` already produces YAML summary with `total_latency` + per-group `latency` + `pct`
- `test_resnet_patterns.py` has existing test infrastructure (accelerator path, imports, patterns)

### Established Patterns
- Tests use `tempfile.TemporaryDirectory()` for output
- `@pytest.mark.timeout(120)` for sub-graph tests, need `@pytest.mark.timeout(900)` for full E2E
- YAML summary format: `{total_latency: float, groups: [{group: int, latency: float, pct: float}]}`

### Integration Points
- Add `slow` marker to `conftest.py`
- Extend YAML summary format in `main_stream_co.py` with timing data
- Test in `test_resnet_patterns.py` (same file as other ResNet tests)

</code_context>

<specifics>
## Specific Ideas

- The timing analysis should show which of the 11 groups takes the longest wall-clock time — this reveals if ZigZag or MILP is the bottleneck for specific groups
- Log format is `2026-05-14 21:16:14 - stream.stages.generation.fusion_group_iteration.run +55 - INFO - Group 0 latency: 6044.0` — the timestamp prefix gives wall-clock timing for free
- The `slow` marker should be in `conftest.py` with `pytest.ini_options` or as a registered marker

</specifics>

<deferred>
## Deferred Ideas

None — this is the final phase of v1.6. Anything beyond full ResNet18 E2E belongs in a future milestone.

</deferred>

---

*Phase: 28-resnet18-full-workload-e2e*
*Context gathered: 2026-05-15*
