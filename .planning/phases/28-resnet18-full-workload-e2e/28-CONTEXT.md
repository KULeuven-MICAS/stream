# Phase 28: ResNet18 Full Workload E2E - Context

**Gathered:** 2026-05-15
**Status:** Ready for planning

<domain>
## Phase Boundary

Run the complete ResNet18 ONNX model end-to-end through the CO pipeline on TPU hardware. The Phase 27 fusion strategy splits it into 11 groups — this phase proves the entire pipeline works at full scale. Add a timing breakdown to the YAML summary using existing log timestamps for bottleneck analysis.

</domain>

<decisions>
## Implementation Decisions

### Test Strategy
- **D-01:** Single `test_resnet18_full_e2e()` integration test. Runs `optimize_allocation_co_generic` on `resnet18.onnx`, asserts: positive `total_latency`, exactly 11 `group_latencies` entries, all positive, `total_latency == sum(group_latencies)`. Sub-graph patterns are already tested individually in Phase 25.

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
- Log parsing approach (regex on timestamps vs structured log parsing)
- Exact YAML summary format for timing data
- Whether to update main_stream_co.py directly or add a helper function
- Test file location (extend test_resnet_patterns.py or new file)

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
