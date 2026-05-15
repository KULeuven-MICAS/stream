# Phase 28: ResNet18 Full Workload E2E - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-15
**Phase:** 28-resnet18-full-workload-e2e
**Areas discussed:** Test strategy, Timeout & performance, Failure handling, Timing analysis

---

## Test Strategy

| Option | Description | Selected |
|--------|-------------|----------|
| Single E2E test with comprehensive assertions | One test running full pipeline, assert latency + 11 groups | ✓ |
| Staged test (parse + split + allocate) | Separate tests per pipeline phase | |
| You decide | Claude picks | |

**User's choice:** Single E2E test. Sub-graph patterns already tested in Phase 25.

---

## Timeout & Performance

| Option | Description | Selected |
|--------|-------------|----------|
| Separate slow marker with 15min timeout | @pytest.mark.slow + timeout(900), skip by default | ✓ |
| Long timeout, always runs | timeout(900) but no skip marker | |
| You decide | Claude picks | |

**User's choice:** Separate slow marker.

---

## Failure Handling

| Option | Description | Selected |
|--------|-------------|----------|
| Fix in Phase 28 | Groups match Phase 25 patterns, fix any integration bugs | ✓ |
| Log and continue | Report partial results | |
| You decide | Claude judges per failure | |

**User's choice:** Free text — "As the determination of the different groups should be identical to the different groups we tested in the previous phase, this should also work here and if not it should be fixed holistically."

---

## Timing Analysis

| Option | Description | Selected |
|--------|-------------|----------|
| Use existing log timestamps | Parse log output for per-group/per-stage durations | ✓ |
| Timing decorators on Stage.run() | Lightweight but slightly invasive | |
| Timing in FusionGroupIterationStage only | Per-group only, localized | |

**User's choice:** Use existing log timestamps.

### Timing Output (follow-up)

| Option | Description | Selected |
|--------|-------------|----------|
| In the YAML summary | Extend main_stream_co.py YAML output with timing | ✓ |
| Separate utility | Standalone log parser | |

**User's choice:** In the YAML summary. Extends Phase 24 D-04 bottleneck analysis intent.

---

## Claude's Discretion

- Log parsing approach
- Exact YAML timing format
- Helper function location
- Test file location (extend test_resnet_patterns.py)

## Deferred Ideas

None — final phase of v1.6.
