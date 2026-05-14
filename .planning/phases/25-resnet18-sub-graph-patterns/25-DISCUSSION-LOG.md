# Phase 25: ResNet18 Sub-Graph Patterns - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-14
**Phase:** 25-resnet18-sub-graph-patterns
**Areas discussed:** Sub-graph selection, Workload builder approach, Test structure & assertions, Failure handling

---

## Sub-Graph Selection

| Option | Description | Selected |
|--------|-------------|----------|
| Basic residual block | Conv->Relu->Conv->Add with skip connection fan-out | ✓ (part of all) |
| Stride-2 residual with downsample | Conv(s=2)->Relu->Conv + downsample Conv(1x1,s=2)->Add | ✓ (part of all) |
| Front-end path | Conv(7x7,s=2)->Relu->MaxPool(s=2) | ✓ (part of all) |
| All three above | Cover all distinct patterns | ✓ |

**User's choice:** All three, plus a fourth: two back-to-back residual blocks split into two fusion groups (one per block), with the split after the Add operator. User noted this explores automatic cut-point detection at Add nodes for Phase 26.

### Split Method (follow-up)

| Option | Description | Selected |
|--------|-------------|----------|
| Fixed mapping YAML per test | Builder produces both ONNX and companion mapping YAML | ✓ |
| Split after Add heuristic | Implement auto-split now | |
| Manual split_fusion_groups override | Add cut-point parameter to API | |

**User's choice:** Fixed mapping YAML per test. Auto-detection deferred to Phase 26.

---

## Workload Builder Approach

| Option | Description | Selected |
|--------|-------------|----------|
| One parametric builder | Single make_resnet_subgraph() with config enum selecting pattern | ✓ |
| Separate builder per pattern | 4 distinct make_* functions | |
| You decide | Claude picks | |

**User's choice:** One parametric builder. DRY approach with shared Conv/Relu/Add logic.

### Mapping Return (follow-up)

| Option | Description | Selected |
|--------|-------------|----------|
| Builder returns (onnx_path, mapping_path) | Builder generates both ONNX and mapping for multi-group patterns | ✓ |
| Separate mapping builder | Dedicated function for mapping generation | |
| You decide | Claude picks | |

**User's choice:** Builder returns (onnx_path, mapping_path) tuple.

---

## Test Structure & Assertions

| Option | Description | Selected |
|--------|-------------|----------|
| New test_resnet_patterns.py | Dedicated test file in tests/ | ✓ |
| Extend test_generic_mapping.py | Add to existing file | |
| You decide | Claude picks | |

**User's choice:** New dedicated test file.

### Assertions

| Option | Description | Selected |
|--------|-------------|----------|
| Positive per-group latency + total | Basic latency assertions | ✓ |
| Correct group count | Verify expected number of fusion groups | ✓ |
| Core type allocation | Verify pooling ops land on pooling cores | ✓ |
| Fan-out transfer count | Verify expected skip connection transfers | |

**User's choice:** Positive latency + correct group count + core type allocation. Fan-out transfer count not selected.

---

## Failure Handling

| Option | Description | Selected |
|--------|-------------|----------|
| Fix in Phase 25 | Debug and fix all failures in this phase | ✓ |
| Skip with xfail marker | Mark failing tests as xfail, defer to later phases | |
| Hybrid | Fix simple issues, xfail deep ones | |

**User's choice:** Fix all in Phase 25. The goal is to prove patterns work, not document limitations.

---

## Claude's Discretion

- Internal builder implementation (helpers, node creation)
- Exact dimension values for sub-graphs
- How to inspect core type allocation in assertions
- Whether front-end path needs fixed mapping for MaxPool→pooling core

## Deferred Ideas

- Automatic cut-point detection at Add nodes (Phase 26)
- max_group_depth parameter (Phase 26)
- Full ResNet18 E2E test (Phase 27)
