---
phase: 22-onnx-parser-completions
plan: "01"
subsystem: workload-graph
tags: [FusionEdge, workload, node-types, split-fusion-groups]
dependency_graph:
  requires: []
  provides: [FusionEdge class, split_fusion_groups method, FusionEdge Workload integration]
  affects: [stream/workload/node.py, stream/workload/workload.py]
tech_stack:
  added: []
  patterns: [frozen-dataclass, isinstance-branch, graph-split]
key_files:
  created: []
  modified:
    - stream/workload/node.py
    - stream/workload/workload.py
decisions:
  - "FusionEdge(HasInputs, HasOutputs) — not HasIterationSpace — so dimension_relations skips it automatically (D-03, D-07)"
  - "split_fusion_groups() consumes FusionEdge nodes, inserting OutEdge/InEdge boundary pairs into adjacent groups (D-10)"
  - "with_modified_dimension_sizes passes FusionEdge tensors unchanged via tensor_map.get(name, original) fallback (D-08)"
  - "visualize tensor dim lookup wrapped in try/except for tensors only reachable through FusionEdge nodes"
  - "get_timeslots priority 2 for FusionEdge (between TransferNode=1 and InEdge=3) — no resource bucket, gets earliest slot"
metrics:
  duration_minutes: 2
  completed_date: "2026-05-11"
  tasks_completed: 2
  tasks_total: 2
  files_modified: 2
---

# Phase 22 Plan 01: FusionEdge Node Type and Workload Integration Summary

**One-liner:** FusionEdge frozen dataclass (HasInputs+HasOutputs, not HasIterationSpace) added to node.py, integrated into all six Workload methods, with split_fusion_groups() splitting graphs at FusionEdge boundaries into self-contained sub-workloads.

## What Was Built

### Task 1: FusionEdge class in node.py

Added `FusionEdge(HasInputs, HasOutputs)` frozen dataclass between `OutEdge` and `TransferType` in `stream/workload/node.py`. Fields: `name` (inherited via Node), `inputs` (from HasInputs), `outputs` (from HasOutputs), `op_type: str` (new — preserves original ONNX op name like "Flatten"). Not a subclass of `HasIterationSpace`, so no `operand_mapping` or affine iteration space.

### Task 2: Workload integration + split_fusion_groups in workload.py

Six integration points updated in `stream/workload/workload.py`:

1. **Import**: Added `FusionEdge` to the import from `stream.workload.node`.

2. **`get_fusion_edges()`**: New convenience method mirroring `get_computation_nodes()` and `get_transfer_nodes()`.

3. **`split_fusion_groups()`**: New method. Traverses the workload in lexicographical topological order, assigns each non-FusionEdge node to a group index (incremented at each FusionEdge), then synthesizes `OutEdge` and `InEdge` boundary nodes for each FusionEdge. Returns `list[Workload]` of self-contained sub-workloads. Returns `[self]` when no FusionEdge nodes exist.

4. **`with_modified_dimension_sizes()`**: Added `elif isinstance(node, FusionEdge)` branch before the final `else: raise TypeError`. FusionEdge tensors pass through via `tensor_map.get(name, original)` fallback — if the tensor was not resized (because it's not reachable via an iteration space node), the original tensor is kept.

5. **`visualize()`**: Added `elif isinstance(node, FusionEdge)` branch before the final `else: raise ValueError`. FusionEdge renders as a diamond shape in light purple (`#d9b3ff`), labeled with `name\n[op_type]`. Tensor dimension lookup in the tensor-styling loop wrapped in `try/except (StopIteration, KeyError)` to handle tensors that only flow through FusionEdge (not reachable via `get_iteration_space_nodes()`).

6. **`get_timeslots()` priority function**: Added `FusionEdge` at priority 2 (between `TransferNode=1` and `InEdge=3`). FusionEdge has no resource bucket (`slot_transfers` / `slot_computes`), so it always gets the earliest available slot without contention checks.

7. **`get_ir()`**: Added `if isinstance(node, FusionEdge): node_data["fusion_op_type"] = node.op_type` after the TransferNode check in the nodes_info loop.

**Note:** `dimension_relations()` required no changes — it filters for `HasIterationSpace` edges only, so FusionEdge edges are automatically excluded (D-07).

## Test Results

178 existing tests pass without modification. No regressions.

```
178 passed in 18.05s
```

## Deviations from Plan

None — plan executed exactly as written.

## Known Stubs

None — FusionEdge is fully implemented with all six Workload method branches and split_fusion_groups. No placeholder or hardcoded values.

## Self-Check: PASSED

Files exist:
- stream/workload/node.py — FOUND (contains `class FusionEdge(HasInputs, HasOutputs):`)
- stream/workload/workload.py — FOUND (contains `split_fusion_groups`, `get_fusion_edges`, all six FusionEdge branches)

Commits exist:
- 7e02591 — feat(22-01): add FusionEdge frozen dataclass to node.py
- 314f0a9 — feat(22-01): integrate FusionEdge into all Workload methods + add split_fusion_groups
