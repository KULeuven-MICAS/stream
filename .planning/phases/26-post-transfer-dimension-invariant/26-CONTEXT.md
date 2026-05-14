# Phase 26: Post-Transfer Dimension Invariant - Context

**Gathered:** 2026-05-14
**Status:** Ready for planning

<domain>
## Phase Boundary

Debug and fix a pipeline invariant violation: dimension sizes (z-variable sizes from `unique_dimensions()` RREF) should NOT change after transfer-graph construction in `build_transfer_graph()`. The current fallback in `_insert_kernel_iteration_variables()` (fall back to `spatial_unrolling=1` when `dim_size % spatial_unrolling != 0`) is a band-aid that masks the real bug. This phase removes the band-aid, finds the root cause, and fixes it properly. Also filters Reshape shape tensors from data-flow parsing.

</domain>

<decisions>
## Implementation Decisions

### Root Cause Investigation
- **D-01:** Debug-first approach. Add targeted logging to the BASIC_RESIDUAL test case to capture the dimension decomposition (`unique_dimensions()` z-variable sizes) before and after `build_transfer_graph()`. Identify which specific transfer node(s) cause the divergence and WHY. The hypothesis that identity AffineMaps cause RREF pivot changes needs concrete evidence — if maps are truly identity, sizes should be invariant regardless of pivot selection.
- **D-02:** Autonomous debug with checkpoints. Claude investigates autonomously but pauses at key findings to present the before/after dimension decomposition diffs. User approves direction before any fix is applied.

### Reshape Shape Tensor Filtering
- **D-03:** Filter non-data-type initializers (INT64, INT32 shape/axis constants) in ONNXModelParser during initializer parsing — don't add them to `name_to_tensor_dict` at all. They're ONNX op metadata, not data-flow tensors. This is the cleanest approach, preventing them from ever entering the workload graph. Remove the `i64` type mapping that was added as a workaround.

### Band-Aid Removal
- **D-04:** After the root cause is fixed, restore the original `assert rem == 0` in `_insert_kernel_iteration_variables()` and remove the fallback-to-1 logic. The assert is correct — it catches a real invariant.

### Claude's Discretion
- Specific logging locations and format for the debug investigation
- Whether the fix is in `build_transfer_graph()`, `unique_dimensions()`, or the TransferNode construction
- Whether additional test assertions are needed beyond the restored assert

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Transfer-Graph Construction
- `stream/cost_model/steady_state_scheduler.py` §`build_transfer_graph()` (line 271) — Main entry point for transfer node insertion
- `stream/cost_model/steady_state_scheduler.py` §`generate_transfer_node()` (line 519) — Creates TransferNode with `AffineMap.identity(len(tensor.shape))` operand_mapping
- `stream/cost_model/steady_state_scheduler.py` §`update_destination_node_inputs()` (line 327) — Recreates ComputationNodes with updated input tensors

### Dimension Decomposition
- `stream/workload/workload.py` §`unique_dimensions()` (line 299) — RREF-based dimension decomposition producing z-variables
- `stream/workload/workload.py` §`get_dimension_sizes()` (line 237) — Two-pass dimension size inference (direct + probe)
- `stream/workload/workload.py` §`dimension_relations()` — Stacks affine maps from all HasIterationSpace nodes

### Spatial Unrolling (band-aid location)
- `stream/workload/utils.py` §`_insert_kernel_iteration_variables()` (line 109) — Where the band-aid fallback was added; assert to restore

### ONNX Parser (Reshape filtering)
- `stream/parser/onnx/model.py` — Where `name_to_tensor_dict` is populated from initializers
- `stream/parser/onnx/utils.py` — Where `i64` type mapping was added (to be removed)
- `stream/parser/onnx/fusion_edge.py` — FusionEdgeParser (already takes only `input[0]`)

### Test Cases (reproduction)
- `tests/test_resnet_patterns.py` — 4 pattern tests; BASIC_RESIDUAL is the simplest reproduction case
- `stream/inputs/testing/workload/make_resnet_subgraph.py` — Parametric builder for test workloads

</canonical_refs>

<code_context>
## Existing Code Insights

### Key Mechanism
- `build_transfer_graph()` iterates all workload tensors, inserting TransferNode(s) between src/dst
- TransferNode operand_mapping is `AffineMap.identity(len(tensor.shape))` — should add no new dimension constraints
- Destination ComputationNodes are recreated with updated input tensors (new Tensor objects with `_1` suffix names, same shape)
- `Workload(new_nodes.values())` constructs the post-transfer workload
- `unique_dimensions()` recomputes RREF on the full stacked relation matrix including TransferNodes

### The Invariant
- Pre-transfer: `workload.unique_dimensions()` → z-variables with specific sizes
- Post-transfer: `ssw.unique_dimensions()` → z-variables that SHOULD have the same sizes
- The mapping's `inter_core_tiling` and `spatial_unrolling` assume pre-transfer sizes
- If post-transfer sizes differ, `dim_size % spatial_unrolling != 0` fires

### Suspicious Points
- TransferNodes implement `HasIterationSpace` and contribute rows to `dimension_relations()`
- Identity maps are linearly dependent on existing rows, but RREF on a larger matrix may assign different pivot columns
- `update_destination_node_inputs()` creates NEW ComputationNode objects with different input Tensor references — do the affine maps still reference the same dimension positions?

</code_context>

<specifics>
## Specific Ideas

- The user's intuition: "If it adds identity maps, even with a different pivot it should still be the same dimension size in the end." This suggests the bug may NOT be in RREF pivot selection but in something else — perhaps tensor shapes actually changing, or dimension indices shifting.
- Start debugging with BASIC_RESIDUAL (simplest fan-out case: Conv→Relu→Conv→Add with skip connection)
- Compare z-variable sizes before/after `build_transfer_graph()` — if they differ, trace which TransferNode(s) caused the change

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 26-post-transfer-dimension-invariant*
*Context gathered: 2026-05-14*
