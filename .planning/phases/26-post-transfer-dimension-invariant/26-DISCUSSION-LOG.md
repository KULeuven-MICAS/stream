# Phase 26: Post-Transfer Dimension Invariant - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-14
**Phase:** 26-post-transfer-dimension-invariant
**Areas discussed:** Root cause hypothesis, Reshape shape tensor filtering, Debug approach

---

## Root Cause Hypothesis

| Option | Description | Selected |
|--------|-------------|----------|
| Exclude TransferNodes from dimension_relations() | TransferNodes are pass-through, identity maps add no info | |
| Fix RREF to be order-independent | Normalize RREF output so redundant rows don't change result | |
| Debug first then decide | Investigate actual z-variable divergence, then determine fix | ✓ |

**User's choice:** Free text — "Let's first do some more debugging for a specific case where this occurred. If it adds identity maps, even with a different pivot it should still be the same dimension size in the end?" User's intuition: identity maps should NOT cause size changes, so the root cause may be elsewhere.

---

## Reshape Shape Tensor Filtering

| Option | Description | Selected |
|--------|-------------|----------|
| Filter in ONNXModelParser during initializer parsing | Don't add non-data-type initializers to name_to_tensor_dict | ✓ |
| Filter in FusionEdgeParser only | Already done (takes input[0]), keep i64 as defense | |
| You decide | Claude picks cleanest approach | |

**User's choice:** Filter in ONNXModelParser during initializer parsing. Cleanest — prevents metadata tensors from entering the workload graph.

---

## Debug Approach

| Option | Description | Selected |
|--------|-------------|----------|
| Interactive debug session | Claude adds logging, shares output, user guides next step | |
| Autonomous debug with checkpoints | Claude investigates autonomously, pauses at key findings for approval | ✓ |
| Script-based reproduction | Standalone debug script user runs themselves | |

**User's choice:** Autonomous debug with checkpoints. Claude investigates but pauses to present dimension decomposition diffs before applying any fix.

---

## Claude's Discretion

- Specific logging locations and format
- Whether fix is in build_transfer_graph(), unique_dimensions(), or TransferNode construction
- Additional test assertions beyond the restored assert

## Deferred Ideas

None — discussion stayed within phase scope.
