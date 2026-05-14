# Phase 27: ResNet18 Fusion Strategy - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-14
**Phase:** 27-resnet18-fusion-strategy
**Areas discussed:** Split criteria, Split mechanism, Group sizing constraints, GenericMappingGenerator integration

---

## Split Criteria

| Option | Description | Selected |
|--------|-------------|----------|
| After each residual Add+Relu | Split at natural residual block boundaries, ~10 groups | ✓ |
| Fixed max_group_depth | Split when group exceeds N nodes | |
| At dimension changes (stride-2) | Split at spatial dimension change boundaries | |
| You decide | Claude picks | |

**User's choice:** After each residual Add+Relu.

### Front-end Group (follow-up)

| Option | Description | Selected |
|--------|-------------|----------|
| Separate group | Front-end (Conv1→Relu→MaxPool) is its own group | ✓ |
| Merge with layer1.0 | Fewer groups, mixed core allocation | |

**User's choice:** Separate group.

---

## Split Mechanism

| Option | Description | Selected |
|--------|-------------|----------|
| Pre-processing pass inserts Reshape at cut points | Same as Phase 25 DUAL_RESIDUAL pattern | |
| Extend split_fusion_groups() with cut-point parameter | Direct split logic, no synthetic nodes | ✓ |
| Make Add a configurable FusionEdge trigger | Extends FUSION_EDGE_OPS dynamically | |
| You decide | Claude picks | |

**User's choice:** Extend split_fusion_groups() to accept cut-point nodes.

### Cut-point Source (follow-up)

| Option | Description | Selected |
|--------|-------------|----------|
| Separate analyzer function | determine_fusion_cut_points(workload) returns cut-point names | ✓ |
| Caller provides list directly | split_fusion_groups(cut_points=[...]) | |
| You decide | Claude picks | |

**User's choice:** Separate analyzer function. Separates policy from mechanism.

---

## Group Sizing Constraints

| Option | Description | Selected |
|--------|-------------|----------|
| Variable sizes OK | Let heuristic produce naturally-sized groups (3-6 nodes) | ✓ |
| Add max_group_depth guard | Safety net assert for oversized groups | |
| Enforce roughly equal sizes | Merge/split to balance | |

**User's choice:** Variable sizes OK.

---

## GenericMappingGenerator Integration

| Option | Description | Selected |
|--------|-------------|----------|
| No major changes | Existing per-group generation scales to 10+ groups | |
| Add group-aware optimizations | Inter-group tiling coordination | |
| You decide | Claude evaluates | |

**User's choice:** Free text — No major changes, but: (1) implementation must be readable and modular, (2) should not break other flows, (3) skeleton should allow future extension, (4) update `.claude/skills/` documentation and API specs.

---

## Claude's Discretion

- Internal determine_fusion_cut_points() implementation (Add+Relu pattern detection)
- split_fusion_groups() cut-point parameter handling
- OutEdge/InEdge boundary pair creation for non-FusionEdge cut points
- Which skill files to update

## Deferred Ideas

- max_group_depth as complementary heuristic
- Dimension-change splits
- Inter-group tiling coordination
- Full ResNet18 E2E test (Phase 28)
