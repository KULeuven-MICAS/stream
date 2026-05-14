# Phase 27: ResNet18 Fusion Strategy - Context

**Gathered:** 2026-05-14
**Status:** Ready for planning

<domain>
## Phase Boundary

Implement bounded fusion group splitting so ResNet18 produces multiple manageable groups instead of one 47-node monolith. Add a `determine_fusion_cut_points()` analyzer that identifies residual Add+Relu boundaries, extend `split_fusion_groups()` to accept explicit cut-point nodes, and verify the full ResNet18 workload splits into ~10 groups that each pass through the CO pipeline. Keep the implementation modular and extensible for future workloads. Update documentation.

</domain>

<decisions>
## Implementation Decisions

### Split Criteria
- **D-01:** Split after each residual Add+Relu node. Each `Add` is a synchronization point where both the main path and skip connection have converged. For ResNet18 this produces ~10 groups of 3-6 nodes each, aligned with hardware reuse patterns.
- **D-02:** The front-end path (Conv1→Relu→MaxPool) is its own separate group. MaxPool runs on the pooling core — keeping it separate avoids mixing core allocation strategies.

### Split Mechanism
- **D-03:** Extend `split_fusion_groups()` to accept an optional list of cut-point node names. When provided, it splits after those nodes in addition to the existing FusionEdge splits. The split logic creates OutEdge/InEdge boundary pairs at cut points (same as FusionEdge boundaries). No synthetic Reshape nodes inserted — cleaner graph.
- **D-04:** A separate analyzer function `determine_fusion_cut_points(workload) -> list[str]` walks the graph and returns cut-point node names based on the "split after Add+Relu" heuristic. Separates policy (where to split) from mechanism (how to split).
- **D-05:** `GenericMappingGenerationStage` calls `determine_fusion_cut_points()`, then passes the result to `split_fusion_groups(cut_points=...)`. The existing pipeline (FusionGroupIterationStage → per-group inner pipeline) handles the rest.

### Group Sizing
- **D-06:** Variable-sized groups are acceptable. Let the residual boundary heuristic produce naturally-sized groups (3-6 nodes). No artificial merging, splitting, or max_group_depth constraint.

### GenericMappingGenerator Integration
- **D-07:** No major changes to GenericMappingGenerator. It already handles per-sub-workload mapping generation with operator_types-aware core selection. 10 groups means 10 independent mapping YAMLs — the existing pipeline scales naturally.

### Extensibility & Documentation
- **D-08:** The fusion strategy implementation should be modular and extensible. `determine_fusion_cut_points()` is currently only used for the ResNet example, but the skeleton should allow extension to other heuristics in the future (e.g., max_group_depth, dimension-change splits). It should NOT break existing flows (2-conv, Conv-Relu-Flatten-Gemm, etc.).
- **D-09:** Update `.claude/skills/` documentation to reflect the fusion strategy functionality. Update API specs in the skills to cover the new `determine_fusion_cut_points()` function and the extended `split_fusion_groups()` signature.

### Claude's Discretion
- Internal implementation of `determine_fusion_cut_points()` (how to detect Add+Relu patterns in the graph)
- How `split_fusion_groups()` handles the new cut-point parameter alongside existing FusionEdge logic
- Exact OutEdge/InEdge boundary pair creation for non-FusionEdge cut points
- Which skill files to update and what content to add

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Fusion Group Splitting
- `stream/workload/workload.py` §`split_fusion_groups()` (line 143) — Current split logic, only handles FusionEdge nodes
- `stream/workload/node.py` — FusionEdge, ComputationNode, InEdge, OutEdge node types

### Pipeline Stages
- `stream/stages/generation/generic_mapping_generation.py` — GenericMappingGenerationStage calls split_fusion_groups()
- `stream/stages/generation/fusion_group_iteration.py` — FusionGroupIterationStage iterates per group
- `stream/api.py` — optimize_allocation_co_generic() pipeline wiring

### Mapping Generator
- `stream/mapping/generic_generator.py` — GenericMappingGenerator (per-group YAML generation)

### Phase 25 Sub-Graph Tests
- `tests/test_resnet_patterns.py` — 4 pattern tests including DUAL_RESIDUAL (2-group split)
- `stream/inputs/testing/workload/make_resnet_subgraph.py` — Parametric builder

### ResNet18 Model
- `stream/inputs/examples/workload/resnet18.onnx` — Full ResNet18 (20 Conv, 17 Relu, 8 Add, 1 MaxPool, 1 GlobalAveragePool, 1 Flatten, 1 Gemm)

### Skills Documentation
- `.claude/skills/api-testing/SKILL.md` — API reference trigger
- `.claude/skills/pipeline/SKILL.md` — Pipeline stages documentation trigger
- `.claude/skills/` — All skill directories to check for relevance

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `split_fusion_groups()` already creates OutEdge/InEdge boundary pairs at FusionEdge nodes — same pattern needed for explicit cut points
- `FusionGroupIterationStage` iterates per group and aggregates latencies — works unchanged with more groups
- `GenericMappingGenerator.generate_all_groups()` produces per-group YAML — scales to 10+ groups

### Established Patterns
- FusionEdge-based splitting: walk graph, find FusionEdge nodes, create OutEdge (preceding group) + InEdge (following group) with the FusionEdge's input/output tensors
- InEdge duplication: when an InEdge (model input or initializer) is consumed by nodes in multiple groups, it's duplicated into each group
- `Workload(nodes)` constructor: builds the graph from a node list, preserving edge connectivity

### Integration Points
- `GenericMappingGenerationStage.run()` calls `split_fusion_groups()` — add `determine_fusion_cut_points()` call before it
- `split_fusion_groups()` signature needs an optional `cut_points` parameter
- No changes to FusionGroupIterationStage or downstream pipeline stages

</code_context>

<specifics>
## Specific Ideas

- The "split after Add+Relu" heuristic can be implemented as: find all ComputationNode with type=="Add", then check if the successor is a Relu — if so, the Relu is the cut point (split AFTER the Relu, not after the Add, so the Relu is in the preceding group)
- For ResNet18's front-end: MaxPool is the cut point (split after MaxPool, before layer1.0)
- The analyzer should also detect FusionEdge nodes as cut points to maintain backward compatibility
- Keep the `determine_fusion_cut_points()` function in `stream/workload/` (near `split_fusion_groups`) — it's workload analysis, not a pipeline stage
- The user explicitly wants the implementation to be readable, modular, and not break other flows

</specifics>

<deferred>
## Deferred Ideas

- **max_group_depth parameter** — could be added as an alternative/complementary heuristic in `determine_fusion_cut_points()` later
- **Dimension-change splits** (at stride-2 boundaries) — another future heuristic option
- **Inter-group tiling coordination** — adjacent groups sharing spatial dimensions could benefit from coordinated tiling (future optimization)
- **Full ResNet18 E2E test** — Phase 28 scope

</deferred>

---

*Phase: 27-resnet18-fusion-strategy*
*Context gathered: 2026-05-14*
