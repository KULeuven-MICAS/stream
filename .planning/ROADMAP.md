# Roadmap: Stream AIE — TETRA Constraint Optimization

## Milestones

- ✅ **v1.0 OR-Tools TETRA Backend** - Phases 1-4 (shipped 2026-05-07)
- ✅ **v1.1 Selective Constraints** - Phases 5-8 (shipped 2026-05-08)
- ✅ **v1.2 Codebase Documentation** - Phases 9-14 (shipped 2026-05-10)
- ✅ **v1.3 MCP Server & Intermediate Representations** - Phases 15-18 (shipped 2026-05-11)
- ✅ **v1.4 Robust Non-AIE Support** - Phases 19-21 (shipped 2026-05-11)
- ✅ **v1.5 Multi-Group CO Pipeline** - Phases 22-24 (shipped 2026-05-14)
- 🚧 **v1.6 ResNet18 Full Workload** - Phases 25-28 (in progress)

## Phases

<details>
<summary>✅ v1.0 OR-Tools TETRA Backend (Phases 1-4) - SHIPPED 2026-05-07</summary>

- [x] Phase 1: Solver Facade (4/4 plans) — completed 2026-05-07
- [x] Phase 2: ORToolsBackend (1/2 plans) — completed 2026-05-07
- [x] Phase 3: Linearized Division — completed 2026-05-07
- [x] Phase 4: Verification & Config (2/2 plans) — completed 2026-05-07

</details>

<details>
<summary>✅ v1.1 Selective Constraints (Phases 5-8) - SHIPPED 2026-05-08</summary>

- [x] Phase 5: ConstraintSelection Dataclass (2/2 plans) — completed 2026-05-08
- [x] Phase 6: Pipeline & API Surface (2/2 plans) — completed 2026-05-08
- [x] Phase 7: End-to-End Validation (1/1 plans) — completed 2026-05-08
- [x] Phase 8: Constraint Toggle Study Script (1/1 plans) — completed 2026-05-08

</details>

<details>
<summary>✅ v1.2 Codebase Documentation (Phases 9-14) - SHIPPED 2026-05-10</summary>

- [x] Phase 9: Dead Code Cleanup (1/1 plans) — completed 2026-05-09
- [x] Phase 10: CLAUDE.md & Skill Scaffolding (2/2 plans) — completed 2026-05-09
- [x] Phase 11: Solver System Skills (1/1 plans) — completed 2026-05-09
- [x] Phase 12: Pipeline Skills (1/1 plans) — completed 2026-05-09
- [x] Phase 13: MILP & Constraint Skills (1/1 plans) — completed 2026-05-10
- [x] Phase 14: API & Testing Skills (1/1 plans) — completed 2026-05-10

</details>

<details>
<summary>✅ v1.3 MCP Server & Intermediate Representations (Phases 15-18) - SHIPPED 2026-05-11</summary>

- [x] Phase 15: Pre-flight Cleanup (2/2 plans) — completed 2026-05-10
- [x] Phase 16: IR Models (2/2 plans) — completed 2026-05-10
- [x] Phase 17: MCP Server Skeleton (2/2 plans) — completed 2026-05-10
- [x] Phase 18: MCP Tools (2/2 plans) — completed 2026-05-10

</details>

<details>
<summary>✅ v1.4 Robust Non-AIE Support (Phases 19-21) - SHIPPED 2026-05-11</summary>

- [x] Phase 19: GA Removal (1/1 plans) — completed 2026-05-11
- [x] Phase 20: Mapping Format Fixes (1/1 plans) — completed 2026-05-11
- [x] Phase 21: TPU End-to-End Test (1/1 plans) — completed 2026-05-11

</details>

### ✅ v1.5 Multi-Group CO Pipeline (Shipped 2026-05-14)

**Milestone Goal:** Build and verify the multi-group CO pipeline infrastructure — generic mapping generation, fusion group iteration, and multi-destination transfer handling — using fast synthetic workloads.

- [x] **Phase 22: ONNX Parser Completions** - Fix ConvParser bias crash, add shape inference, register Add/Relu/Pool ops so all 49 ResNet18 nodes parse (completed 2026-05-11)
- [x] **Phase 23: Generic Mapping Generator** - Auto-infer fused_groups, allocations, and tilings from workload+hardware; wire stage into pipeline; fix FMT-05 YAML validation (completed 2026-05-11)
- [x] **Phase 24: Multi-Group Pipeline Integration** - Fix pipeline bugs (inverse_permutation, fan-out, ZigZag), verify multi-group CO with synthetic workload, wire main_stream_co.py generic entry point (completed 2026-05-14)

### ⬚ v1.6 ResNet18 Full Workload (Planned)

**Milestone Goal:** Progressively verify ResNet18 sub-graph patterns through the CO pipeline, then run the complete workload end-to-end.

- [x] **Phase 25: ResNet18 Sub-Graph Patterns** - Test key ResNet18 patterns (stride-2 conv, residual skip connections, pooling→flatten boundary) as isolated multi-node sub-graphs through the full CO pipeline (completed 2026-05-14)
- [ ] **Phase 26: Post-Transfer Dimension Invariant** - Debug and fix why unique_dimensions() RREF produces different z-variable sizes after transfer-graph construction; remove spatial unrolling band-aid; filter Reshape shape tensor from data-flow parsing
- [ ] **Phase 27: ResNet18 Fusion Strategy** - Implement smarter fusion group splitting (bounded aperture, memory-aware group sizing) so ResNet18 produces manageable groups instead of one 47-node group
- [ ] **Phase 28: ResNet18 Full Workload E2E** - Run the complete ResNet18 ONNX through the CO pipeline end-to-end, verify positive latency, confirm main_stream_co.py produces valid YAML summary

## Phase Details

### Phase 22: ONNX Parser Completions
**Goal**: All ResNet18 ONNX nodes parse without error into valid ComputationNode objects
**Depends on**: Phase 21 (TPU E2E baseline)
**Requirements**: PARSE-01, PARSE-02, PARSE-03, PARSE-04, PARSE-05
**Success Criteria** (what must be TRUE):
  1. A 3-input Conv node (weight + bias) produces a ComputationNode without assertion error
  2. Intermediate tensor shapes are available after parsing (shape inference runs before node parsing)
  3. Add and Relu nodes each produce a valid ComputationNode via registered parsers
  4. MaxPool and GlobalAveragePool nodes produce valid ComputationNodes via rewritten PoolingParser
  5. Parsing the ResNet18 ONNX model completes with all 49 nodes accounted for (ComputationNode or passthrough)
**Plans:** 3/3 plans complete
Plans:
- [x] 22-01-PLAN.md — FusionEdge node type + Workload integration
- [x] 22-02-PLAN.md — ConvParser/GemmParser bias fix + shape inference + Relu registration
- [x] 22-03-PLAN.md — New parsers (Add, MaxPool, GlobalAveragePool, FusionEdge) + ResNet18 test

### Phase 23: Generic Mapping Generator
**Goal**: A deterministic GenericMappingGenerator produces a schema-valid mapping for any workload+hardware pair and is wired into the pipeline
**Depends on**: Phase 22
**Requirements**: MAP-01, MAP-02, MAP-03, MAP-04, FMT-05
**Success Criteria** (what must be TRUE):
  1. `GenericMappingGenerator` produces a Mapping with core_allocation, inter_core_tiling, fused_groups, and intra_core_tiling populated from workload+hardware inputs
  2. The generated mapping contains exactly one FusedGroup with non-empty intra_core_tiling entries (no assertion error from determine_fusion_splits)
  3. `MappingValidator` accepts the generated mapping without errors (nested-list format, all nodes covered)
  4. `GenericMappingGenerationStage` executes between ONNXModelParserStage and MappingParserStage in the pipeline
  5. The TPU mapping YAML validates against the current schema (FMT-05)
**Plans:** 3/3 plans complete
Plans:
- [x] 23-01-PLAN.md — Hardware operator_types + GenericMappingGenerator class
- [x] 23-02-PLAN.md — Pipeline stages (GenericMappingGenerationStage + FusionGroupIterationStage) + api.py wiring
- [x] 23-03-PLAN.md — Test suite (MAP-01 through MAP-04, FMT-05 validation)

### Phase 24: Multi-Group Pipeline Integration
**Goal**: The multi-group CO pipeline completes end-to-end for a synthetic multi-group workload (Conv->Relu->Flatten->Gemm producing 2 fusion groups) and main_stream_co.py provides both manual and generic entry points with YAML summary output
**Depends on**: Phase 23
**Requirements**: RES-01, RES-02, RES-03
**Success Criteria** (what must be TRUE):
  1. The CO pipeline completes for a multi-group workload with FusionEdge splitting and the SteadyStateScheduler returns positive per-group latencies
  2. Fan-out transfer handling uses `get_node_with_largest_resource_allocation` for multi-destination tensors (no `dsts[0]` assumption)
  3. `main_stream_co.py` has both `optimize_allocation_co_with_mapping` (manual) and `optimize_allocation_co_generic` (auto) entry points and prints YAML summary with per-group latency breakdown
**Plans:** 2/2 plans complete
Plans:
- [x] 24-01-PLAN.md — Core pipeline bug fixes (inverse_permutation + fan-out tiling)
- [x] 24-02-PLAN.md — ZigZag estimation fixes + synthetic workload builder + multi-group test + main_stream_co.py integration

### Phase 25: ResNet18 Sub-Graph Patterns
**Goal**: Key ResNet18 structural patterns (stride-2 conv, residual add with fan-out, pooling→flatten transitions) each run successfully through the full CO pipeline as isolated multi-node sub-graphs
**Depends on**: Phase 24
**Requirements**: RNET-01, RNET-02, RNET-03
**Success Criteria** (what must be TRUE):
  1. A stride-2 Conv + BN + ReLU sub-graph completes CO allocation with positive latency
  2. A residual block sub-graph (two conv paths + Add with fan-out input) completes CO allocation with fan-out transfers correctly routed
  3. A MaxPool → Conv → GlobalAveragePool sub-graph completes on mixed core types (compute + pooling)
**Plans:** 2/2 plans complete
Plans:
- [x] 25-01-PLAN.md — FusionEdgeParser Reshape fix + parametric ResNet18 sub-graph builder (4 patterns)
- [x] 25-02-PLAN.md — Integration tests for all 4 patterns + pipeline fixes (D-07)

### Phase 26: Post-Transfer Dimension Invariant
**Goal**: Diagnose and fix why unique_dimensions() RREF produces different z-variable sizes after transfer-graph construction for fan-out workloads; restore the spatial unrolling assert; filter Reshape shape tensors from data-flow parsing
**Depends on**: Phase 25
**Requirements**: RNET-07, RNET-08
**Success Criteria** (what must be TRUE):
  1. The spatial unrolling assert (`dim_size % spatial_unrolling == 0`) passes for all 4 Phase 25 sub-graph patterns WITHOUT the fallback — dimension sizes are invariant across transfer node insertion
  2. Reshape ONNX shape tensors (INT64 initializers) are not included in the workload's data-flow graph (filtered before or during FusionEdge parsing)
  3. All 191 existing tests pass with the band-aid removed
**Plans:** TBD

### Phase 27: ResNet18 Fusion Strategy
**Goal**: Implement bounded fusion group splitting so ResNet18 produces multiple manageable groups (not one 47-node monolith) with controllable aperture depth
**Depends on**: Phase 26
**Requirements**: RNET-04, RNET-05
**Success Criteria** (what must be TRUE):
  1. A `max_group_depth` parameter limits fusion group size (e.g., max 8 layers per group)
  2. ResNet18 splits into N groups (N > 2) where each group has at most `max_group_depth` computation nodes
  3. The generated per-group mappings all pass MappingValidator
**Plans:** TBD

### Phase 28: ResNet18 Full Workload E2E
**Goal**: The complete ResNet18 ONNX model runs end-to-end through the CO pipeline on TPU hardware with bounded fusion groups and produces a valid YAML summary
**Depends on**: Phase 27
**Requirements**: RNET-06
**Success Criteria** (what must be TRUE):
  1. `optimize_allocation_co_generic` completes for `resnet18.onnx` on TPU hardware with positive total_latency
  2. All fusion groups produce positive per-group latencies in the YAML summary
  3. `python main_stream_co.py` exits without error and prints the full ResNet18 allocation summary
**Plans:** TBD

## Progress

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1. Solver Facade | v1.0 | 4/4 | Complete | 2026-05-07 |
| 2. ORToolsBackend (linear) | v1.0 | 1/2 | Complete | 2026-05-07 |
| 3. Linearized Division | v1.0 | TBD | Complete | 2026-05-07 |
| 4. Verification & Config | v1.0 | 2/2 | Complete | 2026-05-07 |
| 5. ConstraintSelection Dataclass | v1.1 | 2/2 | Complete | 2026-05-08 |
| 6. Pipeline & API Surface | v1.1 | 2/2 | Complete | 2026-05-08 |
| 7. End-to-End Validation | v1.1 | 1/1 | Complete | 2026-05-08 |
| 8. Constraint Toggle Study Script | v1.1 | 1/1 | Complete | 2026-05-08 |
| 9. Dead Code Cleanup | v1.2 | 1/1 | Complete | 2026-05-09 |
| 10. CLAUDE.md & Skill Scaffolding | v1.2 | 2/2 | Complete | 2026-05-09 |
| 11. Solver System Skills | v1.2 | 1/1 | Complete | 2026-05-09 |
| 12. Pipeline Skills | v1.2 | 1/1 | Complete | 2026-05-09 |
| 13. MILP & Constraint Skills | v1.2 | 1/1 | Complete | 2026-05-10 |
| 14. API & Testing Skills | v1.2 | 1/1 | Complete | 2026-05-10 |
| 15. Pre-flight Cleanup | v1.3 | 2/2 | Complete | 2026-05-10 |
| 16. IR Models | v1.3 | 2/2 | Complete | 2026-05-10 |
| 17. MCP Server Skeleton | v1.3 | 2/2 | Complete | 2026-05-10 |
| 18. MCP Tools | v1.3 | 2/2 | Complete | 2026-05-10 |
| 19. GA Removal | v1.4 | 1/1 | Complete | 2026-05-11 |
| 20. Mapping Format Fixes | v1.4 | 1/1 | Complete | 2026-05-11 |
| 21. TPU End-to-End Test | v1.4 | 1/1 | Complete | 2026-05-11 |
| 22. ONNX Parser Completions | v1.5 | 3/3 | Complete | 2026-05-11 |
| 23. Generic Mapping Generator | v1.5 | 3/3 | Complete | 2026-05-11 |
| 24. Multi-Group Pipeline Integration | v1.5 | 2/2 | Complete | 2026-05-14 |
| 25. ResNet18 Sub-Graph Patterns | v1.6 | 2/2 | Complete   | 2026-05-14 |
| 26. Post-Transfer Dimension Invariant | v1.6 | TBD | Not started | - |
| 27. ResNet18 Fusion Strategy | v1.6 | TBD | Not started | - |
| 28. ResNet18 Full Workload E2E | v1.6 | TBD | Not started | - |
