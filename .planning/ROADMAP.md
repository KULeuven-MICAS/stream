# Roadmap: Stream AIE — TETRA Constraint Optimization

## Milestones

- ✅ **v1.0 OR-Tools TETRA Backend** - Phases 1-4 (shipped 2026-05-07)
- ✅ **v1.1 Selective Constraints** - Phases 5-8 (shipped 2026-05-08)
- ✅ **v1.2 Codebase Documentation** - Phases 9-14 (shipped 2026-05-10)
- ✅ **v1.3 MCP Server & Intermediate Representations** - Phases 15-18 (shipped 2026-05-11)
- ✅ **v1.4 Robust Non-AIE Support** - Phases 19-21 (shipped 2026-05-11)
- 🚧 **v1.5 ResNet18 TPU CO Flow** - Phases 22-24 (in progress)

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

### 🚧 v1.5 ResNet18 TPU CO Flow (In Progress)

**Milestone Goal:** Run ResNet18 end-to-end through the CO pipeline on TPU hardware with auto-generated mapping.

- [x] **Phase 22: ONNX Parser Completions** - Fix ConvParser bias crash, add shape inference, register Add/Relu/Pool ops so all 49 ResNet18 nodes parse (completed 2026-05-11)
- [x] **Phase 23: Generic Mapping Generator** - Auto-infer fused_groups, allocations, and tilings from workload+hardware; wire stage into pipeline; fix FMT-05 YAML validation (completed 2026-05-11)
- [ ] **Phase 24: ResNet18 End-to-End Flow** - Fix fan-out transfer handling, run ResNet18 CO on TPU, verify main_stream_co.py produces output

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
- [ ] 22-03-PLAN.md — New parsers (Add, MaxPool, GlobalAveragePool, FusionEdge) + ResNet18 test

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

### Phase 24: ResNet18 End-to-End Flow
**Goal**: ResNet18 runs fully through the CO pipeline on TPU hardware and main_stream_co.py produces a valid allocation result
**Depends on**: Phase 23
**Requirements**: RES-01, RES-02, RES-03
**Success Criteria** (what must be TRUE):
  1. The CO pipeline completes for ResNet18 on TPU hardware and the SteadyStateScheduler returns a positive latency value
  2. All 8 fan-out transfer points in ResNet18 are handled without index errors (multi-destination tensors routed correctly)
  3. `python main_stream_co.py` with the ResNet18 workload and auto-generated mapping exits without error and prints allocation output
**Plans:** 1/2 plans executed
Plans:
- [ ] 24-01-PLAN.md — Core pipeline bug fixes (inverse_permutation + fan-out tiling)
- [x] 24-02-PLAN.md — ZigZag estimation fixes + main_stream_co.py integration + YAML output + test verification

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
| 22. ONNX Parser Completions | v1.5 | 2/3 | Complete    | 2026-05-11 |
| 23. Generic Mapping Generator | v1.5 | 3/3 | Complete    | 2026-05-11 |
| 24. ResNet18 End-to-End Flow | v1.5 | 1/2 | In Progress|  |
