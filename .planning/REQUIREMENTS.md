# Requirements: Stream AIE — Codebase Documentation

**Defined:** 2026-05-09
**Core Value:** Make stream_aie navigable for both human developers and AI agents via structured documentation as Claude Code skills

## v1.2 Requirements

### Dead Code Cleanup

- [x] **CLEAN-01**: Remove unused stage files (StreamCostModelEvaluationStage, SetFixedAllocationStage, UserDefinedModelParserStage) and any other dead imports referencing them

### CLAUDE.md & Navigation

- [x] **NAV-01**: CLAUDE.md exists at repo root with codebase overview, directory structure, key entry points, and conventions
- [x] **NAV-02**: CLAUDE.md references `.claude/skills/` for topic-specific deep dives

### Solver System

- [ ] **SOLVER-01**: Skill doc covers SolverModel ABC, GurobiBackend, ORToolsBackend, factory pattern, and when to use each
- [ ] **SOLVER-02**: Skill doc covers ConstraintSelection, its relationship to NamespaceConstraints, and which constraints apply to which hardware

### Pipeline Stages

- [ ] **STAGE-01**: Skill doc covers each active pipeline stage (AcceleratorParser, ONNXModelParser, MappingParser, TilingGeneration, CoreCostEstimation, ConstraintOptimizationAllocation, MemoryAccessesEstimation, MappingGeneration) with responsibility, inputs/outputs, and where it fits in the flow
- [ ] **STAGE-02**: Skill doc covers StageContext and the MainStage/LeafStage execution model

### MILP & Constraints

- [ ] **MILP-01**: Skill doc covers TransferAndTensorAllocator MILP structure (variables, constraint groups, objective function, ConstraintSelection guards)
- [ ] **MILP-02**: Skill doc covers NamespaceConstraints pattern, AIE2Constraints, and how hardware-specific constraints are dispatched

### API & Testing

- [ ] **API-01**: Skill doc covers public API (optimize_allocation_co, optimize_mapping), CLI flags, and common usage patterns
- [ ] **API-02**: Skill doc covers testing patterns (unit vs integration, backend patching, study scripts)

### Skill Structure

- [x] **SKILL-01**: Each skill has a SKILL.md with trigger description so AI agents know when to load it
- [x] **SKILL-02**: Skills are self-contained — each readable independently without requiring other skills

## Out of Scope

| Feature | Reason |
|---------|--------|
| Auto-generated API docs (Sphinx/mkdocs) | Focus on AI-agent-friendly skills, not hosted docs |
| User tutorials / getting started guide | Separate concern — this milestone is reference documentation |
| Inline code comments overhaul | Documentation lives in skills, not scattered through source |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| CLEAN-01 | Phase 9 | Complete |
| NAV-01 | Phase 10 | Complete |
| NAV-02 | Phase 10 | Complete |
| SOLVER-01 | Phase 11 | Pending |
| SOLVER-02 | Phase 11 | Pending |
| STAGE-01 | Phase 12 | Pending |
| STAGE-02 | Phase 12 | Pending |
| MILP-01 | Phase 13 | Pending |
| MILP-02 | Phase 13 | Pending |
| API-01 | Phase 14 | Pending |
| API-02 | Phase 14 | Pending |
| SKILL-01 | Phase 10 (cross-cutting: enforced in Phases 11-14) | Complete |
| SKILL-02 | Phase 10 (cross-cutting: enforced in Phases 11-14) | Complete |

**Coverage:**
- v1.2 requirements: 13 total
- Mapped to phases: 13
- Unmapped: 0

---
*Requirements defined: 2026-05-09*
*Last updated: 2026-05-09*
