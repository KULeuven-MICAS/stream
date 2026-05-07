# Requirements: Stream AIE — Selective Constraints

**Defined:** 2026-05-07
**Core Value:** Enable users to explore TETRA design space by toggling hardware resource constraints

## v1.1 Requirements

Requirements for selective constraint toggling. Each maps to roadmap phases.

### Constraint Selection

- [x] **SEL-01**: ConstraintSelection frozen dataclass with 4 bool fields (memory_capacity, object_fifo_depth, buffer_descriptors, dma_channels), all defaulting to True
- [ ] **SEL-02**: TransferAndTensorAllocator skips disabled constraint groups via if-guards in _create_constraints()
- [ ] **SEL-03**: DMA toggle skips only context.add_dma_usage_constraints() dispatch, keeps accounting variables for objective
- [ ] **SEL-04**: Objective conditionally excludes DMA terms when dma_channels=False
- [x] **SEL-05**: __post_init__ emits WARNING for nonsensical constraint combinations (e.g. memory off + FIFO on)

### Pipeline Integration

- [ ] **PIPE-01**: constraint_selection parameter threads from API through StageContext to both allocators

### User Interface

- [ ] **UI-01**: optimize_allocation_co() and optimize_mapping() accept constraint_selection kwarg
- [ ] **UI-02**: Main scripts accept --disable-constraints CLI flag parsing into ConstraintSelection

### Verification

- [ ] **TEST-01**: Tight-instance tests per constraint group (infeasibility-flip on toggle)
- [ ] **TEST-02**: Cross-backend parity (Gurobi and OR-Tools agree within tolerance with same constraint selection)

## v1.0 Requirements (Validated)

All v1.0 requirements were validated in the previous milestone:
- ABS-01 through ABS-05 (Solver abstraction)
- ORT-01, ORT-02, ORT-03 (OR-Tools implementation)
- VER-01, VER-02, VER-03 (Verification & config)

## Out of Scope

| Feature | Reason |
|---------|--------|
| Toggling structural constraints (link contention, reuse, slot latency) | Model-validity constraints — disabling breaks the formulation |
| DSE sweep utility (all 16 combinations) | Nice-to-have, defer to v1.2 |
| Per-constraint infeasibility diagnosis (N+1 retry) | Advanced diagnostic, defer to v1.2 |
| Custom resource limit overrides (e.g. set memory cap to 50%) | Different feature — modifying constraints, not toggling |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| SEL-01 | Phase 5 | Complete |
| SEL-02 | Phase 5 | Pending |
| SEL-03 | Phase 5 | Pending |
| SEL-04 | Phase 5 | Pending |
| SEL-05 | Phase 5 | Complete |
| PIPE-01 | Phase 6 | Pending |
| UI-01 | Phase 6 | Pending |
| UI-02 | Phase 6 | Pending |
| TEST-01 | Phase 7 | Pending |
| TEST-02 | Phase 7 | Pending |

**Coverage:**
- v1.1 requirements: 10 total
- Mapped to phases: 10
- Unmapped: 0

---
*Requirements defined: 2026-05-07*
*Last updated: 2026-05-07*
