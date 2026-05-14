# Stream AIE — TETRA Constraint Optimization

## What This Is

Stream is a multi-core accelerator design space exploration framework with layer-fused scheduling. The TETRA constraint optimization (Transfer and Tensor Allocation) uses MILP to decide tensor placement and transfer paths on multi-core accelerators (AIE, TPU-like). The solver abstraction layer supports multiple backends (Gurobi, OR-Tools GSCIP/HiGHS) behind a unified API. Comprehensive documentation is available as Claude Code skills in `.claude/skills/`.

## Core Value

Enable users to explore the TETRA design space efficiently — selecting solver backends, toggling constraint groups, and understanding the impact of hardware constraints on schedule optimality.

## Current Milestone: v1.6 ResNet18 Full Workload

**Goal:** Progressively verify ResNet18 sub-graph patterns, implement bounded fusion strategy, then run the complete workload end-to-end.

**Target features:**
- ResNet18 structural patterns (stride-2 conv, residual skip, mixed pooling) verified as sub-graphs
- Bounded fusion group splitting with configurable `max_group_depth`
- Full ResNet18 end-to-end CO on TPU hardware with valid multi-group scheduler result
- `main_stream_co.py` produces complete ResNet18 YAML allocation summary

## Shipped

Shipped v1.5 Multi-Group CO Pipeline (2026-05-14). The project now has:
- Generic mapping generation stage with operator_types-aware core selection
- Multi-group FusionGroupIterationStage with per-group latency tracking
- `main_stream_co.py` with manual + generic entry points and YAML summary output
- Synthetic Conv->Relu->Flatten->Gemm test workload for fast multi-group validation
- 4 pipeline bugs fixed: inverse_permutation, fan-out, ZigZag fallback, operand assert
- 187 tests passing

Shipped v1.4 Robust Non-AIE Support (2026-05-11). The project now has:
- Solver abstraction layer with Gurobi and OR-Tools backends, switchable via `--backend` CLI flag
- Selective constraint toggling via `ConstraintSelection` dataclass + `--disable-constraints` CLI flag
- CLAUDE.md navigation hub + 9 skill files across 5 domain groups in `.claude/skills/`
- FastMCP server (`stream/mcp/server.py`) with 6 tools exposing TETRA optimization to AI agents
- Pydantic IR models (`stream/ir/`) for workloads, allocations, and hardware with per-persona views
- TPU (non-AIE) end-to-end CO pipeline verified with proper pytest test
- 195 tests across unit, integration, and regression suites

## Requirements

### Validated

- [x] Solver abstraction layer — Validated in Phase 1: Solver Facade
- [x] OR-Tools TETRA implementation — Validated in Phase 2: ORToolsBackend (MathOpt API, GSCIP/HiGHS)
- [x] Backend switching configuration — Validated in Phase 4: --backend CLI arg, pipeline wiring, default OR-Tools
- [x] Cross-backend verification — Validated in Phase 4: tests/verify_backends.py, SolveStats dataclass
- [x] ConstraintSelection dataclass — Validated in Phase 5: frozen dataclass with 4 bool fields, TTA guards
- [x] Selective constraint toggling — Validated in Phase 6: pipeline threading complete
- [x] CLI --disable-constraints flag — Validated in Phase 6: all 4 main scripts
- [x] Integration with both solver backends — Validated in Phase 7: cross-backend parity confirmed
- [x] CLAUDE.md navigation hub — Validated in Phase 10: overview, directory tree, entry points, conventions, skills index
- [x] Skill directory scaffolding — Validated in Phase 10: 4 domain groups with SKILL.md triggers
- [x] Solver system documentation — Validated in Phase 11: solver-backends.md, constraint-selection.md
- [x] Pipeline documentation — Validated in Phase 12: pipeline-stages.md, stage-execution.md
- [x] MILP & constraint documentation — Validated in Phase 13: milp-formulation.md, namespace-constraints.md
- [x] API & testing documentation — Validated in Phase 14: api-reference.md, testing-patterns.md
- [x] Stdout cleanup for MCP — Validated in Phase 15: print→logger, configure_logging() helper
- [x] Serializable get_ir() methods — Validated in Phase 15: SteadyStateScheduler + Mapping
- [x] Pydantic IR models — Validated in Phase 16: WorkloadIR, AcceleratorIR, AllocationIR with persona views
- [x] FastMCP server — Validated in Phase 17: STDIO transport, async job pattern, content-addressed IDs
- [x] MCP tools — Validated in Phase 18: 6 tools (run/poll optimization, 4 inspection)
- [x] GA code removal — Validated in Phase 19: all GA files, imports, deap dependency removed
- [x] Mapping format fixes — Validated in Phase 20: nested-list format, NameError/IndexError/rsplit bugs fixed, 194 tests pass
- [x] Generic mapping generation stage — Validated in Phase 23: GenericMappingGenerator, operator_types-aware core selection, memory-aware tiling
- [x] Multi-group CO pipeline — Validated in Phase 24: FusionGroupIterationStage, group_latencies, synthetic workload test
- [x] `main_stream_co.py` with generic entry point — Validated in Phase 24: optimize_allocation_co_generic + YAML summary
- [x] TPU mapping YAML schema validation (FMT-05) — Validated in Phase 23: MappingValidator accepts generated YAML

### Active (v1.6)

- [ ] ResNet18 sub-graph patterns verified through CO pipeline (stride-2 conv, residual, pooling)
- [ ] Bounded fusion group splitting with configurable max_group_depth
- [ ] Full ResNet18 end-to-end CO on TPU hardware with multi-group scheduling
- [ ] `main_stream_co.py` produces complete ResNet18 YAML allocation summary

### Out of Scope

- Full removal of gurobipy — coexistence first, migration later
- Toggling structural constraints (link contention, reuse forcing, slot latency) — model-validity constraints
- OR-Tools CP-SAT solver — using linear solver for MILP parity
- Auto-generated API docs (Sphinx/mkdocs) — documentation lives in AI-friendly skills
- User tutorials / getting started guide — separate concern from reference documentation

## Context

- TETRA uses a solver abstraction layer (`stream/opt/solver/solver.py`) with SolverModel ABC, GurobiBackend, and ORToolsBackend
- All CO files use the facade — no direct gurobipy imports outside the backend
- Gurobipy requires a commercial license; OR-Tools is open-source (Apache 2.0)
- GurobiBackend uses `addGenConstrNL` for division encoding; linear-only backends use piecewise linearization
- Documentation is structured as `.claude/skills/` with SKILL.md triggers for AI agent auto-discovery
- 195 tests across unit, integration, and regression suites

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Abstraction at model level | Wrap Model, Var, Constraint creation rather than expression trees | ✓ Good (Phase 1) |
| GurobiBackend uses addGenConstrNL for division | Exact NL constraint preserves identical results | ✓ Good (Phase 1) |
| supports_nonlinear dispatch | GurobiBackend routes through addGenConstrNL; linear backends use piecewise enumeration | ✓ Good (Phase 1) |
| ConstraintSelection as frozen dataclass | Immutable config, clear defaults, nonsensical-combo warnings | ✓ Good (Phase 5) |
| Documentation as .claude/skills/ | AI agent auto-discovery via SKILL.md triggers, self-contained files | ✓ Good (Phase 10) |
| Conceptual guide style (no code examples) | Avoids staleness, keeps skill files shorter, readers go to source | ✓ Good (Phase 11) |
| 4 domain groups matching phases | optimization, pipeline, constraints, api-testing | ✓ Good (Phase 10) |
| FastMCP with STDIO transport | Local subprocess, no auth, ms-latency, auto JSON Schema from types | ✓ Good (Phase 17) |
| Async job pattern for MILP solves | asyncio.to_thread avoids blocking; 60s timeout handled via poll | ✓ Good (Phase 17) |
| Content-addressed experiment IDs | SHA-256 of file contents + config; deterministic cache hits | ✓ Good (Phase 17) |
| Pydantic IR with persona views | Methods on IR class (.algorithmic_view() etc.), max code reuse | ✓ Good (Phase 16) |
| Remove GA path entirely | Dead code, unused DEAP dependency, simplified api.py | ✓ Good (Phase 19) |
| Memory-less hardware fallbacks | Single direct transfer when no on-chip memory tiles (TPU) | ✓ Good (Phase 21) |
| Offchip as shim-equivalent | `"offchip"` core type treated like `"shim"` for transfer classification | ✓ Good (Phase 21) |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition:**
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone:**
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-05-11 after v1.5 milestone start*
