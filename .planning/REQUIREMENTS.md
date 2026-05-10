# Requirements: Stream AIE — MCP Server & Intermediate Representations

**Defined:** 2026-05-10
**Core Value:** Enable AI agents to drive TETRA design space exploration via structured MCP tools with clean, serializable intermediate representations

## v1.3 Requirements

### Pre-flight Cleanup

- [x] **CLEAN-02**: All bare `print()` calls in the MILP solve path are replaced with logger calls so stdout remains clean for MCP stdio transport
- [x] **CLEAN-03**: Missing `get_ir()` methods added to SteadyStateScheduler and Mapping, returning serializable dict representations of allocation results
- [x] **CLEAN-04**: Module-level `logging.basicConfig()` in api.py moved to a callable helper so MCP server can configure logging independently

### IR Design

- [x] **IR-01**: Pydantic BaseModel IR classes (WorkloadIR, AllocationIR, AcceleratorIR) exist with `schema_version` field and produce valid JSON Schema via `.model_json_schema()`
- [x] **IR-02**: Per-persona IR views available: algorithmic (latency/objective focus), hardware (per-core resource utilization), compiler (node-to-core mapping + transfer routing)

### MCP Server

- [x] **MCP-01**: FastMCP server with STDIO transport boots in under 1.5s, registers tools discoverable by Claude Code, and uses lifespan-based state management
- [x] **MCP-02**: `run_optimization` tool uses async job pattern — returns job ID immediately, results retrievable via polling tool (handles 2-15 min solves without timeout)
- [x] **MCP-03**: Experiment IDs are content-addressed (hash of hardware + workload + mapping + backend + constraints) enabling deterministic cache hits

### Tools

- [x] **TOOL-01**: `run_optimization` tool accepts workload path, hardware YAML, backend, and constraint selection — launches async MILP solve and returns job ID
- [ ] **TOOL-02**: `get_workload_ir` and `get_accelerator_ir` tools return parsed workload DAG and hardware topology as structured JSON matching Pydantic IR schemas
- [x] **TOOL-03**: `get_allocation_ir` and `get_solve_stats` tools return optimization results (tensor placements, latencies, solve statistics) as structured JSON

## Out of Scope

| Feature | Reason |
|---------|--------|
| HTTP/SSE transport | MCP server runs as local subprocess; network transport is a separate concern |
| Authentication / API keys | Local subprocess model, no multi-user scenarios |
| compare_backends tool | Can be composed by calling run_optimization twice; defer to v1.4 |
| Transfer path traceability in IR | MulticastPathPlan serialization is complex; P2 feature |
| Per-core resource breakdown IR | Requires deep TTA internals work; P2 feature |
| MCP Resources / Prompts primitives | Tools are the right primitive for active computation |
| Web platform integration | Existing web demo is a separate architecture |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| CLEAN-02 | Phase 15 | Complete |
| CLEAN-03 | Phase 15 | Complete |
| CLEAN-04 | Phase 15 | Complete |
| IR-01 | Phase 16 | Complete |
| IR-02 | Phase 16 | Complete |
| MCP-01 | Phase 17 | Complete |
| MCP-02 | Phase 17 | Complete |
| MCP-03 | Phase 17 | Complete |
| TOOL-01 | Phase 18 | Complete |
| TOOL-02 | Phase 18 | Pending |
| TOOL-03 | Phase 18 | Complete |

**Coverage:**
- v1.3 requirements: 11 total
- Mapped to phases: 11 (100%)
- Unmapped: 0

---
*Requirements defined: 2026-05-10*
*Last updated: 2026-05-10 — traceability mapped after roadmap creation*
