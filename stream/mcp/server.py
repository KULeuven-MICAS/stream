"""stream.mcp.server — FastMCP server with lifespan, 6 tool stubs, and async job pattern.

This module boots the stream-aie MCP server. It is intentionally minimal at module level
to keep cold-start import time under budget (MCP-01). Heavy imports (stream.api, solver
backends) are NOT imported here — they belong in Phase 18 tool handlers only.

No stdout writes — stdout is owned by JSON-RPC (see Phase 15 stdout cleanup).
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastmcp import Context, FastMCP

from stream.mcp.jobs import ServerState, make_experiment_id

# ---------------------------------------------------------------------------
# Lifespan — creates ServerState once per server lifetime (Pattern 1)
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(server: FastMCP) -> Any:
    """Initialize server-wide state on startup; yield to tool handlers; clean up on shutdown."""
    state = ServerState()
    yield {"state": state}


# ---------------------------------------------------------------------------
# FastMCP app instance
# ---------------------------------------------------------------------------

mcp = FastMCP("stream-aie", lifespan=lifespan)


# ---------------------------------------------------------------------------
# State accessor helper (centralizes lifespan context access — Pitfall 3)
# ---------------------------------------------------------------------------


def _get_state(mcp_ctx: Context) -> ServerState:
    """Extract ServerState from the FastMCP lifespan context.

    Centralised here so that a FastMCP API version change only requires a
    single fix rather than touching every tool handler.
    """
    return mcp_ctx.lifespan_context["state"]


# ---------------------------------------------------------------------------
# Tool 1: run_optimization
# ---------------------------------------------------------------------------


@mcp.tool()
async def run_optimization(  # noqa: PLR0913
    hardware: str,
    workload: str,
    mapping: str,
    output_path: str,
    backend: str = "ortools_gscip",
    memory_capacity: bool = True,
    object_fifo_depth: bool = True,
    buffer_descriptors: bool = True,
    dma_channels: bool = True,
    mcp_ctx: Context = None,
) -> dict[str, Any]:
    """Submit a TETRA constraint-optimization job. Returns a job_id immediately.

    The solve runs in the background (Phase 18 will wire in asyncio.to_thread()).
    Poll with poll_optimization(job_id) to retrieve status and results.
    Do not call get_allocation_ir or get_solve_stats until status is 'complete'.

    Args:
        hardware: Path to hardware architecture YAML file.
        workload: Path to workload ONNX file.
        mapping: Path to mapping YAML file.
        output_path: Directory where optimization outputs and pickle cache will be written.
        backend: Solver backend identifier. One of: ortools_gscip, ortools_highs,
            ortools_gurobi, gurobi. Default is 'ortools_gscip'.
        memory_capacity: Enable memory capacity constraints (default True).
        object_fifo_depth: Enable object FIFO depth constraints (default True).
        buffer_descriptors: Enable buffer descriptor constraints (default True).
        dma_channels: Enable DMA channel constraints (default True).
        mcp_ctx: FastMCP context injected automatically by the framework.

    Returns:
        Dict with 'job_id' (12-char hex), 'status' ('pending' or 'complete'), and
        optionally 'cache_hit' (True if result was already cached).
    """
    constraint_dict = {
        "buffer_descriptors": buffer_descriptors,
        "dma_channels": dma_channels,
        "memory_capacity": memory_capacity,
        "object_fifo_depth": object_fifo_depth,
    }
    job_id = make_experiment_id(hardware, workload, mapping, backend, constraint_dict)
    state = _get_state(mcp_ctx)

    # Cache hit: same experiment already completed — return without re-solving (D-08)
    if job_id in state.jobs and state.jobs[job_id]["status"] == "complete":
        return {"job_id": job_id, "status": "complete", "cache_hit": True}

    # Register job as pending
    state.jobs[job_id] = {"status": "pending", "result": None, "error": None}

    # Phase 18 replaces the stub below with:
    #   asyncio.create_task(_run_solve_background(job_id, state, hardware, workload,
    #                                              mapping, output_path, backend,
    #                                              constraint_dict))
    return {"job_id": job_id, "status": "pending", "message": "not implemented yet"}


# ---------------------------------------------------------------------------
# Tool 2: poll_optimization
# ---------------------------------------------------------------------------


@mcp.tool()
async def poll_optimization(
    job_id: str,
    mcp_ctx: Context = None,
) -> dict[str, Any]:
    """Check the status of a previously submitted optimization job.

    Returns the current job state from the in-memory registry. Status values:
    - 'pending': job is queued, solve not yet started
    - 'running': solve is in progress
    - 'complete': result is available (access via get_allocation_ir, get_solve_stats)
    - 'failed': solve encountered an error (check 'error' field)
    - 'not_found': no job with this ID exists in the current server session

    Args:
        job_id: 12-char hex job ID returned by run_optimization.
        mcp_ctx: FastMCP context injected automatically by the framework.

    Returns:
        Dict with 'status' and, when complete, 'result'. When not found,
        returns {'status': 'not_found', 'job_id': job_id}.
    """
    state = _get_state(mcp_ctx)
    job = state.jobs.get(job_id)
    if job is None:
        return {"status": "not_found", "job_id": job_id}
    return job


# ---------------------------------------------------------------------------
# Tool 3: get_workload_ir
# ---------------------------------------------------------------------------


@mcp.tool()
async def get_workload_ir(
    workload: str,
    mcp_ctx: Context = None,
) -> dict[str, Any]:
    """Return the workload DAG as structured JSON matching the WorkloadIR schema.

    Phase 18 will implement this tool by loading the workload ONNX file,
    parsing it through the stream pipeline, and returning a WorkloadIR Pydantic
    model serialized as JSON. The returned structure includes operator nodes,
    tensor edges, layer dimensions, and tiling configurations.

    Args:
        workload: Path to workload ONNX file.
        mcp_ctx: FastMCP context injected automatically by the framework.

    Returns:
        Dict with WorkloadIR fields when implemented. Currently returns
        {'status': 'not_implemented', 'message': ...}.
    """
    return {"status": "not_implemented", "message": "Phase 18 will implement this tool"}


# ---------------------------------------------------------------------------
# Tool 4: get_accelerator_ir
# ---------------------------------------------------------------------------


@mcp.tool()
async def get_accelerator_ir(
    hardware: str,
    mcp_ctx: Context = None,
) -> dict[str, Any]:
    """Return the hardware accelerator model as structured JSON matching the AcceleratorIR schema.

    Phase 18 will implement this tool by loading the hardware YAML, parsing it
    through the stream hardware model, and returning an AcceleratorIR Pydantic model
    serialized as JSON. The returned structure includes cores, memory hierarchy,
    DMA channels, and interconnect topology.

    Args:
        hardware: Path to hardware architecture YAML file.
        mcp_ctx: FastMCP context injected automatically by the framework.

    Returns:
        Dict with AcceleratorIR fields when implemented. Currently returns
        {'status': 'not_implemented', 'message': ...}.
    """
    return {"status": "not_implemented", "message": "Phase 18 will implement this tool"}


# ---------------------------------------------------------------------------
# Tool 5: get_allocation_ir
# ---------------------------------------------------------------------------


@mcp.tool()
async def get_allocation_ir(
    job_id: str,
    mcp_ctx: Context = None,
) -> dict[str, Any]:
    """Return the TETRA allocation result as structured JSON matching the AllocationIR schema.

    Phase 18 will implement this tool by loading the completed optimization result
    (from the pickle cache at {output_path}/{job_id}/ctx.pickle) and returning an
    AllocationIR Pydantic model with three persona views:
    - algorithmic_view: tensor placement per operator
    - hardware_view: per-core allocation and memory usage
    - compiler_view: slot-indexed DMA schedule

    Requires a completed optimization job. Call run_optimization first and wait
    for poll_optimization to return status 'complete'.

    Args:
        job_id: 12-char hex job ID returned by run_optimization.
        mcp_ctx: FastMCP context injected automatically by the framework.

    Returns:
        Dict with AllocationIR fields when implemented. Currently returns
        {'status': 'not_implemented', 'message': ...}.
    """
    return {"status": "not_implemented", "message": "Phase 18 will implement this tool"}


# ---------------------------------------------------------------------------
# Tool 6: get_solve_stats
# ---------------------------------------------------------------------------


@mcp.tool()
async def get_solve_stats(
    job_id: str,
    mcp_ctx: Context = None,
) -> dict[str, Any]:
    """Return MILP solve statistics for a completed optimization job.

    Phase 18 will implement this tool by extracting SolveStats from the pickle
    cache and returning structured metrics including objective value, solve time,
    gap, number of variables, number of constraints, and solver backend used.

    Requires a completed optimization job. Call run_optimization first and wait
    for poll_optimization to return status 'complete'.

    Args:
        job_id: 12-char hex job ID returned by run_optimization.
        mcp_ctx: FastMCP context injected automatically by the framework.

    Returns:
        Dict with SolveStats fields when implemented. Currently returns
        {'status': 'not_implemented', 'message': ...}.
    """
    return {"status": "not_implemented", "message": "Phase 18 will implement this tool"}
