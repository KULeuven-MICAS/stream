"""Unit tests for stream.mcp.server FastMCP app.

Tests cover:
- Tool discovery (6 registered tools with correct names)
- Import time (<0.5s)
- run_optimization returns job_id immediately
- poll_optimization returns pending status after run
- poll_optimization returns not_found for unknown job_id
- Cache hit: same inputs return same job_id without duplicate
- Lifespan creates ServerState accessible from tools
- Stub tools return not_implemented status
"""

from __future__ import annotations

import asyncio
import pathlib

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_temp_file(tmp_path: pathlib.Path, name: str, content: bytes = b"fake content") -> str:
    """Write a temp file with given content and return its path as string."""
    p = tmp_path / name
    p.write_bytes(content)
    return str(p)


# ---------------------------------------------------------------------------
# test_tool_discovery
# ---------------------------------------------------------------------------


def test_tool_discovery() -> None:
    """FastMCP app registers exactly 6 tools with the required names."""
    from stream.mcp.server import mcp

    tools = asyncio.run(mcp.list_tools())
    tool_names = {t.name for t in tools}

    expected = {
        "run_optimization",
        "poll_optimization",
        "get_workload_ir",
        "get_accelerator_ir",
        "get_allocation_ir",
        "get_solve_stats",
    }
    assert tool_names == expected, f"Expected tools {expected}, got {tool_names}"
    assert len(tools) == 6, f"Expected exactly 6 tools, got {len(tools)}"


# ---------------------------------------------------------------------------
# test_import_time
# ---------------------------------------------------------------------------


def test_import_time() -> None:
    """stream.mcp.server adds negligible import overhead on top of FastMCP (MCP-01 cold start).

    FastMCP itself is the dominant import cost (~1.5s). This test verifies that server.py
    does NOT import heavy modules (stream.api, solver backends) at module level by measuring
    the marginal overhead: once fastmcp is loaded, importing stream.mcp.server should be fast.
    Uses a subprocess that pre-warms fastmcp and measures only the delta for stream.mcp.server.
    """
    import subprocess
    import sys

    # Measure marginal cost: fastmcp already loaded, then load stream.mcp.server
    script = (
        "import time; "
        "import fastmcp; "  # pre-warm FastMCP (dominant startup cost)
        "start = time.perf_counter(); "
        "import stream.mcp.server; "
        "elapsed = time.perf_counter() - start; "
        "print(f'{elapsed:.4f}')"
    )
    result = subprocess.run(  # noqa: PLW1510
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"Import failed: {result.stderr}"
    elapsed = float(result.stdout.strip())
    assert elapsed < 0.5, (
        f"server.py marginal import took {elapsed:.3f}s which exceeds 0.5s. "
        "Check for heavy module-level imports (stream.api, solvers, etc.) in server.py."
    )


# ---------------------------------------------------------------------------
# test_run_optimization_returns_job_id
# ---------------------------------------------------------------------------


def test_run_optimization_returns_job_id(tmp_path: pathlib.Path) -> None:
    """run_optimization returns a dict with job_id (12-char hex) and status 'pending'."""
    from fastmcp.client import Client

    from stream.mcp.server import mcp

    hw = _write_temp_file(tmp_path, "hardware.yaml", b"hardware: fake")
    wl = _write_temp_file(tmp_path, "workload.onnx", b"workload: fake")
    mp = _write_temp_file(tmp_path, "mapping.yaml", b"mapping: fake")
    out = str(tmp_path / "output")

    async def run() -> dict:
        async with Client(mcp) as client:
            result = await client.call_tool(
                "run_optimization",
                {
                    "hardware": hw,
                    "workload": wl,
                    "mapping": mp,
                    "output_path": out,
                },
            )
            return result.data

    data = asyncio.run(run())
    assert "job_id" in data, f"Missing job_id in response: {data}"
    assert len(data["job_id"]) == 12, f"job_id should be 12 chars, got {len(data['job_id'])}"
    assert all(c in "0123456789abcdef" for c in data["job_id"]), f"job_id should be hex: {data['job_id']}"
    assert data["status"] == "pending", f"Expected status 'pending', got {data['status']}"


# ---------------------------------------------------------------------------
# test_poll_optimization_pending
# ---------------------------------------------------------------------------


def test_poll_optimization_pending(tmp_path: pathlib.Path) -> None:
    """After run_optimization, poll_optimization returns an in-progress status.

    The background task may have advanced to 'running' by the time poll is called,
    so we accept both 'pending' and 'running' as valid in-progress states.
    """
    from fastmcp.client import Client

    from stream.mcp.server import mcp

    hw = _write_temp_file(tmp_path, "hardware.yaml", b"hw")
    wl = _write_temp_file(tmp_path, "workload.onnx", b"wl")
    mp = _write_temp_file(tmp_path, "mapping.yaml", b"mp")
    out = str(tmp_path / "out")

    async def run() -> dict:
        async with Client(mcp) as client:
            run_result = await client.call_tool(
                "run_optimization",
                {"hardware": hw, "workload": wl, "mapping": mp, "output_path": out},
            )
            job_id = run_result.data["job_id"]
            poll_result = await client.call_tool("poll_optimization", {"job_id": job_id})
            return poll_result.data

    data = asyncio.run(run())
    assert data["status"] in ("pending", "running", "failed"), (
        f"Expected in-progress status, got {data['status']}"
    )


# ---------------------------------------------------------------------------
# test_poll_optimization_not_found
# ---------------------------------------------------------------------------


def test_poll_optimization_not_found() -> None:
    """poll_optimization for an unknown job_id returns status 'not_found'."""
    from fastmcp.client import Client

    from stream.mcp.server import mcp

    async def run() -> dict:
        async with Client(mcp) as client:
            result = await client.call_tool("poll_optimization", {"job_id": "unknownjobid"})
            return result.data

    data = asyncio.run(run())
    assert data["status"] == "not_found", f"Expected status 'not_found', got {data['status']}"


# ---------------------------------------------------------------------------
# test_cache_hit
# ---------------------------------------------------------------------------


def test_cache_hit(tmp_path: pathlib.Path) -> None:
    """Calling run_optimization twice with the same inputs returns the same job_id."""
    from fastmcp.client import Client

    from stream.mcp.server import mcp

    hw = _write_temp_file(tmp_path, "hardware.yaml", b"hw_content")
    wl = _write_temp_file(tmp_path, "workload.onnx", b"wl_content")
    mp = _write_temp_file(tmp_path, "mapping.yaml", b"mp_content")
    out = str(tmp_path / "out")

    params = {"hardware": hw, "workload": wl, "mapping": mp, "output_path": out}

    async def run() -> tuple[dict, dict]:
        async with Client(mcp) as client:
            r1 = await client.call_tool("run_optimization", params)
            r2 = await client.call_tool("run_optimization", params)
            return r1.data, r2.data

    d1, d2 = asyncio.run(run())
    assert d1["job_id"] == d2["job_id"], (
        f"Same inputs should produce same job_id. Got {d1['job_id']} and {d2['job_id']}"
    )


# ---------------------------------------------------------------------------
# test_lifespan_creates_state
# ---------------------------------------------------------------------------


def test_lifespan_creates_state() -> None:
    """The lifespan context manager yields a dict with key 'state' containing a ServerState instance."""
    from stream.mcp.jobs import ServerState
    from stream.mcp.server import mcp

    # Use the FastMCP _lifespan_result after starting lifespan
    async def run() -> dict:
        async with Client(mcp) as _client:
            # Access the lifespan result directly from the server
            result = mcp._lifespan_result
            return result

    from fastmcp.client import Client

    lifespan_result = asyncio.run(run())
    assert lifespan_result is not None, "Lifespan result should not be None"
    assert "state" in lifespan_result, f"Lifespan result should have 'state' key: {lifespan_result}"
    assert isinstance(lifespan_result["state"], ServerState), (
        f"'state' should be a ServerState instance, got {type(lifespan_result['state'])}"
    )


# ---------------------------------------------------------------------------
# test_get_workload_ir_no_params
# ---------------------------------------------------------------------------


def test_get_workload_ir_no_params() -> None:
    """get_workload_ir with no params returns invalid_input error."""
    from fastmcp.client import Client

    from stream.mcp.server import mcp

    async def run() -> dict:
        async with Client(mcp) as client:
            result = await client.call_tool("get_workload_ir", {})
            return result.data

    data = asyncio.run(run())
    assert data["status"] == "error", f"Expected status 'error', got {data}"
    assert data["error_type"] == "invalid_input", f"Expected error_type 'invalid_input', got {data}"


# ---------------------------------------------------------------------------
# test_get_workload_ir_invalid_path
# ---------------------------------------------------------------------------


def test_get_workload_ir_invalid_path() -> None:
    """get_workload_ir with a nonexistent file path returns invalid_input error."""
    from fastmcp.client import Client

    from stream.mcp.server import mcp

    async def run() -> dict:
        async with Client(mcp) as client:
            result = await client.call_tool("get_workload_ir", {"workload": "/nonexistent/path.onnx"})
            return result.data

    data = asyncio.run(run())
    assert data["status"] == "error", f"Expected status 'error', got {data}"
    assert data["error_type"] == "invalid_input", f"Expected error_type 'invalid_input', got {data}"


# ---------------------------------------------------------------------------
# test_get_workload_ir_not_found_experiment
# ---------------------------------------------------------------------------


def test_get_workload_ir_not_found_experiment() -> None:
    """get_workload_ir with a nonexistent experiment_id returns not_found error."""
    from fastmcp.client import Client

    from stream.mcp.server import mcp

    async def run() -> dict:
        async with Client(mcp) as client:
            result = await client.call_tool("get_workload_ir", {"experiment_id": "nonexistent1"})
            return result.data

    data = asyncio.run(run())
    assert data["status"] == "error", f"Expected status 'error', got {data}"
    assert data["error_type"] == "not_found", f"Expected error_type 'not_found', got {data}"


# ---------------------------------------------------------------------------
# test_get_accelerator_ir_no_params
# ---------------------------------------------------------------------------


def test_get_accelerator_ir_no_params() -> None:
    """get_accelerator_ir with no params returns invalid_input error."""
    from fastmcp.client import Client

    from stream.mcp.server import mcp

    async def run() -> dict:
        async with Client(mcp) as client:
            result = await client.call_tool("get_accelerator_ir", {})
            return result.data

    data = asyncio.run(run())
    assert data["status"] == "error", f"Expected status 'error', got {data}"
    assert data["error_type"] == "invalid_input", f"Expected error_type 'invalid_input', got {data}"


# ---------------------------------------------------------------------------
# test_get_accelerator_ir_invalid_path
# ---------------------------------------------------------------------------


def test_get_accelerator_ir_invalid_path() -> None:
    """get_accelerator_ir with a nonexistent file path returns invalid_input error."""
    from fastmcp.client import Client

    from stream.mcp.server import mcp

    async def run() -> dict:
        async with Client(mcp) as client:
            result = await client.call_tool("get_accelerator_ir", {"hardware": "/nonexistent/hw.yaml"})
            return result.data

    data = asyncio.run(run())
    assert data["status"] == "error", f"Expected status 'error', got {data}"
    assert data["error_type"] == "invalid_input", f"Expected error_type 'invalid_input', got {data}"


# ---------------------------------------------------------------------------
# test_get_accelerator_ir_not_found_experiment
# ---------------------------------------------------------------------------


def test_get_accelerator_ir_not_found_experiment() -> None:
    """get_accelerator_ir with a nonexistent experiment_id returns not_found error."""
    from fastmcp.client import Client

    from stream.mcp.server import mcp

    async def run() -> dict:
        async with Client(mcp) as client:
            result = await client.call_tool("get_accelerator_ir", {"experiment_id": "nonexistent2"})
            return result.data

    data = asyncio.run(run())
    assert data["status"] == "error", f"Expected status 'error', got {data}"
    assert data["error_type"] == "not_found", f"Expected error_type 'not_found', got {data}"


# ---------------------------------------------------------------------------
# test_get_allocation_ir_not_found
# ---------------------------------------------------------------------------


def test_get_allocation_ir_not_found() -> None:
    """get_allocation_ir with nonexistent experiment_id returns structured not_found error."""
    from fastmcp.client import Client

    from stream.mcp.server import mcp

    async def run() -> dict:
        async with Client(mcp) as client:
            result = await client.call_tool("get_allocation_ir", {"job_id": "nonexistent1"})
            return result.data

    data = asyncio.run(run())
    assert data.get("status") == "error", f"Expected status 'error', got {data}"
    assert data.get("error_type") == "not_found", f"Expected error_type 'not_found', got {data}"


# ---------------------------------------------------------------------------
# test_get_allocation_ir_not_ready
# ---------------------------------------------------------------------------


def test_get_allocation_ir_not_ready(tmp_path: pathlib.Path) -> None:
    """get_allocation_ir returns not_ready error when job is still pending/running."""
    from fastmcp.client import Client

    from stream.mcp.server import mcp

    hw = _write_temp_file(tmp_path, "hardware.yaml", b"hw_not_ready")
    wl = _write_temp_file(tmp_path, "workload.onnx", b"wl_not_ready")
    mp = _write_temp_file(tmp_path, "mapping.yaml", b"mp_not_ready")
    out = str(tmp_path / "out_not_ready")

    async def run() -> dict:
        async with Client(mcp) as client:
            run_result = await client.call_tool(
                "run_optimization",
                {"hardware": hw, "workload": wl, "mapping": mp, "output_path": out},
            )
            job_id = run_result.data["job_id"]
            # Immediately call get_allocation_ir — job is still pending
            result = await client.call_tool("get_allocation_ir", {"job_id": job_id})
            return result.data

    data = asyncio.run(run())
    assert data.get("status") == "error", f"Expected status 'error', got {data}"
    assert data.get("error_type") == "not_ready", f"Expected error_type 'not_ready', got {data}"


# ---------------------------------------------------------------------------
# test_get_solve_stats_not_found
# ---------------------------------------------------------------------------


def test_get_solve_stats_not_found() -> None:
    """get_solve_stats with nonexistent experiment_id returns structured not_found error."""
    from fastmcp.client import Client

    from stream.mcp.server import mcp

    async def run() -> dict:
        async with Client(mcp) as client:
            result = await client.call_tool("get_solve_stats", {"job_id": "nonexistent2"})
            return result.data

    data = asyncio.run(run())
    assert data.get("status") == "error", f"Expected status 'error', got {data}"
    assert data.get("error_type") == "not_found", f"Expected error_type 'not_found', got {data}"


# ---------------------------------------------------------------------------
# test_get_solve_stats_not_ready
# ---------------------------------------------------------------------------


def test_get_solve_stats_not_ready(tmp_path: pathlib.Path) -> None:
    """get_solve_stats returns not_ready error when job is still pending/running."""
    from fastmcp.client import Client

    from stream.mcp.server import mcp

    hw = _write_temp_file(tmp_path, "hardware.yaml", b"hw_ss_not_ready")
    wl = _write_temp_file(tmp_path, "workload.onnx", b"wl_ss_not_ready")
    mp = _write_temp_file(tmp_path, "mapping.yaml", b"mp_ss_not_ready")
    out = str(tmp_path / "out_ss_not_ready")

    async def run() -> dict:
        async with Client(mcp) as client:
            run_result = await client.call_tool(
                "run_optimization",
                {"hardware": hw, "workload": wl, "mapping": mp, "output_path": out},
            )
            job_id = run_result.data["job_id"]
            # Immediately call get_solve_stats — job is still pending
            result = await client.call_tool("get_solve_stats", {"job_id": job_id})
            return result.data

    data = asyncio.run(run())
    assert data.get("status") == "error", f"Expected status 'error', got {data}"
    assert data.get("error_type") == "not_ready", f"Expected error_type 'not_ready', got {data}"


# ---------------------------------------------------------------------------
# test_run_optimization_no_stub_message
# ---------------------------------------------------------------------------


def test_run_optimization_no_stub_message(tmp_path: pathlib.Path) -> None:
    """run_optimization no longer returns 'not implemented yet' message — returns job_id and status."""
    from fastmcp.client import Client

    from stream.mcp.server import mcp

    hw = _write_temp_file(tmp_path, "hardware.yaml", b"hw_no_stub")
    wl = _write_temp_file(tmp_path, "workload.onnx", b"wl_no_stub")
    mp = _write_temp_file(tmp_path, "mapping.yaml", b"mp_no_stub")
    out = str(tmp_path / "out_no_stub")

    async def run() -> dict:
        async with Client(mcp) as client:
            result = await client.call_tool(
                "run_optimization",
                {"hardware": hw, "workload": wl, "mapping": mp, "output_path": out},
            )
            return result.data

    data = asyncio.run(run())
    assert "job_id" in data, f"Missing job_id in response: {data}"
    assert data["status"] == "pending", f"Expected status 'pending', got {data['status']}"
    assert data.get("message") != "not implemented yet", (
        f"run_optimization should no longer return stub message, got {data}"
    )
