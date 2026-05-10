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
import time
import tempfile
import pathlib

import pytest


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
    """Importing stream.mcp.server completes in under 0.5 seconds (cold-start budget)."""
    import subprocess
    import sys

    start = time.perf_counter()
    result = subprocess.run(
        [sys.executable, "-c", "import stream.mcp.server"],
        capture_output=True,
        text=True,
    )
    elapsed = time.perf_counter() - start

    assert result.returncode == 0, f"Import failed: {result.stderr}"
    assert elapsed < 0.5, (
        f"Import took {elapsed:.3f}s which exceeds the 0.5s budget. "
        "Check for heavy module-level imports (stream.api, solvers, etc.)."
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
    """After run_optimization, poll_optimization returns status 'pending'."""
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
    assert data["status"] == "pending", f"Expected status 'pending', got {data['status']}"


# ---------------------------------------------------------------------------
# test_poll_optimization_not_found
# ---------------------------------------------------------------------------


def test_poll_optimization_not_found() -> None:
    """poll_optimization for an unknown job_id returns status 'not_found'."""
    from fastmcp.client import Client
    from stream.mcp.server import mcp

    async def run() -> dict:
        async with Client(mcp) as client:
            result = await client.call_tool(
                "poll_optimization", {"job_id": "unknownjobid"}
            )
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
# test_stub_tools_return_not_implemented
# ---------------------------------------------------------------------------


def test_stub_tools_return_not_implemented(tmp_path: pathlib.Path) -> None:
    """Stub tools (get_workload_ir, get_accelerator_ir, get_allocation_ir, get_solve_stats)
    return dicts with status 'not_implemented'."""
    from fastmcp.client import Client
    from stream.mcp.server import mcp

    hw = _write_temp_file(tmp_path, "hardware.yaml", b"hw")
    wl = _write_temp_file(tmp_path, "workload.onnx", b"wl")

    async def run() -> list[dict]:
        async with Client(mcp) as client:
            results = []
            r1 = await client.call_tool("get_workload_ir", {"workload": wl})
            results.append(r1.data)
            r2 = await client.call_tool("get_accelerator_ir", {"hardware": hw})
            results.append(r2.data)
            r3 = await client.call_tool("get_allocation_ir", {"job_id": "somejobid1"})
            results.append(r3.data)
            r4 = await client.call_tool("get_solve_stats", {"job_id": "somejobid2"})
            results.append(r4.data)
            return results

    results = asyncio.run(run())
    tool_names = ["get_workload_ir", "get_accelerator_ir", "get_allocation_ir", "get_solve_stats"]
    for tool_name, data in zip(tool_names, results):
        assert data.get("status") == "not_implemented", (
            f"{tool_name} should return status 'not_implemented', got {data}"
        )
