"""Unit tests for stream.mcp.jobs: ServerState lifecycle and make_experiment_id determinism."""

from __future__ import annotations

import re

from stream.mcp.jobs import ServerState, make_experiment_id


class TestMakeExperimentId:
    """Tests for content-addressed experiment ID generation (MCP-03)."""

    def _write_files(self, tmp_path):
        """Helper: create three small files with known content."""
        hardware = tmp_path / "hardware.yaml"
        workload = tmp_path / "workload.onnx"
        mapping = tmp_path / "mapping.yaml"
        hardware.write_bytes(b"hardware: npu2\ncols: 4\n")
        workload.write_bytes(b"onnx_model_bytes_placeholder")
        mapping.write_bytes(b"mapping: default\n")
        return str(hardware), str(workload), str(mapping)

    def test_content_addressed_id_deterministic(self, tmp_path):
        """Calling make_experiment_id twice with identical files and params produces the same ID."""
        hardware, workload, mapping = self._write_files(tmp_path)
        backend = "ortools_gscip"
        constraints = {
            "memory_capacity": True,
            "object_fifo_depth": True,
            "buffer_descriptors": True,
            "dma_channels": True,
        }
        id1 = make_experiment_id(hardware, workload, mapping, backend, constraints)
        id2 = make_experiment_id(hardware, workload, mapping, backend, constraints)
        assert id1 == id2

    def test_content_addressed_id_different_backend(self, tmp_path):
        """Same files but different backend string produces a different ID."""
        hardware, workload, mapping = self._write_files(tmp_path)
        constraints = {
            "memory_capacity": True,
            "object_fifo_depth": True,
            "buffer_descriptors": True,
            "dma_channels": True,
        }
        id_gscip = make_experiment_id(hardware, workload, mapping, "ortools_gscip", constraints)
        id_highs = make_experiment_id(hardware, workload, mapping, "ortools_highs", constraints)
        assert id_gscip != id_highs

    def test_content_addressed_id_different_constraints(self, tmp_path):
        """Same files but one constraint flipped produces a different ID."""
        hardware, workload, mapping = self._write_files(tmp_path)
        backend = "ortools_gscip"
        constraints_all_true = {
            "memory_capacity": True,
            "object_fifo_depth": True,
            "buffer_descriptors": True,
            "dma_channels": True,
        }
        constraints_one_false = {
            "memory_capacity": False,
            "object_fifo_depth": True,
            "buffer_descriptors": True,
            "dma_channels": True,
        }
        id_all = make_experiment_id(hardware, workload, mapping, backend, constraints_all_true)
        id_one = make_experiment_id(hardware, workload, mapping, backend, constraints_one_false)
        assert id_all != id_one

    def test_content_addressed_id_different_content(self, tmp_path):
        """Different file content at same path produces a different ID."""
        hardware = tmp_path / "hardware.yaml"
        workload = tmp_path / "workload.onnx"
        mapping = tmp_path / "mapping.yaml"
        workload.write_bytes(b"onnx_model_bytes_placeholder")
        mapping.write_bytes(b"mapping: default\n")
        backend = "ortools_gscip"
        constraints = {
            "memory_capacity": True,
            "object_fifo_depth": True,
            "buffer_descriptors": True,
            "dma_channels": True,
        }
        hardware.write_bytes(b"hardware: npu2\ncols: 4\n")
        id_v1 = make_experiment_id(str(hardware), str(workload), str(mapping), backend, constraints)
        hardware.write_bytes(b"hardware: npu1\ncols: 2\n")
        id_v2 = make_experiment_id(str(hardware), str(workload), str(mapping), backend, constraints)
        assert id_v1 != id_v2

    def test_content_addressed_id_length(self, tmp_path):
        """Result is exactly 12 lowercase hex characters."""
        hardware, workload, mapping = self._write_files(tmp_path)
        result = make_experiment_id(
            hardware,
            workload,
            mapping,
            "ortools_gscip",
            {"memory_capacity": True, "object_fifo_depth": True, "buffer_descriptors": True, "dma_channels": True},
        )
        assert len(result) == 12
        assert re.match(r"^[0-9a-f]{12}$", result) is not None


class TestServerState:
    """Tests for ServerState job registry."""

    def test_server_state_default(self):
        """ServerState() initializes with empty jobs dict."""
        state = ServerState()
        assert isinstance(state.jobs, dict)
        assert len(state.jobs) == 0

    def test_server_state_job_lifecycle(self):
        """Can add a job as pending, update to running, update to complete with result."""
        state = ServerState()
        job_id = "abc123def456"

        # Add as pending
        state.jobs[job_id] = {"status": "pending", "result": None, "error": None}
        assert state.jobs[job_id]["status"] == "pending"
        assert state.jobs[job_id]["result"] is None
        assert state.jobs[job_id]["error"] is None

        # Update to running
        state.jobs[job_id]["status"] = "running"
        assert state.jobs[job_id]["status"] == "running"

        # Update to complete with result
        result_payload = {"latency_ns": 42000, "tiles": 16}
        state.jobs[job_id]["status"] = "complete"
        state.jobs[job_id]["result"] = result_payload
        assert state.jobs[job_id]["status"] == "complete"
        assert state.jobs[job_id]["result"] == result_payload

    def test_server_state_job_failed(self):
        """Can add a job and set status to failed with error string."""
        state = ServerState()
        job_id = "deadbeef0001"
        state.jobs[job_id] = {"status": "pending", "result": None, "error": None}
        state.jobs[job_id]["status"] = "failed"
        state.jobs[job_id]["error"] = "Solver timed out after 900s"
        assert state.jobs[job_id]["status"] == "failed"
        assert state.jobs[job_id]["error"] == "Solver timed out after 900s"
        assert state.jobs[job_id]["result"] is None
