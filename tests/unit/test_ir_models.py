"""Unit tests for WorkloadIR and AcceleratorIR Pydantic models.

Covers IR-01 and IR-02 from Phase 16 requirements:
  - IR-01: Pydantic BaseModel IR classes exist with schema_version field and produce valid JSON Schema
  - IR-02: Per-persona IR views available (algorithmic, hardware, compiler)

Tests use synthetic dicts matching get_ir() shapes — no real Workload/Accelerator objects required.
Mock objects provide .get_ir() -> dict for from_internal() tests.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from stream.ir import AcceleratorIR, WorkloadIR
from stream.ir.accelerator import (
    AcceleratorCompilerView,
    AcceleratorHardwareView,
    CoreIR,
)
from stream.ir.workload import (
    NodeIR,
    WorkloadAlgorithmicView,
    WorkloadCompilerView,
)

# ---------------------------------------------------------------------------
# Fixtures: synthetic dicts matching get_ir() shapes
# ---------------------------------------------------------------------------

WORKLOAD_RAW: dict = {
    "num_nodes": 3,
    "num_edges": 2,
    "num_unique_dimensions": 4,
    "unique_dimensions": {
        "K": {"index": 0, "size": 128},
        "M": {"index": 1, "size": 64},
        "N": {"index": 2, "size": 32},
        "C": {"index": 3, "size": None},
    },
    "dimension_expressions": ["K*M", "M*N"],
    "dimension_relations": ["K == M"],
    "nodes": [
        {
            "name": "MatMul",
            "type": "ComputationNode",
            "dimensions": {"K": 128, "M": 64},
            "global_dim_indices": [0, 1],
            "inputs": [{"name": "A", "shape": [128, 64], "operand_type": "I"}],
            "outputs": [{"name": "O", "shape": [64, 32], "operand_type": "O"}],
            "computation_type": "Gemm",
        },
        {
            "name": "Transfer1",
            "type": "TransferNode",
            "inputs": [{"name": "A", "shape": [128, 64], "operand_type": "I"}],
            "outputs": [{"name": "A_out", "shape": [128, 64], "operand_type": "O"}],
            "transfer_type": "DMA",
        },
        {
            "name": "ReLU",
            "type": "ComputationNode",
            "dimensions": {"N": 32, "M": 64},
            "global_dim_indices": [2, 1],
            "inputs": [{"name": "O", "shape": [64, 32], "operand_type": "I"}],
            "outputs": [{"name": "R", "shape": [64, 32], "operand_type": "O"}],
            "computation_type": "Relu",
        },
    ],
    "edges": [
        {"source": "MatMul", "target": "ReLU", "shared_tensors": ["O"]},
        {"source": "Transfer1", "target": "MatMul", "shared_tensors": ["A"]},
    ],
    "tensors": {
        "A": {
            "shape": [128, 64],
            "relevant_dimensions": ["K", "M"],
            "strides_per_dimension": {"K": [64, 1], "M": [1, 0]},
        },
        "O": {
            "shape": [64, 32],
            "relevant_dimensions": ["M", "N"],
            "strides_per_dimension": {"M": [32, 1], "N": [1, 0]},
        },
    },
    "generations": {"MatMul": 0, "Transfer1": 0, "ReLU": 1},
}

ACCELERATOR_RAW: dict = {
    "name": "test_aie2_accel",
    "num_cores": 3,
    "offchip_core_id": 2,
    "nb_shared_mem_groups": 1,
    "cores": [
        {
            "id": 0,
            "name": "core_0",
            "core_type": "aie2.compute",
            "type": "compute",
            "row_id": 0,
            "col_id": 0,
            "utilization": 0.85,
            "memory": {"capacity_bits": 65536},
            "max_object_fifo_depth": 4,
        },
        {
            "id": 1,
            "name": "core_1",
            "core_type": "aie2.compute",
            "type": "compute",
            "row_id": 0,
            "col_id": 1,
            "utilization": 0.72,
            "memory": {"capacity_bits": 65536},
            "max_object_fifo_depth": 4,
        },
        {
            "id": 2,
            "name": "offchip",
            "core_type": "offchip",
            "type": "offchip",
            "row_id": -1,
            "col_id": -1,
            "utilization": 0.0,
        },
    ],
    "core_connectivity": [
        {
            "type": "bus",
            "cores": [0, 1],
            "bandwidth": 32.0,
            "unit_energy_cost": 0.5,
        },
        {
            "type": "link",
            "from_core": 1,
            "to_core": 2,
            "bandwidth": 64.0,
            "unit_energy_cost": 1.0,
        },
    ],
}


# ---------------------------------------------------------------------------
# TestWorkloadIR
# ---------------------------------------------------------------------------


class TestWorkloadIR:
    def test_json_schema(self):
        """WorkloadIR.model_json_schema() must include schema_version const '1.0'."""
        schema = WorkloadIR.model_json_schema()
        assert "schema_version" in schema["properties"]
        sv = schema["properties"]["schema_version"]
        assert sv.get("const") == "1.0", f"Expected const='1.0', got: {sv}"

    def test_from_dict(self):
        """WorkloadIR constructed from a raw get_ir() dict validates without error."""
        raw = WORKLOAD_RAW
        nodes = [
            NodeIR(
                name=n["name"],
                node_type=n["type"],
                dimensions=n.get("dimensions"),
                global_dim_indices=n.get("global_dim_indices"),
                inputs=[
                    {"name": t["name"], "shape": t["shape"], "operand_type": t["operand_type"]}
                    for t in n.get("inputs", [])
                ]
                if n.get("inputs")
                else None,
                outputs=[
                    {"name": t["name"], "shape": t["shape"], "operand_type": t["operand_type"]}
                    for t in n.get("outputs", [])
                ]
                if n.get("outputs")
                else None,
                computation_type=n.get("computation_type"),
                transfer_type=n.get("transfer_type"),
            )
            for n in raw["nodes"]
        ]
        ir = WorkloadIR(
            num_nodes=raw["num_nodes"],
            num_edges=raw["num_edges"],
            num_unique_dimensions=raw["num_unique_dimensions"],
            unique_dimensions=raw["unique_dimensions"],
            dimension_expressions=raw["dimension_expressions"],
            dimension_relations=raw["dimension_relations"],
            nodes=nodes,
            edges=raw["edges"],
            tensors=raw["tensors"],
            generations=raw["generations"],
        )
        assert ir.num_nodes == 3
        assert ir.num_unique_dimensions == 4
        assert len(ir.nodes) == 3
        assert ir.schema_version == "1.0"

    def test_from_internal(self):
        """WorkloadIR.from_internal(mock_workload) constructs a valid model."""
        mock_workload = MagicMock()
        mock_workload.get_ir.return_value = WORKLOAD_RAW

        ir = WorkloadIR.from_internal(mock_workload)

        mock_workload.get_ir.assert_called_once()
        assert ir.num_nodes == 3
        assert ir.num_edges == 2
        assert ir.schema_version == "1.0"
        assert len(ir.nodes) == 3

    def test_algorithmic_view(self):
        """WorkloadIR.algorithmic_view() returns WorkloadAlgorithmicView with correct fields."""
        mock_workload = MagicMock()
        mock_workload.get_ir.return_value = WORKLOAD_RAW
        ir = WorkloadIR.from_internal(mock_workload)

        view = ir.algorithmic_view()

        assert isinstance(view, WorkloadAlgorithmicView)
        assert view.num_nodes == 3
        assert view.num_unique_dimensions == 4
        assert view.num_edges == 2
        assert view.schema_version == "1.0"
        assert "K*M" in view.dimension_expressions

    def test_compiler_view(self):
        """WorkloadIR.compiler_view() returns WorkloadCompilerView with nodes, edges, generations."""
        mock_workload = MagicMock()
        mock_workload.get_ir.return_value = WORKLOAD_RAW
        ir = WorkloadIR.from_internal(mock_workload)

        view = ir.compiler_view()

        assert isinstance(view, WorkloadCompilerView)
        assert len(view.nodes) == 3
        assert len(view.edges) == 2
        assert view.generations == {"MatMul": 0, "Transfer1": 0, "ReLU": 1}
        assert view.schema_version == "1.0"

    def test_json_round_trip(self):
        """model_dump_json() on WorkloadIR produces valid JSON that round-trips through json.loads."""
        mock_workload = MagicMock()
        mock_workload.get_ir.return_value = WORKLOAD_RAW
        ir = WorkloadIR.from_internal(mock_workload)

        json_str = ir.model_dump_json()
        parsed = json.loads(json_str)

        assert parsed["num_nodes"] == 3
        assert parsed["schema_version"] == "1.0"
        assert len(parsed["nodes"]) == 3

    def test_optional_node_fields(self):
        """NodeIR with optional fields (computation_type=None, transfer_type=None) validates correctly."""
        # TransferNode has transfer_type but no computation_type
        # ComputationNode has computation_type but no transfer_type
        node_compute = NodeIR(name="MatMul", node_type="ComputationNode", computation_type="Gemm")
        assert node_compute.transfer_type is None
        assert node_compute.dimensions is None

        node_transfer = NodeIR(name="Transfer1", node_type="TransferNode", transfer_type="DMA")
        assert node_transfer.computation_type is None
        assert node_transfer.dimensions is None

    def test_node_type_mapping_from_raw(self):
        """from_internal() maps dict key 'type' to NodeIR field 'node_type' correctly."""
        mock_workload = MagicMock()
        mock_workload.get_ir.return_value = WORKLOAD_RAW
        ir = WorkloadIR.from_internal(mock_workload)

        assert ir.nodes[0].node_type == "ComputationNode"
        assert ir.nodes[1].node_type == "TransferNode"
        assert ir.nodes[2].node_type == "ComputationNode"


# ---------------------------------------------------------------------------
# TestAcceleratorIR
# ---------------------------------------------------------------------------


class TestAcceleratorIR:
    def test_json_schema(self):
        """AcceleratorIR.model_json_schema() must include schema_version const '1.0'."""
        schema = AcceleratorIR.model_json_schema()
        assert "schema_version" in schema["properties"]
        sv = schema["properties"]["schema_version"]
        assert sv.get("const") == "1.0", f"Expected const='1.0', got: {sv}"

    def test_from_dict(self):
        """AcceleratorIR constructed from raw get_ir() dict validates without error."""
        raw = ACCELERATOR_RAW
        common_fields = {"id", "name", "core_type", "type", "row_id", "col_id", "utilization"}
        cores = []
        for c in raw["cores"]:
            extra = {k: v for k, v in c.items() if k not in common_fields}
            cores.append(
                CoreIR(
                    id=c["id"],
                    name=c["name"],
                    core_type=c["core_type"],
                    type=c["type"],
                    row_id=c["row_id"],
                    col_id=c["col_id"],
                    utilization=c["utilization"],
                    extra_fields=extra,
                )
            )
        ir = AcceleratorIR(
            name=raw["name"],
            num_cores=raw["num_cores"],
            offchip_core_id=raw["offchip_core_id"],
            nb_shared_mem_groups=raw["nb_shared_mem_groups"],
            cores=cores,
            core_connectivity=raw["core_connectivity"],
        )
        assert ir.name == "test_aie2_accel"
        assert ir.num_cores == 3
        assert ir.schema_version == "1.0"

    def test_from_internal(self):
        """AcceleratorIR.from_internal(mock_accelerator) constructs a valid model."""
        mock_accelerator = MagicMock()
        mock_accelerator.get_ir.return_value = ACCELERATOR_RAW

        ir = AcceleratorIR.from_internal(mock_accelerator)

        mock_accelerator.get_ir.assert_called_once()
        assert ir.name == "test_aie2_accel"
        assert ir.num_cores == 3
        assert ir.offchip_core_id == 2
        assert ir.schema_version == "1.0"
        assert len(ir.cores) == 3

    def test_hardware_view(self):
        """AcceleratorIR.hardware_view() returns AcceleratorHardwareView with cores and connectivity."""
        mock_accelerator = MagicMock()
        mock_accelerator.get_ir.return_value = ACCELERATOR_RAW
        ir = AcceleratorIR.from_internal(mock_accelerator)

        view = ir.hardware_view()

        assert isinstance(view, AcceleratorHardwareView)
        assert view.name == "test_aie2_accel"
        assert view.num_cores == 3
        assert len(view.cores) == 3
        assert len(view.core_connectivity) == 2
        assert view.schema_version == "1.0"

    def test_compiler_view(self):
        """AcceleratorIR.compiler_view() returns AcceleratorCompilerView with topology info."""
        mock_accelerator = MagicMock()
        mock_accelerator.get_ir.return_value = ACCELERATOR_RAW
        ir = AcceleratorIR.from_internal(mock_accelerator)

        view = ir.compiler_view()

        assert isinstance(view, AcceleratorCompilerView)
        assert view.name == "test_aie2_accel"
        assert len(view.cores) == 3
        # compiler view should have core topology data
        assert view.cores[0].id == 0
        assert view.cores[0].core_type == "aie2.compute"
        assert view.schema_version == "1.0"

    def test_json_round_trip(self):
        """model_dump_json() on AcceleratorIR produces valid JSON that round-trips through json.loads."""
        mock_accelerator = MagicMock()
        mock_accelerator.get_ir.return_value = ACCELERATOR_RAW
        ir = AcceleratorIR.from_internal(mock_accelerator)

        json_str = ir.model_dump_json()
        parsed = json.loads(json_str)

        assert parsed["name"] == "test_aie2_accel"
        assert parsed["schema_version"] == "1.0"
        assert len(parsed["cores"]) == 3

    def test_extra_fields_for_core_types(self):
        """AcceleratorIR cores with extra_fields dict for type-specific data validate correctly."""
        mock_accelerator = MagicMock()
        mock_accelerator.get_ir.return_value = ACCELERATOR_RAW
        ir = AcceleratorIR.from_internal(mock_accelerator)

        # aie2 compute core should have extra_fields with memory and max_object_fifo_depth
        compute_core = ir.cores[0]
        assert "memory" in compute_core.extra_fields
        assert "max_object_fifo_depth" in compute_core.extra_fields
        assert compute_core.extra_fields["memory"]["capacity_bits"] == 65536

        # offchip core should have empty extra_fields
        offchip_core = ir.cores[2]
        assert offchip_core.extra_fields == {}

    def test_offchip_core_id_none(self):
        """AcceleratorIR with offchip_core_id=None validates correctly."""
        raw = {**ACCELERATOR_RAW, "offchip_core_id": None}
        mock_accelerator = MagicMock()
        mock_accelerator.get_ir.return_value = raw
        ir = AcceleratorIR.from_internal(mock_accelerator)
        assert ir.offchip_core_id is None
