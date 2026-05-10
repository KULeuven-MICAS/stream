"""WorkloadIR Pydantic model with per-persona view methods.

Wraps the output of Workload.get_ir() in a typed, versioned Pydantic model.
Construction is always via the from_internal() classmethod.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from stream.workload.workload import Workload


class TensorOperandIR(BaseModel):
    """A single tensor operand (input or output) for a workload node."""

    name: str = Field(description="Tensor identifier")
    shape: list[int] = Field(description="Tensor shape as a list of dimension sizes")
    operand_type: str = Field(description="Operand role: 'I' for input, 'O' for output, etc.")


class NodeIR(BaseModel):
    """IR representation of a single workload node (ComputationNode or TransferNode)."""

    name: str = Field(description="Node identifier, unique within the workload DAG")
    node_type: str = Field(description="'ComputationNode' or 'TransferNode'")
    dimensions: dict[str, int] | None = Field(
        default=None, description="Loop dimensions and their sizes; only present for HasIterationSpace nodes"
    )
    global_dim_indices: list[int] | None = Field(
        default=None,
        description="Indices into the workload's unique dimension list; only for HasIterationSpace nodes",
    )
    inputs: list[TensorOperandIR] | None = Field(
        default=None, description="Input tensor operands; only for HasInputs nodes"
    )
    outputs: list[TensorOperandIR] | None = Field(
        default=None, description="Output tensor operands; only for HasOutputs nodes"
    )
    computation_type: str | None = Field(
        default=None, description="Computation operator type (e.g. 'Gemm', 'Relu'); only for ComputationNode"
    )
    transfer_type: str | None = Field(
        default=None, description="Transfer mechanism (e.g. 'DMA'); only for TransferNode"
    )


class WorkloadAlgorithmicView(BaseModel):
    """Algorithmic-persona projection of WorkloadIR.

    Contains high-level workload shape: node counts, dimension counts, and symbolic
    dimension expressions. Suitable for algorithmic engineers reasoning about workload size.
    """

    schema_version: Literal["1.0"] = "1.0"
    num_nodes: int = Field(description="Total number of nodes in the workload DAG")
    num_edges: int = Field(description="Total number of edges in the workload DAG")
    num_unique_dimensions: int = Field(description="Number of unique loop dimensions across all nodes")
    dimension_expressions: list[str] = Field(description="Symbolic expressions relating workload dimensions")


class WorkloadCompilerView(BaseModel):
    """Compiler-persona projection of WorkloadIR.

    Contains the full node graph (with types, dimensions, tensor operands), edges,
    tensor metadata, and generation assignments. Suitable for compiler engineers
    performing node-to-core mapping and transfer routing.
    """

    schema_version: Literal["1.0"] = "1.0"
    nodes: list[NodeIR] = Field(description="All workload nodes with their types, dimensions, and tensor operands")
    edges: list[dict[str, Any]] = Field(description="DAG edges with source, target, and shared tensor info")
    tensors: dict[str, dict[str, Any]] = Field(
        description="Tensor metadata: shape, relevant dimensions, and strides per dimension"
    )
    generations: dict[str, int] = Field(description="Timeslot (generation) assignment per node name")


class WorkloadIR(BaseModel):
    """Typed Pydantic model wrapping Workload.get_ir() output.

    schema_version '1.0': minor bumps (1.1) for additive fields, major bumps (2.0) for
    removed/renamed fields. Construction is always via from_internal().
    """

    model_config = ConfigDict(
        json_schema_extra={
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": "stream_aie/workload_ir/v1",
        }
    )

    schema_version: Literal["1.0"] = "1.0"
    num_nodes: int = Field(description="Total number of nodes in the workload DAG")
    num_edges: int = Field(description="Total number of edges in the workload DAG")
    num_unique_dimensions: int = Field(description="Number of unique loop dimensions across all nodes")
    unique_dimensions: dict[str, dict[str, Any]] = Field(
        description="Map from dimension string to its index and size (size may be None if unknown)"
    )
    dimension_expressions: list[str] = Field(description="Symbolic expressions relating workload dimensions")
    dimension_relations: list[str] = Field(description="Constraint expressions between workload dimensions")
    nodes: list[NodeIR] = Field(description="All workload nodes in topological order")
    edges: list[dict[str, Any]] = Field(description="DAG edges with source, target, and shared tensor info")
    tensors: dict[str, dict[str, Any]] = Field(
        description="Tensor metadata: shape, relevant dimensions, and strides per dimension"
    )
    generations: dict[str, int] = Field(description="Timeslot (generation) assignment per node name")

    @classmethod
    def from_internal(cls, workload: Workload) -> WorkloadIR:
        """Construct WorkloadIR from a Workload internal object.

        Calls workload.get_ir() once, maps the resulting dict fields to Pydantic types,
        and validates on construction. The dict key 'type' maps to NodeIR field 'node_type'.
        """

        raw = workload.get_ir()
        nodes = []
        for n in raw["nodes"]:
            inputs = (
                [TensorOperandIR(**t) for t in n["inputs"]] if n.get("inputs") is not None else None
            )
            outputs = (
                [TensorOperandIR(**t) for t in n["outputs"]] if n.get("outputs") is not None else None
            )
            nodes.append(
                NodeIR(
                    name=n["name"],
                    node_type=n["type"],
                    dimensions=n.get("dimensions"),
                    global_dim_indices=n.get("global_dim_indices"),
                    inputs=inputs,
                    outputs=outputs,
                    computation_type=n.get("computation_type"),
                    transfer_type=n.get("transfer_type"),
                )
            )
        return cls(
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

    def algorithmic_view(self) -> WorkloadAlgorithmicView:
        """Return algorithmic-persona projection: high-level workload shape and dimension info."""
        return WorkloadAlgorithmicView(
            num_nodes=self.num_nodes,
            num_edges=self.num_edges,
            num_unique_dimensions=self.num_unique_dimensions,
            dimension_expressions=self.dimension_expressions,
        )

    def compiler_view(self) -> WorkloadCompilerView:
        """Return compiler-persona projection: full node graph with tensors and generation assignments."""
        return WorkloadCompilerView(
            nodes=self.nodes,
            edges=self.edges,
            tensors=self.tensors,
            generations=self.generations,
        )
