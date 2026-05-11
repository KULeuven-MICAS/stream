from abc import ABC
from dataclasses import dataclass
from enum import Flag

from xdsl.ir.affine import AffineMap

from stream.datatypes import LayerDim
from stream.workload.tensor import Tensor


@dataclass(frozen=True, repr=False)
class Node(ABC):
    name: str

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"


@dataclass(frozen=True, repr=False)
class HasOutputs(Node, ABC):
    outputs: tuple[Tensor, ...]

    @property
    def output(self):
        assert len(self.outputs) == 1
        return self.outputs[0]


@dataclass(frozen=True, repr=False)
class HasInputs(Node, ABC):
    inputs: tuple[Tensor, ...]


@dataclass(frozen=True, repr=False)
class InEdge(HasOutputs): ...


@dataclass(frozen=True, repr=False)
class OutEdge(HasInputs): ...


@dataclass(frozen=True, repr=False)
class FusionEdge(HasInputs, HasOutputs):
    """A graph boundary node for shape-only ops (Flatten, Reshape).

    FusionEdge marks a fusion group boundary in the workload graph.
    It is NOT HasIterationSpace -- it has no affine iteration space
    or operand_mapping. Its tensors pass through unchanged during
    dimension resizing.
    """

    op_type: str  # original ONNX op type, e.g. "Flatten"


class TransferType(Flag):
    """Flags for different types of data transfer operations (can be combined)."""

    NONE = 0
    MEM_TO_MEM = 1
    MEM_TO_COMPUTE = 2
    COMPUTE_TO_MEM = 3
    COMPUTE_TO_COMPUTE = 4
    CONSTANT = 5
    NONCONSTANT = 6


@dataclass(frozen=True, repr=False)
class HasIterationSpace(HasInputs, HasOutputs):
    operand_mapping: tuple[AffineMap, ...]

    @property
    def num_dims(self) -> int:
        # Dimensionality of all maps should be equal
        return self.operand_mapping[0].num_dims

    def get_mapping(self, tensor: Tensor) -> AffineMap:
        if tensor in self.tensors:
            idx = self.tensors.index(tensor)
            return self.operand_mapping[idx]
        raise RuntimeError(f"Tensor {tensor.name} not found in node {self.name}")

    def get_dimension_size(self, layer_dim: LayerDim) -> int:
        dim_index = layer_dim.get_idx()
        return self.outputs[-1].shape[dim_index]  # TODO: Probably not always of output tensor

    @property
    def tensors(self) -> tuple[Tensor, ...]:
        return self.inputs + self.outputs


@dataclass(frozen=True, repr=False)
class TransferNode(HasIterationSpace):
    transfer_type: TransferType


@dataclass(frozen=True, repr=False)
class ComputationNode(HasIterationSpace):
    type: str  # e.g., "Conv", "Gemm", etc.

    def has_same_performance(self, other: "ComputationNode") -> bool:
        """Check if this computation node has the same performance characteristics as another node.
        This is a simple check based on operand data types and shapes.
        More sophisticated checks may be needed in the future."""
        if len(self.inputs) != len(other.inputs):
            return False
        for inp_self, inp_other in zip(self.inputs, other.inputs, strict=True):
            if inp_self.operand_type != inp_other.operand_type:
                return False
            if inp_self.shape != inp_other.shape:
                return False
        if self.outputs[0].operand_type != other.outputs[0].operand_type:
            return False
        if self.outputs[0].shape != other.outputs[0].shape:
            return False
        return True
