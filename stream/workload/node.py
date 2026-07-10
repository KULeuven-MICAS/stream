from abc import ABC
from dataclasses import dataclass
from enum import Flag

from xdsl.ir.affine import AffineMap

from stream.workload.node_key import node_key
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
    """A graph boundary node for layout-only ops (Flatten, Reshape, Transpose, Squeeze, Unsqueeze):
    pure re-indexing with no compute, which splits the fusion group (see
    ``Workload.split_fusion_groups``).

    A FusionEdge is NOT ``HasIterationSpace`` -- it has no ``operand_mapping``. It is the escape hatch
    the affine IR reserves for operators that cannot be expressed as one affine node: rather than
    force a lossy affine map, the operator becomes an explicit fusion boundary, its tensors passing
    through unchanged during dimension resizing. (Normalizations like Softmax are NOT FusionEdges --
    they parse to schedulable ``NormalizationNode``s that decompose for fusion analysis.)
    """

    op_type: str  # original ONNX op type, e.g. "Transpose" or "Reshape"


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
        """Whether two nodes cost the same: identical op type, operand precisions/shapes, and affine
        maps. Delegates to the canonical :func:`~stream.workload.node_key.node_key` so that op type
        and the operand maps are part of the identity (a Conv and a Gemm with matching tensor shapes
        are no longer treated as equal)."""
        return node_key(self) == node_key(other)


@dataclass(frozen=True, repr=False)
class NormalizationNode(ComputationNode):
    """A normalization (Softmax, LpNormalization, LayerNormalization, …): one *schedulable* node
    handled by a single native kernel, but internally a reduce-then-broadcast over ``reduction_axes``.

    A normalization is not one affine access relation -- its output at index ``j`` along the
    normalized axis depends on the *whole* slice over that axis (a reduction, then a broadcast). So a
    single ``operand_mapping`` cannot express it faithfully. We resolve the tension with two views:

    - **scheduling view (this node):** identity ``operand_mapping`` over the full shape, i.e. a shaped
      element-wise op that a fused softmax/norm kernel evaluates natively. It costs, dedups and
      schedules as one node.
    - **fusion-analysis view (derived):** :func:`stream.workload.normalization.decompose_normalization`
      expands it into its affine sub-operators (e.g. max → exp → sum → div), which makes explicit that
      the block's *other* axes are PARALLEL (freely fusible with the producer/consumer, as in flash
      attention) and only ``reduction_axes`` carry the intra-op reduction (kept resident, or streamed).

    ``reduction_axes`` are the positions (into the node's iteration space) that the normalization
    reduces over -- the one axis a single affine map cannot capture, hence stored, not derived.
    """

    reduction_axes: tuple[int, ...] = ()
