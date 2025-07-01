# --------------------------------------------------------------------------- #
#  steady_state_tensor.py  ── extended with “whole-tensor” support            #
# --------------------------------------------------------------------------- #
from enum import Flag, auto
from math import prod
from typing import Any, Sequence, Tuple

from zigzag.datatypes import LayerOperand

from stream.hardware.architecture.core import Core
from stream.workload.steady_state_iteration_space import SteadyStateIterationSpace
from stream.workload.steady_state_node import SteadyStateNode


# ................................................ tensor-type bit-flags ...
class TensorFlag(Flag):
    CONSTANT = auto()
    NONCONSTANT = auto()
    INPUT = auto()
    OUTPUT = auto()


# ............................................................ main class ...
class SteadyStateTensor(SteadyStateNode):
    """
    Node representing a tensor (slice) participating in the steady-state graph.
    full_shape : tuple[int, ...] | None
        Shape of the *entire* layer-tensor, e.g. (N, C, H, W).
    slices_per_full : int | None
        How many of the steady-state slices make up one full tensor;
        equals 1 if you do not want to distinguish.
    """

    def __init__(
        self,
        *,
        id: int,
        node_name: str,
        size: int,  # slice size (elements / bits)
        type: TensorFlag,
        operand: LayerOperand,
        steady_state_iteration_space: SteadyStateIterationSpace,
        possible_resource_allocation: list[Core | None],
        full_shape: Sequence[int] | None = None,
        slices_per_full: int | None = None,
    ):
        super().__init__(
            id=id,
            node_name=node_name,
            type="tensor",
            steady_state_iteration_space=steady_state_iteration_space,
            possible_resource_allocation=possible_resource_allocation,
        )
        self.chosen_resource_allocation = (
            None if len(possible_resource_allocation) > 1 else possible_resource_allocation[0]
        )
        self.possible_resource_allocation: list[Core]

        # --------- existing attributes ----------
        self.size: int = size  # slice size
        self.tensor_flag: TensorFlag = type
        self.operand: LayerOperand = operand
        self.runtime: int | float = 0  # kept as before

        # --------- NEW attributes ---------------
        self.full_shape: Tuple[int, ...] | None = tuple(full_shape) if full_shape else None
        self.slices_per_full: int | None = slices_per_full
        self.full_size: int | None = prod(self.full_shape) if self.full_shape else None

    # ..................................................... helpers ...
    # slice == “one steady-state chunk”      whole == “complete layer tensor”
    @property
    def slice_size(self) -> int:
        return self.size

    @property
    def whole_size(self) -> int:
        """
        Size of the full tensor (elements / bits).
        Falls back to slice_size if caller never supplied full_shape.
        """
        return self.full_size or self.size

    @property
    def slices_per_full_tensor(self) -> int:
        """How many steady-state slices make up one full tensor."""
        return self.slices_per_full or 1

    def __str__(self) -> str:
        return f"Tensor({self.node_name})"

    __repr__ = __str__  # same textual form

    def __hash__(self) -> int:
        return hash((self.id, self.node_name, self.tensor_flag, self.chosen_resource_allocation))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, SteadyStateTensor):
            return False
        return (
            self.id == other.id
            and self.node_name == other.node_name
            and self.tensor_flag == other.tensor_flag
            and self.chosen_resource_allocation == other.chosen_resource_allocation
        )

    @property
    def plot_name(self) -> str:
        return self.node_name
