from collections.abc import Sequence
from dataclasses import dataclass
from math import prod
from typing import TYPE_CHECKING, TypeAlias

from xdsl.dialects.builtin import MemRefType
from xdsl.dialects.memref import AllocOp, SubviewOp
from zigzag.datatypes import LayerDim, LayerOperand

if TYPE_CHECKING:
    from zigzag.hardware.architecture.memory_instance import MemoryInstance

    from stream.cost_model.memory_manager import MemoryManager
    from stream.hardware.architecture.accelerator import Accelerator

TensorHash: TypeAlias = int


@dataclass
class SubviewTensorInputs:
    """Inputs to the SubviewTensor constructor for easier instantiation."""

    subview: SubviewOp
    sizes: Sequence[int]
    cn_source: "ComputationNode"
    layer_operand: LayerOperand
    loop_dimensions: list[LayerDim]
    loop_ranges: tuple[tuple[int, int], ...]


class SubviewTensor:
    """Class to represent a data tensor.
    TODO: Add from which layer this tensor originates and its dimension ranges
    """

    def __init__(  # noqa: PLR0913
        self,
        subview: SubviewOp,
        sizes: Sequence[int],
        cn_source: "ComputationNode",
        layer_operand: LayerOperand,
        loop_dimensions: list[LayerDim],
        loop_ranges: tuple[tuple[int, int], ...],
        name: str = "",
    ):
        """Initialize the SubviewTensor instance.

        Args:
            size: the size of the tensor in bits
            origin (ComputationNode): The computation node that consumes/produces this tensor
            layer_operand (str, optional): The layer operand to which this tensor belongs
            loop_dimensions (tuple, optional): The loop dimensions for this tensor
            loop_ranges (tuple, optional): The loop range span for the different dimensions of this operand
        """
        self.subview = subview
        self.sizes = sizes
        self.size = prod(sizes)
        self.cn_source = cn_source
        self.__layer_operand = layer_operand
        self.memory_operand = self.cn_source.memory_operand_links.layer_to_mem_op(layer_operand)
        self.loop_dimensions = loop_dimensions
        self.__loop_ranges = loop_ranges
        self.base_priority: None | int = None  # Will be set when we know how many successors this node has (static)
        self.instance_priorities: dict[MemoryInstance, int] = {}
        self.id = (self.cn_source.id, self.cn_source.sub_id, layer_operand)
        self.name = name

    def __str__(self) -> str:
        return f"SubviewTensor{self.id}"

    def __repr__(self) -> str:
        return str(self)

    def __len__(self) -> int:
        return self.size

    def __hash__(self) -> int:
        return hash((self.cn_source, self.layer_operand))

    def __lt__(self, __o: object) -> bool:
        return isinstance(__o, SubviewTensor) and self.size < __o.size

    def get_inputs(self) -> SubviewTensorInputs:
        """Get the inputs to the SubviewTensor constructor."""
        memref_type = self.subview.source.type
        assert isinstance(memref_type, MemRefType)
        return SubviewTensorInputs(
            subview=self.subview,
            sizes=self.sizes,
            cn_source=self.cn_source,
            layer_operand=self.layer_operand,
            loop_dimensions=self.loop_dimensions,
            loop_ranges=self.loop_ranges,
        )

    def equality_hash(self):
        return hash((self.name, self.loop_ranges))

    def set_base_priorities(self, base_priority: int):
        self.base_priority = base_priority

    def get_instance_priority(self, top_instance: "MemoryInstance", memory_manager: "MemoryManager"):
        if top_instance in self.instance_priorities:
            return self.instance_priorities[top_instance]
        else:
            # If the top_instance is not in the dict. it means the core_id is the core that generates the tensor.
            # We  then return as priority the sum of all priorities of top instances that are not sotring the tensor.
            storing_core_ids, top_instance_idxs, _ = memory_manager.find_tensor(self)
            storing_instances = []
            for storing_core_id, top_instance_idx in zip(storing_core_ids, top_instance_idxs, strict=False):
                core = memory_manager.accelerator.get_core(storing_core_id)
                storing_instance = memory_manager.top_instances_per_core[core][top_instance_idx]
                storing_instances.append(storing_instance)
            not_storing_instances = list(set(self.instance_priorities.keys()) - set(storing_instances))
            not_storing_priority = sum(
                self.instance_priorities[not_storing_instance] for not_storing_instance in not_storing_instances
            )
            return not_storing_priority

    def initialize_instance_priorities(
        self, g: "ComputationNodeWorkload", node: "ComputationNode", accelerator: "Accelerator"
    ):
        if self.layer_operand == node.output_operand:
            out_edges = [(succ, d) for n, succ, d in g.out_edges(node, data=True) if succ.id != n.id]
            for successor, data in out_edges:
                assert successor.chosen_core_allocation is not None, (
                    f"Chosen core allocation for {successor} is None. "
                    "This should not happen, as the chosen core allocation should be set before this method is called."
                )
                core = accelerator.get_core(successor.chosen_core_allocation)
                layer_operand = data["operand"]
                memory_operand = successor.memory_operand_links.layer_to_mem_op(layer_operand)
                top_instance = core.get_top_memory_instance(memory_operand)
                if top_instance in self.instance_priorities:
                    self.instance_priorities[top_instance] += 1
                else:  # first time we see this instance
                    self.instance_priorities[top_instance] = 1

        else:
            if self.base_priority is None:
                return  # No base priority set, this tensor will not be spawned as it will use previous layer outputs
            assert node.chosen_core_allocation is not None, (
                f"Chosen core allocation for {node} is None. "
                "This should not happen, as the chosen core allocation should be set before this method is called."
            )
            core = accelerator.get_core(node.chosen_core_allocation)
            top_instance = core.get_top_memory_instance(self.memory_operand)
            self.instance_priorities[top_instance] = self.base_priority

    def get_total_priority(self):
        return sum(self.instance_priorities.values())

    @property
    def source(self) -> SubviewOp | AllocOp:
        return self.subview.source.op

    @property
    def original_shape(self) -> list[int]:
        """Get the original shape of the tensor before subviewing."""
        source = self.source
        while isinstance(source, SubviewOp):
            source = source.source.op
        assert isinstance(source, AllocOp)
        results_type: MemRefType = source.results[0].type
        return results_type.get_shape()

    @property
    def origin(self):
        """Protect property so static hash can't be altered"""
        return self.cn_source

    @property
    def loop_ranges(self):
        """Protect property so static hash can't be altered"""
        return self.__loop_ranges

    @property
    def layer_operand(self):
        """Protect property so static hash can't be altered"""
        return self.__layer_operand

    @property
    def loop_ranges_per_dim(self) -> "LOOP_RANGES_T":
        """Same format as ComputationNode.loop_ranges"""
        return {dim: loop_range for dim, loop_range in zip(self.loop_dimensions, self.loop_ranges, strict=False)}
