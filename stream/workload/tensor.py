from typing import TYPE_CHECKING, TypeAlias

from zigzag.datatypes import LayerDim, LayerOperand

if TYPE_CHECKING:
    from zigzag.hardware.architecture.memory_instance import MemoryInstance

    from stream.cost_model.memory_manager import MemoryManager
    from stream.hardware.architecture.accelerator import Accelerator
    from stream.workload.computation.computation_node import LOOP_RANGES_T, ComputationNode
    from stream.workload.onnx_workload import ComputationNodeWorkload

TensorHash: TypeAlias = int


class Tensor:
    """Class to represent a data tensor.
    TODO: Add from which layer this tensor originates and its dimension ranges
    """

    def __init__(
        self,
        size: int,
        origin: "ComputationNode",
        layer_operand: LayerOperand,
        loop_dimensions: list[LayerDim],
        loop_ranges: tuple[tuple[int, int], ...],
    ):
        """Initialize the Tensor instance.

        Args:
            size: the size of the tensor in bits
            origin (ComputationNode): The computation node that consumes/produces this tensor
            layer_operand (str, optional): The layer operand to which this tensor belongs
            loop_dimensions (tuple, optional): The loop dimensions for this tensor
            loop_ranges (tuple, optional): The loop range span for the different dimensions of this operand
        """
        self.size = size
        self.__origin = origin
        self.__layer_operand = layer_operand
        self.memory_operand = self.origin.memory_operand_links.layer_to_mem_op(layer_operand)
        self.loop_dimensions = loop_dimensions
        self.__loop_ranges = loop_ranges
        self.base_priority: None | int = None  # Will be set when we know how many successors this node has (static)
        self.instance_priorities: dict[MemoryInstance, int] = {}
        self.id = (self.origin.id, self.origin.sub_id, layer_operand)
        self.equality_hash = hash((self.origin.id, self.layer_operand, self.loop_ranges))
        self.__static_hash = hash((origin, layer_operand))

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

    def __str__(self) -> str:
        return f"Tensor{self.id}"

    def __repr__(self) -> str:
        return str(self)

    def __len__(self) -> int:
        return self.size

    def __hash__(self) -> int:
        return self.__static_hash

    def __lt__(self, __o: object) -> bool:
        return isinstance(__o, Tensor) and self.size < __o.size

    @property
    def origin(self):
        """Protect property so static hash can't be altered"""
        return self.__origin

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
