from typing import Any

from zigzag.datatypes import MemoryOperand
from zigzag.mapping.spatial_mapping import SpatialMapping
from zigzag.utils import DiGraphWrapper

from stream.cost_model.communication_manager import CommunicationManager
from stream.hardware.architecture.core import Core


class CoreGraph(DiGraphWrapper[Core]):
    """Represents the core structure of an accelerator"""


class Accelerator:
    """
    The Accelerator class houses a set of Cores with an additional Global Buffer.
    This Global Buffer sits above the cores, and can optionally be disabled.
    In this Stream version, the cores are actually a graph with directed edges representing communication links.
    """

    def __init__(
        self,
        name: str,
        cores: CoreGraph,
        nb_shared_mem_groups: int,
        offchip_core_id: int | None = None,
    ):
        """ """
        self.name = name
        self.cores = cores
        self.offchip_core_id = offchip_core_id
        self.nb_shared_mem_groups = nb_shared_mem_groups
        self.communication_manager = CommunicationManager(self)

    def get_core(self, core_id: int) -> Core:
        """s
        Return the core with id 'core_id'.
        Raises ValueError() when a core_id is not found in the available cores.
        """
        return self.cores.get_node_with_id(core_id)

    def get_offchip_core(self) -> Core:
        """Return the offchip core."""
        assert self.offchip_core_id, "This accelerator has no offchip core id."
        return self.get_core(self.offchip_core_id)

    def get_spatial_mapping_from_core(self, core_allocation: list[int]) -> SpatialMapping:
        """Iff the dataflows of all given cores is the same, return that dataflow. Otherwise, throw an error"""
        all_dataflows = [self.get_core(core_id).dataflows for core_id in core_allocation]
        some_dataflow = all_dataflows.pop()

        # All cores have same dataflow
        if some_dataflow is not None and all(some_dataflow == dataflow for dataflow in all_dataflows):
            return some_dataflow

        raise ValueError("Unclear which dataflow to return or no valid dataflow found.")

    def has_shared_memory(self, core_id_a: int, core_id_b: int, mem_op_a: MemoryOperand, mem_op_b: MemoryOperand):
        """Check whether two cores have a shared top level memory instance for a given memory operand.

        Args:
            core_id_a : The first core id.
            core_id_b : The second core id.
            mem_op_a : The memory operand for the tensor in core a.
            mem_op_b : The memory operand for the tensor in core b.
        """
        core_a = self.get_core(core_id_a)
        core_b = self.get_core(core_id_b)
        top_memory_instance_a = next(
            (
                ml.memory_instance
                for ml, out_degree in core_a.memory_hierarchy.out_degree()
                if out_degree == 0 and mem_op_a in ml.operands
            )
        )
        top_memory_instance_b = next(
            (
                ml.memory_instance
                for ml, out_degree in core_b.memory_hierarchy.out_degree()
                if out_degree == 0 and mem_op_b in ml.operands
            )
        )
        return top_memory_instance_a is top_memory_instance_b

    @property
    def core_list(self) -> list[Core]:
        return list(self.cores.node_list)

    def __str__(self) -> str:
        return f"Accelerator({self.name})"

    def __repr__(self) -> str:
        return str(self)

    def __jsonrepr__(self) -> dict[str, Any]:
        """
        JSON representation used for saving this object to a json file.
        """
        return {"name": self.name, "cores": self.cores}
