import logging
import os

from zigzag.cost_model.cost_model import CostModelEvaluation
from zigzag.datatypes import MemoryOperand
from zigzag.hardware.architecture.Core import Core
from zigzag.hardware.architecture.memory_level import MemoryLevel
from zigzag.hardware.architecture.memory_port import DataDirection, PortAllocation
from zigzag.mapping.spatial_mapping import SpatialMapping
from zigzag.stages.CostModelStage import CostModelStage
from zigzag.stages.LomaStage import LomaStage
from zigzag.stages.MainStage import MainStage
from zigzag.stages.reduce_stages import MinimalLatencyStage
from zigzag.stages.SpatialMappingGeneratorStage import SpatialMappingGeneratorStage
from zigzag.stages.Stage import Stage
from zigzag.utils import pickle_deepcopy
from zigzag.workload.Workload import Workload

from stream.classes.hardware.architecture.accelerator import Accelerator
from stream.classes.workload.computation_node import ComputationNode
from stream.utils import load_scme, save_scme
from stream.visualization.node_hw_performances import (
    visualize_node_hw_performances_pickle,
)

logger = logging.getLogger(__name__)


class IntraCoreMappingStage(Stage):
    """
    Class that saves the optimal CME for each valid node-core allocation to the node.
    """

    def __init__(
        self, list_of_callables, *, workload: Workload, accelerator: Accelerator, loma_lpf_limit: int, **kwargs
    ):
        """
        Initialize the stage by:
        - extracting all the unique nodes that will have to be evaluated
        - initializing the valid node-core allocations (which are used later by the InterCoreMappingStage)
        """
        super().__init__(list_of_callables, **kwargs)
        self.workload = workload
        self.accelerator = accelerator
        self.loma_lpf_limit = loma_lpf_limit
        self.loma_show_progress_bar: bool = kwargs.get("loma_show_progress_bar", False)
        self.node_hw_performances_path: str | None = kwargs.get("node_hw_performances_path", None)

        # Extract all unique nodes that will have to be evaluated
        self.unique_nodes: list[ComputationNode] = []
        for node in self.workload.nodes():
            if not isinstance(node, ComputationNode):
                continue
            equal_nodes = list(
                (
                    unique_node
                    for unique_node in self.unique_nodes
                    if node == unique_node and node.group == unique_node.group
                )
            )
            if not equal_nodes:
                self.unique_nodes.append(node)

        # Initialize the valid node-core allocations.
        self.valid_allocations: dict[ComputationNode, list[int]] = {}
        for node in self.unique_nodes:
            assert isinstance(node, ComputationNode), f"IntraCoreMapingStage received node {node} of type {type(node)}."
            assert isinstance(node.possible_core_allocation, list), f"Core allocation is not a list for node {node}."
            self.valid_allocations[node] = node.possible_core_allocation

        # Initialize dict that will store for all unique nodes their intra-core HW performance for the valid node-core
        #  allocations.
        self.node_hw_performances: dict[ComputationNode, dict[Core, CostModelEvaluation]] = {
            unique_node: {} for unique_node in self.unique_nodes
        }  # look-up table

    def run(self):
        logger.info("Start IntraCoreMappingStage.")
        if self.node_hw_performances_path:
            try:
                self.given_node_hw_performances = load_scme(self.node_hw_performances_path)
            except FileNotFoundError:
                self.given_node_hw_performances = None
        else:
            self.given_node_hw_performances = None

        for node in self.unique_nodes:
            # TODO This should never evaluate to true: enforce core_allocation as list everywhere
            if isinstance(self.valid_allocations[node], tuple):
                raise ValueError
                # try:
                #     core_ids = (self.valid_allocations[node][node.group],)
                # except IndexError:
                #     nb_groups = len(set((n.group for n in self.workload.nodes() if n.id == node)))
                #     assert (
                #         len(self.valid_allocations[node]) == 1
                #     ), f"Fixed mapping for {node.name} should contain {nb_groups} entries."
                #     core_ids = (self.valid_allocations[node][0],)
            else:
                core_ids = self.valid_allocations[node]
            for core_id in core_ids:
                core = self.accelerator.get_core(core_id)
                # Offchip memory core doesn't have operational units
                if core.operational_array.total_area == 0:
                    continue
                # It's possible this node might not fully fit within the core's top level memories. If so, we update
                #  the core
                too_large_operands_for_cme = self.check_core_capacity_for_node(core, node)
                # Check if this (node, core) combination is present in the loaded performances pickle
                if (
                    self.given_node_hw_performances is not None
                    and node in self.given_node_hw_performances
                    and self.given_node_hw_performances[node] is not None
                    and core in self.given_node_hw_performances[node]
                ):
                    if not isinstance(self.given_node_hw_performances[node][core], CostModelEvaluation):
                        raise TypeError(f"Given node_hw_performances for node {node} and core {core} is not a CME.")
                    cme = self.given_node_hw_performances[node][core]
                    self.node_hw_performances[node][core] = cme
                # Check if this (node, core) combination has already been optimized during this run
                elif self.node_hw_performances and any(
                    (node == processed_node for processed_node in self.node_hw_performances)
                ):
                    equal_node = next(
                        processed_node for processed_node in self.node_hw_performances if node == processed_node
                    )
                    if self.node_hw_performances[equal_node] and any(
                        (core == processed_core for processed_core in self.node_hw_performances[equal_node])
                    ):
                        # Find the core that is equal
                        equal_core = next(
                            processed_core
                            for processed_core in self.node_hw_performances[equal_node]
                            if core == processed_core
                        )
                        cme = self.node_hw_performances[equal_node][equal_core]
                        self.node_hw_performances[node][core] = cme
                        self.save_node_hw_performances()
                    # Compute this (node, core) combination's optimal mapping
                    else:
                        # Set the node's core allocation to the core_id we want to extract hw performance for
                        node.set_core_allocation(core_id)
                        # Set the node's spatial mapping to the possible spatial mappings of the current core
                        node.spatial_mapping = core.dataflows if core.dataflows is not None else SpatialMapping.empty()
                        # Initialize the flow that will be followed to extract the optimal HW performance of every
                        #  unique node-core allocation
                        main_stage = self.get_intra_core_mapping_flow(
                            node=node,
                            too_large_operands=too_large_operands_for_cme,
                            core_id=core_id,
                        )
                        answers = main_stage.run()
                        assert len(answers) == 1, "IntraCoreMappingStage's subflow returned more than one CME"
                        cme = answers[0][0]
                        node.core_allocation = None  # Reset the node's core allocation
                        self.node_hw_performances[node][core] = cme
                        self.save_node_hw_performances()  # Save the hw performances dict after every node is finished
        self.visualize_node_hw_performances()
        kwargs = self.kwargs.copy()
        kwargs["workload"] = self.workload
        kwargs["accelerator"] = self.accelerator
        kwargs["node_hw_performances"] = self.node_hw_performances

        logger.info("Finished IntraCoreMappingStage.")
        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], **kwargs)
        for cme, extra_info in sub_stage.run():
            yield cme, extra_info

    def save_node_hw_performances(self):
        if self.node_hw_performances_path:
            parent = os.path.dirname(self.node_hw_performances_path)
            os.makedirs(parent, exist_ok=True)
            save_scme(self.node_hw_performances, self.node_hw_performances_path)
            logger.debug(f"Saved unique CN node HW performance to {self.node_hw_performances_path}.")

    def visualize_node_hw_performances(self):
        if "visualize_node_hw_performances_path" in self.kwargs:
            if "visualize_node_hw_performances_path":
                # Get the scale factors
                scale_factors = {
                    n.id: len(list(cn for cn in self.workload if cn == n)) for n in self.node_hw_performances
                }
                # Run the visualization
                visualize_node_hw_performances_pickle(
                    self.node_hw_performances,
                    scale_factors,
                    self.kwargs["visualize_node_hw_performances_path"],
                )

    def get_intra_core_mapping_flow(self, node: ComputationNode, too_large_operands: list[MemoryOperand], core_id: int):
        logger.info(f"Launching intra-core mapping optimization for {node} -> core {core_id} ...")

        if too_large_operands:
            accelerator = self.add_offchip_to_core(core_id, too_large_operands, node.id)
        else:
            accelerator = self.accelerator

        main_stage = MainStage(
            [  # Initializes the MainStage as entry point
                MinimalLatencyStage,
                SpatialMappingGeneratorStage,  # Generates multiple spatial mappings (SM)
                MinimalLatencyStage,  # Reduces all CMEs, returning minimal latency one
                LomaStage,  # Generates multiple temporal mappings (TM)
                CostModelStage,  # Evaluates generated SM and TM through cost model
            ],
            layer=node,
            accelerator=accelerator,  # required by a number of stages
            loma_lpf_limit=self.loma_lpf_limit,  # required by LomaStage
            loma_show_progress_bar=self.loma_show_progress_bar,
        )
        return main_stage

    def check_given_node_hw_performances(self):
        """Check if the given node_hw_performances nodes have the same characteristics as our current given mapping.
        Right now, this just checks if every unique node is present in this dictionary. If not, the user should rerun.
        A more accurate check would be to check the CME's inside of node_hw_performances and make sure the cores are the
          same as the ones we have now.
        In this function, we can't use the equality operator on the two nodes, as the current nodes might not yet have a
          core allocation.
        Args:
            node_hw_performances (dict): A dictionary containing for every unique node a dict like {core:optimal_CME for
              core in valid_cores}
        """
        for unique_node in self.unique_nodes:
            if unique_node not in self.given_node_hw_performances.keys():
                raise ValueError(
                    f"Given node_hw_performances don't match current unique nodes. There is no entry for unique_node "
                    f"{unique_node}."
                )

    def check_core_capacity_for_node(self, core: Core, node: ComputationNode) -> list[MemoryOperand]:
        """Check if we need to add a DRAM memory to the given core for the given node.
        The DRAM can span one or more operands, based on the total size of available on-chip memory
        and the stored operands inside each memory.

        Args:
            core (Core): The core onto which we want to map the node
            node (ComputationNode): The node we want to map onto the core

        Returns:
            list: A list of memory operands for which the capacity on the core is insufficient.
        """
        too_large_operands_for_cme = []

        # Step 1: get all the unique top level memories of the core
        memory_hierarchy_dict = core.mem_hierarchy_dict
        top_memories = [memory[-1] for (mem_op, memory) in memory_hierarchy_dict.items()]
        unique_top_memories = set(top_memories)

        # Step 2: for each top level memory, for each operand this memory holds, calculate the required capacity
        # (in bit) for holding them
        memory_operand_link = node.memory_operand_links
        constant_operands = node.constant_operands
        output_operand = node.output_operand
        for top_memory in unique_top_memories:
            top_level_capacity = top_memory.memory_instance.size
            memory_operands = list(top_memory.mem_level_of_operands.keys())
            layer_operands = [memory_operand_link.mem_to_layer_op(mem_operand) for mem_operand in memory_operands]
            bits_to_be_stored_in_top_level = {}
            for layer_operand, memory_operand in zip(layer_operands, memory_operands):
                # if the operand is constant operand (e.g. 'W' and the first layer's 'I') or output operand, required
                # capacity is gotten from the node directly
                if layer_operand in constant_operands + [output_operand]:
                    bits_to_be_stored_in_top_level[memory_operand] = node.operand_size_bit[layer_operand]
                # if the operand is variable input operand, required capacity is calculated by summing up the the total
                #  data amount on the in edges,
                # which can be larger than the ideal required data size
                else:
                    bits_to_be_stored_in_top_level[memory_operand] = 0
                    in_edges_data = [data for (_, _, data) in self.workload.in_edges(node, data=True)]
                    for edge_data in (d for d in in_edges_data if "operand" in d and d["operand"] == layer_operand):
                        bits_to_be_stored_in_top_level[memory_operand] += edge_data["bits"]
            total_required_capacity = sum(bits_to_be_stored_in_top_level.values())

            # Step 3: compare the total required capacity with the top level memory capacity
            if total_required_capacity <= top_level_capacity:
                pass
            else:
                # when the memory capacity is smaller than the requirement,
                # sort the required capacity of each operand that shares this memory based on the operand's required
                #  size, from small to large
                # fit the operands to the memory from small to large
                bits_to_be_stored_in_top_level = {
                    k: v for k, v in sorted(bits_to_be_stored_in_top_level.items(), key=lambda item: item[1])
                }
                nb_operands_in_top_level = len(bits_to_be_stored_in_top_level)
                while top_level_capacity < sum(
                    list(bits_to_be_stored_in_top_level.values())[:nb_operands_in_top_level]
                ):
                    nb_operands_in_top_level -= 1
                    if nb_operands_in_top_level == 0:
                        break
                operands_stored_in_top_level = list(bits_to_be_stored_in_top_level.keys())[:nb_operands_in_top_level]
                operands_stored_in_offchip = list(bits_to_be_stored_in_top_level.keys())[nb_operands_in_top_level:]

                # Step 4: Check when some operand(s) fit in the top level core memory, and some cannot fit
                # (too_large_operands), the top level core memory has enough space for supporting the SU of not-fitted
                #  operands
                if not operands_stored_in_top_level or not operands_stored_in_offchip:
                    pass
                else:
                    rest_capacity = self.get_top_level_memory_rest_capacity(
                        operands_stored_in_top_level,
                        bits_to_be_stored_in_top_level,
                        top_level_capacity,
                    )
                    required_capacity = self.get_too_large_operands_minimal_required_capacity_in_top_level_memory(
                        operands_stored_in_offchip, core.dataflows, node, core
                    )
                    while rest_capacity < required_capacity:
                        # put_the_largest operands_stored_in_top_level to operands_stored_in_offchip
                        nb_operands_in_top_level -= 1
                        operands_stored_in_top_level = list(bits_to_be_stored_in_top_level.keys())[
                            :nb_operands_in_top_level
                        ]
                        operands_stored_in_offchip = list(bits_to_be_stored_in_top_level.keys())[
                            nb_operands_in_top_level:
                        ]
                        if not operands_stored_in_top_level:
                            break
                        rest_capacity = self.get_top_level_memory_rest_capacity(
                            operands_stored_in_top_level,
                            bits_to_be_stored_in_top_level,
                            top_level_capacity,
                        )
                        required_capacity = self.get_too_large_operands_minimal_required_capacity_in_top_level_memory(
                            operands_stored_in_offchip, core.dataflows, node
                        )

                too_large_operands_for_cme += operands_stored_in_offchip
        return too_large_operands_for_cme

    @staticmethod
    def get_top_level_memory_rest_capacity(
        operands_stored_in_top_level,
        bits_to_be_stored_in_top_level,
        top_level_capacity_bits,
    ) -> int:
        """Calculate the remaining capacity in the top level core memory after storing the operands_stored_in_top_level

        Args:
            operands_stored_in_top_level (list): list of operands that can fit in the top memory level of the core
            bits_to_be_stored_in_top_level (dict): the data size in bit for each variable operands
            top_level_capacity_bits (int): the total capacity of the top level core memory

        Returns:
            int: the memory capacity left after storing the operands_stored_in_top_level
        """
        rest_capacity = top_level_capacity_bits
        for mem_operand in operands_stored_in_top_level:
            rest_capacity -= bits_to_be_stored_in_top_level[mem_operand]
        return rest_capacity

    def get_too_large_operands_minimal_required_capacity_in_top_level_memory(
        self,
        operands_stored_in_offchip: list[MemoryOperand],
        dataflows: SpatialMapping,
        node: ComputationNode,
        core: Core,
    ) -> int:
        """Calculate the required capacity in the top level core memory for operands_stored_in_offchip due to spatial
        unrolling

        Args:
            operands_stored_in_offchip (list): list of operands that cannot fit in the top memory level of the core
            dataflows (list of dict): the dataflows (spatial mappings) that current core supports
            node (ComputationNode): The computational node we want to map onto the core

        Returns:
            int: the required memory capacity in the top memory of the core for operands_stored_in_offchip
        """

        def get_lowest_level_unrolled_memory_capacity(memory_operand: MemoryOperand):
            memory_level = core.memory_hierarchy.get_memory_levels(memory_operand)[0]
            return memory_level.memory_instance.size * memory_level.unroll_count

        unroll_dict: dict[MemoryOperand, int] = {}
        for mem_operand in operands_stored_in_offchip:
            capacity = get_lowest_level_unrolled_memory_capacity(mem_operand)
            unroll_dict[mem_operand] = capacity
        return round(sum(unroll_dict.values()))

    def add_offchip_to_core(self, core_id: int, too_large_operands: list[MemoryOperand], layer_idx: int):
        """Add the offchip memory as the top level memory of the core with core_id in a copy of the accelerator

        Args:
            core_id (int): The id of the core to which we want to add the off-chip memory for cost evaluation.
            too_large_operands (list): The memory operands the off-chip memory should store.
            layer_idx (int): workload layer index.
        """
        logger.warning(
            f"Adding offchip memory for core_id={core_id}, layer={layer_idx}, memory_operands={too_large_operands}."
        )
        updated_accelerator: Accelerator = pickle_deepcopy(self.accelerator)
        core: Core = updated_accelerator.get_core(core_id)
        offchip_core: Core = pickle_deepcopy(self.accelerator.get_core(self.accelerator.offchip_core_id))
        # Sanity checks
        # Make sure that there is only one offchip memory
        offchip_memory_levels = offchip_core.memory_hierarchy.mem_level_list
        assert (
            len(offchip_memory_levels) == 1
        ), "There is more than one offchip memory, unsure which one to take for intra core mapping"
        offchip_memory_level: MemoryLevel = pickle_deepcopy(offchip_memory_levels[0])
        offchip_memory_instance = offchip_memory_level.memory_instance
        offchip_memory_operands = too_large_operands
        # Recreate the port allocation
        offchip_port_alloc_raw: dict[MemoryOperand, dict[DataDirection, str]] = {}
        for memory_operand in offchip_memory_operands:
            offchip_port_alloc_raw[memory_operand] = offchip_memory_level.port_alloc_raw.get_alloc_for_mem_op(
                memory_operand
            )

        offchip_port_alloc = PortAllocation(offchip_port_alloc_raw)
        offchip_served_dimensions = offchip_memory_level.served_dimensions

        # Easiest way to add the offchip memory level is to do an 'add_memory()' call to the MemoryHierarchy
        core.memory_hierarchy.add_memory(
            offchip_memory_instance,
            offchip_memory_operands,
            offchip_port_alloc,
            offchip_served_dimensions,
        )
        core.recalculate_memory_hierarchy_information()  # Recalculates some internals of the Core object

        return updated_accelerator
