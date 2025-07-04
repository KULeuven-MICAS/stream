import logging
import os
from math import ceil, prod
from typing import Any

from zigzag.cost_model.cost_model import CostModelEvaluation
from zigzag.datatypes import Constants, MemoryOperand
from zigzag.hardware.architecture.memory_level import MemoryLevel
from zigzag.hardware.architecture.memory_port import DataDirection, PortAllocation
from zigzag.mapping.temporal_mapping import TemporalMappingType
from zigzag.stages.evaluation.cost_model_evaluation import CostModelStage
from zigzag.stages.mapping.spatial_mapping_generation import SpatialMappingGeneratorStage
from zigzag.stages.mapping.temporal_mapping_generator_stage import TemporalMappingGeneratorStage
from zigzag.utils import pickle_deepcopy

from stream.hardware.architecture.accelerator import Accelerator
from stream.hardware.architecture.core import Core
from stream.stages.generation.layer_stacks_generation import STACK_T
from stream.stages.stage import MainStage, Stage, StageCallable
from stream.utils import CostModelEvaluationLUT, contains_wildcard, get_top_level_inst_bandwidth, get_unique_nodes
from stream.visualization.cost_model_evaluation_lut import (
    visualize_cost_lut_pickle,
)
from stream.workload.computation.computation_node import ComputationNode
from stream.workload.onnx_workload import ComputationNodeWorkload

logger = logging.getLogger(__name__)


class ZigZagCoreMappingEstimationStage(Stage):
    """
    Class that saves the optimal CME for each valid node-core allocation to the node.
    """

    def __init__(
        self,
        list_of_callables: list[StageCallable],
        *,
        workload: ComputationNodeWorkload,
        accelerator: Accelerator,
        loma_lpf_limit: int,
        cost_lut_path: str,
        **kwargs: Any,
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
        self.cost_lut_path = cost_lut_path
        self.visualize_cost_lut_path = os.path.splitext(self.cost_lut_path)[0] + ".png"
        self.loma_show_progress_bar: bool = kwargs.get("loma_show_progress_bar", False)
        self.layer_stacks: list[STACK_T] = kwargs["layer_stacks"]
        self.temporal_mapping_type: TemporalMappingType = kwargs["temporal_mapping_type"]

        # Extract all unique nodes that will have to be evaluated
        self.unique_nodes = get_unique_nodes(self.workload)

        assert all(isinstance(node, ComputationNode) for node in self.unique_nodes), (
            "ZigZagCoreMappingEstimationStage received a non-ComputationNode."
        )
        assert all(isinstance(node.possible_core_allocation, list) for node in self.unique_nodes), (
            "ZigZagCoreMappingEstimationStage received a node with a non-list core allocation."
        )

        self.valid_allocations: dict[ComputationNode, list[int]] = {
            node: node.possible_core_allocation for node in self.unique_nodes
        }
        self.cost_lut = CostModelEvaluationLUT(self.cost_lut_path)

    def run(self):
        logger.info("Start ZigZagCoreMappingEstimationStage.")
        self.update_cost_lut()
        self.visualize_cost_lut()
        logger.info("Finished ZigZagCoreMappingEstimationStage.")

        kwargs = self.kwargs.copy()
        kwargs["workload"] = self.workload
        kwargs["accelerator"] = self.accelerator
        kwargs["cost_lut"] = self.cost_lut
        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], **kwargs)
        yield from sub_stage.run()

    def update_cost_lut(self):
        for node in self.unique_nodes:
            core_ids = self.valid_allocations[node]
            for core_id in core_ids:
                core = self.accelerator.get_core(core_id)
                # Offchip memory core doesn't have operational units
                if core.operational_array.total_unit_count == 0:
                    continue
                # If the (node, core) combination has already been optimized, we skip it
                if self.cost_lut.has_cme(node, core):
                    continue
                # If an equal performance has already been computed, we take it
                equal_node = self.cost_lut.get_equal_node(node)
                equal_core = self.cost_lut.get_equal_core(equal_node, core) if equal_node else None
                if equal_node and equal_core:
                    cme = pickle_deepcopy(self.cost_lut.get_cme(equal_node, equal_core))
                    # Update the CME attributes for this node-core combination
                    cme.layer.core_allocation = [core_id]
                    cme.core_id = core_id
                    self.cost_lut.add_cme(node, core, cme, allow_overwrite=False)
                    continue
                else:
                    node_duplicate = pickle_deepcopy(node)
                    # Remove duplicate cores with same id in case the core definition has changed
                    self.cost_lut.remove_cores_with_same_id(node, core)
                    # Compute the optimal performance for this node-core combination. If this node does not fully fit
                    # within the core's top level memories, we update the core to include an offchip memory.
                    too_large_operands_for_cme = self.check_core_capacity_for_node(core, node_duplicate)
                    # # ! --- ensure all constant weights are accessed via blocking behavior i.s.o. transfer -RG
                    # for layer_op in node.constant_operands:
                    #     mem_op = node.memory_operand_links.layer_to_mem_op(layer_op)
                    #     if mem_op not in too_large_operands_for_cme and node.operand_precision[layer_op] > 0:
                    #         too_large_operands_for_cme.append(mem_op)
                    # # ! ---
                    node_duplicate.set_chosen_core_allocation(core_id)

                    # Attempt to override the node's spatial mapping based on the core's dataflow
                    if core.dataflows:
                        node_duplicate.spatial_mapping = core.dataflows

                    cme = self.run_zigzag(node_duplicate, too_large_operands_for_cme, core_id)
                    cme = self.increase_cc_per_op(cme, node.type)

                    node_duplicate.set_chosen_core_allocation(None)  # Reset the node's chosen core allocation
                    self.cost_lut.add_cme(node, core, cme, allow_overwrite=False)
            self.cost_lut.save()

    def get_cc_per_op(self, op_type: str):
        """Return the number of cycles that the operational units need to finish the given operation."""
        match op_type:
            case "silu":
                return 4
            case "sigmoid":
                return 4
            case "exp":
                return 4
            case _:
                return 1

    def increase_cc_per_op(self, cme: CostModelEvaluation, op_type: str):
        """Given a ZigZag that assumes each operation takes one cycle, generate a new one that takes into account that
        the operation might take more than one cycle."""
        cc_per_op = self.get_cc_per_op(op_type)
        if cc_per_op > 1:
            logger.warning(f"Setting cycles per mac of {op_type} node to {cc_per_op}")

        new_cme = CostModelEvaluation(
            accelerator=cme.accelerator,
            layer=cme.layer,
            spatial_mapping=cme.spatial_mapping,
            spatial_mapping_int=cme.spatial_mapping_int,
            temporal_mapping=cme.temporal_mapping,
            access_same_data_considered_as_no_access=cme.access_same_data_considered_as_no_access,
            cycles_per_op=cc_per_op,
        )

        return new_cme

    def visualize_cost_lut(self):
        scale_factors = {
            n: len([cn for cn in self.workload.node_list if cn.has_same_performance(n)])
            for n in self.cost_lut.get_nodes()
        }
        visualize_cost_lut_pickle(self.cost_lut, scale_factors, self.visualize_cost_lut_path)

    def run_zigzag(
        self, node: ComputationNode, too_large_operands: list[MemoryOperand], core_id: int
    ) -> CostModelEvaluation:
        """Run the ZigZag flow to estimate performance of a given node on a core."""

        main_stage = self.instantiate_zigzag_flow(node, too_large_operands, core_id)
        logger.info(f"Launching intra-core mapping optimization for {node} -> core {core_id} ...")
        answers = main_stage.run()
        assert len(answers) == 1, "ZigZagCoreMappingEstimationStage's subflow returned more than one CME"
        cme: CostModelEvaluation = answers[0][0]  # type: ignore
        return cme

    def instantiate_zigzag_flow(self, node: ComputationNode, too_large_operands: list[MemoryOperand], core_id: int):
        """Instantiate a runnable ZigZag mainstage"""
        core = self.accelerator.get_core(core_id)
        nb_parallel_nodes: int = (
            1 if contains_wildcard(node.inter_core_tiling) else prod(size for _, size in node.inter_core_tiling)
        )  # type: ignore

        if too_large_operands:
            core = self.add_offchip_to_core(core, too_large_operands, node.id)

        main_stage = MainStage(
            [  # Initializes the MainStage as entry point
                MinimalBandwidthLatencyStage,  # type: ignore
                SpatialMappingGeneratorStage,  # Generates multiple spatial mappings (SM)
                MinimalBandwidthLatencyStage,  # Reduces all CMEs, returning minimal EDP one
                TemporalMappingGeneratorStage,  # Generates multiple temporal mappings (TM)
                CostModelStage,  # Evaluates generated SM and TM through cost model
            ],
            layer=node,
            accelerator=core,  # Accelerator in zigzag corresponds to Core in stream
            loma_lpf_limit=self.loma_lpf_limit,  # required by LomaEngine
            loma_show_progress_bar=self.loma_show_progress_bar,
            temporal_mapping_type=self.temporal_mapping_type,
            nb_parallel_nodes=nb_parallel_nodes,
            has_dram_level=(len(too_large_operands) > 0),
        )
        return main_stage

    def check_core_capacity_for_node(self, core: Core, node: ComputationNode) -> list[MemoryOperand]:
        """Check if we need to add a DRAM memory to the given core for the given node.
        The DRAM can span one or more operands, based on the total size of available on-chip memory
        and the stored operands inside each memory.

        Args:
            core: The core onto which we want to map the node
            node: The node we want to map onto the core

        Returns:
            A list of memory operands for which the capacity on the core is insufficient.
        """
        too_large_operands_for_cme: list[MemoryOperand] = []

        # ! Always use blocking for single-stack nodes -RG
        stack_this_node = next(stack for stack in self.layer_stacks if node.id in stack)
        node_is_alone_in_stack = len(stack_this_node) == 1
        if node_is_alone_in_stack:
            return node.memory_operand_links.mem_operands

        # Step 1: get all the unique top level memories of the core
        memory_hierarchy_dict = core.mem_hierarchy_dict
        top_memories = [memory[-1] for (_, memory) in memory_hierarchy_dict.items()]
        unique_top_memories = set(top_memories)

        # Step 2: for each top level memory, for each operand this memory holds, calculate the required capacity
        # (in bit) for holding them
        memory_operand_link = node.memory_operand_links
        for top_memory in unique_top_memories:
            top_level_capacity = top_memory.memory_instance.size
            memory_operands = list(top_memory.mem_level_of_operands.keys())
            layer_operands = [memory_operand_link.mem_to_layer_op(mem_operand) for mem_operand in memory_operands]
            bits_to_be_stored_in_top_level: dict[MemoryOperand, int] = {}
            for layer_operand, memory_operand in zip(layer_operands, memory_operands, strict=False):
                nb_bits = node.operand_size_bit[layer_operand]
                bits_to_be_stored_in_top_level[memory_operand] = nb_bits
            total_required_capacity = sum(bits_to_be_stored_in_top_level.values())

            # Step 3: compare the total required capacity with the top level memory capacity
            if top_level_capacity < total_required_capacity:
                # when the memory capacity is smaller than the requirement, sort the required capacity of each operand
                # that shares this memory based on the operand's required size, from small to large
                # Fit the operands to the memory from small to large
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
                        operands_stored_in_offchip, core
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
                            operands_stored_in_offchip, core
                        )

                too_large_operands_for_cme += operands_stored_in_offchip
        return too_large_operands_for_cme

    @staticmethod
    def get_top_level_memory_rest_capacity(
        operands_stored_in_top_level: list[MemoryOperand],
        bits_to_be_stored_in_top_level: dict[MemoryOperand, int],
        top_level_capacity_bits: int,
    ) -> int:
        """Calculate the remaining capacity in the top level core memory after storing the operands_stored_in_top_level

        Args:
            operands_stored_in_top_level: list of operands that can fit in the top memory level of the core
            bits_to_be_stored_in_top_level: the data size in bit for each variable operands
            top_level_capacity_bits: the total capacity of the top level core memory

        Returns:
            The memory capacity left after storing the operands_stored_in_top_level
        """
        rest_capacity = top_level_capacity_bits
        for mem_operand in operands_stored_in_top_level:
            rest_capacity -= bits_to_be_stored_in_top_level[mem_operand]
        return rest_capacity

    def get_too_large_operands_minimal_required_capacity_in_top_level_memory(
        self,
        operands_stored_in_offchip: list[MemoryOperand],
        core: Core,
    ) -> int:
        """Calculate the required capacity in the top level core memory for operands_stored_in_offchip due to spatial
        unrolling

        Args:
            operands_stored_in_offchip: list of operands that cannot fit in the top memory level of the core
            dataflows (list of dict): the dataflows (spatial mappings) that current core supports
            node (ComputationNode): The computational node we want to map onto the core

        Returns:
            The required memory capacity in the top memory of the core for operands_stored_in_offchip
        """

        def get_lowest_level_unrolled_memory_capacity(memory_operand: MemoryOperand):
            memory_level = core.memory_hierarchy.get_memory_levels(memory_operand)[0]
            return memory_level.memory_instance.size * memory_level.unroll_count

        unroll_dict: dict[MemoryOperand, int] = {}
        for mem_operand in operands_stored_in_offchip:
            capacity = get_lowest_level_unrolled_memory_capacity(mem_operand)
            unroll_dict[mem_operand] = capacity
        return round(sum(unroll_dict.values()))

    def add_offchip_to_core(self, core: Core, too_large_operands: list[MemoryOperand], layer_idx: int):
        """Add the offchip memory as the top level memory of the core with core_id in a copy of the accelerator

        Args:
            core_id: The id of the core to which we want to add the off-chip memory for cost evaluation.
            too_large_operands: The memory operands the off-chip memory should store.
            layer_idx: workload layer index.
        """
        assert self.accelerator.offchip_core_id is not None
        logger.warning(f"Adding offchip memory for {core}, layer={layer_idx}, memory_operands={too_large_operands}.")
        offchip_core: Core = pickle_deepcopy(self.accelerator.get_core(self.accelerator.offchip_core_id))

        # Sanity checks: make sure that there is only one offchip memory
        offchip_memory_levels = offchip_core.memory_hierarchy.mem_level_list
        assert len(offchip_memory_levels) == 1, (
            "There is more than one offchip memory, unsure which one to take for intra core mapping"
        )

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

        # Create new core with updated memory hierarchy
        updated_core: Core = pickle_deepcopy(core)
        updated_core.memory_hierarchy.add_memory(
            offchip_memory_instance,
            offchip_memory_operands,
            offchip_port_alloc,
            offchip_served_dimensions,
        )
        updated_core.recalculate_memory_hierarchy_information()  # Recalculates some internals of the Core object

        return updated_core


class MinimalBandwidthLatencyStage(Stage):
    """Class that keeps yields only the cost model evaluation that has minimal objective function of all cost model
    evaluations generated by it's substages created by list_of_callables.
    The objective function is defined as:
        `ceil(nb_parallel_nodes * required_dram_bandwidth / total_dram_bandwidth) * latency`
    """

    def __init__(
        self,
        list_of_callables: list[StageCallable],
        *,
        reduce_minimal_keep_others: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the compare stage.
        """
        super().__init__(list_of_callables, **kwargs)
        self.keep_others = reduce_minimal_keep_others
        accelerator: Core = kwargs["accelerator"]
        self.nb_parallel_nodes: int = kwargs.get("nb_parallel_nodes", 1)
        self.has_dram_level: bool = kwargs.get("has_dram_level", False)

        self.mem_ops_with_dram: list[MemoryOperand] = []
        self.mem_ops = list(accelerator.memory_hierarchy.operands)
        self.total_dram_bandwidth: int | None = None

        if self.has_dram_level:
            nb_levels_per_op = [len(accelerator.memory_hierarchy.get_memory_levels(op)) for op in self.mem_ops]
            dram_mem_level = max(nb_levels_per_op)  # start at 1
            self.mem_ops_with_dram = [
                op for op, nb_levels in zip(self.mem_ops, nb_levels_per_op, strict=False) if nb_levels == dram_mem_level
            ]
            ports = accelerator.get_top_memory_instance(self.mem_ops_with_dram[0]).ports
            self.total_dram_bandwidth = max((port.bw_max for port in ports), default=0)

    def get_used_dram_bandwidth_for_op(self, cme: CostModelEvaluation, mem_op: MemoryOperand):
        if mem_op not in self.mem_ops_with_dram:
            return 0
        bw_per_direction = get_top_level_inst_bandwidth(cme, mem_op)
        total_bw = sum(bw_per_direction.data.values())
        return total_bw

    def objective_function(self, cme: CostModelEvaluation) -> float:
        """
        # TODO this does not cover all cases
        """
        latency: int = int(cme.latency_total2)

        if not self.has_dram_level:
            return latency

        assert self.total_dram_bandwidth is not None

        match len(self.mem_ops_with_dram):
            case 1:
                total_used_dram_bw = self.nb_parallel_nodes * self.get_used_dram_bandwidth_for_op(
                    cme, self.mem_ops_with_dram[0]
                )
            case 2:
                # Assume that 1 operand is broadcasted to all cores and only needs 1 simultaneous transfer for all cores
                # We don't know which operand is broadcasted, so just pick one that is not the output
                broadcast_op = next(op for op in self.mem_ops_with_dram if op != Constants.OUTPUT_MEM_OP)
                other_op = next(op for op in self.mem_ops_with_dram if op != broadcast_op)
                bw_for_broadcasting = 1 * self.get_used_dram_bandwidth_for_op(cme, broadcast_op)
                bw_for_blocking = self.nb_parallel_nodes * self.get_used_dram_bandwidth_for_op(cme, other_op)
                total_used_dram_bw = bw_for_blocking + bw_for_broadcasting
            case 3:
                # We don't know broadcast op, just pick one that is not the output
                broadcast_op = next(op for op in self.mem_ops_with_dram if op != Constants.OUTPUT_MEM_OP)
                other_ops = [op for op in self.mem_ops_with_dram if op != broadcast_op]

                bw_for_broadcasting = 1 * self.get_used_dram_bandwidth_for_op(cme, broadcast_op)
                bw_for_blocking = self.nb_parallel_nodes * sum(
                    self.get_used_dram_bandwidth_for_op(cme, mem_op) for mem_op in other_ops
                )
                total_used_dram_bw = bw_for_blocking + bw_for_broadcasting
            case _:
                raise NotImplementedError

        return ceil(total_used_dram_bw / self.total_dram_bandwidth) * latency

    def run(self):
        """! Run the compare stage by comparing a new cost model output with the current best found result."""
        sub_list_of_callables = self.list_of_callables[1:]
        substage: Stage = self.list_of_callables[0](sub_list_of_callables, **self.kwargs)

        other_cmes: list[tuple[CostModelEvaluation, Any]] = []
        best_cme: CostModelEvaluation | None = None
        for cme, extra_info in substage.run():
            assert isinstance(cme, CostModelEvaluation)
            if best_cme is None or self.objective_function(cme) < self.objective_function(best_cme):
                best_cme = cme
            if self.keep_others:
                other_cmes.append((cme, extra_info))

        assert best_cme is not None
        yield best_cme, other_cmes
