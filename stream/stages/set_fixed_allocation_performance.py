import logging
from math import prod
from typing import Any

from zigzag.cost_model.cost_model import CostModelEvaluation
from zigzag.datatypes import MemoryOperand
from zigzag.mapping.data_movement import FourWayDataMoving

from stream.hardware.architecture.accelerator import Accelerator
from stream.stages.stage import Stage, StageCallable
from stream.utils import (
    CostModelEvaluationLUT,
    get_too_large_operands,
    get_top_level_inst_bandwidth,
)
from stream.workload.computation.computation_node import ComputationNode
from stream.workload.onnx_workload import ComputationNodeWorkload

logger = logging.getLogger(__name__)


class SetFixedAllocationPerformanceStage(Stage):
    def __init__(
        self,
        list_of_callables: list[StageCallable],
        *,
        workload: ComputationNodeWorkload,
        accelerator: Accelerator,
        cost_lut: CostModelEvaluationLUT,
        **kwargs: Any,
    ):
        super().__init__(list_of_callables, **kwargs)
        self.accelerator = accelerator
        self.workload = workload
        self.cost_lut = cost_lut
        self.latency_attr = kwargs.get("latency_attr", "latency_total2")
        self.fix_all = kwargs.get("fix_all", False)

    def run(self):
        logger.info("Start SetFixedAllocationPerformanceStage.")

        self.check_and_fix_chosen_core_allocation()

        # Set the performance of all nodes that have a fixed allocation
        self.set_fixed_allocation_performance()
        logger.info("Finished SetFixedAllocationPerformanceStage.")

        kwargs = self.kwargs.copy()
        kwargs["workload"] = self.workload
        kwargs["accelerator"] = self.accelerator
        kwargs["cost_lut"] = self.cost_lut
        sub_stage = self.list_of_callables[0](
            self.list_of_callables[1:],
            **kwargs,
        )
        yield from sub_stage.run()

    def set_fixed_allocation_performance(self):
        for node in self.workload.node_list:
            # ! It is still possible nodes don't have a core allocation here! -RG
            if isinstance(node.chosen_core_allocation, int):
                core_id = node.chosen_core_allocation
                if core_id is None:
                    raise ValueError(f"Node {node} has fixed allocation but the chosen_core_allocation was not set.")
                equal_node = self.cost_lut.get_equal_node(node)
                assert equal_node is not None, f"{node} has fixed allocation but no equal node found."
                core = self.accelerator.get_core(core_id)
                cme = self.cost_lut.get_cme(equal_node, core)
                latency = getattr(cme, self.latency_attr)
                too_large_operands = get_too_large_operands(cme, self.accelerator, core_id=core_id)
                onchip_energy, offchip_energy = self.get_energy_distribution(cme, too_large_operands)

                # Get the required offchip bandwidth during the execution of the node for all directions
                bandwidth_scaling = cme.ideal_temporal_cycle / latency
                offchip_bandwidth_per_op: dict[MemoryOperand, FourWayDataMoving] = {
                    mem_op: get_top_level_inst_bandwidth(cme, mem_op, bandwidth_scaling)
                    for mem_op in too_large_operands
                }
                self.set_hw_performance_node(node, onchip_energy, offchip_energy, latency, core_id)
                node.set_too_large_operands(too_large_operands.copy())
                node.set_offchip_bandwidth(offchip_bandwidth_per_op)

    def get_energy_distribution(
        self, cme: CostModelEvaluation, too_large_operands: list[MemoryOperand]
    ) -> tuple[float, float]:
        onchip_energy = cme.energy_total  # initialize the on-chip energy as total energy
        # If there is a too_large_operand, we separate the off-chip energy.
        offchip_energy = 0
        for too_large_operand in too_large_operands:
            layer_operand = cme.layer.memory_operand_links.mem_to_layer_op(too_large_operand)
            layer_operand_offchip_energy = cme.mem_energy_breakdown[layer_operand][-1]
            offchip_energy += layer_operand_offchip_energy
            onchip_energy -= layer_operand_offchip_energy
        return onchip_energy, offchip_energy

    @staticmethod
    def set_hw_performance_node(
        node: ComputationNode,
        onchip_energy: float,
        offchip_energy: float,
        runtime: int,
        chosen_core_allocation: int,
    ):
        """Set the hardware performance and core_allocation of the given node.

        Args:
            node (Node): The node of which to set the
            onchip_energy (float): on-chip energy of executing this node
            offchip_energy (float): off-chip energy of executing this node
            runtime: runtime of executing this node
            core_allocation: the core_id on which this node will be ran
        """
        node.set_onchip_energy(onchip_energy)
        node.set_offchip_energy(offchip_energy)
        node.set_runtime(runtime)
        node.set_chosen_core_allocation(chosen_core_allocation)

    def check_and_fix_chosen_core_allocation(self):
        """! Check that all nodes in the workload have a chosen_core_allocation.
        If self.fix_all is True, chosen alloc is fixed for all nodes depending on the group and possible alloc.
        If self.fix_all is False, only nodes where the inter_core_tiling match the possible alloc length are fixed.
        """
        for node in self.workload.node_list:
            if node.chosen_core_allocation is None:
                if not self.fix_all:
                    # Check if there is a wildcard in the inter_core_tiling
                    inter_core_factors = [tile_size for _, tile_size in node.inter_core_tiling]
                    if "*" in inter_core_factors:
                        raise ValueError(f"{node} has a wildcard in its inter_core_tiling {node.inter_core_tiling}.")
                    # Only fix the chosen_core_allocation if the inter_core_tiling matches possible_core_allocation len
                    if len(node.possible_core_allocation) != prod(inter_core_factors):
                        if len(inter_core_factors) != 1:
                            raise ValueError(
                                f"{node} has a chosen_core_allocation of None, but the inter_core_tiling "
                                f"{node.inter_core_tiling} does not match the possible_core_allocation length "
                                f"{len(node.possible_core_allocation)}."
                            )
                        continue
                try:
                    core_id = node.possible_core_allocation[node.group]
                except IndexError as exc:
                    raise IndexError(
                        f"{node} has a group {node.group} that is out of bounds for the possible_core_allocation "
                        f"{node.possible_core_allocation}. Allowed: {list(range(len(node.possible_core_allocation)))}."
                    ) from exc
                node.set_chosen_core_allocation(core_id)
                logger.warning(
                    f"{node} does not have a chosen_core_allocation. Setting to {core_id} out of "
                    f"possible allocations {node.possible_core_allocation}."
                )
