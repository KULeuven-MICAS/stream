import logging
from math import prod

from zigzag.cost_model.cost_model import CostModelEvaluation
from zigzag.datatypes import MemoryOperand
from zigzag.mapping.data_movement import FourWayDataMoving

from stream.stages.context import StageContext
from stream.stages.stage import Stage, StageCallable
from stream.utils import get_too_large_operands, get_top_level_inst_bandwidth
from stream.workload.computation.computation_node import ComputationNode

logger = logging.getLogger(__name__)


class SetFixedAllocationPerformanceStage(Stage):
    REQUIRED_FIELDS = ("workload", "accelerator", "cost_lut")

    def __init__(
        self,
        list_of_callables: list[StageCallable],
        ctx: StageContext,
    ):
        super().__init__(list_of_callables, ctx)
        self.accelerator = self.ctx.require_value("accelerator", self.__class__.__name__)
        self.workload = self.ctx.require_value("workload", self.__class__.__name__)
        self.cost_lut = self.ctx.require_value("cost_lut", self.__class__.__name__)
        self.latency_attr = self.ctx.get("latency_attr", "latency_total2")
        self.fix_all = self.ctx.get("fix_all", False)

    def run(self):
        logger.info("Start SetFixedAllocationPerformanceStage.")

        self.check_and_fix_chosen_core_allocation()

        # Set the performance of all nodes that have a fixed allocation
        self.set_fixed_allocation_performance()
        logger.info("Finished SetFixedAllocationPerformanceStage.")

        self.ctx.set(workload=self.workload, accelerator=self.accelerator, cost_lut=self.cost_lut)
        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], self.ctx)
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
                cme = self.cost_lut.get_cost(equal_node, core)
                latency = int(cme.ideal_cycle)
                # Scale latency based on utilization of core
                latency = int(latency * 100 / node.kernel.utilization)
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
        onchip_energy = getattr(cme, "energy_total", 0)  # initialize the on-chip energy as total energy
        # If there is a too_large_operand, we separate the off-chip energy.
        offchip_energy = 0
        mem_energy_breakdown = getattr(cme, "mem_energy_breakdown", {}) or {}
        for too_large_operand in too_large_operands:
            if not mem_energy_breakdown:
                continue
            layer_operand = cme.layer.memory_operand_links.mem_to_layer_op(too_large_operand)
            layer_operand_offchip_energy = mem_energy_breakdown[layer_operand][-1]
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
                                f"{len(node.possible_core_allocation)}. "
                                f"Make sure the layer is large enough to be tiled across the cores."
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
