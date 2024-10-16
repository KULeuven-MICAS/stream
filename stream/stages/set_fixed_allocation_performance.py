import logging
from typing import Any

from zigzag.cost_model.cost_model import CostModelEvaluation
from zigzag.datatypes import MemoryOperand
from zigzag.mapping.data_movement import MemoryAccesses

from stream.hardware.architecture.accelerator import Accelerator
from stream.stages.stage import Stage, StageCallable
from stream.utils import CostModelEvaluationLUT, get_too_large_operands
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
        node_hw_performances: CostModelEvaluationLUT,
        **kwargs: Any,
    ):
        super().__init__(list_of_callables, **kwargs)
        self.accelerator = accelerator
        self.workload = workload
        self.node_hw_performances = node_hw_performances
        self.latency_attr = kwargs.get("latency_attr", "latency_total2")

    def run(self):
        logger.info("Start SetFixedAllocationPerformanceStage.")
        # Set the performance of all nodes that have a fixed allocation
        self.set_fixed_allocation_performance()
        logger.info("Finished SetFixedAllocationPerformanceStage.")

        kwargs = self.kwargs.copy()
        kwargs["workload"] = self.workload
        kwargs["accelerator"] = self.accelerator
        kwargs["node_hw_performances"] = self.node_hw_performances
        sub_stage = self.list_of_callables[0](
            self.list_of_callables[1:],
            **kwargs,
        )
        for cme, extra_info in sub_stage.run():
            yield cme, extra_info

    def set_fixed_allocation_performance(self):
        for node in self.workload.node_list:
            if isinstance(node.core_allocation, list) and len(node.core_allocation) == 1:
                node.core_allocation_is_fixed = True
                node.set_chosen_core_allocation(node.core_allocation[0])
            if node.core_allocation_is_fixed:
                core_id = node.chosen_core_allocation
                if core_id is None:
                    raise ValueError(f"Node {node} has fixed allocation but the chosen_core_allocation was not set.")
                equal_node = self.node_hw_performances.get_equal_node(node)
                assert equal_node is not None, f"{node} has fixed allocation but no equal node found."
                core = self.accelerator.get_core(core_id)
                cme = self.node_hw_performances.get_cme(equal_node, core)
                latency = getattr(cme, self.latency_attr)
                too_large_operands = get_too_large_operands(cme, self.accelerator, core_id=core_id)
                onchip_energy, offchip_energy = self.get_energy_distribution(cme, too_large_operands)
                # Get the required offchip bandwidth during the execution of the node for all directions
                offchip_bandwidth = self.get_offchip_bandwidth(cme, too_large_operands)
                self.set_hw_performance_node(node, onchip_energy, offchip_energy, latency, core_id)
                node.set_too_large_operands(too_large_operands.copy())
                node.set_offchip_bandwidth(offchip_bandwidth)

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

    def get_offchip_bandwidth(
        self, cme: CostModelEvaluation, too_large_operands: list[MemoryOperand]
    ) -> MemoryAccesses:
        if not too_large_operands:
            return MemoryAccesses(0, 0, 0, 0)
        # If there was offchip memory added for some operands, get the offchip bandwidth required
        assert self.accelerator.offchip_core_id is not None, "Off-chip core id is not set."
        offchip_core = self.accelerator.get_core(self.accelerator.offchip_core_id)
        offchip_instance = next(iter(offchip_core.mem_hierarchy_dict.values()))[-1].memory_instance
        offchip_bandwidth = cme.get_total_inst_bandwidth(offchip_instance)
        return offchip_bandwidth

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
            runtime (int): runtime of executing this node
            core_allocation (int): the core_id on which this node will be ran
        """
        node.set_onchip_energy(onchip_energy)
        node.set_offchip_energy(offchip_energy)
        node.set_runtime(runtime)
        node.set_chosen_core_allocation(chosen_core_allocation)
