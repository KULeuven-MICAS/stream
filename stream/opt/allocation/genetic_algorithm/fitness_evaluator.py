from zigzag.datatypes import LayerOperand, MemoryOperand
from zigzag.mapping.data_movement import FourWayDataMoving
from zigzag.utils import pickle_deepcopy

from stream.cost_model.cost_model import StreamCostModelEvaluation
from stream.hardware.architecture.accelerator import Accelerator
from stream.utils import CostModelEvaluationLUT, get_too_large_operands, get_top_level_inst_bandwidth
from stream.workload.computation.computation_node import ComputationNode
from stream.workload.onnx_workload import ComputationNodeWorkload


class FitnessEvaluator:
    def __init__(
        self,
        workload: ComputationNodeWorkload,
        accelerator: Accelerator,
        cost_lut: CostModelEvaluationLUT,
    ) -> None:
        self.workload = workload
        self.accelerator = accelerator
        self.cost_lut = cost_lut
        # self.num_cores = len(inputs.accelerator.cores)

    def get_fitness(self):
        raise NotImplementedError


class StandardFitnessEvaluator(FitnessEvaluator):
    """The standard fitness evaluator considers latency, max buffer occupancy and energy equally."""

    def __init__(
        self,
        workload: ComputationNodeWorkload,
        accelerator: Accelerator,
        cost_lut: CostModelEvaluationLUT,
        layer_groups_flexible,
        operands_to_prefetch: list[LayerOperand],
        scheduling_order: list[tuple[int, int]],
        latency_attr: str,
    ) -> None:
        super().__init__(workload, accelerator, cost_lut)

        self.weights = (-1.0, -1.0)
        self.metrics = ["energy", "latency"]

        self.layer_groups_flexible = layer_groups_flexible
        self.operands_to_prefetch = operands_to_prefetch
        self.scheduling_order = scheduling_order
        self.latency_attr = latency_attr

    def get_fitness(self, core_allocations: list[int], return_scme: bool = False):
        """Get the fitness of the given core_allocations

        Args:
            core_allocations (list): core_allocations
        """
        self.set_node_core_allocations(core_allocations)
        scme = StreamCostModelEvaluation(
            pickle_deepcopy(self.workload),
            pickle_deepcopy(self.accelerator),
            self.operands_to_prefetch,
            self.scheduling_order,
        )
        scme.evaluate()
        energy = scme.energy
        latency = scme.latency
        if not return_scme:
            return energy, latency
        return energy, latency, scme

    def set_node_core_allocations(self, core_allocations: list[int]):
        """Sets the core allocation of all nodes in self.workload according to core_allocations.
        This will only set the energy, runtime and core_allocation of the nodes which are flexible in their core
        allocation.
        We assume the energy, runtime and core_allocation of the other nodes are already set.

        Args:
            core_allocations (list): list of the node-core allocations
        """
        for i, core_allocation in enumerate(core_allocations):
            core = self.accelerator.get_core(core_allocation)
            (layer_id, group_id) = self.layer_groups_flexible[i]
            # Find all nodes of this coarse id and set their core_allocation, energy and runtime
            nodes = (
                node
                for node in self.workload.node_list
                if isinstance(node, ComputationNode) and node.id == layer_id and node.group == group_id
            )
            for node in nodes:
                equal_unique_node = self.cost_lut.get_equal_node(node)
                assert equal_unique_node is not None, "Node not found in CostModelEvaluationLUT"
                cme = self.cost_lut.get_cme(equal_unique_node, core)
                onchip_energy = cme.energy_total  # Initialize on-chip energy as total energy
                latency = getattr(cme, self.latency_attr)
                too_large_operands = get_too_large_operands(cme, self.accelerator, core_id=core_allocation)
                # If there is a too_large_operand, we separate the off-chip energy.
                offchip_energy = 0
                for too_large_operand in too_large_operands:
                    layer_operand = next(
                        k for (k, v) in cme.layer.memory_operand_links.data.items() if v == too_large_operand
                    )
                    layer_operand_offchip_energy = cme.mem_energy_breakdown[layer_operand][-1]
                    offchip_energy += layer_operand_offchip_energy
                    onchip_energy -= layer_operand_offchip_energy
                # Get the required offchip bandwidth during the execution of the node for all directions
                bandwidth_scaling = cme.ideal_temporal_cycle / latency
                offchip_bandwidth_per_op: dict[MemoryOperand, FourWayDataMoving] = {
                    mem_op: get_top_level_inst_bandwidth(cme, mem_op, bandwidth_scaling)
                    for mem_op in too_large_operands
                }
                node.set_onchip_energy(onchip_energy)
                node.set_offchip_energy(offchip_energy)
                node.set_runtime(int(latency))
                node.set_chosen_core_allocation(core_allocation)
                node.set_too_large_operands(too_large_operands)
                node.set_offchip_bandwidth(offchip_bandwidth_per_op)
