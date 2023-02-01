from stream.classes.cost_model.cost_model import StreamCostModelEvaluation
from stream.classes.workload.computation_node import ComputationNode
from zigzag.utils import pickle_deepcopy


class FitnessEvaluator:
    def __init__(self, workload=None, accelerator=None, node_hw_performances=None) -> None:
        self.workload = workload
        self.accelerator = accelerator
        self.node_hw_performances = node_hw_performances
        # self.num_cores = len(inputs.accelerator.cores)

    def get_fitness(self):
        raise NotImplementedError

class StandardFitnessEvaluator(FitnessEvaluator):
    """The standard fitness evaluator considers latency, max buffer occupancy and energy equally.
    """
    def __init__(self, workload, accelerator, node_hw_performances, coarse_node_ids_flexible, scheduler_candidate_selection) -> None:
        super().__init__(workload, accelerator, node_hw_performances)

        self.weights = (-1.0, -1.0)
        self.metrics = ["energy", "latency"]

        self.coarse_node_ids_flexible = coarse_node_ids_flexible
        self.scheduler_candidate_selection = scheduler_candidate_selection

    def get_fitness(self, core_allocations: list, return_scme=False):
        """Get the fitness of the given core_allocations

        Args:
            core_allocations (list): core_allocations
        """
        self.set_node_core_allocations(core_allocations)
        scme = StreamCostModelEvaluation(pickle_deepcopy(self.workload), pickle_deepcopy(self.accelerator), self.scheduler_candidate_selection)
        scme.run()
        energy = scme.energy
        latency = scme.latency
        if not return_scme:
            return energy, latency
        return energy, latency, scme

    def set_node_core_allocations(self, core_allocations):
        """Sets the core allocation of all nodes in self.workload according to core_allocations.
        This will only set the energy, runtime and core_allocation of the nodes which are flexible in their core allocation.
        We assume the energy, runtime and core_allocation of the other nodes are already set.

        Args:
            core_allocations (list): list of the node-core allocations
        """
        for i, core_allocation in enumerate(core_allocations):
            core = self.accelerator.get_core(core_allocation)
            coarse_id = self.coarse_node_ids_flexible[i]
            # Find all nodes of this coarse id and set their core_allocation, energy and runtime
            nodes = (node for node in self.workload.nodes() if isinstance(node, ComputationNode) and node.id[0] == coarse_id)
            for node in nodes:
                try:
                    equivalent_unique_node = next((n for n in self.node_hw_performances.keys() if node == n))
                except StopIteration:
                    raise ValueError(f"The given node_hw_performances doesn't have run information for node={node}")
                try:
                    cme = self.node_hw_performances[equivalent_unique_node][core]
                except KeyError:
                    raise KeyError(f"The given node_hw_performances doesn't have information for core_allocation={core_allocation} of node={node}")
                energy = cme.energy_total
                latency = cme.latency_total1
                too_large_operands = self.get_too_large_operands(cme, core_id=core_allocation)
                node.set_energy(energy)
                node.set_runtime(latency)
                node.set_core_allocation(core_allocation)
                node.set_too_large_operands(too_large_operands)

    def get_too_large_operands(self, cme, core_id):
        """Create a list of memory operands for which an extra memory level (i.e. offchip) was added.

        Args:
            cme (CostModelEvaluation): The CostModelEvaluation containing information wrt the memory utilization.
        """
        too_large_operands = []
        core = self.accelerator.get_core(core_id)
        core_nb_memory_levels = core.memory_hierarchy.nb_levels
        for (layer_operand, l) in cme.mapping.data_elem_per_level.items():
            memory_operand = cme.layer.memory_operand_links[layer_operand]
            if len(l) > core_nb_memory_levels[memory_operand] + 1:  # +1 because of spatial level
                too_large_operands.append(memory_operand)
        return too_large_operands
