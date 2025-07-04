import logging
from typing import Any

from zigzag.datatypes import LayerOperand

from stream.hardware.architecture.accelerator import Accelerator
from stream.opt.allocation.genetic_algorithm.fitness_evaluator import StandardFitnessEvaluator
from stream.opt.allocation.genetic_algorithm.genetic_algorithm import GeneticAlgorithm
from stream.stages.stage import Stage, StageCallable
from stream.utils import CostModelEvaluationLUT, get_unique_nodes
from stream.workload.computation.computation_node import ComputationNode
from stream.workload.onnx_workload import ComputationNodeWorkload

logger = logging.getLogger(__name__)


class GeneticAlgorithmAllocationStage(Stage):
    """
    Class that finds the best inter-core mapping using a genetic algorithm.
    From the IntraCoreMappingStage we receive the `CostModelEvaluationLUT`, containing for each node and its valid core
      allocations the best CME.
    We then initialize the genetic algorithm.
    TODO A separate "GeneticAlgorithmStage" should be added where we parse all GA-related info and this stage then calls
    TODO that stage.
    """

    def __init__(
        self,
        list_of_callables: list[StageCallable],
        *,
        workload: ComputationNodeWorkload,
        accelerator: Accelerator,
        cost_lut: CostModelEvaluationLUT,
        nb_ga_generations: int,
        nb_ga_individuals: int,
        operands_to_prefetch: list[LayerOperand],
        scheduling_order: list[tuple[int, int]],
        **kwargs: Any,
    ):
        """Initialize the InterCoreMappingStage.

        Args:
            list_of_callables (list): List of the substages to be called. This should be empty as this is a leaf stage.
            workload (DiGraph): The NetworkX DiGraph representing the workload to be scheduled
            accelerator (Accelerator): The hardware accelerator onto which we schedule the workload
            cost_lut (CostModelEvaluationLUT): A LUT of CMEs for each unique node and their valid cores
            nb_ga_generations: The number of generations considered by the genetic algorithm
            nb_ga_individuals: The number of individuals in each genetic algorithm generation
        """
        super().__init__(list_of_callables, **kwargs)
        self.workload = workload
        self.accelerator = accelerator
        self.cost_lut = cost_lut
        self.nb_generations = nb_ga_generations
        self.nb_individuals = nb_ga_individuals
        self.operands_to_prefetch = operands_to_prefetch
        self.scheduling_order = scheduling_order
        self.latency_attr = kwargs.get("latency_attr", "latency_total2")

        # Determine the set of all (layer, group) combinations to be allocated separately
        self.layer_groups: list[tuple[int, int]] = sorted(set((n.id, n.group) for n in self.workload.node_list))

        # self.coarse_node_ids contains all the original node (aka layers) ids of the original graph
        self.unique_nodes = get_unique_nodes(self.workload)
        self.coarse_node_ids: list[int] = [id for id, _ in self.layer_groups]
        # allocated to more than one core
        # TODO is this sorting key correct?
        self.unique_nodes_flexible: list[ComputationNode] = []
        for n in self.unique_nodes:
            if not isinstance(n.chosen_core_allocation, int):
                self.unique_nodes_flexible.append(n)

        # For each unique node get the possible core allocations by getting the ids of the cores in cost_lut
        self.valid_allocations: list[list[int]] = []
        # Save all the layer group combinations that are flexible
        self.layer_groups_flexible: list[tuple[int, int]] = []
        for layer_id, group_id in self.layer_groups:
            # Find the unique node that corresponds to this layer
            # This assumes all the nodes of this layer are identical
            unique_node = next(n for n in self.unique_nodes if n.id == layer_id)
            if unique_node in self.unique_nodes_flexible:
                cores = self.cost_lut.get_cores(unique_node)
                valid_core_ids = [core.id for core in cores if core.id < len(self.unique_nodes_flexible)]
                self.layer_groups_flexible.append((layer_id, group_id))
                self.valid_allocations.append(valid_core_ids)

        # Initialize the fitness evaluator of different core allocations
        self.fitness_evaluator = StandardFitnessEvaluator(
            self.workload,
            self.accelerator,
            self.cost_lut,
            self.layer_groups_flexible,
            self.operands_to_prefetch,
            self.scheduling_order,
            self.latency_attr,
        )

        # Extract the length of an individual.
        # This is the number of unique original nodes that have more than one possible core allocation
        self.individual_length = len(self.layer_groups_flexible)
        # Extract the value range each gene in the individual can have.
        # This ranges from 0 to the max core index.
        # TODO There might be some case where a core is not possible, so it shouldnt be tried by the GA
        core_ids: list[int] = sorted([core.id for core in self.accelerator.cores.node_list])
        self.core_id_range = (min(core_ids), max(core_ids))
        self.nb_cores = max(core_ids) - min(core_ids) + 1  # Assuming they are incrementing with step size 1

    def run(self):
        """Run the InterCoreMappingStage by checking if we have a fixed core_allocation.
        - if yes: evaluate fixed core allocation
        - if no: initialize and run the genetic algorithm
        """

        logger.info("Start GeneticAlgorithmAllocationStage.")
        if self.individual_length == 0:
            logger.info("Evaluating fixed layer-core allocation.")
            core_allocations = []
            (_, _, scme) = self.fitness_evaluator.get_fitness(core_allocations, return_scme=True)  # type: ignore
            yield scme, None
        else:
            logger.info(
                f"Running Genetic Algorithm with {self.nb_generations} "
                f"generations and {self.nb_individuals} individuals."
            )
            flexible_layer_names = [f"{n.name}" for n in self.unique_nodes_flexible]
            logger.info(
                f"Exploring allocation for {len(self.unique_nodes_flexible)} flexible layers: {flexible_layer_names}"
            )
            # Initialize the genetic algorithm
            self.genetic_algorithm = GeneticAlgorithm(
                self.fitness_evaluator,
                self.individual_length,
                self.valid_allocations,
                self.nb_generations,
                self.nb_individuals,
            )
            # Run the genetic algorithm and get the results
            pop, hof = self.genetic_algorithm.run()
            logger.info("Finished Genetic Algorithm.")
            # Return the SCME of the last individual in the hall of fame
            best_core_allocations = hof[-1]
            results = self.fitness_evaluator.get_fitness(best_core_allocations, return_scme=True)
            scme = results[-1]
            yield scme, None
        logger.info("Finished GeneticAlgorithmAllocationStage.")

    def is_leaf(self) -> bool:
        return True
