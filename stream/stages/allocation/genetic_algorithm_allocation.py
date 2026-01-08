import logging

from stream.opt.allocation.genetic_algorithm.fitness_evaluator import StandardFitnessEvaluator
from stream.opt.allocation.genetic_algorithm.genetic_algorithm import GeneticAlgorithm
from stream.stages.context import StageContext
from stream.stages.stage import Stage, StageCallable
from stream.utils import get_unique_nodes
from stream.workload.workload import ComputationNode

logger = logging.getLogger(__name__)


class GeneticAlgorithmAllocationStage(Stage):
    """
    Class that finds the best inter-core mapping using a genetic algorithm.
    From the IntraCoreMappingStage we receive the `CoreCostLUT`, containing for each node and its valid core
      allocations the best CME.
    We then initialize the genetic algorithm.
    TODO A separate "GeneticAlgorithmStage" should be added where we parse all GA-related info and this stage then calls
    TODO that stage.
    """

    REQUIRED_FIELDS = (
        "workload",
        "accelerator",
        "cost_lut",
        "nb_ga_generations",
        "nb_ga_individuals",
        "operands_to_prefetch",
        "scheduling_order",
    )

    def __init__(
        self,
        list_of_callables: list[StageCallable],
        ctx: StageContext,
    ):
        """Initialize the InterCoreMappingStage.

        Args:
            list_of_callables (list): List of the substages to be called. This should be empty as this is a leaf stage.
            workload (DiGraph): The NetworkX DiGraph representing the workload to be scheduled
            accelerator (Accelerator): The hardware accelerator onto which we schedule the workload
            cost_lut (CoreCostLUT): A LUT of cost entries for each unique node and their valid cores
            nb_ga_generations: The number of generations considered by the genetic algorithm
            nb_ga_individuals: The number of individuals in each genetic algorithm generation
        """
        super().__init__(list_of_callables, ctx)
        self.workload = self.ctx.require_value("workload", self.__class__.__name__)
        self.accelerator = self.ctx.require_value("accelerator", self.__class__.__name__)
        self.cost_lut = self.ctx.require_value("cost_lut", self.__class__.__name__)
        self.nb_generations = self.ctx.require_value("nb_ga_generations", self.__class__.__name__)
        self.nb_individuals = self.ctx.require_value("nb_ga_individuals", self.__class__.__name__)
        self.operands_to_prefetch = self.ctx.require_value("operands_to_prefetch", self.__class__.__name__)
        self.scheduling_order = self.ctx.require_value("scheduling_order", self.__class__.__name__)
        self.latency_attr = self.ctx.get("latency_attr", "latency_total2")

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
