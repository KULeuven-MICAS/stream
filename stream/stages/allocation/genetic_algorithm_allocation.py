import logging
from typing import Any

from zigzag.datatypes import LayerOperand
from zigzag.stages.stage import Stage, StageCallable

from stream.hardware.architecture.accelerator import Accelerator
from stream.opt.allocation.genetic_algorithm.fitness_evaluator import (
    StandardFitnessEvaluator,
)
from stream.opt.allocation.genetic_algorithm.genetic_algorithm import (
    GeneticAlgorithm,
)
from stream.utils import CostModelEvaluationLUT
from stream.workload.computation_node import ComputationNode
from stream.workload.onnx_workload import ComputationNodeWorkload

logger = logging.getLogger(__name__)


class GeneticAlgorithmAllocationStage(Stage):
    """
    Class that finds the best inter-core mapping using a genetic algorithm.
    From the IntraCoreMappingStage we receive the `node_hw_performances`, containing for each node and its valid core
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
        node_hw_performances: CostModelEvaluationLUT,
        nb_ga_generations: int,
        nb_ga_individuals: int,
        plot_hof: bool,
        plot_file_name: bool,
        plot_full_schedule: bool = False,
        plot_data_transfer: bool = False,
        operands_to_prefetch: list[LayerOperand],
        scheduling_order: list[tuple[int, int]],
        **kwargs: Any,
    ):
        """Initialize the InterCoreMappingStage.

        Args:
            list_of_callables (list): List of the substages to be called. This should be empty as this is a leaf stage.
            workload (DiGraph): The NetworkX DiGraph representing the workload to be scheduled
            accelerator (Accelerator): The hardware accelerator onto which we schedule the workload
            node_hw_performances (CostModelEvaluationLUT): A LUT of CMEs for each unique node and their valid cores
            nb_ga_generations (int): The number of generations considered by the genetic algorithm
            nb_ga_individuals (int): The number of individuals in each genetic algorithm generation
        """
        super().__init__(list_of_callables, **kwargs)
        self.workload = workload
        self.accelerator = accelerator
        self.node_hw_performances = node_hw_performances
        self.nb_generations = nb_ga_generations
        self.nb_individuals = nb_ga_individuals
        self.plot_hof = plot_hof
        self.fig_path = plot_file_name
        self.plot_full_schedule = plot_full_schedule
        self.plot_data_transfer = plot_data_transfer
        self.operands_to_prefetch = operands_to_prefetch
        self.scheduling_order = scheduling_order

        # Determine the set of all (layer, group) combinations to be allocated separately
        self.layer_groups: list[tuple[int, int]] = sorted(set((n.id, n.group) for n in self.workload.node_list))

        # self.coarse_node_ids contains all the original node (aka layers) ids of the original graph
        self.unique_nodes = list(self.node_hw_performances.get_nodes())
        self.coarse_node_ids: list[int] = [id for id, _ in self.layer_groups]
        # self.coarse_node_ids_flexible contains only those original node ids that have flexibility: they can be
        # allocated to more than one core
        # TODO is this sorting key correct?
        self.unique_nodes_flexible: list[ComputationNode] = []
        for n in self.node_hw_performances.get_nodes():
            if len(self.node_hw_performances.get_cores(n)) > 1:
                self.unique_nodes_flexible.append(n)

        self.coarse_node_ids_flexible: list[int] = [n.id for n in self.unique_nodes_flexible]
        # For each unique node get the possible core allocations by getting the ids of the cores in node_hw_performances
        self.valid_allocations: list[list[int]] = []
        # Save all the layer group combinations that are flexible
        self.layer_groups_flexible: list[tuple[int, int]] = []
        for layer_id, group_id in self.layer_groups:
            # Find the unique node that corresponds to this layer
            # This assumes all the nodes of this layer are identical
            unique_node = next((n for n in self.unique_nodes if n.id == layer_id))
            if unique_node in self.unique_nodes_flexible:
                cores = self.node_hw_performances.get_cores(unique_node)
                valid_core_ids = [core.id for core in cores if core.id < len(self.unique_nodes_flexible)]
                self.layer_groups_flexible.append((layer_id, group_id))
                self.valid_allocations.append(valid_core_ids)

        # Initialize the fitness evaluator of different core allocations
        self.fitness_evaluator = StandardFitnessEvaluator(
            self.workload,
            self.accelerator,
            self.node_hw_performances,
            self.layer_groups_flexible,
            self.operands_to_prefetch,
            self.scheduling_order,
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

        logger.info("Start InterCoreMappingStage.")
        if self.individual_length == 0:
            logger.info("Evaluating fixed layer-core allocation.")
            core_allocations = []
            (energy, latency, scme) = self.fitness_evaluator.get_fitness(core_allocations, return_scme=True)
            yield scme, None
        else:
            logger.info("Running Inter-Core Allocation Optimization with Genetic Algorithm.")
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
            if self.plot_hof:
                for i, core_allocations in enumerate(hof):
                    results = self.fitness_evaluator.get_fitness(core_allocations, return_scme=True)
                    scme = results[-1]
                    # scme.plot_schedule(plot_full_schedule=self.plot_full_schedule,
                    #                    plot_data_transfer=self.plot_data_transfer,
                    #                    fig_path=f"outputs/schedule_plot{self.fig_path}{i}.png")
                    # scme.plot_memory_usage(fig_path=f"outputs/memory_usage_plot{self.fig_path}{i}.png")
            yield scme, None
        logger.info("Finished InterCoreMappingStage.")

    def is_leaf(self) -> bool:
        return True
