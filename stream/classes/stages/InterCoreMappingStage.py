from operator import attrgetter
import logging

from yaml import Node

from stream.classes.cost_model.cost_model import StreamCostModelEvaluation
from stream.classes.hardware.architecture.accelerator import Accelerator
from zigzag.cost_model.cost_model import CostModelEvaluation
from zigzag.datatypes import LayerOperand
from zigzag.hardware.architecture.Core import Core
from zigzag.stages.Stage import Stage, StageCallable
from stream.classes.workload.computation_node import ComputationNode
from stream.classes.opt.allocation.genetic_algorithm.genetic_algorithm import (
    GeneticAlgorithm,
)
from stream.classes.opt.allocation.genetic_algorithm.fitness_evaluator import (
    StandardFitnessEvaluator, FitnessEvaluator
)
from stream.utils import get_too_large_operands
from zigzag.workload.Workload import Workload
from typing import Type, TypeVar

logger = logging.getLogger(__name__)

TFitnessEvaluator = TypeVar("TFitnessEvaluator", bound=FitnessEvaluator)

class InterCoreMappingStage(Stage):
    """
    Class that finds the best inter-core mapping using a genetic algorithm.
    From the IntraCoreMappingStage we receive the `node_hw_performances`, containing for each node and its valid core allocations the best CME.
    We then initialize the genetic algorithm.
    TODO A separate "GeneticAlgorithmStage" should be added where we parse all GA-related info and this stage then calls that stage.
    """

    def __init__(
        self,
        list_of_callables: list[StageCallable],
        *,
        workload: Workload,
        accelerator: Accelerator,
        node_hw_performances: dict[ComputationNode, dict[Core, CostModelEvaluation]],
        nb_ga_generations: int,
        nb_ga_individuals: int,
        plot_hof: bool,
        plot_file_name: bool,
        plot_full_schedule: bool = False,
        plot_data_transfer: bool = False,
        operands_to_prefetch: list[LayerOperand],
        custom_fitness_evaluator: Type[TFitnessEvaluator] | None = None,
        **kwargs,
    ):
        """Initialize the InterCoreMappingStage.

        Args:
            list_of_callables (list): List of the substages to be called. This should be empty as this is a leaf stage.
            workload (DiGraph): The NetworkX DiGraph representing the workload to be scheduled
            accelerator (Accelerator): The hardware accelerator onto which we schedule the workload
            node_hw_performances (dict): A nested dict containing for each node a dict with for each valid core its best HW performance
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
        self.scheduling_order = kwargs.get("scheduling_order", None)
        self.original_workload = kwargs["original_workload"]
        self.custom_fitness_evaluator = custom_fitness_evaluator

        # Determine the set of all (layer, group) combinations to be allocated separately
        self.layer_groups: list[tuple[int, int]] = sorted(set((n.id, n.group) for n in self.workload.nodes()))

        # self.coarse_node_ids contains all the original node (aka layers) ids of the original graph
        self.unique_nodes = list(set((n for n, _ in self.node_hw_performances.items())))
        self.coarse_node_ids: list[int] = [id for id in self.layer_groups]
        # self.coarse_node_ids_flexible contains only those original node ids that have flexibility: they can be allocated to more than one core
        # TODO is this sorting key correct?
        self.unique_nodes_flexible: list[ComputationNode] = sorted(
            set((n for n, hw_performances in self.node_hw_performances.items() if len(hw_performances.keys()) > 1)),
            key=lambda x: (x.id, x.sub_id),
        )
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
                hw_performances = self.node_hw_performances[unique_node]
                valid_core_ids = [
                    core.id for core in hw_performances.keys() if core.id < len(self.unique_nodes_flexible)
                ]
                self.layer_groups_flexible.append((layer_id, group_id))
                self.valid_allocations.append(valid_core_ids)

        # Set the hardware performance and core_allocation of nodes in the workload that only have a single possible core allocation
        self.set_hw_performance_non_flexible_nodes()

        # Initialize the fitness evaluator of different core allocations
        if self.custom_fitness_evaluator is not None:
            self.fitness_evaluator = self.custom_fitness_evaluator(
                self.workload,
                self.accelerator,
                self.node_hw_performances,
                self.layer_groups_flexible,
                self.scheduling_order,
                self.operands_to_prefetch,
                self.original_workload,
            )
        else:
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
        core_ids: list[int] = sorted([core.id for core in self.accelerator.cores.nodes()])
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
            # scme.plot_schedule(plot_full_schedule=self.plot_full_schedule,
            #                    plot_data_transfer=self.plot_data_transfer,
            #                    fig_path=f"outputs/schedule_plot{self.fig_path}fixed.png")
            # scme.plot_memory_usage(fig_path=f"outputs/memory_usage_plot{self.fig_path}fixed.png")
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

    def set_hw_performance_non_flexible_nodes(self):
        """Set the energy, runtime and core_allocation of the nodes in self.workload that only have a single possible core allocation."""
        non_flexible_unique_nodes: set[ComputationNode] = set(self.unique_nodes) - set(self.unique_nodes_flexible)
        for non_flexible_unique_node in non_flexible_unique_nodes:
            hw_performances = self.node_hw_performances[non_flexible_unique_node]
            assert (
                len(hw_performances.keys()) == 1
            ), f"Non-flexible unique node {non_flexible_unique_node} has more than one entry in node_hw_performances."
            (core, cme) = next((key, val) for key, val in hw_performances.items())
            onchip_energy = cme.energy_total  # Initialize the on-chip energy as total energy
            latency = cme.latency_total1

            too_large_operands = get_too_large_operands(cme, self.accelerator, core_id=core.id)
            # If there is a too_large_operand, we separate the off-chip energy.
            offchip_energy = 0
            for too_large_operand in too_large_operands:
                layer_operand = cme.layer.memory_operand_links.mem_to_layer_op(too_large_operand)
                layer_operand_offchip_energy = cme.mem_energy_breakdown[layer_operand][-1]
                offchip_energy += layer_operand_offchip_energy
                onchip_energy -= layer_operand_offchip_energy
            # If there was offchip memory added for too_large_operands, get the offchip bandwidth
            offchip_core = self.accelerator.get_core(self.accelerator.offchip_core_id)
            offchip_instance = next(v for k, v in offchip_core.mem_hierarchy_dict.items())[-1].memory_instance
            offchip_bw = cme.get_total_inst_bandwidth(offchip_instance)

            nodes: list[ComputationNode] = [
                n
                for n in self.workload.nodes()
                if n == non_flexible_unique_node and n.group == non_flexible_unique_node.group
            ]
            for node in nodes:
                self.set_hw_performance_node(node, onchip_energy, offchip_energy, latency, core.id)
                node.set_too_large_operands(too_large_operands.copy())
                node.set_offchip_bandwidth(offchip_bw)

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

    def is_leaf(self) -> bool:
        return True
