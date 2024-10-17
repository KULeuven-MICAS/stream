import itertools
import logging
import os
from time import time
from typing import Any, TypeAlias

import networkx as nx
import numpy as np
from networkx import DiGraph
from zigzag.utils import pickle_deepcopy, pickle_load, pickle_save

from stream.cost_model.cost_model import StreamCostModelEvaluation
from stream.hardware.architecture.accelerator import Accelerator
from stream.opt.allocation.constraint_optimization.allocation import ALLOCATION_T, get_optimal_allocations
from stream.stages.estimation.stream_cost_model_evaluation import StreamCostModelEvaluationStage
from stream.stages.estimation.zigzag_core_mapping_estimation import ZigZagCoreMappingEstimationStage
from stream.stages.generation.tiled_workload_generation import (
    TiledWorkloadGenerationStage,
)
from stream.stages.set_fixed_allocation_performance import SetFixedAllocationPerformanceStage
from stream.stages.stage import MainStage, Stage, StageCallable
from stream.utils import CostModelEvaluationLUT
from stream.visualization.constraint_optimization import visualize_waco
from stream.workload.computation.computation_node import ComputationNode
from stream.workload.dnn_workload import DNNWorkloadStream
from stream.workload.mapping import TILING_T
from stream.workload.onnx_workload import ComputationNodeWorkload

logger = logging.getLogger(__name__)

STACK_T: TypeAlias = tuple[int, ...]
SCHEDULE_ORDER_T: TypeAlias = list[tuple[int, int]]


class ConstraintOptimizationAllocationStage(Stage):
    """
    Class that finds the best workload allocation for the workload using constraint optimization.
    This stages requires a CostModelEvaluationLUT, containing for each node and its valid core allocations the best CME.
    """

    def __init__(
        self,
        list_of_callables: list[StageCallable],
        *,
        workload: ComputationNodeWorkload,
        accelerator: Accelerator,
        node_hw_performances: CostModelEvaluationLUT,
        layer_stacks: list[tuple[int, ...]],
        node_hw_performances_path_with_split: str,
        **kwargs: Any,
    ):
        """Initialize the ResourceAllocationStage.

        Args:
            list_of_callables (list): List of the substages to be called. This should be empty as this is a leaf stage.
            workload (DiGraph): The NetworkX DiGraph representing the workload to be scheduled
            accelerator (Accelerator): The hardware accelerator onto which we schedule the workload
            node_hw_performances (dict): A nested dict containing for each node a dict with for each valid core its best HW performance
            layer_stacks (list): List of tuples with each tuple containing the layer ids to fuse together
        """
        super().__init__(list_of_callables, **kwargs)
        self.workload = workload
        self.accelerator = accelerator
        self.node_hw_performances = node_hw_performances
        self.layer_stacks = layer_stacks
        self.original_workload: ComputationNodeWorkload = kwargs["original_workload"]
        self.mode = kwargs.get("mode", "fused")  # assume default is fused

        self.steady_state_visualization_path = kwargs.get("steady_state_visualization_path", "outputs/")
        self.node_hw_performances_path_with_split = node_hw_performances_path_with_split
        if "visualize_node_hw_performances_path_with_split" in kwargs:
            self.visualize_node_hw_performances_path_with_split = kwargs[
                "visualize_node_hw_performances_path_with_split"
            ]
        else:
            node_hw_performances_visualization_path = (
                os.path.splitext(self.node_hw_performances_path_with_split)[0] + ".png"
            )
            self.visualize_node_hw_performances_path_with_split = node_hw_performances_visualization_path
        self.co_time_limit: int = kwargs.get("co_time_limit", 600)

        # Which CME attribute to use for the node latencies
        self.latency_attr = kwargs.get("latency_attr", "latency_total1")

        # Attributes that will be assigned throughout the stage
        self.ss_to_computes: dict[STACK_T, set[ComputationNode]] = {}
        self.hashes_per_sink_node: dict[STACK_T, dict[ComputationNode, int]] = {}
        self.steady_state_hashes: dict[STACK_T, int] = {}
        self.compute_per_sink_node: dict[STACK_T, dict[ComputationNode, set[ComputationNode]]] = {}
        self.ss_iterations_per_stack: dict[STACK_T, int] = {}
        self.optimal_allocation_per_stack: dict[STACK_T, ALLOCATION_T] = {}
        self.nb_macs_per_stack: dict[STACK_T, int] = {}
        self.nb_macs_in_ss_per_stack: dict[STACK_T, int] = {}
        self.ss_mac_percentages_per_stack: dict[STACK_T, int] = {}

    def run(self):
        logger.info("Start ConstraintOptimizationAllocationStage.")
        # For each layer stack, determine for each node in the last layer(s) how many nodes have to be computed and cached

        self.extract_steady_state_per_stack()
        self.find_best_allocation_per_stack()
        scme = self.run_coala()

        logger.info("End ConstraintOptimizationAllocationStage.")
        yield (scme, None)

    def run_coala(self):
        combined_allocation: ALLOCATION_T = []
        timestep_offset = 0
        for optimal_allocation in self.optimal_allocation_per_stack.values():
            # Update all timesteps in this allocation with the offset and add it to the combined allocation
            for t, a, id in optimal_allocation:
                new_t = t + timestep_offset
                combined_allocation.append((new_t, a, id))
            max_timestep = max(list(zip(*combined_allocation))[0])
            timestep_offset = max_timestep + 1
        scme = self.schedule_allocation(combined_allocation)
        return scme

    def extract_steady_state_per_stack(self):
        for i, stack in enumerate(self.layer_stacks):
            nodes = [n for n in self.workload.node_list if n.id in stack]
            if len(nodes) == 0:
                logger.warning(f"Stack {i} is empty.")
                continue

            # TODO initialize the subgraph of type DiGraphWrapper[ComputationNode]
            sg: DiGraph = self.workload.subgraph(nodes)
            sink_nodes: list[ComputationNode] = sorted(
                n for n in sg.nodes() if len(self.get_real_successors(n, sg)) == 0  # type: ignore
            )
            sink_layer_ids = sorted(set(n.id for n in sink_nodes))
            sink_layer_nodes = [tuple(sorted(n for n in sink_nodes if n.id == layer_id)) for layer_id in sink_layer_ids]
            interlaced = [tuple(filter(lambda x: x is not None, t)) for t in itertools.zip_longest(*sink_layer_nodes)]
            computed: set[ComputationNode] = set()
            to_compute_sets: dict[int, set[ComputationNode]] = dict()
            memoization_hashes: dict[int, frozenset[ComputationNode]] = dict()
            to_compute_counts: dict[int, int] = dict()
            state_ids: dict[int, list[int]] = dict()
            to_compute_unique: dict[tuple[ComputationNode, ...], set[ComputationNode]] = dict()
            hashes_per_sink_pair = dict()
            for pair in interlaced:
                needed_compute: set[ComputationNode] = set()
                for sink_node in pair:
                    needed_compute |= nx.ancestors(sg, sink_node) | {sink_node}  # type: ignore
                to_compute: set[ComputationNode] = needed_compute - computed  # type: ignore
                to_compute_unique[pair] = to_compute
                to_compute_ids = [n.id for n in to_compute]
                to_compute_per_layer = {id: to_compute_ids.count(id) for id in stack}
                to_compute_set = frozenset(sorted(to_compute_per_layer.items()))
                memoization_hash = hash(to_compute_set)
                hashes_per_sink_pair[pair] = memoization_hash
                if memoization_hash in memoization_hashes:
                    to_compute_counts[memoization_hash] += 1
                else:
                    to_compute_sets[memoization_hash] = to_compute
                    memoization_hashes[memoization_hash] = to_compute_set
                    to_compute_counts[memoization_hash] = 1
                state_ids[memoization_hash] = state_ids.get(memoization_hash, []) + [n.id for n in to_compute]
                computed |= needed_compute

            scaled_counts: dict[int, int] = {}
            # Get the most important sink node to optimize for by looking at the importance
            total_nb_macs = 0
            for memoization_hash, count in to_compute_counts.items():
                nb_macs = sum(n.total_mac_count for n in to_compute_sets[memoization_hash])
                scaled_counts[memoization_hash] = nb_macs * count
                total_nb_macs += nb_macs * count
            max_count = max(scaled_counts.values())
            self.ss_mac_percentages_per_stack[stack] = max_count / total_nb_macs
            self.nb_macs_per_stack[stack] = total_nb_macs
            self.nb_macs_in_ss_per_stack[stack] = max_count
            max_memoization_hash = next(k for k, v in scaled_counts.items() if v == max_count)
            steady_state_pair = next(k for k, v in hashes_per_sink_pair.items() if v == max_memoization_hash)

            compute_for_ss = to_compute_unique[steady_state_pair]
            self.ss_to_computes[stack] = compute_for_ss
            to_compute_ids_ss = [n.id for n in compute_for_ss]
            to_compute_per_layer_ss = {id: to_compute_ids_ss.count(id) for id in stack}
            memoization_hash_ss = hash(frozenset(sorted(to_compute_per_layer_ss.items())))
            self.ss_iterations_per_stack[stack] = to_compute_counts[memoization_hash_ss]

            self.hashes_per_sink_node[stack] = hashes_per_sink_pair
            self.steady_state_hashes[stack] = memoization_hash_ss
            self.compute_per_sink_node[stack] = to_compute_unique

        nb_steady_state_nodes = sum(list(len(v) for v in self.ss_to_computes.values()))
        nb_nodes = self.workload.number_of_nodes()
        percentage_nodes = nb_steady_state_nodes / nb_nodes * 100
        logger.info(f"Percentage of steady state nodes: {nb_steady_state_nodes}/{nb_nodes} = {percentage_nodes:.2f}%")
        nb_steady_state_macs = sum(self.nb_macs_in_ss_per_stack.values())
        nb_macs = sum(self.nb_macs_per_stack.values())
        percentage_macs = nb_steady_state_macs / nb_macs * 100
        logger.info(f"Percentage of steady state macs: {nb_steady_state_macs}/{nb_macs} = {percentage_macs:.2f}%")

    def find_best_allocation_per_stack(self):
        for stack, to_compute in self.ss_to_computes.items():
            iterations = self.ss_iterations_per_stack[stack]
            t_start = time()
            optimal_allocation = self.find_best_allocation(to_compute, iterations, stack, self.co_time_limit)
            t_end = time()
            logger.info(f"Stack {stack}: {t_end - t_start:.3f} seconds")
            self.optimal_allocation_per_stack[stack] = optimal_allocation

    def find_best_allocation(
        self, to_compute: set[ComputationNode], iterations: int, stack: STACK_T = (0,), time_limit: int = 600
    ):
        """# TODO: Implement overhead of tensor transfers between cores"""
        # Check if the allocation is already cached, if not: find it
        stack_str = "_".join([str(id) for id in stack])
        allocations_path = os.path.join(self.steady_state_visualization_path, f"steady_state-{stack_str}.pickle")
        if os.path.exists(allocations_path):
            allocation = pickle_load(allocations_path)
        else:
            sg = self.workload.subgraph(to_compute)
            logger.info(f"Optimizing allocation for {iterations} iterations of {len(to_compute)} ss nodes.")
            allocation = get_optimal_allocations(
                sg,
                self.accelerator,
                self.node_hw_performances,
                iterations,
                time_limit=time_limit,
            )
            pickle_save(allocation, allocations_path)
        fig_path = os.path.join(self.steady_state_visualization_path, f"steady_state-{stack_str}.html")
        print(f"stack = {stack}")
        visualize_waco(allocation, self.node_hw_performances, self.accelerator, fig_path, iterations)
        return allocation

    def get_scheduling_order(self, unpartitioned_workload: DNNWorkloadStream) -> SCHEDULE_ORDER_T:
        """
        Get the scheduling order of all ids that will exist in the transformed workload.
        Returns a list with all ids where the earlier the higher priority
        The scheduling order is altered to accommodate the inter core tiling of the given workload

        Args:
           unpartitioned_workload: original workload (before partitioning into finder nodes), used to extract the inter-
                                   core tiling loops
        """

        scheduling_order: SCHEDULE_ORDER_T = []
        for stack, compute in self.compute_per_sink_node.items():
            hash_steady_state = self.steady_state_hashes[stack]
            allocation_steady_state = self.optimal_allocation_per_stack[stack]
            hashes_per_sink_node = self.hashes_per_sink_node[stack]
            order = self.get_cn_order(
                allocation=allocation_steady_state,
                compute_per_sink_node=compute,
                hashes_per_sink_node=hashes_per_sink_node,
                memoization_hash_ss=hash_steady_state,
            )

            adjusted_order = self.adjust_order_to_inter_core_tiling(stack, order, unpartitioned_workload)
            scheduling_order += adjusted_order

        return scheduling_order

    def adjust_order_to_inter_core_tiling(
        self, stack: STACK_T, order: SCHEDULE_ORDER_T, unpartitioned_workload: DNNWorkloadStream
    ):
        """Given an allocation order for a given stack, extend the order to extra outer loops that result from the
        inter core tiling. This method anticipates the fact that later on, CNs will be split further to allow for inter-
        core tiling, and adjusts the scheduling beforehand

        Args:
            stack: CN stack for which the order applies
            order: scheduling order for this stack, without inter core tiling
            unpartitioned_workload: original workload (before partitioning into finder nodes), used to extract the inter-
                                    core tiling loops

        """

        for node in self.get_computation_nodes(stack, unpartitioned_workload):
            # NOTE this uses `inter_core_tiling`, because the inter core tiling is added to the intra core tiling
            # in `schedule_allocation` in order to alter the workload
            outer_loops = node.inter_core_tiling

            # try:
            #     outer_loops = hint_loops[(layer_id,)]
            # except KeyError:
            #     # If the layer_id is not present it means it was not in the allocation.
            #     # This happens if all nodes of the layer were not in the steady state
            #     outer_loops = []

            for _, factor in outer_loops:
                assert isinstance(factor, int), "tiling options `*` and `all` should be replaced by now"
                if factor == 1:
                    # In case CO decides to not split up the node across cores
                    continue

                inserted = 0
                for i, ids_in_stack in enumerate(order.copy()):
                    layer_id, node_id = ids_in_stack
                    if layer_id == node.id:
                        nb_nodes_this_layer = self.get_nb_nodes_for_layer(layer_id)
                        for scale in range(1, factor):
                            new_node_id = scale * nb_nodes_this_layer + node_id
                            order.insert(i + inserted + 1, (layer_id, new_node_id))
                            inserted += 1

        return order

    def get_nb_nodes_for_layer(self, layer_id: int):
        return len(list(n for n in self.workload.node_list if n.id == layer_id))

    def get_computation_nodes(self, stack: tuple[int, ...], workload: DNNWorkloadStream) -> list[ComputationNode]:
        nodes = [n for n in workload.node_list if n.id in stack]
        computation_nodes = [n for n in nodes if isinstance(n, ComputationNode)]
        return computation_nodes

    def get_cn_order(
        self,
        allocation: ALLOCATION_T,
        compute_per_sink_node: dict[ComputationNode, set[ComputationNode]],
        hashes_per_sink_node: dict[ComputationNode, int],
        memoization_hash_ss: int,
    ) -> SCHEDULE_ORDER_T:
        """
        Get the scheduling orders of all cns of a stack based on the order in the steady state allocation.
        For nodes belonging to sink nodes that are not steady state, we sort by deepest id.
        For nodes belonging to steady state sink nodes, we sort according to the found allocation
        """
        order: SCHEDULE_ORDER_T = []
        allocation = sorted(allocation, key=lambda x: (x[0], x[2], x[1]))
        allocation_adjusted: ALLOCATION_T = []  # allocation with removed inter core splits (which have same sub id)
        seen_ids: set[tuple[int, int]] = set()
        for t, c, id in allocation:
            if id not in seen_ids:
                allocation_adjusted.append((t, c, id))
                seen_ids.add(id)

        allocation_adjusted = sorted(allocation_adjusted, key=lambda x: (x[0], x[2], x[1]))
        layer_order_steady_state = [layer_id for layer_id, _ in [id for (_, _, id) in allocation_adjusted]]
        for sink_node, to_compute in compute_per_sink_node.items():
            if hashes_per_sink_node[sink_node] == memoization_hash_ss:
                order_i = self.get_order_steady_state(to_compute, layer_order_steady_state)
            else:
                order_i = self.get_order_non_steady_state(to_compute)
            order += order_i
        return order

    def get_order_steady_state(self, to_compute: set[ComputationNode], layer_order_steady_state: list[int]):
        assert len(to_compute) == len(layer_order_steady_state)
        # Obtain for each layer the nodes that have to be scheduled
        nodes_per_layer = {
            layer_id: sorted(n for n in to_compute if n.id == layer_id)
            for layer_id in sorted(set(layer_order_steady_state))
        }
        order: SCHEDULE_ORDER_T = []
        for layer_id in layer_order_steady_state:
            first_node = nodes_per_layer[layer_id].pop(0)
            order.append((first_node.id, first_node.sub_id))
        return order

    def get_order_non_steady_state(self, to_compute: set[ComputationNode]):

        return [(n.id, n.sub_id) for n in sorted(to_compute, key=lambda x: (-x.id, -x.sub_id))]

    def schedule_allocation(self, allocation: ALLOCATION_T) -> StreamCostModelEvaluation:
        # Create a modified sub-workload with the extra inter core splits
        max_layer_id = max(id[0] for _, _, id in allocation)
        sub_nodes = filter(lambda n: n.id <= max_layer_id, self.original_workload.node_list)
        unpartitioned_sub_workload: DNNWorkloadStream = pickle_deepcopy(self.original_workload.subgraph(sub_nodes))

        # Get the involved layer ids we want to schedule and their core allocations
        layer_ids = sorted(set(id[0] for _, _, id in allocation))
        core_strs = [sorted(set((c for _, c, id in allocation if id[0] == layer_id))) for layer_id in layer_ids]
        core_ids = [[int(s.split(" ")[-1]) for s in core_str] for core_str in core_strs]

        # Manually add the wanted core ids for layers not in the steady state
        layer_ids, core_ids = self.add_core_ids_for_layers_not_in_steady_state(
            layer_ids=layer_ids, core_ids=core_ids, sub_workload=unpartitioned_sub_workload
        )

        # Set the correct allocations for the layers in the copied workload
        self.set_fixed_allocations_for_workload(unpartitioned_sub_workload, layer_ids, core_ids)

        # Generate/check inter core mapping for all nodes
        for node in unpartitioned_sub_workload.node_list:
            # No allocation for this node (e.g. DummyNode)
            if node.id not in layer_ids:
                continue

            core_allocation_this_node = next(
                core_ids_this_node for core_ids_this_node, node_id in zip(core_ids, layer_ids) if node_id == node.id
            )
            nb_cores_split = len(core_allocation_this_node)

            # Set correct inter core tiling
            inter_core_tiling = self.replace_wildcard_in_tiling(node.inter_core_tiling, nb_cores_split)
            node.inter_core_tiling = inter_core_tiling
            # Add inter core tiling to intra-core mapping for the next run
            node.intra_core_tiling += inter_core_tiling

        scheduling_order = self.get_scheduling_order(unpartitioned_sub_workload)

        loma_lpf_limit = 7
        kwargs = self.kwargs.copy()
        kwargs["loma_lpf_limit"] = loma_lpf_limit
        kwargs["accelerator"] = self.accelerator
        kwargs["workload"] = unpartitioned_sub_workload
        kwargs["scheduling_order"] = scheduling_order
        kwargs["node_hw_performances_path"] = self.node_hw_performances_path_with_split
        kwargs["visualize_node_hw_performances_path"] = self.visualize_node_hw_performances_path_with_split
        kwargs["latency_attr"] = self.latency_attr

        # Create stages that will run a single cost model evaluation (fixed core allocations)
        main_stage = MainStage(
            [
                TiledWorkloadGenerationStage,  # Splits in intra-core mapping
                ZigZagCoreMappingEstimationStage,
                SetFixedAllocationPerformanceStage,
                StreamCostModelEvaluationStage,
            ],
            **kwargs,
        )
        scme, _ = main_stage.run()
        scme = scme[0]
        return scme

    def add_core_ids_for_layers_not_in_steady_state(
        self, layer_ids: list[int], core_ids: list[list[int]], sub_workload: ComputationNodeWorkload
    ) -> tuple[list[int], list[list[int]]]:
        """Find any layers that might not have been in the steady state allocation and need to be allocated manually
        The nodes of these layers will be allocated across all possible cores in their defined inter core tiling
          dimension
        """

        layer_ids_not_in_ss = [
            layer_id for stack in self.layer_stacks for layer_id in stack if layer_id not in layer_ids
        ]

        for layer_id_not_in_ss in layer_ids_not_in_ss:
            layer_ids_idx = np.searchsorted(layer_ids, layer_id_not_in_ss)
            for node in filter(lambda n: n.id == layer_id_not_in_ss, sub_workload.node_list):
                node.chosen_core_allocation = node.core_allocation
                node.possible_core_allocation = node.core_allocation
                node.core_allocation_is_fixed = True
                layer_ids.insert(layer_ids_idx, layer_id_not_in_ss)
                core_ids.insert(layer_ids_idx, node.core_allocation)
                logger.warning(f"{node} not in steady state allocation; allocated to: {node.core_allocation}.")

        return layer_ids, core_ids

    def set_fixed_allocations_for_workload(
        self, workload: ComputationNodeWorkload, layer_ids: list[int], core_ids: list[list[int]]
    ):
        """! Modify the workload to fix the core allocations to the given core_ids for the given layer_ids."""
        assert len(layer_ids) == len(core_ids)
        for layer_id, cores in zip(layer_ids, core_ids):
            n = next(n for n in workload.node_list if n.id == layer_id)
            n.chosen_core_allocation = list(cores)
            n.possible_core_allocation = list(cores)
            n.core_allocation_is_fixed = True

        # Find any layers that might not have been in the steady state allocation and need to be allocated manually
        # The nodes of these layers will be allocated across all possible cores in the K dimension if possible
        layer_ids_not_in_ss = [
            layer_id for stack in self.layer_stacks for layer_id in stack if layer_id not in layer_ids
        ]

        # TODO this piece of code is outdated
        for layer_id_not_in_ss in layer_ids_not_in_ss:
            layer_ids_idx = np.searchsorted(layer_ids, layer_id_not_in_ss)
            for n in filter(lambda n: n.id == layer_id_not_in_ss, workload.node_list):
                assert isinstance(n.core_allocation, list)
                n.chosen_core_allocation = n.core_allocation
                n.possible_core_allocation = n.core_allocation
                n.core_allocation_is_fixed = True
                assert "K" in n.loop_dim_size
                layer_ids.insert(layer_ids_idx, layer_id_not_in_ss)
                core_ids.insert(layer_ids_idx, n.core_allocation)
                logger.warning(f"{n} not in steady state allocation; allocated to: {n.core_allocation}.")

    def replace_wildcard_in_tiling(self, tiling: TILING_T, nb_cores_split: int):
        """The user can define a wildcard `*` in the inter core tiling, meaning that the value found by the CO
        must be used instead.
        # TODO in case multiple splits are given (e.g. C and D), should the nb_cores_split be scaled?
        """
        if len(tiling) > 1:
            raise ValueError("Only 1 partition dimension should be defined in inter core tiling")
        assert len(tiling) == 1, "No inter core tiling found"

        tiling_replaced: TILING_T = []
        for layer_dim, split_factor in tiling:
            if split_factor == "*":
                split_factor = nb_cores_split
            elif split_factor == "all":
                raise ValueError("inter core tiling should not contain `all`")
            tiling_replaced.append((layer_dim, split_factor))
        return tiling_replaced

    def is_leaf(self) -> bool:
        return True
