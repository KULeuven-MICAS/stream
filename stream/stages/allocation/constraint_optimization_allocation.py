import itertools
import logging
import os
from time import time
from typing import Any, TypeAlias

import networkx as nx
import numpy as np
from networkx import DiGraph
from zigzag.datatypes import LayerDim
from zigzag.stages.main import MainStage
from zigzag.stages.stage import Stage
from zigzag.utils import pickle_deepcopy, pickle_load, pickle_save

from stream.hardware.architecture.accelerator import Accelerator
from stream.opt.allocation.constraint_optimization.allocation import ALLOCATION_T, get_optimal_allocations
from stream.stages.estimation.stream_cost_model_evaluation import StreamCostModelEvaluationStage
from stream.stages.estimation.zigzag_core_mapping_estimation import ZigZagCoreMappingEstimationStage
from stream.stages.generation.hint_loops_partitioned_workload_generation import (
    HintLoopsPartitionedWorkloadGenerationStage,
)
from stream.stages.set_fixed_allocation_performance import SetFixedAllocationPerformanceStage
from stream.utils import CostModelEvaluationLUT
from stream.visualization.constraint_optimization import visualize_waco
from stream.workload.computation.computation_node import ComputationNode
from stream.workload.onnx_workload import ComputationNodeWorkload

logger = logging.getLogger(__name__)

STACK_T: TypeAlias = tuple[int, ...]


class ConstraintOptimizationAllocationStage(Stage):
    """
    Class that finds the best workload allocation for the workload using constraint optimization.
    This stages requires a CostModelEvaluationLUT, containing for each node and its valid core allocations the best CME.
    """

    def __init__(
        self,
        list_of_callables,
        *,
        workload: ComputationNodeWorkload,
        accelerator: Accelerator,
        node_hw_performances: CostModelEvaluationLUT,
        layer_stacks: list[tuple[range, ...]],
        hint_loops: Any,
        node_hw_performances_path_with_split: str,
        **kwargs: dict[str, Any],
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
        self.original_workload = kwargs["original_workload"]
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
        self.hint_loops = hint_loops
        self.co_time_limit: int = kwargs.get("co_time_limit", 600)

        # Which CME attribute to use for the node latencies
        self.latency_attr = kwargs.get("latency_attr", "latency_total1")

        # Attributes that will be assigned throughout the stage
        self.ss_to_computes: dict[STACK_T, set[ComputationNode]] = {}
        self.hashes_per_sink_node: dict[STACK_T, dict[ComputationNode, int]] = {}
        self.steady_state_hashes: dict[STACK_T, int] = {}
        self.compute_per_sink_node: dict[STACK_T, ComputationNode] = {}
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
            nodes = [n for n in self.workload.nodes() if n.id in stack]
            if len(nodes) == 0:
                logger.warning(f"Stack {i} is empty.")
                continue
            sg: DiGraph = self.workload.subgraph(nodes)
            sink_nodes = sorted(n for n in sg.nodes() if len(self.get_real_successors(n, sg)) == 0)
            sink_layer_ids = sorted(set(n.id for n in sink_nodes))
            sink_layer_nodes = [tuple(sorted(n for n in sink_nodes if n.id == layer_id)) for layer_id in sink_layer_ids]
            interlaced = [tuple(filter(lambda x: x is not None, t)) for t in itertools.zip_longest(*sink_layer_nodes)]
            computed = set()
            to_compute_sets = dict()
            memoization_hashes = dict()
            to_compute_counts = dict()
            state_ids = dict()
            to_compute_unique = dict()
            hashes_per_sink_pair = dict()
            for pair in interlaced:
                needed_compute = set()
                for sink_node in pair:
                    needed_compute |= nx.ancestors(sg, sink_node) | {sink_node}
                to_compute = needed_compute - computed
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
                pass
            scaled_counts = {}
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
        # TODO: Implement overhead of tensor transfers between cores
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

    def get_scheduling_order(self, hint_loops):
        """
        Get the scheduling order of all ids that will exist in the transformed workload.
        Returns a list with all ids where the earlier the higher priority
        """
        nb_nodes_per_layer = {
            layer_id: len(list(n for n in self.workload.nodes() if n.id == layer_id))
            for layer_id in sorted(n.id for n in self.workload.nodes())
        }
        pass
        scheduling_order = []
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
            # Adjust the order such that if there is a K split in the transformed workload,
            # those ids are also present
            for layer_id in stack:
                try:
                    outer_loops = hint_loops[(layer_id,)]
                except KeyError:
                    # If the layer_id is not present it means it was not in the allocation.
                    # This happens if all nodes of the layer were not in the steady state
                    outer_loops = []
                if outer_loops:
                    dim, size = outer_loops[-1]
                    if (dim.name == "K" and size > 1) or (dim.name == "G" and size > 1):
                        inserted = 0
                        for i, id in enumerate(order.copy()):
                            l_id, n_id = id
                            if l_id == layer_id:
                                for scale in range(1, size):
                                    new_id = scale * nb_nodes_per_layer[layer_id] + n_id
                                    order.insert(i + inserted + 1, (l_id, new_id))
                                    inserted += 1
            scheduling_order += order
        return scheduling_order

    def get_cn_order(self, allocation, compute_per_sink_node, hashes_per_sink_node, memoization_hash_ss):
        """
        Get the scheduling orders of all cns of a stack based on the order in the steady state allocation.
        For nodes belonging to sink nodes that are not steady state, we sort by deepest id.
        For nodes belonging to steady state sink nodes, we sort according to the found allocation
        """
        order = []
        allocation = sorted(allocation, key=lambda x: (x[0], x[2], x[1]))
        allocation_adjusted = []  # will hold allocation with removed k splits
        seen_ids = set()
        for t, c, id in allocation:
            if id not in seen_ids:
                allocation_adjusted.append((t, c, id))
                seen_ids.add(id)

        allocation_adjusted = sorted(allocation_adjusted, key=lambda x: (x[0], x[2], x[1]))
        layer_order_steady_state = [layer_id for layer_id, node_id in [id for (_, _, id) in allocation_adjusted]]
        for sink_node, to_compute in compute_per_sink_node.items():
            if hashes_per_sink_node[sink_node] == memoization_hash_ss:
                order_i = self.get_order_steady_state(to_compute, layer_order_steady_state)
            else:
                order_i = self.get_order_non_steady_state(to_compute)
            order += order_i
        return order

    def get_order_steady_state(self, to_compute, layer_order_steady_state):
        assert len(to_compute) == len(layer_order_steady_state)
        # Obtain for each layer the nodes that have to be scheduled
        nodes_per_layer = {
            layer_id: sorted(n for n in to_compute if n.id == layer_id)
            for layer_id in sorted(set(layer_order_steady_state))
        }
        order = []
        for layer_id in layer_order_steady_state:
            first_node = nodes_per_layer[layer_id].pop(0)
            order.append((first_node.id, first_node.sub_id))
        return order

    def get_order_non_steady_state(self, to_compute: list[ComputationNode]):
        return [(n.id, n.sub_id) for n in sorted(to_compute, key=lambda x: (-x.id, -x.sub_id))]

    def schedule_allocation(self, allocation: ALLOCATION_T) -> StreamCostModelEvaluationStage:
        # Get the involved layer ids we want to schedule and their core allocations
        layer_ids = sorted(set(id[0] for _, _, id in allocation))
        core_strs = [sorted(set((c for _, c, id in allocation if id[0] == layer_id))) for layer_id in layer_ids]
        core_ids = [[int(s.split(" ")[-1]) for s in core_str] for core_str in core_strs]

        # Create a modified workload with the correct number of k splits
        nodes = filter(lambda n: n.id <= max(layer_ids), self.original_workload.node_list)
        original_workload_copy = pickle_deepcopy(self.original_workload.subgraph(nodes))
        # Set the correct allocations for the layers in the copied workload
        self.set_fixed_allocations_for_workload(original_workload_copy, layer_ids, core_ids)

        # Parameters for stages
        kwargs = self.kwargs.copy()
        cn_define_mode = 3
        hint_loops = self.get_hint_loops(layer_ids, core_ids)
        scheduling_order = self.get_scheduling_order(hint_loops)
        loma_lpf_limit = 7
        kwargs["hint_loops"] = hint_loops
        kwargs["cn_define_mode"] = cn_define_mode
        kwargs["loma_lpf_limit"] = loma_lpf_limit
        kwargs["accelerator"] = self.accelerator
        kwargs["workload"] = original_workload_copy
        kwargs["scheduling_order"] = scheduling_order
        kwargs["node_hw_performances_path"] = self.node_hw_performances_path_with_split
        kwargs["visualize_node_hw_performances_path"] = self.visualize_node_hw_performances_path_with_split
        kwargs["latency_attr"] = self.latency_attr

        # Create stages that will run a single cost model evaluation (fixed core allocations)
        main_stage = MainStage(
            [
                HintLoopsPartitionedWorkloadGenerationStage,
                ZigZagCoreMappingEstimationStage,
                SetFixedAllocationPerformanceStage,
                StreamCostModelEvaluationStage,
            ],
            **kwargs,
        )
        scme, _ = main_stage.run()
        scme = scme[0]
        return scme

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

    def get_hint_loops(self, layer_ids, core_ids):
        hint_loops = {}
        for layer_id, cores in zip(layer_ids, core_ids):
            updated_hint_loops = next(v for k, v in self.hint_loops.items() if layer_id in k).copy()
            if len(cores) > 1:
                nb_cores = len(cores)
                layer_node = next(n for n in self.original_workload.nodes() if n.id == layer_id)
                if layer_node.layer_dim_sizes.data.get(LayerDim("G"), 1) > 1:
                    loop_dim = LayerDim("G")
                elif layer_node.layer_dim_sizes.data.get(LayerDim("K"), 1) > 1:
                    loop_dim = LayerDim("K")
                else:
                    raise ValueError("Unknown what loop dim to split across cores")
                updated_hint_loops.append((loop_dim, nb_cores))
            hint_loops[(layer_id,)] = updated_hint_loops
        return hint_loops

    def get_real_predecessors(self, node, g=None):
        if not g:
            g = self.workload
        return list(n for n in g.predecessors(node) if n.id != node.id)

    def get_real_successors(self, node, g=None):
        if not g:
            g = self.workload
        return list(n for n in g.successors(node) if n.id != node.id)

    def is_leaf(self) -> bool:
        return True
