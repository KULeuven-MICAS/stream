import logging
import os
from time import time
from typing import Any, Optional, TypeAlias

import networkx as nx
import numpy as np
from zigzag.utils import pickle_deepcopy, pickle_load, pickle_save

from stream.cost_model.cost_model import StreamCostModelEvaluation
from stream.cost_model.steady_state_scheduler import SteadyStateScheduler
from stream.hardware.architecture.accelerator import Accelerator
from stream.opt.allocation.constraint_optimization.allocation import (
    NodeType,
    TimeSlotAllocation,
    get_optimal_allocations,
)
from stream.opt.allocation.constraint_optimization.utils import calculate_total_latency
from stream.stages.estimation.stream_cost_model_evaluation import StreamCostModelEvaluationStage
from stream.stages.estimation.zigzag_core_mapping_estimation import ZigZagCoreMappingEstimationStage
from stream.stages.generation.layer_stacks_generation import STACK_T
from stream.stages.generation.tiled_workload_generation import (
    TiledWorkloadGenerationStage,
)
from stream.stages.set_fixed_allocation_performance import SetFixedAllocationPerformanceStage
from stream.stages.stage import MainStage, Stage, StageCallable
from stream.utils import CostModelEvaluationLUT
from stream.workload.computation.computation_node import ComputationNode
from stream.workload.dnn_workload import DNNWorkloadStream
from stream.workload.mapping import TILING_T
from stream.workload.onnx_workload import ComputationNodeWorkload
from stream.workload.utils import get_real_successors

logger = logging.getLogger(__name__)

SCHEDULE_ORDER_T: TypeAlias = list[tuple[int, int]]


class ConstraintOptimizationAllocationStage(Stage):
    """
    Class that finds the best workload allocation for the workload using constraint optimization.
    This stages requires a CostModelEvaluationLUT, containing for each node and its valid core allocations the best CME.
    """

    CO_TIME_LIMIT = 600

    def __init__(
        self,
        list_of_callables: list[StageCallable],
        *,
        workload: ComputationNodeWorkload,
        accelerator: Accelerator,
        cost_lut: CostModelEvaluationLUT,
        layer_stacks: list[tuple[int, ...]],
        allocations_path: str,
        tiled_workload_post_co_path: str,
        cost_lut_post_co_path: str,
        **kwargs: Any,
    ):
        """Initialize the ResourceAllocationStage.

        Args:
            list_of_callables (list): List of the substages to be called. This should be empty as this is a leaf stage.
            workload (DiGraph): The NetworkX DiGraph representing the workload to be scheduled
            accelerator (Accelerator): The hardware accelerator onto which we schedule the workload
            cost_lut (CostModelEvaluationLUT): A lookup table containing for each node the best CME for each core
            layer_stacks (list): List of tuples with each tuple containing the layer ids to fuse together
            allocations_path (str): Path to the directory where the optimal allocations are stored
            cost_lut_post_co_path (str): Path to the file where the cost LUT after CO is stored
        """
        super().__init__(list_of_callables, **kwargs)
        self.workload = workload
        self.accelerator = accelerator
        self.cost_lut = cost_lut
        self.layer_stacks = layer_stacks
        self.original_workload: ComputationNodeWorkload = kwargs["original_workload"]
        self.mode = kwargs.get("mode", "fused")  # assume default is fused

        self.allocations_path = allocations_path
        os.makedirs(self.allocations_path, exist_ok=True)
        self.tiled_workload_post_co_path = tiled_workload_post_co_path
        self.cost_lut_post_co_path = cost_lut_post_co_path
        self.co_time_limit: int = kwargs.get("co_time_limit", self.CO_TIME_LIMIT)

        # Which CME attribute to use for the node latencies
        self.latency_attr = kwargs.get("latency_attr", "latency_total1")

        # Attributes that will be assigned throughout the stage
        self.ss_to_computes: dict[STACK_T, set[ComputationNode]] = {}
        self.hashes_per_sink_node: dict[STACK_T, dict[ComputationNode, int]] = {}
        self.steady_state_hashes: dict[STACK_T, int] = {}
        self.compute_per_sink_node: dict[STACK_T, dict[ComputationNode, set[ComputationNode]]] = {}
        self.ss_iterations_per_stack: dict[STACK_T, int] = {}
        self.optimal_allocation_per_stack: dict[STACK_T, TimeSlotAllocation] = {}
        self.nb_macs_per_stack: dict[STACK_T, int] = {}
        self.nb_macs_in_ss_per_stack: dict[STACK_T, int] = {}
        self.ss_mac_percentages_per_stack: dict[STACK_T, int] = {}

    def run(self):
        logger.info("Start ConstraintOptimizationAllocationStage.")
        # For each layer stack, determine for each node in the last layer(s) how many nodes have to be computed and cached

        self.extract_steady_state_per_stack()
        self.find_best_allocation_per_stack()
        _ = self.run_simple_scheduler()
        scme = self.run_coala()

        logger.info("End ConstraintOptimizationAllocationStage.")
        yield (scme, None)

    def run_coala(self):
        combined_allocation: TimeSlotAllocation = TimeSlotAllocation([], self.accelerator, self.workload)
        timestep_offset = 0
        for optimal_allocation in self.optimal_allocation_per_stack.values():
            # Update all timesteps in this allocation with the offset and add it to the combined allocation
            combined_allocation.merge_with_allocation(optimal_allocation, timestep_offset)
            max_timestep = optimal_allocation.slot_max
            timestep_offset = max_timestep + 1
        scme = self.schedule_allocation(combined_allocation)
        return scme

    def run_simple_scheduler(self):
        """
        Run a simple scheduler that unrolls the steady state allocations of different stacks into a single
        allocation to get a latency and energy estimate.
        """
        # tsa = TimeSlotAllocation([], self.accelerator, self.workload)
        # for stack, optimal_allocation in self.optimal_allocation_per_stack.items():
        #     warmup, already_computed = self.get_warmup_timeslot_allocation(stack, tsa, optimal_allocation)
        #     steady_state, already_computed = self.get_steady_state_timeslot_allocation(stack, warmup, optimal_allocation, already_computed)
        #     cooldown = self.get_cooldown_timeslot_allocation(stack, steady_state, optimal_allocation, already_computed)
        #     cooldown.visualize_allocation()
        #     tsa = cooldown
        # return cooldown
        for stack, optimal_allocation in self.optimal_allocation_per_stack.items():
            stack_subgraph = self.workload.get_subgraph([n for n in self.workload.node_list if n.id in stack])
            scheduler = SteadyStateScheduler(stack_subgraph, self.accelerator, self.original_workload)
            schedule = scheduler.run(optimal_allocation)
        return schedule

    def get_warmup_timeslot_allocation(
        self, stack: STACK_T, initial_tsa: TimeSlotAllocation, optimal_allocation: TimeSlotAllocation
    ) -> tuple[TimeSlotAllocation, set[ComputationNode]]:
        """
        Get the TimeSlotAllocation for the warmup phase of the given stack.
        """
        steady_state_hash = self.steady_state_hashes[stack]
        # Add all sink nodes that are part of the warmup (before steady state starts)
        warmup_sink_nodes = []
        for sink_node, memoization_hash in self.hashes_per_sink_node[stack].items():
            if memoization_hash != steady_state_hash:
                warmup_sink_nodes.append(sink_node)
            else:
                break
        # Build the TimeSlotAllocation for the warmup phase
        already_computed = set()
        warmup_tsa = initial_tsa
        for sink_node in warmup_sink_nodes:
            warmup_tsa, already_computed = self.update_timeslot_allocation_with_ancestors(
                sink_node, warmup_tsa, optimal_allocation, already_computed, NodeType.WARMUP
            )
        return warmup_tsa, already_computed

    def get_steady_state_timeslot_allocation(
        self,
        stack: STACK_T,
        warmup: TimeSlotAllocation,
        optimal_allocation: TimeSlotAllocation,
        already_computed: set[ComputationNode] = set(),
    ) -> tuple[TimeSlotAllocation, set[ComputationNode]]:
        """
        Get the TimeSlotAllocation for the steady state phase of the given stack.
        This is the allocation that is used to compute the steady state latency and energy.
        The given optimal_allocation is a single iteration of the steady state, which is unrolled here for all iterations.
        """
        # Get all the steady state sink nodes
        steady_state_hash = self.steady_state_hashes[stack]
        in_ss = False
        steady_state_sink_nodes = []
        for sink_node, memoization_hash in self.hashes_per_sink_node[stack].items():
            if memoization_hash != steady_state_hash:
                if not in_ss:
                    # We are still in the warmup phase
                    continue
                else:
                    # We are now in cooldown, so break
                    break
            in_ss = True
            steady_state_sink_nodes.append(sink_node)
        # Build the TimeSlotAllocation for the steady state phase
        steady_state_tsa = warmup  # Initialize with the warmup allocation
        for sink_node in steady_state_sink_nodes:
            steady_state_tsa, already_computed = self.update_timeslot_allocation_with_ancestors(
                sink_node, steady_state_tsa, optimal_allocation, already_computed, NodeType.STEADY_STATE
            )
        return steady_state_tsa, already_computed

    def get_cooldown_timeslot_allocation(
        self,
        stack: STACK_T,
        steady_state: TimeSlotAllocation,
        optimal_allocation: TimeSlotAllocation,
        already_computed: set[ComputationNode] = set(),
    ) -> TimeSlotAllocation:
        """
        Get the TimeSlotAllocation for the cooldown phase of the given stack.
        """
        steady_state_hash = self.steady_state_hashes[stack]
        # Add all sink nodes that are part of the cooldown (after steady state starts)
        cooldown_sink_nodes = []
        seen_ss = False
        for sink_node, memoization_hash in self.hashes_per_sink_node[stack].items():
            if memoization_hash != steady_state_hash and seen_ss:
                cooldown_sink_nodes.append(sink_node)
            else:
                seen_ss = True
        # Build the TimeSlotAllocation for the warmup phase
        cooldown_tsa = steady_state  # Initialize with the steady state allocation
        for sink_node in cooldown_sink_nodes:
            cooldown_tsa, already_computed = self.update_timeslot_allocation_with_ancestors(
                sink_node, cooldown_tsa, optimal_allocation, already_computed, NodeType.COOLDOWN
            )
        return cooldown_tsa

    def update_timeslot_allocation_with_ancestors(
        self,
        node: ComputationNode,
        tsa: TimeSlotAllocation,
        optimal_allocation: TimeSlotAllocation,
        already_computed: set[ComputationNode],
        node_type: NodeType,
    ) -> tuple[TimeSlotAllocation, set[ComputationNode]]:
        """
        Update the TimeSlotAllocation with all ancestors of the given node that are not already computed.
        This is used to ensure that all nodes needed for the node are included in the allocation.
        """
        needed_compute = nx.ancestors(self.workload, node) | {node}
        new_compute = needed_compute - already_computed
        subgraph = self.workload.get_subgraph(new_compute)
        sorted_like_steady_state = self.sort_like_steady_state(subgraph, optimal_allocation, node_type)
        for node in sorted_like_steady_state:
            if isinstance(node, ComputationNode):
                already_computed.add(node)
                core_allocations = optimal_allocation.get_cores_for_node_id(node.id)
                # Get the predecessors, and update the minimum slot this can be scheduled in accordingly
                preds = list(self.workload.predecessors(node))
                pred_slots = [tsa.get_timeslot_of_node(pred) for pred in preds]
                latest_pred_slot = max(pred_slots, default=0)
                for core in core_allocations:
                    tsa.add_node_to_next_slot(node, core, min_slot=latest_pred_slot + 1, node_type=node_type)
        return tsa, already_computed

    def sort_like_steady_state(
        self, subgraph: ComputationNodeWorkload, optimal_allocation: TimeSlotAllocation, node_type: NodeType
    ) -> list[ComputationNode]:
        """
        Sort the nodes in the subgraph like they are sorted in the steady state allocation.
        This is used to ensure that the nodes are scheduled in the same order as in the steady state allocation.
        """
        if node_type == NodeType.WARMUP:
            order: list[ComputationNode] = nx.topological_sort(subgraph)
        elif node_type == NodeType.STEADY_STATE:
            # There will be exactly the same amount of nodes in the given subgraph as in the optimal allocation.
            # We go through the nodes topologically and sort them based on the earliest optimal node with same id,
            # that we haven't seen yet.
            seen_optimal_nodes = set()
            seen_idxs = set()
            ss_nodes = optimal_allocation.nodes
            nb_nodes = len(ss_nodes)
            order_tmp: list[Optional[ComputationNode]] = [None] * nb_nodes
            for node in nx.topological_sort(subgraph):
                assert isinstance(node, ComputationNode), "Expected only ComputationNodes in the subgraph."
                eq_node = next(
                    n
                    for n in ss_nodes
                    if n.id == node.id and n not in seen_optimal_nodes and ss_nodes.index(n) not in seen_idxs
                )
                eq_node_idx = ss_nodes.index(eq_node)
                seen_optimal_nodes.add(eq_node)
                seen_idxs.add(eq_node_idx)
                order_tmp[eq_node_idx] = node
            assert None not in order_tmp, "Not all nodes were sorted correctly like in the steady state allocation."
            order = [node for node in order_tmp if node is not None]
        elif node_type == NodeType.COOLDOWN:
            order = nx.topological_sort(subgraph)
        else:
            raise ValueError(f"Unknown node type: {node_type}. Expected WARMUP, STEADY_STATE or COOLDOWN.")
        return order

    def extract_steady_state_per_stack(self):
        for i, stack in enumerate(self.layer_stacks):
            nodes = [n for n in self.workload.node_list if n.id in stack]
            if len(nodes) == 0:
                logger.warning(f"Stack {i} is empty.")
                continue

            sg = self.workload.get_subgraph(nodes)
            sink_nodes: list[ComputationNode] = sorted(
                n for n in sg.nodes() if len(get_real_successors(n, sg)) == 0  # type: ignore
            )
            sink_layer_ids = sorted(set(n.id for n in sink_nodes))
            assert len(sink_layer_ids) == 1, "Expected only one sink layer per layer stack. Update your layer stacks."
            sink_nodes_sorted = sorted(n for n in sink_nodes)
            computed: set[ComputationNode] = set()
            to_compute_sets: dict[int, set[ComputationNode]] = dict()
            memoization_hashes: dict[int, frozenset[tuple[int, int]]] = dict()
            to_compute_counts: dict[int, int] = dict()
            state_ids: dict[int, list[int]] = dict()
            to_compute_unique: dict[ComputationNode, set[ComputationNode]] = dict()
            hashes_per_sink_pair: dict[ComputationNode, int] = dict()
            for sink_node in sink_nodes_sorted:
                needed_compute = nx.ancestors(sg, sink_node) | {sink_node}  # type: ignore
                to_compute: set[ComputationNode] = needed_compute - computed  # type: ignore
                to_compute_unique[sink_node] = to_compute
                to_compute_ids = [n.id for n in to_compute]
                to_compute_per_layer = {id: to_compute_ids.count(id) for id in stack}
                to_compute_set = frozenset(sorted(to_compute_per_layer.items()))
                memoization_hash = hash(to_compute_set)
                hashes_per_sink_pair[sink_node] = memoization_hash
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
        total_ss_latency = 0
        for stack, to_compute in self.ss_to_computes.items():
            iterations = self.ss_iterations_per_stack[stack]
            t_start = time()
            optimal_allocation = self.find_best_allocation(to_compute, iterations, stack, self.co_time_limit)
            ss_latency, _ = calculate_total_latency(
                optimal_allocation, self.cost_lut, self.accelerator, iterations, self.latency_attr
            )
            t_end = time()
            logger.info(
                f"Stack {stack}: Optimization took {t_end - t_start:.3f} seconds; Predicted steady-state latency: {ss_latency} cycles"
            )
            self.optimal_allocation_per_stack[stack] = optimal_allocation
            total_ss_latency += ss_latency
        logger.info(f"Total steady-state latency across stacks: {total_ss_latency} cycles")

    def find_best_allocation(
        self, to_compute: set[ComputationNode], iterations: int, stack: STACK_T = (0,), time_limit: int = 600
    ) -> TimeSlotAllocation:
        """# TODO: Implement overhead of tensor transfers between cores"""
        # Check if the allocation is already cached, if not: find it
        stack_str = "_".join([str(id) for id in stack])
        stack_allocations_path = os.path.join(self.allocations_path, f"steady_state-{stack_str}.pickle")
        sg = self.workload.subgraph(to_compute)
        if os.path.exists(stack_allocations_path):
            allocation = pickle_load(stack_allocations_path)
        else:
            logger.info(f"Optimizing allocation for {iterations} iterations of {len(to_compute)} ss nodes.")
            allocation = get_optimal_allocations(
                sg,
                self.accelerator,
                self.cost_lut,
                iterations,
                time_limit=time_limit,
                latency_attr=self.latency_attr,
            )
            pickle_save(allocation, stack_allocations_path)
        allocation_obj = TimeSlotAllocation(allocation, self.accelerator, self.workload)
        # json_path = stack_allocations_path.replace(".pickle", ".json")
        # to_perfetto_json(allocation, self.cost_lut, self.accelerator, iterations, self.latency_attr, json_path)

        return allocation_obj

    def get_scheduling_order(self, unpartitioned_workload: DNNWorkloadStream) -> SCHEDULE_ORDER_T:
        """
        Get the scheduling order of all ids that will exist in the transformed workload.
        Returns a list with all ids where the earlier the higher priority
        The scheduling order is altered to accommodate the inter core tiling of the given workload

        Args:
           unpartitioned_workload: original workload (before partitioning into finer nodes), used to extract the inter-
                                   core tiling loops
        """

        scheduling_order: SCHEDULE_ORDER_T = []
        for stack in sorted(self.compute_per_sink_node):
            compute_this_stack = self.compute_per_sink_node[stack]
            hash_steady_state = self.steady_state_hashes[stack]
            allocation_steady_state = self.optimal_allocation_per_stack[stack]
            hashes_per_sink_node = self.hashes_per_sink_node[stack]
            order = self.get_cn_order(
                allocation=allocation_steady_state,
                compute_per_sink_node=compute_this_stack,
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
        core tiling, and adjusts the scheduling beforehand.

        Example: [(0, 12), (0, 13)] and inter_core_tiling = 4
            -> [(0, 4*12+0), (0, 49), (0, 50), (0, 51), (0, 4*13+0), ...]
                <------intra-core partition 12------->  <---- partition 13 ---->

        NOTE The ordering given by this method must match the order in which tiles are generated in `get_tiles`

        Args:
            stack: CN stack for which the order applies
            order: scheduling order for this stack, without inter core tiling
            unpartitioned_workload: original workload (before partitioning into finder nodes), used to extract the inter-
                                    core tiling loops

        """
        adjusted_order = order.copy()

        for curr_node in self.get_computation_nodes(stack, unpartitioned_workload):
            # NOTE this uses `inter_core_tiling`, because the inter core tiling is added to the intra core tiling
            # in `schedule_allocation` in order to alter the workload
            outer_loops = curr_node.inter_core_tiling

            for _, inter_core_split_factor in outer_loops:
                assert isinstance(
                    inter_core_split_factor, int
                ), "tiling options `*` and `all` should be replaced by now"
                if inter_core_split_factor == 1:
                    # In case CO decides to not split up the node across cores
                    continue

                i = 0
                while i < len(adjusted_order):
                    layer_id, sub_id = adjusted_order[i]
                    if layer_id == curr_node.id:
                        adjusted_order[i : i + 1] = [
                            (layer_id, sub_id * inter_core_split_factor + j) for j in range(inter_core_split_factor)
                        ]
                        i += inter_core_split_factor
                    else:
                        i += 1

        return adjusted_order

    def get_nb_nodes_for_layer(self, layer_id: int):
        return len(list(n for n in self.workload.node_list if n.id == layer_id))

    def get_computation_nodes(self, stack: tuple[int, ...], workload: DNNWorkloadStream) -> list[ComputationNode]:
        nodes = [n for n in workload.node_list if n.id in stack]
        computation_nodes = [n for n in nodes if isinstance(n, ComputationNode)]
        return computation_nodes

    def get_cn_order(
        self,
        allocation: TimeSlotAllocation,
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
        allocation_adjusted: TimeSlotAllocation = (
            []
        )  # allocation with removed inter core splits (which have same sub id)
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

    def schedule_allocation(self, allocation: TimeSlotAllocation) -> StreamCostModelEvaluation:
        # Get the involved layer ids we want to schedule and their core allocations
        layer_ids = [node.id for node in allocation.nodes]
        core_ids = [core.id for core in allocation.cores]

        # Get the max layer id in the allocation, and a subgraph of the original workload
        max_layer_id = max(layer_ids)
        sub_nodes = filter(lambda n: n.id <= max_layer_id, self.original_workload.node_list)
        unpartitioned_sub_workload: DNNWorkloadStream = pickle_deepcopy(self.original_workload.subgraph(sub_nodes))

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

            # Set correct inter core tiling. Replacing the wildcard will signal to the TiledWorkloadGenerationStage
            # to also spit in the inter core tiling
            inter_core_tiling = self.replace_wildcard_in_tiling(node.inter_core_tiling, nb_cores_split)
            node.inter_core_tiling = inter_core_tiling

        scheduling_order = self.get_scheduling_order(unpartitioned_sub_workload)

        loma_lpf_limit = 7
        kwargs = self.kwargs.copy()
        kwargs["loma_lpf_limit"] = loma_lpf_limit
        kwargs["accelerator"] = self.accelerator
        kwargs["workload"] = unpartitioned_sub_workload
        kwargs["scheduling_order"] = scheduling_order
        kwargs["layer_stacks"] = self.layer_stacks
        kwargs["tiled_workload_path"] = self.tiled_workload_post_co_path
        kwargs["cost_lut_path"] = self.cost_lut_post_co_path
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
            n.possible_core_allocation = list(cores)

    def replace_wildcard_in_tiling(self, tiling: TILING_T, nb_cores_split: int):
        """The user can define a wildcard `*` in the inter core tiling, meaning that the value found by the CO
        must be used instead.
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
