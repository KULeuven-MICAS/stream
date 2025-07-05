import logging
import os
from math import prod
from time import time
from typing import Any, TypeAlias

import networkx as nx
import numpy as np
from zigzag.utils import pickle_deepcopy, pickle_load, pickle_save

from stream.cost_model.cost_model import StreamCostModelEvaluation
from stream.cost_model.steady_state_scheduler import SteadyStateScheduler
from stream.hardware.architecture.accelerator import Accelerator
from stream.hardware.architecture.core import Core
from stream.opt.allocation.constraint_optimization.allocation import ALLOCATION_T, get_optimal_allocations
from stream.opt.allocation.constraint_optimization.timeslot_allocation import NodeType, TimeSlotAllocation
from stream.opt.allocation.constraint_optimization.utils import calculate_total_latency, get_partitioned_nodes
from stream.stages.estimation.stream_cost_model_evaluation import StreamCostModelEvaluationStage
from stream.stages.estimation.zigzag_core_mapping_estimation import ZigZagCoreMappingEstimationStage
from stream.stages.generation.layer_stacks_generation import STACK_T
from stream.stages.generation.tiled_workload_generation import TiledWorkloadGenerationStage
from stream.stages.set_fixed_allocation_performance import SetFixedAllocationPerformanceStage
from stream.stages.stage import MainStage, Stage, StageCallable
from stream.utils import CostModelEvaluationLUT
from stream.workload.computation.computation_node import ComputationNode
from stream.workload.dnn_workload import DNNWorkloadStream
from stream.workload.mapping import TILING_T, TILING_WILDCARD_T
from stream.workload.onnx_workload import ComputationNodeWorkload
from stream.workload.steady_state.computation import SteadyStateComputation
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
        self.ss_mac_percentages_per_stack: dict[STACK_T, float] = {}

    def run(self):
        logger.info("Start ConstraintOptimizationAllocationStage.")
        self.extract_steady_state_per_stack()
        self.find_best_allocation_per_stack()
        # _ = self.run_simple_scheduler()
        scme = self.run_coala()

        logger.info("End ConstraintOptimizationAllocationStage.")
        yield (scme, None)

    def run_coala(self):
        unrolled_allocation = self.get_unrolled_allocation()
        scme = self.schedule_allocation(unrolled_allocation)
        return scme

    def run_simple_scheduler(self):
        """
        Run a simple scheduler that unrolls the steady state allocations of different stacks into a single
        allocation to get a latency and energy estimate.
        """
        for stack, optimal_allocation in self.optimal_allocation_per_stack.items():
            stack_subgraph = self.workload.get_subgraph([n for n in self.workload.node_list if n.id in stack])
            iterations = self.ss_iterations_per_stack[stack]
            scheduler = SteadyStateScheduler(
                stack_subgraph, self.accelerator, self.original_workload, self.cost_lut, iterations
            )
            schedule = scheduler.run(optimal_allocation)
        return schedule

    def get_unrolled_allocation(self) -> TimeSlotAllocation:
        tsa = TimeSlotAllocation([])  # Initialize an empty TimeSlotAllocation
        min_slot = 0
        for stack, optimal_allocation in self.optimal_allocation_per_stack.items():
            warmup, already_computed = self.get_warmup_timeslot_allocation(stack, tsa, optimal_allocation, min_slot)
            steady_state, already_computed = self.get_steady_state_timeslot_allocation(
                stack, warmup, optimal_allocation, already_computed, min_slot
            )
            cooldown = self.get_cooldown_timeslot_allocation(
                stack, steady_state, optimal_allocation, already_computed, min_slot
            )
            tsa = cooldown
            min_slot = tsa.slot_max + 1  # Update the minimum slot for the next stack
        print(f"Unrolled allocation has {tsa.slot_max - tsa.slot_min + 1} slots:")
        return tsa

    def get_warmup_timeslot_allocation(
        self,
        stack: STACK_T,
        initial_tsa: TimeSlotAllocation,
        optimal_allocation: TimeSlotAllocation,
        min_slot: int = 0,
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
                sink_node, warmup_tsa, optimal_allocation, already_computed, stack, NodeType.WARMUP, min_slot
            )
        return warmup_tsa, already_computed

    def get_steady_state_timeslot_allocation(
        self,
        stack: STACK_T,
        warmup: TimeSlotAllocation,
        optimal_allocation: TimeSlotAllocation,
        already_computed: set[ComputationNode] | None = None,
        min_slot: int = 0,
    ) -> tuple[TimeSlotAllocation, set[ComputationNode]]:
        """
        Get the TimeSlotAllocation for the steady state phase of the given stack.
        This is the allocation that is used to compute the steady state latency and energy.
        The given optimal_allocation is a single iteration of the steady state, which is unrolled for all iterations.
        """
        if already_computed is None:
            already_computed = set()
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
                sink_node,
                steady_state_tsa,
                optimal_allocation,
                already_computed,
                stack,
                NodeType.STEADY_STATE,
                min_slot,
            )
        return steady_state_tsa, already_computed

    def get_cooldown_timeslot_allocation(
        self,
        stack: STACK_T,
        steady_state: TimeSlotAllocation,
        optimal_allocation: TimeSlotAllocation,
        already_computed: set[ComputationNode] | None = None,
        min_slot: int = 0,
    ) -> TimeSlotAllocation:
        """
        Get the TimeSlotAllocation for the cooldown phase of the given stack.
        """
        if already_computed is None:
            already_computed = set()
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
                sink_node, cooldown_tsa, optimal_allocation, already_computed, stack, NodeType.COOLDOWN, min_slot
            )
        return cooldown_tsa

    def update_timeslot_allocation_with_ancestors(
        self,
        node: ComputationNode,
        tsa: TimeSlotAllocation,
        optimal_allocation: TimeSlotAllocation,
        already_computed: set[ComputationNode],
        stack: STACK_T,
        node_type: NodeType,
        min_slot: int,
    ) -> tuple[TimeSlotAllocation, set[ComputationNode]]:
        """
        Update the TimeSlotAllocation with all ancestors of the given node that are not already computed.
        This is used to ensure that all nodes needed for the node are included in the allocation.
        """
        needed_compute = nx.ancestors(self.workload, node) | {node}
        new_compute: set[ComputationNode] = needed_compute - already_computed
        new_compute_this_stack: list[ComputationNode] = [n for n in new_compute if n.id in stack]
        subgraph = self.workload.get_subgraph(new_compute_this_stack)
        sorted_like_steady_state = self.sort_like_steady_state(subgraph, optimal_allocation, node_type)
        for n in sorted_like_steady_state:
            if isinstance(n, ComputationNode):
                already_computed.add(n)
                core_allocations = optimal_allocation.get_resources_for_node_id(n.id)
                # Get the predecessors, and update the minimum slot this can be scheduled in accordingly
                preds = list(self.workload.predecessors(n))
                pred_slots = [tsa.get_timeslot_of_node(pred) for pred in preds]  # type: ignore
                latest_pred_slot = max(pred_slots, default=0)
                slot = max(latest_pred_slot + 1, min_slot)
                for core in core_allocations:
                    tsa.add_node_to_next_slot(n, core, min_slot=slot, node_type=node_type)  # type: ignore
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
            order_tmp: list[ComputationNode | None] = [None] * nb_nodes
            for node in nx.topological_sort(subgraph):
                assert isinstance(node, ComputationNode), "Expected only ComputationNodes in the subgraph."
                try:
                    eq_node = next(
                        n
                        for n in ss_nodes
                        if n.id == node.id and n not in seen_optimal_nodes and ss_nodes.index(n) not in seen_idxs
                    )
                except StopIteration as exc:
                    raise ValueError(
                        f"Node {node.id} not found in the steady state allocation. "
                        "This should not happen, please check your steady state allocation."
                    ) from exc
                eq_node_idx = ss_nodes.index(eq_node)
                seen_optimal_nodes.add(eq_node)
                seen_idxs.add(eq_node_idx)
                order_tmp[eq_node_idx] = node
            # assert None not in order_tmp, "Not all nodes were sorted correctly like in the steady state allocation."
            order = [node for node in order_tmp if node is not None]
        elif node_type == NodeType.COOLDOWN:
            order = nx.topological_sort(subgraph)
        else:
            raise ValueError(f"Unknown node type: {node_type}. Expected WARMUP, STEADY_STATE or COOLDOWN.")
        return order

    def extract_steady_state_per_stack(self) -> None:
        """
        Extract steady state computation nodes and related statistics for each stack.
        """
        for i, stack in enumerate(self.layer_stacks):
            nodes = [n for n in self.workload.node_list if n.id in stack]
            if not nodes:
                logger.warning(f"Stack {i} is empty.")
                continue

            sg = self.workload.get_subgraph(nodes)
            sink_nodes = self._get_sorted_sink_nodes(sg)
            self._process_stack_sink_nodes(stack, sink_nodes, sg)

        self._log_steady_state_statistics()

    def _get_sorted_sink_nodes(self, sg: ComputationNodeWorkload) -> list[ComputationNode]:
        sink_nodes = sorted(
            n
            for n in sg.nodes()
            if len(get_real_successors(n, sg)) == 0  # type: ignore
        )
        sink_layer_ids = sorted(set(n.id for n in sink_nodes))
        assert len(sink_layer_ids) == 1, "Expected only one sink layer per layer stack. Update your layer stacks."
        return sorted(sink_nodes)

    def _process_stack_sink_nodes(
        self,
        stack: STACK_T,
        sink_nodes: list[ComputationNode],
        sg: ComputationNodeWorkload,
    ) -> None:
        computed: set[ComputationNode] = set()
        to_compute_sets: dict[int, set[ComputationNode]] = {}
        memoization_hashes: dict[int, frozenset[tuple[int, int]]] = {}
        to_compute_counts: dict[int, int] = {}
        state_ids: dict[int, list[int]] = {}
        to_compute_unique: dict[ComputationNode, set[ComputationNode]] = {}
        hashes_per_sink_pair: dict[ComputationNode, int] = {}

        for sink_node in sink_nodes:
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

        self._finalize_stack_steady_state(
            stack,
            to_compute_sets,
            to_compute_counts,
            to_compute_unique,
            hashes_per_sink_pair,
        )

    def _finalize_stack_steady_state(
        self,
        stack: STACK_T,
        to_compute_sets: dict[int, set[ComputationNode]],
        to_compute_counts: dict[int, int],
        to_compute_unique: dict[ComputationNode, set[ComputationNode]],
        hashes_per_sink_pair: dict[ComputationNode, int],
    ) -> None:
        scaled_counts: dict[int, int] = {}
        total_nb_macs = 0
        for memoization_hash, count in to_compute_counts.items():
            nb_macs = int(sum(n.total_mac_count for n in to_compute_sets[memoization_hash]))
            scaled_counts[memoization_hash] = nb_macs * count
            total_nb_macs += nb_macs * count

        max_count = max(scaled_counts.values())
        self.ss_mac_percentages_per_stack[stack] = max_count / total_nb_macs if total_nb_macs else 0
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

    def _log_steady_state_statistics(self) -> None:
        nb_steady_state_nodes = sum(len(v) for v in self.ss_to_computes.values())
        nb_nodes = self.workload.number_of_nodes()
        percentage_nodes = nb_steady_state_nodes / nb_nodes * 100 if nb_nodes else 0
        logger.info(f"Percentage of steady state nodes: {nb_steady_state_nodes}/{nb_nodes} = {percentage_nodes:.2f}%")
        nb_steady_state_macs = sum(self.nb_macs_in_ss_per_stack.values())
        nb_macs = sum(self.nb_macs_per_stack.values())
        percentage_macs = nb_steady_state_macs / nb_macs * 100 if nb_macs else 0
        logger.info(f"Percentage of steady state macs: {nb_steady_state_macs}/{nb_macs} = {percentage_macs:.2f}%")

    def find_best_allocation_per_stack(self):
        total_ss_latency = 0
        for stack, to_compute in self.ss_to_computes.items():
            iterations = self.ss_iterations_per_stack[stack]
            t_start = time()
            optimal_allocation = self.find_best_allocation(to_compute, iterations, stack, self.co_time_limit)
            ss_latency, _ = calculate_total_latency(optimal_allocation, iterations)
            t_end = time()
            logger.info(f"Stack {stack}: Optimization took {t_end - t_start:.3f} seconds.")
            logger.info(f"Predicted steady-state latency: {ss_latency} cycles.")
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
                iterations=iterations,
                time_limit=time_limit,
                latency_attr=self.latency_attr,
            )
            pickle_save(allocation, stack_allocations_path)  # type: ignore
        steady_state_allocation_list = self.get_steady_state_allocation_list(allocation)
        tsa = TimeSlotAllocation(steady_state_allocation_list)  # type: ignore
        # json_path = stack_allocations_path.replace(".pickle", ".json")
        # to_perfetto_json(allocation, self.cost_lut, self.accelerator, iterations, self.latency_attr, json_path)

        return tsa

    def get_steady_state_allocation_list(
        self, allocation: ALLOCATION_T
    ) -> list[tuple[int, Core, SteadyStateComputation]]:
        """
        Get the SteadyStateComputation for the given node id and sub id.
        This is used to create the SteadyStateComputation object that is used in the TimeSlotAllocation.
        """
        # Get the core allocations for each unique node id and sub id in the allocation
        node_id_to_cores: dict[tuple[int, int], list[Core]] = {}
        node_id_to_slots: dict[tuple[int, int], list[int]] = {}
        for slot, core_id, (n_id, n_sub_id) in allocation:
            # Keep slot
            node_id_to_slots[(n_id, n_sub_id)] = node_id_to_slots.get((n_id, n_sub_id), [])
            node_id_to_slots[(n_id, n_sub_id)].append(slot)
            core = self.accelerator.get_core(core_id)
            node_id_to_cores[(n_id, n_sub_id)] = node_id_to_cores.get((n_id, n_sub_id), [])
            node_id_to_cores[(n_id, n_sub_id)].append(core)
        # Get the partitioned SteadyStateComputation objects for each unique node id and sub id
        steady_state_computations: dict[tuple[int, int], list[SteadyStateComputation]] = {}
        for (n_id, n_sub_id), cores in node_id_to_cores.items():
            # Get the original node from the workload
            node = next(n for n in self.workload.node_list if n.id == n_id and n.sub_id == n_sub_id)
            steady_state_computations[(n_id, n_sub_id)] = get_partitioned_nodes(
                node, cores, self.accelerator, self.cost_lut
            )
        # Create the converted allocation_list with SteadyStateComputation objects
        allocation_list: list[tuple[int, Core, SteadyStateComputation]] = []
        for (n_id, n_sub_id), cores in node_id_to_cores.items():
            slots = node_id_to_slots[(n_id, n_sub_id)]
            computations = steady_state_computations[(n_id, n_sub_id)]
            for slot, core, computation in zip(slots, cores, computations, strict=False):
                allocation_list.append((slot, core, computation))
        return allocation_list

    def get_scheduling_order(self, allocation: TimeSlotAllocation) -> SCHEDULE_ORDER_T:
        """
        Get the scheduling order of all ids that will exist in the transformed workload.
        Returns a list with all ids where the earlier the higher priority
        The scheduling order is altered to accommodate the inter core tiling of the given workload:
        Example: [(0, 12), (0, 13)] and inter_core_tiling = 4
            -> [(0, 4*12+0), (0, 49), (0, 50), (0, 51), (0, 4*13+0), ...]
                <------intra-core partition 12------->  <---- partition 13 ---->

        Args:
           allocation: TimeSlotAllocation for which the scheduling order should be generated
        """
        order: SCHEDULE_ORDER_T = []
        for t in range(allocation.slot_min, allocation.slot_max + 1):
            for node in allocation.get_allocations_in_slot(t).values():
                nb_cores = len(allocation.get_resources_for_node_id(node.id))
                sub_id = node.sub_id  # type: ignore
                adjusted_sub_id = nb_cores * sub_id
                while (node.id, adjusted_sub_id) in order:
                    # Increment the adjusted_sub_id with the sub_id until it is unique
                    adjusted_sub_id += 1
                order.append((node.id, adjusted_sub_id))
        return order

    def get_nb_nodes_for_layer(self, layer_id: int):
        return len(list(n for n in self.workload.node_list if n.id == layer_id))

    def get_computation_nodes(self, stack: tuple[int, ...], workload: DNNWorkloadStream) -> list[ComputationNode]:
        nodes = [n for n in workload.node_list if n.id in stack]
        computation_nodes = [n for n in nodes if isinstance(n, ComputationNode)]
        return computation_nodes

    def get_order_non_steady_state(self, to_compute: set[ComputationNode]):
        return [(n.id, n.sub_id) for n in sorted(to_compute, key=lambda x: (-x.id, -x.sub_id))]

    def schedule_allocation(self, allocation: TimeSlotAllocation) -> StreamCostModelEvaluation:
        # Get the involved layer ids we want to schedule and their core allocations
        # Get the relevant subgraph of the original layer-wise workload
        layer_ids = [node.id for node in allocation.nodes]
        relevant_nodes = list(filter(lambda n: n.id in layer_ids, self.original_workload.node_list))
        unpartitioned_sub_workload: DNNWorkloadStream = pickle_deepcopy(self.original_workload.subgraph(relevant_nodes))

        for n in unpartitioned_sub_workload.node_list:
            if n.id not in layer_ids:
                raise ValueError(
                    f"{n} not in steady state allocation. If this is intended, set their core allocation manually."
                )
        # # Manually add the wanted core ids for layers not in the steady state
        # layer_ids, core_ids = self.add_core_ids_for_layers_not_in_steady_state(
        #     layer_ids=layer_ids, core_ids=core_ids, sub_workload=unpartitioned_sub_workload
        # )

        # Set the correct allocations for the layers in the copied workload
        self.set_fixed_allocations_for_workload(unpartitioned_sub_workload, allocation)
        # Generate/check inter core mapping for all nodes
        self.update_inter_core_mapping(unpartitioned_sub_workload, allocation)
        scheduling_order = self.get_scheduling_order(allocation)

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
        kwargs["fix_all"] = True  # Fix all core allocations to the given ones

        # Create stages that will run a single cost model evaluation (fixed core allocations)
        main_stage = MainStage(
            [
                TiledWorkloadGenerationStage,  # Splits in intra-core mapping
                ZigZagCoreMappingEstimationStage,  # type: ignore
                SetFixedAllocationPerformanceStage,
                StreamCostModelEvaluationStage,
            ],
            **kwargs,
        )
        scme, _ = main_stage.run()
        scme = scme[0]
        return scme

    def update_inter_core_mapping(
        self, unpartitioned_sub_workload: ComputationNodeWorkload, allocation: TimeSlotAllocation
    ):
        for node in unpartitioned_sub_workload.node_list:
            nb_cores = len(allocation.get_resources_for_node_id(node.id))
            # Set correct inter core tiling. Replacing the wildcard will signal to the TiledWorkloadGenerationStage
            # to also split in the inter core tiling
            inter_core_tiling = self.replace_wildcard_in_tiling(node.inter_core_tiling, nb_cores)
            node.inter_core_tiling = inter_core_tiling
            # Check that the length matches the specified inter_core_tiling size
            inter_core_tiling_size = prod([factor for _, factor in inter_core_tiling])
            assert len(node.possible_core_allocation) == inter_core_tiling_size, (
                f"Expected {node} to have {inter_core_tiling_size} "
                f"possible core allocations, but got {len(node.possible_core_allocation)}."
            )

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
                node.chosen_core_allocation = node.core_allocation[0]
                node.possible_core_allocation = node.core_allocation
                layer_ids.insert(layer_ids_idx, layer_id_not_in_ss)
                core_ids.insert(layer_ids_idx, node.core_allocation)
                logger.warning(f"{node} not in steady state allocation; allocated to: {node.core_allocation[0]}.")

        return layer_ids, core_ids

    def set_fixed_allocations_for_workload(self, workload: ComputationNodeWorkload, allocation: TimeSlotAllocation):
        """! Modify the workload to fix the core allocations to the given core_ids for the given layer_ids."""
        for n in workload.node_list:
            cores = allocation.get_resources_for_node_id(n.id)
            n.possible_core_allocation = list(core.id for core in cores)  # type: ignore

    def replace_wildcard_in_tiling(self, tiling: TILING_WILDCARD_T | TILING_T, nb_cores_split: int):
        """The user can define a wildcard `*` in the inter core tiling, meaning that the value found by the CO
        must be used instead.
        """
        if len(tiling) > 1:
            raise ValueError("Only 1 partition dimension should be defined in inter core tiling")
        assert len(tiling) == 1, "No inter core tiling found"

        tiling_replaced: TILING_T = []
        for layer_dim, split_factor in tiling:
            if split_factor == "*":
                split_factor_new = nb_cores_split
            elif split_factor == "all":
                raise ValueError("inter core tiling should not contain `all`")
            else:
                split_factor_new = split_factor
            tiling_replaced.append((layer_dim, split_factor_new))
        return tiling_replaced

    def is_leaf(self) -> bool:
        return True
