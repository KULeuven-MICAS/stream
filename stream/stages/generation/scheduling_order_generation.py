import logging
from math import prod
from typing import Any, TypeAlias

from stream.hardware.architecture.accelerator import Accelerator
from stream.stages.stage import Stage, StageCallable
from stream.utils import contains_wildcard
from stream.workload.computation.computation_node import ComputationNode, GeneratedComputationNode
from stream.workload.mapping import TILING_T
from stream.workload.onnx_workload import ComputationNodeWorkload

logger = logging.getLogger(__name__)


SCHEDULE_ORDER_T: TypeAlias = list[tuple[int, int]]


class SchedulingOrderGenerationStage(Stage):
    def __init__(
        self,
        list_of_callables: list[StageCallable],
        *,
        accelerator: Accelerator,
        workload: ComputationNodeWorkload,
        **kwargs: Any,
    ):
        super().__init__(list_of_callables, **kwargs)
        self.accelerator = accelerator
        self.workload = workload
        self.layer_stacks: list[tuple[int, ...]] | None = kwargs.get("layer_stacks", None)  # optional

        # Mapping from (base_id, gen_id) to layer_id
        self.base_and_gen_to_layer_id: dict[tuple[int, int], int] = {
            (n.base_id, n.gen_id): n.id for n in self.workload.node_list if isinstance(n, GeneratedComputationNode)
        }

    def run(self):
        if self.layer_stacks:
            self.scheduling_order = self.get_scheduling_order_fused()
        else:
            self.scheduling_order = self.get_scheduling_order_lbl()

        self.kwargs["accelerator"] = self.accelerator
        self.kwargs["workload"] = self.workload
        self.kwargs["scheduling_order"] = self.scheduling_order
        sub_stage = self.list_of_callables[0](
            self.list_of_callables[1:],
            **self.kwargs,
        )
        for cme, extra_info in sub_stage.run():
            yield cme, extra_info

    def get_scheduling_order_lbl(self):
        """Generate the scheduling order in case layers are executed sequentially"""
        return sorted(((n.id, n.sub_id) for n in self.workload.node_list), reverse=True)

    def get_scheduling_order_fused(self):
        """Generate the scheduling order for a tiled workload (inter- and intra-core tiled)."""
        # Assumes layer_stacks contain all id's! (responsibility of LayerStacksGenerationStage)
        assert self.layer_stacks

        order: SCHEDULE_ORDER_T = []

        for stack in sorted(self.layer_stacks):
            order += self.get_scheduling_order_for_stack_with_generated_nodes(stack)

        assert len(order) == self.workload.number_of_nodes()
        return order

    def get_scheduling_order_for_stack_no_generated_nodes(self, stack: tuple[int, ...]):
        """
        Example: stack=[0,1] and inter-core tiling is 3 and 4 respectively:
        -> [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2), (1,3), (0,4), (0,5), ...
            <-------------intra-core slot 0--------------->  <---- intra-core slot 1 ---->
            <--all parallel--->
        # TODO remove: get_scheduling_order_for_stack_with_generated_nodes does the exact same
        """
        order: SCHEDULE_ORDER_T = []

        if not stack:
            return order

        some_node_per_layer = [self.get_some_node_for_id(layer_id) for layer_id in stack]
        assert all(not isinstance(n, GeneratedComputationNode) for n in some_node_per_layer)

        # For each layer in stack, get the number of cores and number of intra-core slots over which it is split
        inter_core_tiling_factor_per_layer: list[int] = [
            self.get_total_tiling_size(n.inter_core_tiling) for n in some_node_per_layer
        ]

        nb_intra_core_slots = self._get_and_assert_intra_core_tiling(some_node_per_layer)

        for i in range(nb_intra_core_slots):
            for layer_id, inter_core_tiling in zip(stack, inter_core_tiling_factor_per_layer):
                order += [(layer_id, i * inter_core_tiling + j) for j in range(inter_core_tiling)]

        return order

    def get_scheduling_order_for_stack_with_generated_nodes(self, stack: tuple[int, ...]) -> SCHEDULE_ORDER_T:
        """
        Example: stack=[0,1] and inter-core tiling is 3 and 4 respectively:
        -> [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2), (1,3), (0,4), (0,5), ...
            <-------------intra-core slot 0--------------->  <---- intra-core slot 1 ---->
            <--all parallel--->

        For the generated nodes: main idea is to use `gen_id` as the iter-core tiling slot
        """
        if not stack:
            return []

        # Only single entry for each base_id
        filtered_stack = self.filter_generated_nodes_from_stack(stack)

        some_node_per_layer = [self.get_some_node_for_id(layer_id) for layer_id in filtered_stack]

        # Keep track which nodes are generated and which are not
        non_generated_nodes = [n for n in some_node_per_layer if not isinstance(n, GeneratedComputationNode)]
        generated_base_nodes = [n for n in some_node_per_layer if isinstance(n, GeneratedComputationNode)]
        generated_base_ids = [n.base_id for n in generated_base_nodes]

        # Intra-core tiling: only for non-generated nodes
        nb_intra_core_slots = self._get_and_assert_intra_core_tiling(non_generated_nodes)

        # Intra-core slots for generated nodes
        nb_generated_nodes = self._get_and_assert_nb_generated_nodes(generated_base_nodes)
        assert (
            nb_generated_nodes % nb_intra_core_slots == 0
        ), "Number of generated nodes is not a multiple of the intra-core tiling"
        generated_layer_ids_per_intra_slot = nb_generated_nodes // nb_intra_core_slots

        # Inter-core tiling logic. Same order as `filtered_stack`
        inter_core_tiling_factor_per_layer: list[int] = [
            self.get_total_tiling_size(n.inter_core_tiling) for n in some_node_per_layer
        ]

        return self._generate_scheduling_order_for_stack_with_generated_nodes(
            filtered_stack=filtered_stack,
            inter_core_tiling_factors=inter_core_tiling_factor_per_layer,
            nb_intra_core_slots=nb_intra_core_slots,
            generated_base_ids=generated_base_ids,
            nb_gen_ids_per_slot=generated_layer_ids_per_intra_slot,
        )

    def _generate_scheduling_order_for_stack_with_generated_nodes(
        self,
        filtered_stack: tuple[int, ...],
        inter_core_tiling_factors: list[int],
        nb_intra_core_slots: int,
        generated_base_ids: list[int],
        nb_gen_ids_per_slot: int,
    ):
        """Generate the scheduling order for a stack of layers

        Args:
        filtered_stack: layer ids in stack. Only a single entry for each base_id
        inter_core_tiling_factors: inter-core tiling factor for each layer in stack (same order as `filtered_stack`)
        nb_intra_core_slots: Number of intra-core slots to schedule over. Same number of slots for all layers in stack
        generated_base_ids: Which ids in the filtered_stack are generated nodes
        nb_gen_ids_per_slot: Number of generated nodes with different layer ids per intra-core slot (excl.
                             inter-core tiling)
        """

        order: SCHEDULE_ORDER_T = []

        for i in range(nb_intra_core_slots):
            order_this_slot: SCHEDULE_ORDER_T = []
            for j in range(max(nb_gen_ids_per_slot, 1)):
                # For each j, go through all layers again to make sure generated nodes are nicely interleaved
                for layer_id, inter_core_tiling in zip(filtered_stack, inter_core_tiling_factors):

                    if layer_id in generated_base_ids:
                        gen_id_nb = i * nb_gen_ids_per_slot + j
                        gen_layer_id = self.get_gen_layer_id(base_id=layer_id, gen_id=gen_id_nb)
                        order_this_slot += [(gen_layer_id, k) for k in range(inter_core_tiling)]
                    elif j == 0:
                        # Non-generated nodes should only be scheduled once per intra-core slot (not `nb_gen_ids_per_slot` times)
                        nb_non_generated_nodes_per_slot = self._get_number_of_nodes_per_intra_core_slot(
                            layer_id, nb_intra_core_slots
                        )
                        sub_id_offset = i * (inter_core_tiling * nb_non_generated_nodes_per_slot)
                        order_this_slot += [
                            (layer_id, sub_id_offset + k)
                            for k in range(inter_core_tiling * nb_non_generated_nodes_per_slot)
                        ]

            order += order_this_slot

        return order

    def _get_number_of_nodes_per_intra_core_slot(self, layer_id, nb_intra_core_slots):
        return self.get_total_tiling_size(self.get_some_node_for_id(layer_id).intra_core_tiling) // nb_intra_core_slots

    def _get_and_assert_intra_core_tiling(self, nodes: list[ComputationNode]) -> int:
        """For each node, get the intra-core tiling. Make sure the tiling is the same for all nodes, and return the
        tiling factor"""
        all_intra_core_tiling_factors: list[int] = [self.get_total_tiling_size(n.intra_core_tiling) for n in nodes]
        min_tiling_factor = min(all_intra_core_tiling_factors, default=1)
        assert all(
            tiling_factor % min_tiling_factor == 0 for tiling_factor in all_intra_core_tiling_factors
        ), "Intra-core tiling factors are not multiples of minimum"
        return min_tiling_factor

    def _get_and_assert_nb_generated_nodes(self, generated_base_nodes: list[GeneratedComputationNode]) -> int:
        if not generated_base_nodes:
            return 0
        all_nb_generated_nodes = [self.get_nb_generated_nodes(node) for node in generated_base_nodes]
        nb_generated_nodes = all_nb_generated_nodes.pop()
        assert all(x == nb_generated_nodes for x in all_nb_generated_nodes), "Generated nodes have different nb"
        return nb_generated_nodes

    @staticmethod
    def get_total_tiling_size(tiling: TILING_T) -> int:
        assert not contains_wildcard(tiling)
        return prod(size for _, size in tiling)

    def filter_generated_nodes_from_stack(self, stack: tuple[int, ...]):
        """A stack contains all layer ids that are grouped together. It is possible that the stack contains many
        `GeneratedComputationNodes` who all have a different layer ID but the same `base_id` and the same performance"""

        nodes_in_stack = [next(n for n in self.workload.node_list if n.id == layer_id) for layer_id in stack]
        filtered_stack = tuple(
            n.id for n in nodes_in_stack if not isinstance(n, GeneratedComputationNode) or n.gen_id == 0
        )
        return filtered_stack

    def get_gen_layer_id(self, base_id: int, gen_id: int):
        """Given a base_id and gen_id, return the layer_id of the generated node"""
        return self.base_and_gen_to_layer_id[(base_id, gen_id)]

    def get_some_node_for_id(self, layer_id: int) -> ComputationNode:
        """Get any node (regardless of sub-id) with the given layer_id"""
        return next(n for n in self.workload.node_list if n.id == layer_id)

    def get_nb_generated_nodes(self, node: GeneratedComputationNode) -> int:
        """Get the number of generated nodes for a given base_id"""
        return max(gen_id for base_id, gen_id in self.base_and_gen_to_layer_id if base_id == node.base_id) + 1
