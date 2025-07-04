import logging
from math import prod
from typing import Any, TypeAlias

from stream.hardware.architecture.accelerator import Accelerator
from stream.stages.stage import Stage, StageCallable
from stream.utils import return_tiling_type
from stream.workload.computation.computation_node import ComputationNode, GeneratedComputationNode
from stream.workload.mapping import TILING_T, TILING_WILDCARD_T
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
        yield from sub_stage.run()

    def get_scheduling_order_lbl(self):
        """Generate the scheduling order in case layers are executed sequentially"""
        return sorted(((n.id, n.sub_id) for n in self.workload.node_list), reverse=True)

    def get_scheduling_order_fused(self):
        """Generate the scheduling order for a tiled workload (inter- and intra-core tiled)."""
        # Assumes layer_stacks contain all id's! (responsibility of LayerStacksGenerationStage)
        assert self.layer_stacks

        order: SCHEDULE_ORDER_T = []

        for stack in self.layer_stacks:
            order += self.get_scheduling_order_for_stack_with_generated_nodes(stack)

        assert len(order) == self.workload.number_of_nodes()
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

        nb_intra_core_slots = self._get_and_assert_intra_core_tiling(non_generated_nodes)

        # Inter-core tiling logic. Same order as `filtered_stack`
        inter_core_tiling_factor_per_layer: list[int] = [
            self.get_total_tiling_size(n.inter_core_tiling)
            for n in some_node_per_layer  # type: ignore
        ]

        return self._generate_scheduling_order_for_stack_with_generated_nodes(
            filtered_stack=filtered_stack,
            inter_core_tiling_factors=inter_core_tiling_factor_per_layer,
            nb_intra_core_slots=nb_intra_core_slots,
            generated_base_ids=generated_base_ids,
        )

    def _generate_scheduling_order_for_stack_with_generated_nodes(
        self,
        filtered_stack: tuple[int, ...],
        inter_core_tiling_factors: list[int],
        nb_intra_core_slots: int,
        generated_base_ids: list[int],
        # nb_gen_ids_per_slot: int,
    ):
        """Generate the scheduling order for a stack of layers

        Args:
        filtered_stack: layer ids in stack. Only a single entry for each base_id
        inter_core_tiling_factors: inter-core tiling factor for each layer in stack (same order as `filtered_stack`)
        nb_intra_core_slots: Number of intra-core slots to schedule over. Same number of slots for all layers in stack
        generated_base_ids: Which ids in the filtered_stack are generated nodes
        """

        # This doesn't take gen ids into account yet: it's only the base ids
        order = self._generate_scheduling_order_for_stack(
            filtered_stack=filtered_stack,
            inter_core_tiling_factors=inter_core_tiling_factors,
            nb_intra_core_slots=nb_intra_core_slots,
        )

        # No generated nodes -> nothing to change
        if not generated_base_ids:
            return order

        gen_base_nodes: list[GeneratedComputationNode] = [
            self.get_some_node_for_id(base_id) for base_id in generated_base_ids
        ]  # type: ignore
        gen_dim_size: dict[int, int] = {
            base_node.id: self.get_nb_generated_nodes(base_node) for base_node in gen_base_nodes
        }
        gen_node_inter_core_tiling = {
            base_node.id: self.get_total_tiling_size(base_node.inter_core_tiling)
            for base_node in gen_base_nodes  # type: ignore
        }

        # For generated nodes: convert `(base_id, f(sub_id, gen_id))` to `(gen_layer_id, sub_id)`
        for i, (main_id, sub_id) in enumerate(order):
            if main_id in generated_base_ids:
                # NOTE this is counter intuitive. For gen nodes, we want the following scheduled loop order:
                # for intra-core tiling (excl gen dim)  | Different gen ids should be as close to each other as possible
                #     for gen_dim                       | to allow for fusion of generated nodes
                #         for inter-core tiling         |

                nb_gen_ids = gen_dim_size[main_id]  # size of middle loop
                nb_nodes_in_between_consecutive_gen_ids = gen_node_inter_core_tiling[main_id]  # size of the lowest loop
                nb_nodes_before_same_gen_id_is_seen_again = (  # size of the two innermost loops
                    nb_gen_ids * nb_nodes_in_between_consecutive_gen_ids
                )

                gen_id = sub_id % nb_nodes_before_same_gen_id_is_seen_again

                new_sub_id = sub_id // nb_gen_ids

                gen_layer_id = self.get_gen_layer_id(base_id=main_id, gen_id=gen_id)
                order[i] = (gen_layer_id, new_sub_id)

        return order

    def _generate_scheduling_order_for_stack(
        self,
        filtered_stack: tuple[int, ...],
        inter_core_tiling_factors: list[int],
        nb_intra_core_slots: int,
    ):
        """Generate the scheduling order without taking into account generated nodes. For the generated nodes, this
        method will generate entries in the form `(base_id, f(sub_id, gen_id))`. The base_id and sub_id must
        later be converted to the proper node_id using the `base_and_gen_to_layer_id` mapping."""

        order: SCHEDULE_ORDER_T = []

        for i in range(nb_intra_core_slots):
            order_this_slot: SCHEDULE_ORDER_T = []
            for layer_id, inter_core_tiling in zip(filtered_stack, inter_core_tiling_factors, strict=False):
                nb_intra_nodes_per_slot = self._get_total_nb_intra_core_nodes(layer_id) // nb_intra_core_slots
                assert nb_intra_nodes_per_slot > 0
                nb_sub_ids_per_slot = inter_core_tiling * nb_intra_nodes_per_slot
                order_this_slot += [(layer_id, i * nb_sub_ids_per_slot + j) for j in range(nb_sub_ids_per_slot)]

            order += order_this_slot

        return order

    def _get_total_nb_intra_core_nodes(self, layer_id: int):
        """For the given layer_id, get the total number of intra-core tiled sub-nodes that will be generated.
        NOTE For `GeneratedComputationNode`, this also includes the `gen_id` dimension for the same base_id. The
        `layer_id` must always be a base_id"""
        node = self.get_some_node_for_id(layer_id)
        if isinstance(node, GeneratedComputationNode):
            split_dim = node.gen_split_layer_dim
            intra_core_tiling = [(dim, size) for dim, size in node.intra_core_tiling if dim != split_dim]
            nb_nodes_per_gen_id = self.get_total_tiling_size(intra_core_tiling)
            nb_gen_ids = self.get_nb_generated_nodes(node)
            return nb_nodes_per_gen_id * nb_gen_ids
        return self.get_total_tiling_size(node.intra_core_tiling)

    def _get_and_assert_intra_core_tiling(self, nodes: list[ComputationNode]) -> int:
        """For each node, get the intra-core tiling. Make sure the tiling is the same for all nodes, and return the
        tiling factor"""
        all_intra_core_tiling_factors: list[int] = [self.get_total_tiling_size(n.intra_core_tiling) for n in nodes]
        min_tiling_factor = min(all_intra_core_tiling_factors, default=1)
        assert all(tiling_factor % min_tiling_factor == 0 for tiling_factor in all_intra_core_tiling_factors), (
            "Intra-core tiling factors are not multiples of minimum"
        )
        return min_tiling_factor

    def _get_nb_generated_layer_ids_per_intra_slot(self, generated_base_nodes: list[GeneratedComputationNode]) -> int:
        """Generated nodes are split by default over their `gen_split_layer_dim` and the number of splits is equal to
        the loop dim size of that layer dim. However, this is not necessarily equal to the intra-core tiling of other
        nodes.

        example: gen_split_layer_dim=L, layer has (L, 64) as loop dim, so there will be 64 such generated nodes.
        Other nodes have (L, 4) as intra-core tiling => 64/4=16 generated nodes with different layer id per intra-core
        slot. These nodes can be further split depending on their intra-core tiling (if other than L).

        NOTE we assume that the generated nodes will have the same intra-core tiling as the non-generated nodes!
        e.g. (L, 4) in this example
        """
        nb_layer_ids: int = self._get_and_assert_nb_generated_nodes(generated_base_nodes)
        some_node = generated_base_nodes[0]
        split_layer_dim = some_node.gen_split_layer_dim
        intra_core_tiling = some_node.intra_core_tiling

        if not all(n.gen_split_layer_dim == split_layer_dim for n in generated_base_nodes):
            raise NotImplementedError("Different split layer dims for generated nodes not supported")
        if not all(n.intra_core_tiling == intra_core_tiling for n in generated_base_nodes):
            raise NotImplementedError("Different intra-core tiling for generated nodes not supported")

        tiling_factor = next((size for dim, size in intra_core_tiling if dim == split_layer_dim), 1)

        if not nb_layer_ids % tiling_factor == 0:
            raise ValueError(
                f"Number of generated nodes {nb_layer_ids} in dimension {split_layer_dim} "
                f"is not a multiple of it's intra-core tiling ({split_layer_dim}, {tiling_factor})"
            )

        nb_layer_ids_per_intra_slot = nb_layer_ids // tiling_factor
        return nb_layer_ids_per_intra_slot

    def _get_and_assert_nb_generated_nodes(self, generated_base_nodes: list[GeneratedComputationNode]) -> int:
        if not generated_base_nodes:
            return 0
        all_nb_generated_nodes = [self.get_nb_generated_nodes(node) for node in generated_base_nodes]
        nb_generated_nodes = all_nb_generated_nodes[0]
        assert all(x == nb_generated_nodes for x in all_nb_generated_nodes), "Generated nodes have different nb"
        return nb_generated_nodes

    @staticmethod
    def get_total_tiling_size(tiling: TILING_T | TILING_WILDCARD_T) -> int:
        tiling_converted = return_tiling_type(tiling)
        return prod(size for _, size in tiling_converted)

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
        try:
            return self.base_and_gen_to_layer_id[(base_id, gen_id)]
        except KeyError as exc:
            raise KeyError(
                f"Generated node with {base_id=} and {gen_id=} not found in {self.base_and_gen_to_layer_id=}"
            ) from exc

    def get_some_node_for_id(self, layer_id: int) -> ComputationNode:
        """Get any node (regardless of sub-id) with the given layer_id"""
        return next(n for n in self.workload.node_list if n.id == layer_id)

    def get_nb_generated_nodes(self, node: GeneratedComputationNode) -> int:
        """Get the number of generated nodes for a given base_id"""
        return max(gen_id for base_id, gen_id in self.base_and_gen_to_layer_id if base_id == node.base_id) + 1
