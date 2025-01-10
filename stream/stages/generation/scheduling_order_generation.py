import logging
from math import prod
from typing import Any, TypeAlias

from stream.hardware.architecture.accelerator import Accelerator
from stream.stages.estimation.stream_cost_model_evaluation import StreamCostModelEvaluationStage
from stream.stages.stage import Stage, StageCallable
from stream.utils import contains_wildcard
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

    def run(self):
        if self.layer_stacks:
            self.scheduling_order = self.get_scheduling_order_fused()
        else:
            self.scheduling_order = self.get_scheduling_order_lbl()

        try:
            StreamCostModelEvaluationStage.check_chosen_core_allocation(self.workload)
        except ValueError:
            # Nodes don't have core allocation yet -> Set based on inter-core tiling
            self.set_core_allocation()

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
            order += self.get_scheduling_order_for_stack(stack)

        assert len(order) == self.workload.number_of_nodes()
        return order

    def get_scheduling_order_for_stack(self, stack: tuple[int, ...]):
        """
        Example: stack=[0,1] and inter-core tiling is 3 and 4 respectively:
        -> [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2), (1,3), (0,4), (0,5), ...
            <-------------intra-core slot 0--------------->  <---- intra-core slot 1 ---->
            <--all parallel--->
        """
        order: SCHEDULE_ORDER_T = []

        if not stack:
            return order

        some_node_per_layer = [next(n for n in self.workload.node_list if n.id == layer_id) for layer_id in stack]

        # For each layer in stack, get the number of cores and number of intra-core slots over which it is split
        inter_core_tiling_factor_per_layer: list[int] = [
            self.get_total_tiling_size(n.inter_core_tiling) for n in some_node_per_layer
        ]
        intra_core_tiling_factor_per_layer: list[int] = [
            self.get_total_tiling_size(n.intra_core_tiling) for n in some_node_per_layer
        ]

        nb_intra_core_slots = intra_core_tiling_factor_per_layer.pop()
        assert all(
            x == nb_intra_core_slots for x in intra_core_tiling_factor_per_layer
        ), "Layers in stack have different intra-core tiling"

        for i in range(nb_intra_core_slots):
            for layer_id, inter_core_tiling in zip(stack, inter_core_tiling_factor_per_layer):
                order += [(layer_id, i * inter_core_tiling + j) for j in range(inter_core_tiling)]

        return order

    @staticmethod
    def get_total_tiling_size(tiling: TILING_T) -> int:
        assert not contains_wildcard(tiling)
        return prod(size for _, size in tiling)

    def set_core_allocation(self):
        """For all nodes of the (tiled) workload, set the chosen core allocation based on the sub_id and number of
        inter-core splits for this node.
        # TODO this is only necessary if CO is not being used. Move to something like `COSkipStage`
        """
        for node in self.workload.node_list:
            inter_core_tiling_factor = self.get_total_tiling_size(node.inter_core_tiling)
            core_id = node.sub_id % inter_core_tiling_factor
            node.set_chosen_core_allocation(core_id)
            node.core_allocation_is_fixed = True
