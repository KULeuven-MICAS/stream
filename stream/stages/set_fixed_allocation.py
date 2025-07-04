import logging
from typing import Any

from stream.hardware.architecture.accelerator import Accelerator
from stream.stages.stage import Stage, StageCallable
from stream.utils import (
    CostModelEvaluationLUT,
    get_inter_core_tiling_size,
)
from stream.workload.onnx_workload import ComputationNodeWorkload

logger = logging.getLogger(__name__)


class SetFixedAllocationStage(Stage):
    def __init__(
        self,
        list_of_callables: list[StageCallable],
        *,
        workload: ComputationNodeWorkload,
        accelerator: Accelerator,
        cost_lut: CostModelEvaluationLUT,
        **kwargs: Any,
    ):
        super().__init__(list_of_callables, **kwargs)
        self.accelerator = accelerator
        self.workload = workload
        self.cost_lut = cost_lut

    def run(self):
        logger.info("Start SetFixedAllocationStage.")
        # Set the performance of all nodes that have a fixed allocation
        self.set_fixed_allocation()
        logger.info("Finished SetFixedAllocationStage.")

        kwargs = self.kwargs.copy()
        kwargs["workload"] = self.workload
        kwargs["accelerator"] = self.accelerator
        kwargs["cost_lut"] = self.cost_lut
        sub_stage = self.list_of_callables[0](
            self.list_of_callables[1:],
            **kwargs,
        )
        yield from sub_stage.run()

    def set_fixed_allocation(self):
        for node in self.workload.node_list:
            inter_core_tiling_size = get_inter_core_tiling_size(node)
            if len(node.core_allocation) == inter_core_tiling_size:
                chosen_core_allocation = node.core_allocation[node.group]
                node.set_chosen_core_allocation(chosen_core_allocation)
                # Sanity check: cost_lut should contain an equal_core for all this chosen allocation
                core = self.accelerator.get_core(chosen_core_allocation)
                equal_node = self.cost_lut.get_equal_node(node)
                assert equal_node
                equal_core = self.cost_lut.get_equal_core(equal_node, core)
                assert equal_core
