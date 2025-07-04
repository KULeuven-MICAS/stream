import logging
from collections.abc import Generator
from typing import Any

from zigzag.datatypes import LayerOperand

from stream.cost_model.cost_model import StreamCostModelEvaluation
from stream.hardware.architecture.accelerator import Accelerator
from stream.stages.stage import Stage, StageCallable
from stream.workload.onnx_workload import ComputationNodeWorkload

logger = logging.getLogger(__name__)


class StreamCostModelEvaluationStage(Stage):
    """
    Class that runs a StreamCostModelEvaluation.
    """

    def __init__(
        self,
        list_of_callables: list[StageCallable],
        *,
        workload: ComputationNodeWorkload,
        accelerator: Accelerator,
        operands_to_prefetch: list[LayerOperand],
        **kwargs: Any,
    ):
        """Initialize the InterCoreMappingStage.

        Args:
            list_of_callables (list): List of the substages to be called. This should be empty as this is a leaf stage.
            workload (DiGraph): The NetworkX DiGraph representing the workload to be scheduled
            accelerator (Accelerator): The hardware accelerator onto which we schedule the workload
            operands_to_prefetch (list[LayerOperand]): A list of LayerOperands that whose tensors should be prefetched
        """
        super().__init__(list_of_callables, **kwargs)
        self.workload = workload
        self.accelerator = accelerator
        self.operands_to_prefetch = operands_to_prefetch
        self.scheduling_order = kwargs["scheduling_order"]

    def run(self) -> Generator[tuple[StreamCostModelEvaluation, Any], None, None]:
        """! Run the StreamCostModelEvaluation."""
        logger.info("Start StreamCostModelEvaluationStage.")
        scme = StreamCostModelEvaluation(
            workload=self.workload,
            accelerator=self.accelerator,
            operands_to_prefetch=self.operands_to_prefetch,
            scheduling_order=self.scheduling_order,
        )
        scme.evaluate()
        logger.info("Finished StreamCostModelEvaluationStage.")
        yield scme, None

    def is_leaf(self) -> bool:
        return True

    @staticmethod
    def check_and_fix_chosen_core_allocation(workload: ComputationNodeWorkload):
        """! Check that all nodes in the workload have a chosen_core_allocation."""
        for node in workload.node_list:
            if node.chosen_core_allocation is None:
                node.chosen_core_allocation = node.possible_core_allocation[0]
                logger.warning(
                    f"{node} does not have a chosen_core_allocation. Setting to {node.chosen_core_allocation} out of "
                    f"possible allocations {node.possible_core_allocation}."
                )
