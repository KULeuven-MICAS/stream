import logging
from typing import Any, Generator

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
            node_hw_performances (CostModelEvaluationLUT): A LUT of CMEs for each unique node and their valid cores
            nb_ga_generations (int): The number of generations considered by the genetic algorithm
            nb_ga_individuals (int): The number of individuals in each genetic algorithm generation
        """
        super().__init__(list_of_callables, **kwargs)
        self.workload = workload
        self.accelerator = accelerator
        self.operands_to_prefetch = operands_to_prefetch
        self.scheduling_order = kwargs.get("scheduling_order", None)

        self.check_chosen_core_allocation()

    def run(self) -> Generator[tuple[StreamCostModelEvaluation, Any], None, None]:
        """! Run the StreamCostModelEvaluation."""
        logger.info("Start StreamCostModelEvaluationStage.")
        scme = StreamCostModelEvaluation(
            workload=self.workload,
            accelerator=self.accelerator,
            operands_to_prefetch=self.operands_to_prefetch,
            scheduling_order=self.scheduling_order,
        )
        scme.run()
        logger.info("Finished StreamCostModelEvaluationStage.")
        yield scme, None

    def is_leaf(self) -> bool:
        return True

    def check_chosen_core_allocation(self):
        """! Check that all nodes in the workload have a chosen_core_allocation."""
        for node in self.workload.nodes():
            if not isinstance(node.chosen_core_allocation, int):
                raise ValueError(f"{node} does not have a chosen_core_allocation.")
