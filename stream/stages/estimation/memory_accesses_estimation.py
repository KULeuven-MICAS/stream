import logging
import os

from stream.cost_model.memory_accesses import CoreMemoryAccesses
from stream.cost_model.steady_state_scheduler import SteadyStateScheduler
from stream.hardware.architecture.accelerator import Accelerator
from stream.mapping.mapping import Mapping
from stream.stages.context import StageContext
from stream.stages.stage import Stage, StageCallable
from stream.workload.workload import Workload

logger = logging.getLogger(__name__)


class MemoryAccessesEstimationStage(Stage):
    """
    Stage that computes the number of memory accesses (reads/writes) for each core and tensor in the workload.
        This stage should be run after the ConstraintOptimizationAllocationStage, which determines the mapping of nodes to cores.
        The memory accesses are estimated based on the mapping and the workload, and stored in a CoreMemoryAccesses object.
        The results can be used for further analysis or visualization of memory access patterns.
        The estimated memory accesses are stored in the context for use by subsequent stages or for output.
        NOTE: For now, this stage does not currently distinguish between different memories inside the core.
    """

    REQUIRED_FIELDS = (
        "workload",
        "accelerator",
        "mapping",
        "scheduler",
        "output_path",
    )

    def __init__(
        self,
        list_of_callables: list[StageCallable],
        ctx: StageContext,
    ):
        """
        Initialize the stage by:
        - extracting all the unique nodes that will have to be evaluated
        - initializing the valid node-core allocations (which are used later by the InterCoreMappingStage)
        """
        super().__init__(list_of_callables, ctx)
        self.workload: Workload = self.ctx.get("workload")
        self.accelerator: Accelerator = self.ctx.get("accelerator")
        self.mapping: Mapping = self.ctx.get("mapping")
        self.scheduler: SteadyStateScheduler = self.ctx.get("scheduler")
        self.ssis = self.scheduler.ssis
        self.output_path = self.ctx.get("output_path")
        self.yaml_path: str = os.path.join(self.output_path, "memory_accesses.yaml")

        self.core_memory_accesses: CoreMemoryAccesses = CoreMemoryAccesses()

    def run(self):
        logger.info("Start MemoryAccessesEstimationStage.")
        self.calculate_memory_accesses()
        logger.info("Finished MemoryAccessesEstimationStage.")

        self.ctx.set(workload=self.workload, accelerator=self.accelerator, memory_accesses=self.core_memory_accesses)
        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], self.ctx)
        yield from sub_stage.run()

    def calculate_memory_accesses(self) -> None:
        # Go through all TransferNodes in the workload, and based on the producer-consumer relationships,
        # determine the number of reads/writes for each core and tensor based on the steady state iteration space.
        for tn in self.workload.get_transfer_nodes():
            ssis = self.ssis.get(tn, None)
            if ssis is None:
                raise ValueError(f"No steady state iteration space found for transfer node {tn}.")
            # Memory reads happen on producer cores
            pass
            # Memory writes happen on consumer cores

    def is_leaf(self):
        return True
