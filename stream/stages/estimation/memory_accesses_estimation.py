import logging
import os
from math import ceil, prod

from stream.cost_model.memory_accesses import CoreMemoryAccesses
from stream.cost_model.steady_state_scheduler import SteadyStateScheduler
from stream.hardware.architecture.accelerator import Accelerator
from stream.hardware.architecture.core import Core
from stream.mapping.mapping import Mapping
from stream.stages.context import StageContext
from stream.stages.stage import Stage, StageCallable
from stream.workload.node import InEdge, OutEdge, TransferNode
from stream.workload.steady_state.iteration_space import SteadyStateIterationSpace
from stream.workload.workload import Workload

logger = logging.getLogger(__name__)


def _flatten_cores(allocation) -> list[Core]:
    """Flatten a node's resource/memory allocation to its cores (dropping non-core slot entries)."""
    out: list[Core] = []
    for item in allocation or ():
        candidates = item if isinstance(item, (list, tuple)) else (item,)
        out.extend(c for c in candidates if isinstance(c, Core))
    return out


def _memory_tile_cores(allocation) -> list[Core]:
    """The distinct memory-kind cores a transfer's memory allocation names."""
    tiles: dict[int, Core] = {}
    for core in _flatten_cores(allocation):
        if core.kind == "memory":
            tiles.setdefault(core.id, core)
    return list(tiles.values())


class MemoryAccessesEstimationStage(Stage):
    """
    Stage that computes the number of memory accesses (reads/writes) for each core and tensor in the workload.
        This stage should be run after the ConstraintOptimizationAllocationStage.
        The memory accesses are estimated based on the mapping and the workload.
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
        try:
            self.calculate_memory_accesses()
            accesses = self.core_memory_accesses
        except Exception as exc:  # noqa: BLE001 -- memory-access estimation is observability; never fail the solve
            logger.warning(f"MemoryAccessesEstimationStage: could not estimate memory accesses: {exc}")
            accesses = None  # None means "estimation failed", distinct from "nothing to count"
        logger.info("Finished MemoryAccessesEstimationStage.")

        self.ctx.set(workload=self.workload, accelerator=self.accelerator, memory_accesses=accesses)
        yield from (self.ctx,)

    def calculate_memory_accesses(self) -> None:
        # Go through all TransferNodes in the workload, and based on the producer-consumer relationships,
        # determine the number of reads/writes for each core and tensor based on the steady state iteration space.
        for tn in self.workload.get_transfer_nodes():
            ssis = self.ssis.get(tn, None)
            if ssis is None:
                raise ValueError(f"No steady state iteration space found for transfer node {tn}.")
            # Memory reads happen on producer cores
            self.get_source_accesses(tn, ssis)
            # Memory reads and writes happen on mem tile (if present)
            self.get_mem_core_accesses(tn, ssis)
            # Memory writes happen on consumer cores
            self.get_destination_accesses(tn, ssis)
        # print(self.core_memory_accesses)

    def get_source_accesses(self, tn: TransferNode, ssis: SteadyStateIterationSpace):
        nb_temporal_iterations = prod(ssis.get_applicable_temporal_sizes())
        reuse = ssis.reuse_factor()
        nb_fires = nb_temporal_iterations // reuse
        for tensor in tn.inputs:
            t_core = self.workload.get_tensor_of_transfer_from_single_core(tensor, tn, self.mapping)
            # Get the source core allocation
            src_idx = tn.inputs.index(tensor)
            src = list(self.workload.predecessors(tn))[src_idx]
            if isinstance(src, InEdge):
                cores = [self.accelerator.get_core(self.accelerator.offchip_core_id)]
            else:
                cores = _flatten_cores(self.mapping.get(src).resource_allocation)
            for core in cores:
                # Calculate the number of accesses per fire based on tensor size and bw on the core
                bandwidth = core.get_max_memory_bandwidth(type="read")
                accesses_per_fire = ceil(t_core.size_bits() / bandwidth)
                total_accesses = accesses_per_fire * nb_fires
                self.core_memory_accesses.add_read(core, t_core, total_accesses)

    def get_mem_core_accesses(self, tn: TransferNode, ssis: SteadyStateIterationSpace):
        # A transfer's tensor may reside on several memory tiles; account the accesses on each.
        mem_cores = _memory_tile_cores(self.mapping.get(tn).memory_allocation)
        if not mem_cores:
            return  # No mem tile involved in this transfer, skip
        nb_temporal_iterations = prod(ssis.get_applicable_temporal_sizes())
        reuse = ssis.reuse_factor()
        assert reuse != 0, "Memory core reuse factor cannot be zero if memory core is chosen."
        nb_fires = nb_temporal_iterations // reuse
        for mem_core in mem_cores:
            # For the mem tile we look at both inputs for writes and outputs for reads
            for tensor in tn.inputs:
                t_core = self.workload.get_tensor_of_transfer_from_single_core(tensor, tn, self.mapping)
                bandwidth = mem_core.get_max_memory_bandwidth(type="write")
                accesses_per_fire = ceil(t_core.size_bits() / bandwidth)
                self.core_memory_accesses.add_write(mem_core, t_core, accesses_per_fire * nb_fires)
            for tensor in tn.outputs:
                t_core = self.workload.get_tensor_of_transfer_to_single_core(tensor, tn, self.mapping)
                bandwidth = mem_core.get_max_memory_bandwidth(type="read")
                accesses_per_fire = ceil(t_core.size_bits() / bandwidth)
                self.core_memory_accesses.add_read(mem_core, t_core, accesses_per_fire * nb_fires)

    def get_destination_accesses(self, tn: TransferNode, ssis: SteadyStateIterationSpace):
        nb_temporal_iterations = prod(ssis.get_applicable_temporal_sizes())
        compute_reuse = ssis.reuse_factor()
        assert compute_reuse != 0, "Compute tile reuse factor cannot be zero."
        nb_fires = nb_temporal_iterations // compute_reuse
        for tensor in tn.outputs:
            t_core = self.workload.get_tensor_of_transfer_to_single_core(tensor, tn, self.mapping)
            # Get the destination core allocation
            dst_idx = tn.outputs.index(tensor)
            dst = list(self.workload.successors(tn))[dst_idx]
            if isinstance(dst, OutEdge):
                cores = [self.accelerator.get_core(self.accelerator.offchip_core_id)]
            else:
                cores = _flatten_cores(self.mapping.get(dst).resource_allocation)
            for core in cores:
                # Calculate the number of accesses per fire based on tensor size and bw on the core
                bandwidth = core.get_max_memory_bandwidth(type="write")
                accesses_per_fire = ceil(t_core.size_bits() / bandwidth)
                total_accesses = accesses_per_fire * nb_fires
                self.core_memory_accesses.add_write(core, t_core, total_accesses)

    def is_leaf(self):
        return True
