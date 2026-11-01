import logging
from math import ceil, prod

from stream.cost_model.memory_accesses import CoreMemoryAccesses
from stream.hardware.architecture.accelerator import Accelerator
from stream.hardware.architecture.core import Core
from stream.hardware.architecture.memory import Memory
from stream.hardware.ports import READ, WRITE
from stream.mapping.mapping import Mapping, offchip_memory
from stream.stages.context import StageContext
from stream.stages.stage import Stage, StageCallable
from stream.workload.node import InEdge, OutEdge, TransferNode, TransferType
from stream.workload.steady_state.iteration_space import SteadyStateIterationSpace
from stream.workload.tensor import Tensor
from stream.workload.workload import Workload

logger = logging.getLogger(__name__)


def _flatten_cores(allocation) -> list[Core]:
    """Flatten a node's resource/memory allocation to its cores (dropping non-core slot entries)."""
    out: list[Core] = []
    for item in allocation or ():
        candidates = item if isinstance(item, (list, tuple)) else (item,)
        out.extend(c for c in candidates if isinstance(c, Core))
    return out


def _memory_tile_cores(allocation: tuple[tuple[Memory, ...], ...]) -> list[Core]:
    """The memory-kind cores viewing the staging memories of a transfer's memory allocation, once each. These are
    the cores its accesses are charged to, even when compute cores also view that memory."""
    memories = dict.fromkeys(memory for placement in allocation for memory in placement if memory.staging)
    return list(dict.fromkeys(core for memory in memories for core in memory.cores if core.kind == "memory"))


class MemoryAccessesEstimationStage(Stage):
    """
    Stage that computes the number of memory accesses (reads/writes) for each core and tensor in the workload.
        This stage should be run after the AllocationStage.
        The memory accesses are estimated based on the mapping and the workload.
        The results can be used for further analysis or visualization of memory access patterns.
        The estimated memory accesses are stored in the context for use by subsequent stages or for output.
        NOTE: For now, this stage does not currently distinguish between different memories inside the core.
    """

    reads = ("workload", "accelerator", "mapping", "allocation")
    writes = ("memory_accesses",)

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
        self.ssis = self.ctx.get("allocation").ssis

        self.core_memory_accesses: CoreMemoryAccesses = CoreMemoryAccesses(self.accelerator)

    def run(self):
        logger.info("Start MemoryAccessesEstimationStage.")
        try:
            self.calculate_memory_accesses()
            accesses = self.core_memory_accesses
        except Exception as exc:  # noqa: BLE001 -- memory-access estimation is observability; never fail the solve
            logger.warning(f"MemoryAccessesEstimationStage: could not estimate memory accesses: {exc}")
            accesses = None  # None means "estimation failed", distinct from "nothing to count"
        logger.info("Finished MemoryAccessesEstimationStage.")

        self.ctx.set(memory_accesses=accesses)
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
                for memory, bits in self._offchip_bits(tn, src, t_core, READ):
                    accesses = ceil(bits / self.accelerator.offchip_rate(memory, READ)) * nb_fires
                    self.core_memory_accesses.add_offchip_read(memory, t_core, accesses)
                continue
            cores = _flatten_cores(self.mapping.get(src).resource_allocation)
            for core in cores:
                # Calculate the number of accesses per fire based on tensor size and bw on the core
                bandwidth = core.get_max_memory_bandwidth(type="read")
                accesses_per_fire = ceil(t_core.size_bits() / bandwidth)
                total_accesses = accesses_per_fire * nb_fires
                self.core_memory_accesses.add_read(core, t_core, total_accesses)

    def get_mem_core_accesses(self, tn: TransferNode, ssis: SteadyStateIterationSpace):
        """The writes into and reads out of each memory tile a transfer stages its tensor in. Only transfers into
        memory stage; a transfer into the memory its reader computes from is counted as the reader's write."""
        if tn.transfer_type not in (TransferType.COMPUTE_TO_MEM, TransferType.MEM_TO_MEM):
            return
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
                t_core = self.workload.get_tensor_of_transfer_to_single_core(
                    tensor, tn, self.mapping, ssis=self.ssis.get(tensor)
                )
                bandwidth = mem_core.get_max_memory_bandwidth(type="read")
                accesses_per_fire = ceil(t_core.size_bits() / bandwidth)
                self.core_memory_accesses.add_read(mem_core, t_core, accesses_per_fire * nb_fires)

    def get_destination_accesses(self, tn: TransferNode, ssis: SteadyStateIterationSpace):
        nb_temporal_iterations = prod(ssis.get_applicable_temporal_sizes())
        compute_reuse = ssis.reuse_factor()
        assert compute_reuse != 0, "Compute tile reuse factor cannot be zero."
        nb_fires = nb_temporal_iterations // compute_reuse
        for tensor in tn.outputs:
            t_core = self.workload.get_tensor_of_transfer_to_single_core(
                tensor, tn, self.mapping, ssis=self.ssis.get(tensor)
            )
            # Get the destination core allocation
            dst_idx = tn.outputs.index(tensor)
            dst = list(self.workload.successors(tn))[dst_idx]
            if isinstance(dst, OutEdge):
                for memory, bits in self._offchip_bits(tn, dst, t_core, WRITE):
                    accesses = ceil(bits / self.accelerator.offchip_rate(memory, WRITE)) * nb_fires
                    self.core_memory_accesses.add_offchip_write(memory, t_core, accesses)
                continue
            cores = _flatten_cores(self.mapping.get(dst).resource_allocation)
            for core in cores:
                # Calculate the number of accesses per fire based on tensor size and bw on the core
                bandwidth = core.get_max_memory_bandwidth(type="write")
                accesses_per_fire = ceil(t_core.size_bits() / bandwidth)
                total_accesses = accesses_per_fire * nb_fires
                self.core_memory_accesses.add_write(core, t_core, total_accesses)

    def _offchip_bits(
        self, tn: TransferNode, edge: InEdge | OutEdge, tensor: Tensor, direction: str
    ) -> list[tuple[Memory, float]]:
        """Each off-chip memory ``tn`` reads (or writes) ``tensor`` of ``edge`` in, with the bits of it that memory
        moves per firing: all of them with one off-chip memory, else, for a split placement, its share of what the
        transfer's route delivers over the iterations walking the split axis, or its share of the route's data."""
        if len(self.accelerator.offchip_memories) <= 1:
            return [(offchip_memory(self.accelerator), tensor.size_bits())]
        route, leaving = self.mapping.get(tn).resource_allocation[0], direction == READ
        if (shares := self.workload.placement_shares(edge, tn.inputs[0], self.mapping, route.delivered)) is not None:
            return [(memory, tensor.size_bits() * share) for memory, share in shares.items()]
        side = route.sources if leaving else route.targets
        return [
            (memory, tensor.size_bits() * route.memory_share(memory, leaving))
            for memory in dict.fromkeys(memory for memory in side if memory.offchip)
        ]

    def is_leaf(self):
        return True
