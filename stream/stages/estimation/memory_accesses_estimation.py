import logging
from math import ceil, prod

from stream.cost_model.memory_accesses import CoreMemoryAccesses
from stream.hardware.architecture.accelerator import Accelerator
from stream.hardware.architecture.core import Core
from stream.hardware.architecture.memory import Memory
from stream.hardware.ports import READ, WRITE
from stream.mapping.mapping import Mapping
from stream.stages.context import StageContext
from stream.stages.stage import Stage, StageCallable
from stream.workload.demand import copy_sets, item_bits, moved_elements
from stream.workload.node import ComputationNode, InEdge, OutEdge, TransferNode, TransferType
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
        self.space = self.ctx.get("allocation").space

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
            if isinstance(src, InEdge) and (placement := self.mapping.placement_of(src)) and placement.on_chip:
                self._add_on_chip(tn, src, t_core, READ, nb_fires)
                continue
            if isinstance(src, InEdge):
                for memory, bits, fires in self._edge_bits(tn, src, t_core, READ, nb_fires):
                    accesses = ceil(bits / self.accelerator.offchip_rate(memory, READ)) * fires
                    self.core_memory_accesses.add_offchip_read(memory, t_core, accesses)
                continue
            cores = _flatten_cores(self.mapping.get(src).resource_allocation)
            computed = isinstance(src, ComputationNode)
            tiling = self.workload.get_unique_dims_inter_core_tiling(src, self.mapping) if computed else ()
            positions = prod(f for _, f in tiling) if computed else 0
            for i, core in enumerate(cores):
                taken = self.workload.get_tensor_of_transfer_from_single_core(
                    tensor, tn, self.mapping, i if i < positions else None
                )
                bandwidth = core.get_max_memory_bandwidth(type="read")
                self.core_memory_accesses.add_read(core, taken, ceil(taken.size_bits() / bandwidth) * nb_fires)

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
            if isinstance(dst, OutEdge) and (placement := self.mapping.placement_of(dst)) and placement.on_chip:
                self._add_on_chip(tn, dst, t_core, WRITE, nb_fires)
                continue
            if isinstance(dst, OutEdge):
                for memory, bits, fires in self._edge_bits(tn, dst, t_core, WRITE, nb_fires):
                    accesses = ceil(bits / self.accelerator.offchip_rate(memory, WRITE)) * fires
                    self.core_memory_accesses.add_offchip_write(memory, t_core, accesses)
                continue
            cores = _flatten_cores(self.mapping.get(dst).resource_allocation)
            for core in cores:
                # Calculate the number of accesses per fire based on tensor size and bw on the core
                bandwidth = core.get_max_memory_bandwidth(type="write")
                accesses_per_fire = ceil(t_core.size_bits() / bandwidth)
                total_accesses = accesses_per_fire * nb_fires
                self.core_memory_accesses.add_write(core, t_core, total_accesses)

    def _add_on_chip(
        self, tn: TransferNode, edge: InEdge | OutEdge, tensor: Tensor, direction: str, nb_fires: int
    ) -> None:
        """The reads (or writes) of ``tensor`` in each on-chip memory ``edge`` is placed in, its share
        (:meth:`_edge_bits`), charged to the memory cores viewing it, else to the cores computing from it."""
        add = self.core_memory_accesses.add_read if direction == READ else self.core_memory_accesses.add_write
        for memory, bits, fires in self._edge_bits(tn, edge, tensor, direction, nb_fires):
            for core in _memory_tile_cores(((memory,),)) or memory.cores:
                add(core, tensor, ceil(bits / core.get_max_memory_bandwidth(type=direction)) * fires)

    def _edge_bits(
        self, tn: TransferNode, edge: InEdge | OutEdge, tensor: Tensor, direction: str, nb_fires: int
    ) -> list[tuple[Memory, float, int]]:
        """Each memory ``tn`` reads (or writes) ``tensor`` of ``edge`` in, off chip or placed on chip, with the bits it
        moves per firing and the firings of the ``nb_fires`` that move them: a split placement's share of what the route
        delivers over the walk, else, per copy set the route delivers to, its share of their bits
        (:meth:`MulticastPathPlan.memory_share`) as often as the most often firing of them fires."""
        placement = self.mapping.placement_of(edge)
        on_chip = placement is not None and placement.on_chip
        route, leaving = self.mapping.get(tn).resource_allocation[0], direction == READ
        if (shares := self.workload.placement_shares(placement, tn.inputs[0], route.delivered)) is not None:
            return [(memory, tensor.size_bits() * share, nb_fires) for memory, share in shares.items()]
        side = dict.fromkeys(route.sources if leaving else route.targets)
        memories = [memory for memory in side if memory.offchip != on_chip]
        bitwidth = tensor.operand_type.bitwidth
        if not route.delivered_items:
            # Where its readers read all of it in place, what they read there.
            moved = item_bits(route.demand, bitwidth)
            return [
                (memory, moved * self.space.memory_share(tn, route, memory, leaving), nb_fires) for memory in memories
            ]
        found = []
        for copies in copy_sets(route.delivered_items):
            moved = moved_elements(item for item in route.delivered_items if item.copies == copies) * bitwidth
            fires = max(nb_fires * self._firing_share(tn, c) for c in copies)
            found += [
                (memory, moved * self.space.memory_share(tn, route, memory, leaving, copies), round(fires))
                for memory in memories
            ]
        return found

    def _firing_share(self, tn: TransferNode, copy: int) -> float:
        """The share of ``tn``'s firings at which its copy at index ``copy`` fires: the transfer's reuse over the
        copy's."""
        own = self.ssis.get(tn.outputs[copy])
        return self.ssis[tn].reuse_factor() / own.reuse_factor() if own is not None else 1.0

    def is_leaf(self):
        return True
