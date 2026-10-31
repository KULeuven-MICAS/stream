"""Lowering of a fused group to its steady state: the transfers made explicit, each with the placements and
routes it may take, the iteration spaces, and the timeslots. Pure: nothing it is given is changed."""

from collections.abc import Sequence
from dataclasses import replace
from itertools import combinations, count
from math import ceil, prod
from typing import cast

import numpy as np
from xdsl.ir.affine import AffineMap

from stream.allocation.problem import AllocationProblem
from stream.cost_model.communication_manager import MulticastPathPlan
from stream.cost_model.core_cost_lut import CoreCostLUT
from stream.datatypes import InterCoreTiling, LayerDim
from stream.hardware.architecture.accelerator import Accelerator
from stream.hardware.architecture.core import Core
from stream.hardware.architecture.memory import Memory
from stream.hardware.ports import OUTPUT
from stream.mapping.mapping import Mapping, offchip_memory, placements_of, role_of, the_one
from stream.opt.allocation.constraint_optimization.hardware import build_hardware_facts
from stream.profiling import span
from stream.workload.iterator_type import is_state_operand, streamed_operands
from stream.workload.node import (
    ComputationNode,
    HasInputs,
    HasIterationSpace,
    HasOutputs,
    InEdge,
    Node,
    OutEdge,
    Tensor,
    TransferNode,
    TransferType,
)
from stream.workload.steady_state.iteration_space import IterationVariableType, SteadyStateIterationSpace
from stream.workload.utils import (
    affine_coefficients,
    generate_steady_state_iteration_spaces,
    generate_tensor_ssis,
    get_compute_predecessors_successors,
    get_equivalent_dimension,
    get_node_with_largest_resource_allocation,
)
from stream.workload.workload import Workload, position_of


def reader_positions(
    tensor: Tensor, readers: Sequence[tuple[HasInputs, Sequence[Core | None], Sequence[Memory]]]
) -> tuple[Memory, ...]:
    """The memory a copy of ``tensor`` lands in at each position, one per reading core. Each reader gives its cores
    and the memory each core reads the copy in. Positions keep the readers' core order, since that order pairs a
    spatial index with a core. A core must read the copy in the same memory for every reader. A reader without a core
    (an output edge) gets a position of its own."""
    found: dict[object, tuple[HasInputs, Memory]] = {}
    for reader, cores, memories in readers:
        for core, memory in zip(cores, memories, strict=True):
            key = core if core is not None else object()  # unique key: shared with no other position
            first, held = found.setdefault(key, (reader, memory))
            if held is not memory:
                raise ValueError(
                    f"core {key.id} reads {tensor.name} in {memory.name} for {reader.name} but in {held.name} for "
                    f"{first.name}: a copy lands in one memory per core"
                )
    return tuple(memory for _, memory in found.values())


def largest_divisor_leq(n: int, cap: int) -> int:
    """Largest divisor of ``n`` at most ``cap`` (never below 1)."""
    cap = max(1, min(cap, n))
    for candidate in range(cap, 0, -1):
        if n % candidate == 0:
            return candidate
    return 1


def lower_steady_state(  # noqa: PLR0913
    workload: Workload,
    accelerator: Accelerator,
    mapping: Mapping,
    fusion_splits: dict[LayerDim, int],
    cost_lut: CoreCostLUT,
    nb_cols_to_use: int,
    route_alternatives: int,
) -> AllocationProblem:
    """The steady-state problem of ``workload`` (one fused group) mapped by ``mapping`` on ``accelerator``, each
    placement of a transfer offered up to ``route_alternatives`` routes."""
    lowering = _Lowering(workload, accelerator, mapping, nb_cols_to_use, route_alternatives)
    return lowering.lower(fusion_splits, cost_lut)


class _Lowering:
    """The state the lowering threads through its steps: a mapping it rewrites as the transfer graph grows."""

    def __init__(  # noqa: PLR0913
        self,
        workload: Workload,
        accelerator: Accelerator,
        mapping: Mapping,
        nb_cols_to_use: int,
        route_alternatives: int,
    ):
        self.workload = workload
        self.accelerator = accelerator
        self.mapping = mapping.copy()
        self.nb_cols_to_use = nb_cols_to_use
        self.route_alternatives = route_alternatives
        self.hardware = build_hardware_facts(accelerator, nb_cols_to_use)

    def lower(self, fusion_splits: dict[LayerDim, int], cost_lut: CoreCostLUT) -> AllocationProblem:
        with span("transfer_graph"):
            self.ssw = self.build_transfer_graph()
            self.fusion_splits = self.update_fusion_splits(fusion_splits)
            self.mapping = self.update_mapping()
            cost_lut = cost_lut.with_nodes(self.ssw.get_computation_nodes())
        with span("iteration_spaces"):
            self.ssis = self.generate_ssis()
            self.iterations = self.calculate_iterations()
        with span("timeslots"):
            timeslots = self.ssw.get_timeslots(self.mapping)
        return AllocationProblem(
            source_workload=self.workload,
            workload=self.ssw,
            mapping=self.mapping,
            fusion_splits=self.fusion_splits,
            cost_lut=cost_lut,
            ssis=self.ssis,
            iterations=self.iterations,
            timeslots=timeslots,
            accelerator=self.accelerator,
            hardware=self.hardware,
        )

    def build_transfer_graph(self) -> Workload:
        """Insert the transfers of every tensor but a kernel's carried state, which stays resident on its node's
        cores, so neither it nor an input edge that only delivers it gets a transfer."""
        state = {
            t
            for n in self.workload.nodes
            if isinstance(n, HasIterationSpace)
            for t in n.inputs
            if is_state_operand(n, t)
        }
        new_nodes: dict[str, Node] = {
            node.name: node
            for node in self.workload.nodes
            if not (isinstance(node, InEdge) and all(t in state for t in node.outputs))
        }
        for tensor in self.workload.tensors:
            if tensor in state:
                continue
            srcs = [n for n in self.workload.nodes if isinstance(n, HasOutputs) and tensor in n.outputs]
            assert len(srcs) == 1, f"Expected exactly one source for tensor {tensor}, found {len(srcs)}"
            src = new_nodes[srcs[0].name]
            readers = [n for n in self.workload.nodes if isinstance(n, HasInputs) and tensor in n.inputs]
            dsts = [new_nodes[n.name] for n in readers]
            self.add_transfer_nodes(tensor, src, dsts, new_nodes)
        new_workload = Workload(new_nodes.values())
        return new_workload

    def add_transfer_nodes(self, tensor: Tensor, src: HasOutputs, dsts: list[HasInputs], new_nodes: dict[str, Node]):
        """
        Move ``tensor`` from its source to its readers in one transfer, either directly or staged on a memory tile.

        Staging splits the move in two -- source to the on-chip buffer, buffer to every destination --
        so the tile can hold the tensor across reads and re-lay it out on the way through. See
        :meth:`_stages_on_mem_tile`. A tensor the group computes and a later fused group reads also moves to the
        group's output edge, in a transfer of its own that keeps its name, by which that group reads it; its source
        then writes a renamed copy.
        """
        names = (self._fresh(f"{tensor.name}_{i}") for i in count(1))
        exports = [] if isinstance(src, InEdge) else [dst for dst in dsts if isinstance(dst, OutEdge)]
        assert len(exports) <= 1, f"{tensor.name} leaves its group through more than one output edge"
        readers = [dst for dst in dsts if dst not in exports]
        moves = [(group, self._stages_on_mem_tile(tensor, src, group)) for group in (readers, exports) if group]
        source = tensor
        if exports:
            source = replace(tensor, name=next(names))
            src = self.update_source_tensor(tensor, src, new_nodes, source)
        for group, staged in moves:
            copy = source
            # Named after the tensor it moves, apart from the export's transfer, which reads the same renamed copy.
            name = f"Transfer({tensor.name})" if exports and group is readers else None
            if staged:
                hop_type = self.determine_transfer_type(src, group, tensor, dst_type="memory")
                hop, (copy,) = self.generate_transfer_node([src], copy, hop_type, next(names), name)
                new_nodes[hop.name] = hop
                name = None
            transfer_type = self.determine_transfer_type(src, group, tensor, src_type="memory" if staged else None)
            out_name = tensor.name if group is exports else next(names)
            placements = [self._placement(dst, tensor) for dst in group]
            transfer_node, updated_tensors = self.generate_transfer_node(
                group, copy, transfer_type, out_name, name, placements
            )
            new_nodes[transfer_node.name] = transfer_node
            for dst, updated_tensor in zip(group, updated_tensors, strict=True):
                self.update_destination_node_inputs(tensor, src, new_nodes, dst, updated_tensor)

    def _placement(self, dst: HasInputs, tensor: Tensor) -> object:
        """What and where ``dst`` reads of ``tensor``: its cores, the tensor's index in the workload's unique dims, and
        per core the tile it takes of each dim indexing the tensor. Destinations that read alike share one copy."""
        if not isinstance(dst, ComputationNode):
            return dst.name
        node = self.workload.get_node_by_name(dst.name)
        unique_dims, dim_values = self.workload.unique_dimensions()
        access = self.workload.global_mapping(node, node.get_mapping(tensor))
        index = tuple(expr.replace_dims_and_symbols(dim_values, ()) for expr in access.results)
        indexed = {
            z
            for expr in index
            for z, c in zip(unique_dims, affine_coefficients(expr, len(unique_dims))[1], strict=True)
            if c
        }
        dims, node_mapping = self.workload.get_dims(node), self.mapping.get(dst)
        tiling = [
            (self.workload.leading_dim(dims[dim.position])[0], factor)
            for split in node_mapping.inter_core_tiling[:1]
            for dim, factor in split
        ]
        tiles = tuple(  # per core, in the order codegen unrolls them: the last split dim fastest
            tuple((z, i, f) for (z, f), i in zip(tiling, position, strict=True) if z in indexed)
            for position in np.ndindex(*(f for _, f in tiling))
        )
        return node_mapping.resource_allocation, tuple(map(str, index)), tiles

    def _stages_on_mem_tile(self, tensor: Tensor, src: HasOutputs, dsts: list[HasInputs]) -> bool:
        """Whether a transfer is staged in a memory tile instead of landing straight on the cores.

        Two things ask for a staging buffer. An offchip input, or an output the next fused group reads, is held in
        the tile, which keeps the fan-out off the shim's two DMA channels: one channel feeds the tile, which then
        serves every core from its own. And a producer and consumer that disagree on layout need the
        tensor re-laid out between them, which is the tile's other job.
        """
        if not self._mem_tiles():
            return False
        if isinstance(src, InEdge) or any(isinstance(dst, OutEdge) for dst in dsts):
            return True
        produced = self._declared_layout(src, tensor)
        if produced is None:
            return False
        consumed = (self._declared_layout(dst, tensor) for dst in dsts)
        return any(layout is not None and layout != produced for layout in consumed)

    def _declared_layout(self, node: Node, tensor: Tensor):
        """The layout ``node``'s kernel declares for ``tensor``, None when it declares none.

        None is "not known", never "differs": a node without a kernel, a kernel that declares
        no layouts, and an operand past the end all leave the transfer unstaged.
        """
        kernel = self.mapping.get(node).kernel
        layouts = kernel.operand_layouts() if kernel else ()
        operands = tuple(streamed_operands(node))
        if tensor not in operands:
            return None
        index = operands.index(tensor)
        return layouts[index] if index < len(layouts) else None

    def update_destination_node_inputs(self, tensor, src, new_nodes, dst, updated_tensor):
        # Find corresponding node in new_nodes as it might have already been updated
        dst_new = new_nodes[dst.name]
        # Update the dst input to the second transfer node
        assert len(src.outputs) == 1, "Src must have exactly one output tensor for index below."
        input_idx = dst_new.inputs.index(tensor)
        new_inputs = dst_new.inputs[:input_idx] + (updated_tensor,) + dst_new.inputs[input_idx + 1 :]
        if isinstance(dst_new, ComputationNode):
            new_dst = replace(dst_new, inputs=new_inputs)
        elif isinstance(dst_new, OutEdge):
            new_dst = OutEdge(
                name=dst_new.name,
                inputs=new_inputs,
            )
        else:
            raise ValueError(f"Unexpected dst node type: {type(dst_new)}")
        new_nodes[dst_new.name] = new_dst
        if isinstance(new_dst, OutEdge):
            return  # an output edge has no mapping entry
        # Update the mapping entry for this new_dst node to be the same as the original dst node
        self.mapping.set(new_dst, self.mapping.get(dst))
        # Remove the original dst node from the mapping as it has been updated with new inputs
        self.mapping.remove(dst)

    def update_source_tensor(self, tensor, src, new_nodes, new_tensor) -> ComputationNode:
        output_idx = src.outputs.index(tensor)
        new_outputs = src.outputs[:output_idx] + (new_tensor,) + src.outputs[output_idx + 1 :]
        new_src = replace(src, outputs=new_outputs)
        new_nodes[new_src.name] = new_src
        # Update the mapping entry for this new_src node to be the same as the original src node
        self.mapping.set(new_src, self.mapping.get(src))
        # Remove the original src node from the mapping as it has been updated with new outputs
        self.mapping.remove(src)
        return new_src

    def update_fusion_splits(self, fusion_splits: dict[LayerDim, int]) -> dict[LayerDim, int]:
        # Update the fusion_splits based on the new workload with transfer nodes
        updated_fusion_splits = {}
        for dim, size in fusion_splits.items():
            new_dim = get_equivalent_dimension(self.workload, self.ssw, dim)
            updated_fusion_splits[new_dim] = size
        return updated_fusion_splits

    def update_mapping(self):
        # Update inter_core_tiling of computation node to unique dimensions
        for node in self.ssw.get_computation_nodes():
            unique_dims_tiling = (self.ssw.get_unique_dims_inter_core_tiling(node, self.mapping),)
            self.mapping.update_inter_core_tiling(node, unique_dims_tiling)
        # Add transfer node mappings
        for node in self.ssw.get_transfer_nodes():
            assert len(node.inputs) == 1, "Transfer node must have exactly one input tensor."
            src = cast(HasOutputs, list(self.ssw.predecessors(node))[0])
            dsts = tuple(cast(HasInputs, n) for n in self.ssw.successors(node))
            self.update_mapping_for_transfer(node, src, dsts)
        return self.mapping.with_updated_workload(self.ssw, self.workload)  # updates FusedGroups

    def _fresh(self, name: str) -> str:
        """``name``, suffixed until no tensor of the workload has it, so a copy cannot be taken for another tensor."""
        taken = {t.name for t in self.workload.tensors}
        while name in taken:
            name += "_"
        return name

    def generate_transfer_node(
        self,
        dsts: list[HasInputs],
        tensor: Tensor,
        transfer_type: TransferType,
        out_name: str = "",
        name: str | None = None,
        placements: list[object] | None = None,
    ) -> tuple[TransferNode, list[Tensor]]:
        """The transfer of ``tensor`` to ``dsts``, and the copy each reads: one per distinct placement."""
        copies, transfer_outputs = self.generate_transfer_output_tensors(tensor, placements or dsts, out_name)
        operand_mapping = tuple(AffineMap.identity(len(tensor.shape)) for _ in range(1 + len(copies)))
        transfer_node = TransferNode(
            name=name or f"Transfer({tensor.name})",
            inputs=(tensor,),
            outputs=tuple(copies),
            transfer_type=transfer_type,
            operand_mapping=operand_mapping,
        )
        return transfer_node, transfer_outputs

    def generate_transfer_output_tensors(
        self, tensor: Tensor, placements: list, out_name: str = ""
    ) -> tuple[list[Tensor], list[Tensor]]:
        """The copies of ``tensor``, one per distinct placement, and the copy of each placement."""
        distinct = list(dict.fromkeys(placements))
        copies = [
            Tensor(
                name=f"{out_name}.{i}" if len(distinct) > 1 else out_name,
                operand_type=tensor.operand_type,
                shape=tensor.shape,
                subview=tensor.subview,
            )
            for i in range(len(distinct))
        ]
        return copies, [copies[distinct.index(placement)] for placement in placements]

    def generate_ssis(self) -> dict[HasIterationSpace | Tensor, SteadyStateIterationSpace]:
        ssis = generate_steady_state_iteration_spaces(
            self.ssw,
            self.mapping,
            self.fusion_splits,
        )
        ssis = self.update_tensor_ssis(self.ssw, ssis)
        return ssis

    def update_tensor_ssis(
        self, workload: Workload, ssis: dict[HasIterationSpace | Tensor, SteadyStateIterationSpace]
    ) -> dict[HasIterationSpace | Tensor, SteadyStateIterationSpace]:
        # Generate the tensor SSIS of InEdge(s) output
        for in_edge in workload.get_in_edges():
            for tensor in in_edge.outputs:
                assert tensor not in ssis, (
                    f"Tensor {tensor.name} already has an SSIS, cannot assign the same tensor multiple SSIS."
                )
                succ = next(workload.successors(in_edge))
                tensor_ssis = generate_tensor_ssis(workload, tensor, succ, ssis)
                ssis[tensor] = tensor_ssis
        # Generate the new tensor SSIS of node outputs, and of the state a node keeps: the
        # state's iteration space is the node's own, since it is resident there.
        for node in workload.get_iteration_space_nodes():
            carried = [t for t in node.inputs if is_state_operand(node, t)]
            for tensor in (*node.outputs, *carried):
                assert tensor not in ssis, (
                    f"Tensor {tensor.name} already has an SSIS, cannot assign the same tensor multiple SSIS."
                )
                tensor_ssis = generate_tensor_ssis(workload, tensor, node, ssis)
                if isinstance(node, TransferNode):
                    self.add_halos(tensor, node, tensor_ssis)
                ssis[tensor] = tensor_ssis
        return ssis

    def add_halos(self, tensor: Tensor, transfer: TransferNode, ssis: SteadyStateIterationSpace) -> None:
        """Give each loop that slides the window of a transfer's copy the halo its consecutive tiles share."""
        sliding = (IterationVariableType.TEMPORAL, IterationVariableType.SPATIAL)
        loops = [v for v in ssis if v.relevant and v.type in sliding]
        windows = self.ssw.get_windows(tensor, transfer, self.mapping, [v.dimension for v in loops])
        for loop in loops:
            loop.halo = windows[loop.dimension][1] if loop.dimension in windows else 0

    def update_mapping_for_transfer(self, node: TransferNode, src: HasOutputs, dsts: tuple[HasInputs, ...]) -> None:
        possible_dst_allocs = self.determine_possible_memory_allocations(node, src, dsts)
        possible_inter_core_tiling = self.determine_possible_inter_core_tiling(node, possible_dst_allocs, dsts)
        self.mapping.set_for_node(
            node,
            resource_allocation=(),
            inter_core_tiling=possible_inter_core_tiling,
            memory_allocation=possible_dst_allocs,
        )
        possible_allocations = self.determine_possible_transfer_plans(
            node, src, possible_dst_allocs, possible_inter_core_tiling
        )
        self.mapping.set_for_node(
            node,
            resource_allocation=possible_allocations,
            inter_core_tiling=possible_inter_core_tiling,
            memory_allocation=possible_dst_allocs,
        )

    def determine_possible_memory_allocations(
        self, node: TransferNode, src: HasOutputs, dsts: tuple[HasInputs, ...]
    ) -> tuple[tuple[Memory, ...], ...]:
        """The candidate memories the transfer's copies land in, one per target position. An input transfer's
        MEM_TO_MEM copy and an output transfer's COMPUTE_TO_MEM copy land in memory tiles. Any other copy lands where
        its readers read it."""
        if node.transfer_type in (TransferType.MEM_TO_MEM,) and isinstance(src, InEdge):
            # Find the dst with max number of compute allocations to determine possible memory cores
            compute_dsts = get_compute_predecessors_successors(
                tr=node, workload=self.ssw
            )  # won't have any compute preds
            dst = get_node_with_largest_resource_allocation(compute_dsts, self.mapping)
            return self._mem_tile_placements(dst, node)
        if node.transfer_type in (TransferType.MEM_TO_MEM,) and any(isinstance(dst, OutEdge) for dst in dsts):
            assert len(dsts) == 1, "Currently only support single destination for constant output transfer."
            return (self._target_placement(dsts[0], node),)
        if node.transfer_type in (TransferType.COMPUTE_TO_MEM,):
            staged = self._mem_tile_placements(src, node)
            # No on-chip memory tiles: the copy lands off-chip.
            return staged or ((offchip_memory(self.accelerator),),)
        readers = []
        for dst in dsts:
            # An output edge is read off chip, by no core.
            allocation = ((None,),) if isinstance(dst, OutEdge) else self.mapping.get(dst).resource_allocation
            readers.append((dst, the_one(allocation, dst, "cores"), self._target_placement(dst, node)))
        return (reader_positions(node.inputs[0], readers),)

    def determine_possible_inter_core_tiling(
        self, node: TransferNode, possible_dst_allocs: tuple[tuple[Memory, ...], ...], dsts: tuple[HasInputs, ...]
    ) -> tuple[InterCoreTiling, ...]:
        possible_inter_core_tiling = []
        # A tensor its largest consumer does not split is held whole by every tile it sits on,
        # so its transfer carries no tiling however many copies exist; any other consumer reads
        # its part of a copy. Only the hop that lands on the memory tiles carries copies; the hop
        # from them to the cores still describes how those cores split the work.
        compute_dsts = [dst for dst in dsts if isinstance(dst, ComputationNode)] or [
            dst
            for dst in get_compute_predecessors_successors(tr=node, workload=self.ssw)
            if isinstance(dst, ComputationNode)
        ]
        node_dims = set(self.ssw.get_dims(node))
        largest = get_node_with_largest_resource_allocation(compute_dsts, self.mapping) if compute_dsts else None
        # A copy in memory holds what every consumer reading it needs: it is split only as all of them split the
        # tensor, and held whole where one reads it whole or they split it differently.
        splits = {
            frozenset(
                (dim, factor)
                for dim, factor in self.ssw.get_unique_dims_inter_core_tiling(dst, self.mapping)
                if dim in node_dims
            )
            for dst in compute_dsts
        }
        replicated = (
            node.transfer_type is TransferType.MEM_TO_MEM
            and largest is not None
            and (frozenset() in splits or len(splits) > 1 or self._each_column_reads_whole(largest, node))
        )
        own = self._own_mem_tiles(largest, node) if largest is not None else None
        for dst_allocs in possible_dst_allocs:
            nb_cores = len(dst_allocs)
            if nb_cores == 1 or replicated:
                dst_tiling = tuple()
            elif own and dst_allocs == own[0]:
                dst_tiling = own[1]
            elif all(isinstance(dst, ComputationNode) for dst in dsts):
                # For fan-out: use the destination with the largest resource allocation
                # as the tiling reference (consistent with determine_possible_memory_allocations strategy)
                dst = get_node_with_largest_resource_allocation(dsts, self.mapping)
                dst_tiling = tuple(self.ssw.get_unique_dims_inter_core_tiling(dst, self.mapping))
            else:
                dst_tiling = self.get_inter_core_tiling_for_transfer(node, dst_allocs)
            possible_inter_core_tiling.append(dst_tiling)
        return tuple(possible_inter_core_tiling)

    def get_inter_core_tiling_for_transfer(
        self, node: TransferNode, memory_allocs: tuple[Memory, ...]
    ) -> InterCoreTiling:
        assert isinstance(node, TransferNode), "Node must be a TransferNode for inter-core tiling determination."
        assert node.transfer_type in (TransferType.COMPUTE_TO_MEM, TransferType.MEM_TO_MEM), (
            "This function should only be called for MEM_TO_MEM (input) or COMPUTE_TO_MEM (output) transfers."
        )
        # Get the compute preds and succs
        compute_preds_succs = get_compute_predecessors_successors(tr=node, workload=self.ssw)
        # Get the largest allocation one of these
        largest_alloc_node = get_node_with_largest_resource_allocation(compute_preds_succs, self.mapping)
        # Get its compute tiling and find the tiling loop that matches the number of memory allocs
        largest_alloc_tiling = self.ssw.get_unique_dims_inter_core_tiling(largest_alloc_node, self.mapping)
        node_dims = set(self.ssw.get_dims(node))
        if not any(dim in node_dims for dim, _ in largest_alloc_tiling) and (
            slice_ := self._slice_of(node, len(memory_allocs))
        ):
            return (slice_,)
        return self.get_matching_tiling(largest_alloc_tiling, memory_allocs)

    def get_matching_tiling(self, compute_tiling: InterCoreTiling, dst_allocs: tuple[Memory, ...]) -> InterCoreTiling:
        """The split over ``dst_allocs`` where each memory holds what consecutive cores of ``compute_tiling`` hold."""
        nb_allocs = len(dst_allocs)
        per_tile, rest = divmod(prod(size for _, size in compute_tiling), nb_allocs)
        if not rest and (split := _without_innermost(compute_tiling, per_tile)) is not None:
            return split
        # No clean divisor: split the best loop into the largest even share that fits, instead of raising.
        best = max(compute_tiling, key=lambda loop: largest_divisor_leq(loop[1], nb_allocs), default=None)
        if best is None:
            raise ValueError(f"No tiling loop to reconcile with dst allocs {dst_allocs}")
        dim, size = best
        return ((dim, largest_divisor_leq(size, nb_allocs)),)

    def determine_possible_transfer_plans(
        self,
        node: TransferNode,
        src: HasOutputs,
        possible_dst_allocs: tuple[tuple[Memory, ...], ...],
        possible_tilings: tuple[InterCoreTiling, ...],
    ) -> tuple[MulticastPathPlan, ...]:
        """The route of each placement of ``node``, moving what each target reads from the sources that hold it."""
        all_possible_resource_plans = []
        possible_src_allocs = self._source_placements(src)
        bitwidth = node.inputs[0].operand_type.bitwidth
        for src_allocs in possible_src_allocs:
            for dst_allocs, tiling in zip(possible_dst_allocs, possible_tilings, strict=True):
                demand = self.ssw.get_transfer_demand(
                    node,
                    self.mapping,
                    tiling,
                    len(src_allocs),
                    len(dst_allocs),
                    self.hardware.pairs_by_spatial_index((*src_allocs, *dst_allocs)),
                    dst_allocs,
                    self._reads_in,
                )
                possible_resource_plans = self.accelerator.communication_manager.get_possible_transfer_plan(
                    src_allocs, dst_allocs, demand, bitwidth, self.hardware.shares_memory, self.route_alternatives
                )
                all_possible_resource_plans.extend(possible_resource_plans)
        return tuple(all_possible_resource_plans)

    def calculate_iterations(self) -> int:
        """Calculate the amount of steady state iterations based on all nodes' SSIS."""
        iterations_per_node = {node: prod(ssis.get_temporal_sizes()) for node, ssis in self.ssis.items()}
        # For now, return the minimum number of iterations across all nodes
        return min(iterations_per_node.values())

    def _source_placements(self, node: HasOutputs) -> tuple[tuple[Memory, ...], ...]:
        """The memories ``node`` may hold its output in, one per position."""
        return placements_of(self.accelerator, self.mapping, node, OUTPUT)

    def _target_placement(self, dst: HasInputs, transfer: TransferNode) -> tuple[Memory, ...]:
        """The memories ``dst`` reads ``transfer``'s copy in, one per position."""
        return self._destination_memories(dst, next(t for t in transfer.outputs if t in dst.inputs))

    def _reads_in(self, node: ComputationNode, tensor: Tensor) -> tuple[Memory, ...]:
        """The memory each of ``node``'s cores reads its input ``tensor`` in."""
        return the_one(placements_of(self.accelerator, self.mapping, node, role_of(node, tensor)), node, "placements")

    def determine_transfer_type(
        self,
        src: HasOutputs,
        dsts: list[HasInputs],
        tensor: Tensor,
        src_type: str | None = None,
        dst_type: str | None = None,
    ) -> TransferType:
        """The type of transfer moving ``tensor`` from ``src`` to ``dsts``, by the kinds of memory it leaves and
        lands in."""
        if src_type is None:
            src_allocation = self._source_placements(src)
            assert len(src_allocation) == 1, "TODO: Handle multiple source allocations for transfer type determination."
            src_type = self._effective_allocation_type(src_allocation[0])
        if dst_type is None:
            landing = [m for dst in dsts for m in self._destination_memories(dst, tensor)]
            dst_type = self._effective_allocation_type(landing)
        if src_type == "compute" and dst_type == "compute":
            return TransferType.COMPUTE_TO_COMPUTE
        elif src_type == "compute" and dst_type in ("memory", "offchip"):
            return TransferType.COMPUTE_TO_MEM
        elif src_type in ("memory", "offchip") and dst_type == "compute":
            return TransferType.MEM_TO_COMPUTE
        elif src_type in ("memory", "offchip") and dst_type in ("memory", "offchip"):
            return TransferType.MEM_TO_MEM
        raise ValueError(f"Unsupported transfer type from {src_type} to {dst_type}")

    def _destination_memories(self, dst: HasInputs, tensor: Tensor) -> tuple[Memory, ...]:
        """The memories ``dst`` reads ``tensor`` in: off-chip for an output edge, else the memory each of the
        computation's cores reads it in."""
        if isinstance(dst, OutEdge):
            return (offchip_memory(self.accelerator),)
        assert isinstance(dst, ComputationNode), f"Unexpected destination node type: {type(dst)}"
        return self._reads_in(dst, tensor)

    def _effective_allocation_type(self, allocs: Sequence[Memory]) -> str:
        """The memory kind a set of (possibly mixed) memories serves as: ``compute`` if any is a compute memory (that is
        where the data originates or is needed), else ``offchip``, else ``memory``."""
        kinds = {alloc.kind for alloc in allocs}
        if not kinds:
            raise ValueError("no allocations to determine transfer type")
        return next(kind for kind in ("compute", "offchip", "memory") if kind in kinds)

    def _mem_tiles(self) -> list[Memory]:
        """The memory tiles within the column budget, in column order."""
        return sorted(self.hardware.mem_memories, key=lambda memory: (memory.col_id, memory))

    def _mem_tile_placements(self, src: HasOutputs, node: Node) -> tuple[tuple[Memory, ...], ...]:
        # An input transfer feeds every unrolled destination separately; an output transfer is
        # gathered per column, since a column's compute cores share its memory tile.
        MAX_RELEVANT_FACTOR_PER_TRANSFER_TYPE = {
            TransferType.MEM_TO_MEM: 1,  # for input transfers to mem tile
            TransferType.COMPUTE_TO_MEM: 4,  # for output transfers to mem tile
        }
        inter_core_tiling_entries = self.mapping.get(src).inter_core_tiling
        inter_core_tiling_src = inter_core_tiling_entries[0] if inter_core_tiling_entries else ()
        total_relevant_unrolling = self._relevant_unrolling(src, node)
        columns = self._columns_of(src)
        mem_tiles = self._mem_tiles()
        if node.transfer_type is TransferType.MEM_TO_MEM and self._each_column_reads_whole(src, node):
            return (tuple(c for c in mem_tiles if c.col_id in self._column_ids_of(src)),)
        if own := self._own_mem_tiles(src, node):
            return (own[0],)
        if columns and node.transfer_type in (TransferType.COMPUTE_TO_MEM, TransferType.MEM_TO_MEM):
            # A column's cores share its one memory tile, so both need one tile per occupied column, not per core.
            required_nb_mem_tiles = min(columns, total_relevant_unrolling)
        else:
            required_nb_mem_tiles = ceil(
                total_relevant_unrolling / MAX_RELEVANT_FACTOR_PER_TRANSFER_TYPE[node.transfer_type]
            )
        # Snap the count down to a divisor of the compute split so the transfer is one even inter-core tiling.
        if total_relevant_unrolling > 1:
            required_nb_mem_tiles = largest_divisor_leq(total_relevant_unrolling, required_nb_mem_tiles)
        # Unrolling binds allocation position to spatial index, so column order keeps a tile with its own cores.
        candidates = [tuple(combo) for combo in combinations(mem_tiles, required_nb_mem_tiles)]
        # A tensor its consumers do not split is held whole, so a single tile can be left
        # feeding every column. One copy per occupied column is the alternative: each tile
        # serves its own column, and the shim broadcasts the tensor to them.
        if total_relevant_unrolling == 1 and node.transfer_type is TransferType.MEM_TO_MEM:
            per_column = tuple(c for c in mem_tiles if c.col_id in self._column_ids_of(src))
            if len(per_column) > 1 and per_column not in candidates:
                candidates.append(per_column)
        # Partial sums of a reduction the source splits across columns are completed a slice per column's memory tile,
        # a reduce-scatter: no link carries more than when one tile completes them all, and the slices leave in
        # parallel.
        if node.transfer_type is TransferType.COMPUTE_TO_MEM and self.ssw.splits_reduction(src, inter_core_tiling_src):
            per_column = tuple(c for c in mem_tiles if c.col_id in self._column_ids_of(src))
            if len(per_column) > 1 and self._slice_of(node, len(per_column)):
                return (per_column,)
        return tuple(candidates)

    def _own_mem_tiles(self, src: HasOutputs, node: TransferNode) -> tuple[tuple[Memory, ...], InterCoreTiling] | None:
        """The memory tiles ``src``'s cores work in, where each holds a block of ``node``'s tensor of its own cores, in
        block order, and the split of the tensor over them; None where they do not."""
        allocations = self.mapping.get(src).resource_allocation
        cores = allocations[0] if allocations else ()
        loops = tuple(self.ssw.get_unique_dims_inter_core_tiling(src, self.mapping))
        mem_tiles = set(self._mem_tiles())
        seen = [[memory for memory in self.accelerator.memories_of(core) if memory in mem_tiles] for core in cores]
        if any(len(tiles) != 1 for tiles in seen) or len(cores) != prod(f for _, f in loops):
            return None
        tile_of = [tile for (tile,) in seen]
        if len(set(tile_of)) <= 1:
            return None
        node_dims = set(self.ssw.get_dims(node))
        axes = [i for i, (dim, _) in enumerate(loops) if dim in node_dims]
        held: dict[Memory, set[tuple[int, ...]]] = {}
        for tile, position in zip(tile_of, np.ndindex(*(f for _, f in loops)), strict=True):
            held.setdefault(tile, set()).add(tuple(position[i] for i in axes))
        spans = {tuple(len({p[a] for p in slices}) for a in range(len(axes))) for slices in held.values()}
        if len(spans) != 1:
            return None
        (span,) = spans
        starts = {tile: tuple(min(p[a] for p in slices) for a in range(len(axes))) for tile, slices in held.items()}
        blocks = all(
            len(slices) == prod(span)
            and all(start % width == 0 for start, width in zip(starts[tile], span, strict=True))
            and all(p[a] - starts[tile][a] < span[a] for p in slices for a in range(len(axes)))
            for tile, slices in held.items()
        )
        split = tuple((loops[i][0], loops[i][1] // width) for i, width in zip(axes, span, strict=True))
        if not blocks or len(set(starts.values())) != len(held) or prod(f for _, f in split) != len(held):
            return None
        order = sorted(held, key=lambda tile: tuple(start // w for start, w in zip(starts[tile], span, strict=True)))
        return tuple(order), tuple((dim, f) for dim, f in split if f > 1)

    def _each_column_reads_whole(self, dst: HasOutputs, node: TransferNode) -> bool:
        """Whether in every column ``dst`` runs in, its cores read every tile of ``node``'s tensor between them, so a
        whole copy in each column's memory serves them without data crossing columns, where the code generator does
        not pair the cores by spatial index."""
        allocations, tilings = self.mapping.get(dst).resource_allocation, self.mapping.get(dst).inter_core_tiling
        if not allocations or not tilings or self.hardware.pairs_by_spatial_index(self._source_placements(dst)[0]):
            return False
        cores, tiling = allocations[0], tilings[0]
        node_dims = set(self.ssw.get_dims(node))
        relevant = [i for i, (dim, _) in enumerate(tiling) if dim in node_dims]
        tiles = prod(tiling[i][1] for i in relevant)
        read: dict[int | None, set[tuple[int, ...]]] = {}
        for p, core in enumerate(cores):
            position = np.unravel_index(position_of(p, len(cores), tiling), [f for _, f in tiling])
            read.setdefault(core.col_id, set()).add(tuple(int(position[i]) for i in relevant))
        return tiles > 1 and len(read) > 1 and all(len(seen) == tiles for seen in read.values())

    def _relevant_unrolling(self, src: HasOutputs, node: TransferNode) -> int:
        """How many ways ``src``'s inter-core tiling splits the dimensions of ``node``."""
        tilings = self.mapping.get(src).inter_core_tiling
        node_dims = self.ssw.get_dims(node)
        return prod(size for dim, size in (tilings[0] if tilings else ()) if dim in node_dims)

    def _slice_of(self, node: TransferNode, parts: int) -> tuple[LayerDim, int] | None:
        """The split of ``node``'s largest dimension that ``parts`` divides, if any."""
        dims = [dim for dim in self.ssw.get_dims(node) if self.ssw.get_dimension_size(dim) % parts == 0]
        return (max(dims, key=self.ssw.get_dimension_size), parts) if dims else None

    def _column_ids_of(self, src: HasOutputs) -> set[int]:
        """The array columns the source's cores sit in, empty if they carry no coordinates."""
        allocations = self.mapping.get(src).resource_allocation
        cores = allocations[0] if allocations else ()
        return {core.col_id for core in cores if getattr(core, "col_id", None) is not None}

    def _columns_of(self, src: HasOutputs) -> int:
        """How many array columns the source's cores sit in, 0 if unknown.

        Accelerator descriptions need not give their cores coordinates; without them the
        caller falls back to the rows-per-column estimate.
        """
        return len(self._column_ids_of(src))


def _without_innermost(tiling: InterCoreTiling, factor: int) -> InterCoreTiling | None:
    """``tiling`` with ``factor`` taken off its innermost loops, None where they do not factor it."""
    split = list(tiling)
    while factor > 1 and split:
        dim, size = split.pop()
        if factor % size == 0:
            factor //= size
        elif size % factor == 0:
            split.append((dim, size // factor))
            factor = 1
        else:
            return None
    return tuple(split) if factor == 1 else None
