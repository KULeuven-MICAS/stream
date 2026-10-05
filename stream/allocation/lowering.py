"""Lowering of a fused group to its steady state: the transfers made explicit, each with the placements and
routes it may take, the iteration spaces, and the timeslots. Pure: nothing it is given is changed."""

from dataclasses import replace
from itertools import combinations
from math import ceil, prod
from typing import cast

from xdsl.ir.affine import AffineMap

from stream.allocation.problem import AllocationProblem
from stream.cost_model.communication_manager import MulticastPathPlan
from stream.cost_model.core_cost_lut import CoreCostLUT
from stream.datatypes import InterCoreTiling, LayerDim
from stream.hardware.architecture.accelerator import Accelerator
from stream.hardware.architecture.core import Core
from stream.mapping.mapping import Mapping
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
    generate_steady_state_iteration_spaces,
    generate_tensor_ssis,
    get_compute_predecessors_successors,
    get_equivalent_dimension,
    get_node_with_largest_resource_allocation,
)
from stream.workload.workload import Workload


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
) -> AllocationProblem:
    """The steady-state problem of ``workload`` (one fused group) mapped by ``mapping`` on ``accelerator``."""
    return _Lowering(workload, accelerator, mapping, nb_cols_to_use).lower(fusion_splits, cost_lut)


class _Lowering:
    """The state the lowering threads through its steps: a mapping it rewrites as the transfer graph grows."""

    def __init__(self, workload: Workload, accelerator: Accelerator, mapping: Mapping, nb_cols_to_use: int):
        self.workload = workload
        self.accelerator = accelerator
        self.mapping = mapping.copy()
        self.nb_cols_to_use = nb_cols_to_use
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
        new_nodes: dict[str, Node] = {node.name: node for node in self.workload.nodes}
        # Go through the tensors of the workload to find sources and destinations of the tensor
        for tensor in self.workload.tensors:
            # A kernel's carried state is resident on the cores its node runs on: it is read
            # where it already sits, from one step of that node's own loop to the next, so
            # there is nothing to move and no source to move it from.
            if any(
                isinstance(n, HasIterationSpace) and tensor in n.inputs and is_state_operand(n, tensor)
                for n in self.workload.nodes
            ):
                continue
            srcs = [n for n in self.workload.nodes if isinstance(n, HasOutputs) and tensor in n.outputs]
            assert len(srcs) == 1, f"Expected exactly one source for tensor {tensor}, found {len(srcs)}"
            src = new_nodes[srcs[0].name]
            dsts = [new_nodes[n.name] for n in self.workload.nodes if isinstance(n, HasInputs) and tensor in n.inputs]
            is_constant_o_transfer = any(isinstance(dst, OutEdge) for dst in dsts)
            if is_constant_o_transfer and not isinstance(src, InEdge):
                self.add_two_transfer_nodes_for_constant_output_transfer(tensor, src, dsts, new_nodes)
            else:
                self.add_transfer_nodes(tensor, src, dsts, new_nodes)
        new_workload = Workload(new_nodes.values())
        return new_workload

    def add_transfer_nodes(self, tensor: Tensor, src: HasOutputs, dsts: list[HasInputs], new_nodes: dict[str, Node]):
        """
        Move ``tensor`` from its source to its destinations, either directly or staged on a memory tile.

        Staging splits the move in two -- source to the on-chip buffer, buffer to every destination --
        so the tile can hold the tensor across reads and re-lay it out on the way through. See
        :meth:`_stages_on_mem_tile`.
        """
        if not self._stages_on_mem_tile(tensor, src, dsts):
            transfer_type = self.determine_transfer_type(src, dsts)
            out_name = f"{tensor.name}_1"
            transfer_node, updated_tensors = self.generate_transfer_node(dsts, tensor, transfer_type, out_name)
            new_nodes[transfer_node.name] = transfer_node
            for dst, updated_tensor in zip(dsts, updated_tensors, strict=True):
                self.update_destination_node_inputs(tensor, src, new_nodes, dst, updated_tensor)
            return
        # First transfer node from source to on-chip buffer
        transfer_type_1 = self.determine_transfer_type(src, dsts, dst_type="memory")
        out_name_1 = f"{tensor.name}_1"
        transfer_node_1, updated_tensors_1 = self.generate_transfer_node([src], tensor, transfer_type_1, out_name_1)
        new_nodes[transfer_node_1.name] = transfer_node_1
        # Second transfer node from on-chip buffer to destinations
        out_name_2 = f"{tensor.name}_2"
        transfer_type_2 = self.determine_transfer_type(src, dsts, src_type="memory")
        transfer_node_2, updated_tensors_2 = self.generate_transfer_node(
            dsts, updated_tensors_1[0], transfer_type_2, out_name_2
        )
        new_nodes[transfer_node_2.name] = transfer_node_2
        for dst, updated_tensor in zip(dsts, updated_tensors_2, strict=True):
            self.update_destination_node_inputs(tensor, src, new_nodes, dst, updated_tensor)

    def _stages_on_mem_tile(self, tensor: Tensor, src: HasOutputs, dsts: list[HasInputs]) -> bool:
        """Whether a transfer is staged in a memory tile instead of landing straight on the cores.

        Two things ask for a staging buffer. An offchip input is held in the tile,
        which keeps the fan-out off the shim's two DMA channels: one channel feeds the tile, which then
        serves every core from its own. And a producer and consumer that disagree on layout need the
        tensor re-laid out between them, which is the tile's other job.
        """
        if not self._get_accelerator_memory_cores():
            return False
        if isinstance(src, InEdge):
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
        # Update the mapping entry for this new_dst node to be the same as the original dst node
        self.mapping.set(new_dst, self.mapping.get(dst))
        # Remove the original dst node from the mapping as it has been updated with new inputs
        self.mapping.remove(dst)

    def add_two_transfer_nodes_for_constant_output_transfer(
        self, tensor: Tensor, src: HasOutputs, dsts: list[HasInputs], new_nodes: dict[str, Node]
    ):
        """
        For constant output transfers, we add two transfer nodes:
        - one from the source to the on-chip memory buffer,
        - a second one from the on-chip memory buffer to the destination.
        This is to ensure that the constant tensor is properly allocated in memory and can be reused across iterations.

        If the accelerator has no on-chip memory tiles (e.g. TPU-like hardware), falls back to a single
        direct transfer from the source to the destinations (COMPUTE_TO_MEM).
        """
        assert len(dsts) == 1, "Currently only support single destination for constant output transfer."
        dst = dsts[0]
        assert isinstance(dst, OutEdge), (
            f"Expected destination of constant transfer to be an OutEdge, found {type(dst)}"
        )
        # Fall back to a single direct transfer when no memory tiles are available
        if not self._get_accelerator_memory_cores():
            transfer_type = self.determine_transfer_type(src, dsts)
            new_tensor = self.generate_transfer_input_tensor(tensor, src, name_suffix="_1")
            out_name = f"{tensor.name}"
            transfer_node, updated_tensors = self.generate_transfer_node([dst], new_tensor, transfer_type, out_name)
            new_nodes[transfer_node.name] = transfer_node
            new_src = self.update_source_tensor(tensor, src, new_nodes, new_tensor)
            self.update_destination_tensor(tensor, new_src, new_nodes, dst, updated_tensors)
            return
        # First transfer node from source to on-chip buffer
        transfer_type_1 = self.determine_transfer_type(src, dsts, dst_type="memory")
        new_tensor = self.generate_transfer_input_tensor(tensor, src, name_suffix="_1")
        out_name_1 = f"{tensor.name}_2"
        transfer_node_1, updated_tensors_1 = self.generate_transfer_node([src], new_tensor, transfer_type_1, out_name_1)
        new_nodes[transfer_node_1.name] = transfer_node_1
        new_src = self.update_source_tensor(tensor, src, new_nodes, new_tensor)
        # Second transfer node from on-chip buffer to destination
        transfer_type_2 = self.determine_transfer_type(new_src, dsts, src_type="memory")
        out_name_2 = f"{tensor.name}"
        transfer_node_2, updated_tensors_2 = self.generate_transfer_node(
            [dst], updated_tensors_1[0], transfer_type_2, out_name_2
        )
        new_nodes[transfer_node_2.name] = transfer_node_2
        # Update the dst input to the second transfer node
        self.update_destination_tensor(tensor, new_src, new_nodes, dst, updated_tensors_2)

    def update_destination_tensor(self, tensor, src, new_nodes, dst, updated_tensors_2):
        dst_new = new_nodes[dst.name]
        assert len(src.outputs) == 1, "Src must have exactly one output tensor for index below."
        input_idx = dst_new.inputs.index(tensor)
        new_inputs = dst_new.inputs[:input_idx] + (updated_tensors_2[0],) + dst_new.inputs[input_idx + 1 :]
        new_dst = OutEdge(
            name=dst_new.name,
            inputs=new_inputs,
        )
        new_nodes[new_dst.name] = new_dst
        # No need to update mapping of dst as it's an OutEdge

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

    def generate_transfer_node(
        self, dsts: list[HasInputs], tensor: Tensor, transfer_type: TransferType, out_name: str = ""
    ) -> tuple[TransferNode, list[Tensor]]:
        transfer_outputs = self.generate_transfer_output_tensors(tensor, dsts, out_name)
        operand_mapping = tuple(AffineMap.identity(len(tensor.shape)) for _ in range(1 + len(dsts)))
        transfer_node = TransferNode(
            name=f"Transfer({tensor.name})",
            inputs=(tensor,),
            outputs=tuple(transfer_outputs),
            transfer_type=transfer_type,
            operand_mapping=operand_mapping,
        )
        return transfer_node, transfer_outputs

    def generate_transfer_input_tensor(self, tensor: Tensor, src: HasOutputs, name_suffix: str = "") -> Tensor:
        assert len(src.outputs) == 1, "Src must have exactly one output tensor for index below."
        input_tensor = Tensor(
            name=f"{tensor.name}{name_suffix}",
            operand_type=tensor.operand_type,
            shape=tensor.shape,
            subview=tensor.subview,
        )
        return input_tensor

    def generate_transfer_output_tensors(
        self, tensor: Tensor, dsts: list[HasInputs], out_name: str = ""
    ) -> list[Tensor]:
        transfer_outputs = []
        for i, _ in enumerate(dsts):
            suffix = f".{i}" if len(dsts) > 1 else ""
            transfer_output = Tensor(
                name=f"{out_name}{suffix}",
                operand_type=tensor.operand_type,
                shape=tensor.shape,
                subview=tensor.subview,
            )
            transfer_outputs.append(transfer_output)
        return transfer_outputs

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
        possible_allocations = self.determine_possible_transfer_plans(src, possible_dst_allocs)
        self.mapping.set_for_node(
            node,
            resource_allocation=possible_allocations,
            inter_core_tiling=possible_inter_core_tiling,
            memory_allocation=possible_dst_allocs,
        )

    def determine_possible_memory_allocations(
        self, node: TransferNode, src: HasOutputs, dsts: tuple[HasInputs, ...]
    ) -> tuple[tuple[Core, ...], ...]:
        """
        Determine the memory allocation of the transfer node.
        The memory alloc is always for the destination side tensors. For input transfers, the
        MEM_TO_MEM output tensor is allocated on memory cores; for output transfers the
        COMPUTE_TO_MEM output tensor is. Otherwise, allocations follow destination nodes.
        """
        if node.transfer_type in (TransferType.MEM_TO_MEM,) and isinstance(src, InEdge):
            # Find the dst with max number of compute allocations to determine possible memory cores
            compute_dsts = get_compute_predecessors_successors(
                tr=node, workload=self.ssw
            )  # won't have any compute preds
            dst = get_node_with_largest_resource_allocation(compute_dsts, self.mapping)
            possible_memory_cores = self._get_possible_memory_core_allocations(dst, node)
        elif node.transfer_type in (TransferType.MEM_TO_MEM,) and any(isinstance(dst, OutEdge) for dst in dsts):
            assert len(dsts) == 1, "Currently only support single destination for constant output transfer."
            dst = dsts[0]
            possible_memory_cores = self._retrieve_core_allocation(dst)
        elif node.transfer_type in (TransferType.COMPUTE_TO_MEM,):
            possible_memory_cores = self._get_possible_memory_core_allocations(src, node)
            if not possible_memory_cores:
                # No on-chip memory tiles — fall back to offchip core as the destination
                offchip_core = self.accelerator.get_core(self.accelerator.offchip_core_id)
                possible_memory_cores = ((offchip_core,),)
        else:
            # The destination order is the order the consumers declared, because that is what
            # pairs a spatial index with a core. Sorting here would silently hand each index a
            # different core than the computation node it feeds whenever a layer's cores are
            # not listed in ascending id order.
            destinations: dict[Core, None] = {}
            for dst in dsts:
                assert len(self._retrieve_core_allocation(dst)) == 1, "TODO: Support multiple compute allocations."
                destinations.update(dict.fromkeys(self._retrieve_core_allocation(dst)[0]))
            possible_memory_cores = (tuple(destinations),)
        return possible_memory_cores

    def determine_possible_inter_core_tiling(
        self, node: TransferNode, possible_dst_allocs: tuple[tuple[Core, ...], ...], dsts: tuple[HasInputs, ...]
    ) -> tuple[InterCoreTiling, ...]:
        possible_inter_core_tiling = []
        node_dims = set(self.ssw.get_dims(node))
        # The first hop of a transfer chain lands on a memory tile, so its own destinations
        # are transfers; the consumers that decide the split are the compute nodes behind it.
        compute_dsts = [dst for dst in dsts if isinstance(dst, ComputationNode)] or [
            dst
            for dst in get_compute_predecessors_successors(tr=node, workload=self.ssw)
            if isinstance(dst, ComputationNode)
        ]
        # A tensor whose dimensions the consumers do not split is held whole by every tile
        # it sits on, so its transfer carries no tiling however many copies exist.
        # Only the hop that lands on the memory tiles carries copies; the hop from them to
        # the cores still describes how those cores split the work.
        replicated = (
            node.transfer_type is TransferType.MEM_TO_MEM
            and bool(compute_dsts)
            and not any(
                dim in node_dims
                for dst in compute_dsts
                for dim, _ in self.ssw.get_unique_dims_inter_core_tiling(dst, self.mapping)
            )
        )
        for dst_allocs in possible_dst_allocs:
            nb_cores = len(dst_allocs)
            if nb_cores == 1 or replicated:
                dst_tiling = tuple()
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
        self, node: TransferNode, memory_allocs: tuple[tuple[Core, ...], ...]
    ) -> tuple[InterCoreTiling, ...]:
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
        mem_tiling = self.get_matching_tiling(largest_alloc_tiling, memory_allocs)
        return (mem_tiling,)

    def get_matching_tiling(
        self, compute_tiling: InterCoreTiling, dst_allocs: tuple[Core, ...]
    ) -> tuple[LayerDim, int]:
        # TODO: Make sure that the selected tiling_loop is relevant for the transfer node
        nb_allocs = len(dst_allocs)
        for tiling_loop in compute_tiling:
            _, size = tiling_loop
            if size == nb_allocs:
                return tiling_loop
        # No size with exact match found, try to find one that is a multiple of the number of dst allocs
        for tiling_loop in compute_tiling:
            dim, size = tiling_loop
            if size % nb_allocs == 0:
                return (dim, nb_allocs)
        # No clean divisor: split the best loop into the largest even share that fits, instead of raising.
        best = max(compute_tiling, key=lambda loop: largest_divisor_leq(loop[1], nb_allocs), default=None)
        if best is None:
            raise ValueError(f"No tiling loop to reconcile with dst allocs {dst_allocs}")
        dim, size = best
        return (dim, largest_divisor_leq(size, nb_allocs))

    def determine_possible_transfer_plans(
        self, src: HasOutputs, possible_dst_allocs: tuple[tuple[Core, ...], ...]
    ) -> tuple[MulticastPathPlan, ...]:
        all_possible_resource_plans = []
        possible_src_allocs = self._retrieve_core_allocation(src)
        for src_allocs in possible_src_allocs:
            for dst_allocs in possible_dst_allocs:
                possible_resource_plans = self.accelerator.communication_manager.get_possible_transfer_plan(
                    src_allocs=src_allocs,
                    dst_allocs=dst_allocs,
                )
                all_possible_resource_plans.extend(possible_resource_plans)
        return tuple(all_possible_resource_plans)

    def calculate_iterations(self) -> int:
        """Calculate the amount of steady state iterations based on all nodes' SSIS."""
        iterations_per_node = {node: prod(ssis.get_temporal_sizes()) for node, ssis in self.ssis.items()}
        # For now, return the minimum number of iterations across all nodes
        return min(iterations_per_node.values())

    def _retrieve_core_allocation(self, node: Node) -> tuple[tuple[Core, ...], ...]:
        if isinstance(node, InEdge):
            return ((self.accelerator.get_core(self.accelerator.offchip_core_id),),)
        if isinstance(node, OutEdge):
            return ((self.accelerator.get_core(self.accelerator.offchip_core_id),),)
        if isinstance(node, HasOutputs):
            if isinstance(node, TransferNode):
                return self.mapping.get(node).memory_allocation
            return self.mapping.get(node).resource_allocation
        raise ValueError(f"Unexpected source node type: {type(node)}")

    def determine_transfer_type(
        self, src: HasOutputs, dsts: tuple[HasInputs, ...], src_type: str | None = None, dst_type: str | None = None
    ) -> TransferType:  # noqa: PLR0912
        """Determine the type of transfer needed based on the allocation types of src and dst nodes."""
        if src_type is None:
            src_allocation = self._retrieve_core_allocation(src)
            assert len(src_allocation) == 1, "TODO: Handle multiple source allocations for transfer type determination."
            src_type = self._effective_allocation_type(src_allocation[0])
        if dst_type is None:
            dst_allocations = [self._retrieve_core_allocation(dst)[0] for dst in dsts]
            # A constant input/output may be spread across a mix of allocations -- a compute core that
            # produces/consumes it and a memory/offchip core that stages it. Resolve to the type the
            # transfer must actually serve (compute when any allocation is compute; the PE is where the
            # data originates or is needed) instead of asserting the allocations are homogeneous.
            dst_type = self._effective_allocation_type([alloc for dst_alloc in dst_allocations for alloc in dst_alloc])
        if src_type == "compute" and dst_type == "compute":
            return TransferType.COMPUTE_TO_COMPUTE
        elif src_type == "compute" and dst_type in ("memory", "shim", "offchip"):
            return TransferType.COMPUTE_TO_MEM
        elif src_type in ("memory", "shim", "offchip") and dst_type == "compute":
            return TransferType.MEM_TO_COMPUTE
        elif src_type in ("memory", "shim", "offchip") and dst_type in ("memory", "shim", "offchip"):
            return TransferType.MEM_TO_MEM
        raise ValueError(f"Unsupported transfer type from {src_type} to {dst_type}")

    def _effective_allocation_type(self, allocs: list[Core]) -> str:
        """The transfer type a set of (possibly mixed) allocations must serve.

        Homogeneous allocations return their single type. Mixed allocations -- a constant that is on a
        compute core *and* staged in memory/offchip -- resolve to ``compute`` when any allocation is
        compute (that is where the data originates or is needed), else to the memory-family type. This
        keeps the transfer graph well-defined for the valid allocations the MILP can produce, instead
        of asserting homogeneity.
        """
        alloc_types = set(alloc.type for alloc in allocs)
        if not alloc_types:
            raise ValueError("no allocations to determine transfer type")
        if len(alloc_types) == 1:
            return alloc_types.pop()
        if "compute" in alloc_types:
            return "compute"
        for preferred in ("offchip", "shim", "memory"):
            if preferred in alloc_types:
                return preferred
        return alloc_types.pop()

    def _get_accelerator_memory_cores(self) -> set[Core]:
        """
        Get all memory cores in the accelerator.
        """
        memory_cores = set()
        for core in self.accelerator.core_list:
            if (
                core.type == "memory"
                and not core.id == self.accelerator.offchip_core_id
                and core.col_id < self.nb_cols_to_use
            ):
                memory_cores.add(core)
        return memory_cores

    def _get_possible_memory_core_allocations(self, src: HasOutputs, node: Node) -> tuple[tuple[Core, ...], ...]:
        # An input transfer feeds every unrolled destination separately; an output transfer is
        # gathered per column, since a column's compute cores share its memory tile.
        MAX_RELEVANT_FACTOR_PER_TRANSFER_TYPE = {
            TransferType.MEM_TO_MEM: 1,  # for input transfers to mem tile
            TransferType.COMPUTE_TO_MEM: 4,  # for output transfers to mem tile
        }
        # Check the dims of node and find their unrolling factors in inter_core_tiling of src
        node_dims = self.ssw.get_dims(node)
        inter_core_tiling_entries = self.mapping.get(src).inter_core_tiling
        if not inter_core_tiling_entries:
            inter_core_tiling_src = ()
        else:
            inter_core_tiling_src = inter_core_tiling_entries[0]
        total_relevant_unrolling = 1
        for dim in node_dims:
            for tiling_dim, size in inter_core_tiling_src:
                if tiling_dim == dim:
                    total_relevant_unrolling *= size
        columns = self._columns_of(src)
        if columns and node.transfer_type in (TransferType.COMPUTE_TO_MEM, TransferType.MEM_TO_MEM):
            # A column's cores share its one memory tile, so both need one tile per occupied column, not per core.
            required_nb_memory_cores = min(columns, total_relevant_unrolling)
        else:
            required_nb_memory_cores = ceil(
                total_relevant_unrolling / MAX_RELEVANT_FACTOR_PER_TRANSFER_TYPE[node.transfer_type]
            )
        # Snap the count down to a divisor of the compute split so the transfer is one even inter-core tiling.
        if total_relevant_unrolling > 1:
            required_nb_memory_cores = largest_divisor_leq(total_relevant_unrolling, required_nb_memory_cores)
        # Unrolling binds allocation position to spatial index, so column order keeps a tile with its own cores.
        all_mem_cores = sorted(self._get_accelerator_memory_cores(), key=lambda core: (core.col_id, core.id))
        candidates = [tuple(combo) for combo in combinations(all_mem_cores, required_nb_memory_cores)]
        # A tensor its consumers do not split is held whole, so a single tile can be left
        # feeding every column. One copy per occupied column is the alternative: each tile
        # serves its own column, and the shim broadcasts the tensor to them.
        if total_relevant_unrolling == 1 and node.transfer_type is TransferType.MEM_TO_MEM:
            per_column = tuple(c for c in all_mem_cores if c.col_id in self._column_ids_of(src))
            if len(per_column) > 1 and per_column not in candidates:
                candidates.append(per_column)
        return tuple(candidates)

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
