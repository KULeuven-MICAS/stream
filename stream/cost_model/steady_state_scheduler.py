import logging
import os
from itertools import combinations
from math import ceil, prod
from typing import cast

from xdsl.ir.affine import AffineMap

# if TYPE_CHECKING:
from stream.cost_model.communication_manager import MulticastPathPlan
from stream.cost_model.core_cost_lut import CoreCostLUT
from stream.datatypes import InterCoreTiling, LayerDim
from stream.hardware.architecture.accelerator import Accelerator
from stream.hardware.architecture.core import Core
from stream.mapping.mapping import Mapping
from stream.opt.allocation.constraint_optimization.transfer_and_tensor_allocation import (
    MemoryAlloc,
    TensorDepths,
    TensorReuseLevels,
    TransferAlloc,
    TransferAndTensorAllocator,
)
from stream.opt.solver import ConstraintSelection, SolveStats
from stream.visualization.steady_state_trace import export_steady_state_trace
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
from stream.workload.steady_state.computation import SteadyStateComputation
from stream.workload.steady_state.iteration_space import (
    IterationVariable,
    IterationVariableType,
    LoopEffect,
    Reuse,
    SteadyStateIterationSpace,
)
from stream.workload.utils import (
    generate_steady_state_iteration_spaces,
    get_compute_predecessors_successors,
    get_equivalent_dimension,
    get_node_with_largest_resource_allocation,
)
from stream.workload.workload import Workload

logger = logging.getLogger(__name__)


class SteadyStateScheduler:
    def __init__(  # noqa: PLR0913
        self,
        workload: Workload,
        accelerator: "Accelerator",
        mapping: Mapping,
        fusion_splits: dict[LayerDim, int],
        cost_lut: CoreCostLUT,
        nb_cols_to_use: int = 4,
        output_path: str = "",
        backend: str = "ORTOOLS_GSCIP",
        constraint_selection: ConstraintSelection | None = None,
    ):
        """
        Initialize the SteadyStateScheduler with the allocation and accelerator.

        Args:
            workload (ComputationNodeWorkload): The workload to be scheduled.
        """
        self.workload = workload  # Only contains nodes that are part of the current fusion stack
        self.accelerator = accelerator
        self.mapping = mapping
        self.fusion_splits = fusion_splits
        self.cost_lut = cost_lut
        self.partitioned_nodes: dict[ComputationNode, list[SteadyStateComputation]] = {}
        self.constant_tensors: dict[int, InEdge | OutEdge] = {}
        self.ssw: Workload | None = None

        # Cost model parameters
        self.latency_total = -1
        self.latency_per_iteration = -1
        self.overlap_between_iterations = -1
        self.tensor_depths: TensorDepths = {}

        self.nb_cols_to_use = nb_cols_to_use
        self.backend = backend
        self.constraint_selection = constraint_selection

        self.output_path = output_path
        if self.output_path:
            os.makedirs(self.output_path, exist_ok=True)

        self.solve_stats: SolveStats | None = None

    def get_ir(self) -> dict:
        """Return a dictionary representation of the scheduler state for serialization/inspection.

        This captures:
        - Latency metrics (total, per-iteration, overlap)
        - Backend and constraint configuration used for the solve
        - Fusion splits applied
        - Mapping with node-to-resource allocations
        """
        cs = self.constraint_selection
        constraint_selection_ir = (
            {
                "memory_capacity": cs.memory_capacity,
                "object_fifo_depth": cs.object_fifo_depth,
                "buffer_descriptors": cs.buffer_descriptors,
                "dma_channels": cs.dma_channels,
            }
            if cs is not None
            else None
        )
        return {
            "latency": {
                "total": self.latency_total,
                "per_iteration": self.latency_per_iteration,
                "overlap_between_iterations": self.overlap_between_iterations,
            },
            "backend": self.backend,
            "constraint_selection": constraint_selection_ir,
            "fusion_splits": {str(dim): size for dim, size in self.fusion_splits.items()},
            "mapping": self.mapping.get_ir(),
        }

    def run(self) -> Workload:
        """
        Run the steady state scheduler on the given workload.

        Returns:
            TimeSlotAllocation: The scheduled workload.
        """
        # Update the workload graph to include transfer nodes
        self.ssw = self.build_transfer_graph()
        # Update the fusion_splits based on the new workload with transfer nodes
        self.fusion_splits = self.update_fusion_splits()
        # Save the new workload with transfers
        # self.ssw.visualize(os.path.join(self.output_path, "tiled_workload_with_transfers.png"))
        # Update the mapping for the new workload graph
        self.mapping = self.update_mapping()
        # Update the cost lut for the new workload graph
        self.cost_lut = self.update_cost_lut()
        # Update the steady state iteration spaces to include transfer nodes and tensors
        self.ssis = self.generate_ssis()
        # Calculate the number of iterations based on the steady state iteration spaces
        self.iterations = self.calculate_iterations()
        # Calculate the multiplicity of each node's execution in the steady state workload
        multiplicities = self.calculate_multiplicities()
        # Get the timeslots for all nodes (resource-aware: same slot allowed iff a
        # disjoint core/link assignment exists across same-class nodes in that slot).
        timeslots = self.ssw.get_timeslots(self.mapping)
        # timeslots = self.ssw.get_timeslots_simple()  # baseline: one slot per node, no resource awareness
        # At this point, the only nodes without an allocation are the transfer nodes
        tta = TransferAndTensorAllocator(
            self.ssw,
            timeslots,
            accelerator=self.accelerator,
            iterations=self.iterations,
            ssis=self.ssis,
            multiplicities=multiplicities,
            mapping=self.mapping,
            cost_lut=self.cost_lut,
            nb_cols_to_use=self.nb_cols_to_use,
            output_path=self.output_path,
            backend=self.backend,
            constraint_selection=self.constraint_selection,
        )
        (
            tensor_reuse_levels,
            tensor_depths,
            tensor_allocations,
            transfer_allocations,
            memory_allocations,
            total_latency,
            overlap,
            latency_per_iteration,
        ) = tta.solve()
        # Capture solve statistics before tta goes out of scope (tta.model is a local variable)
        self.solve_stats = tta.model.solve_stats()
        # total, per_iter, ov = tsa_upd.compute_latency(iterations=self.iterations, offchip_core_id=offchip_core_id)
        # assert total == total_latency_solver, (
        #     f"Calculated total latency {total} does not match total latency from solver {total_latency_solver}."
        # )
        self.latency_total, self.latency_per_iteration, self.overlap_between_iterations = (
            total_latency,
            latency_per_iteration,
            overlap,
        )
        # Export Perfetto-compatible JSON traces of the solved schedule
        try:
            for compact, fname in [(True, "steady_state_trace_compact.json"), (False, "steady_state_trace.json")]:
                trace_path = export_steady_state_trace(
                    tta=tta,
                    iterations=self.iterations,
                    overlap=overlap,
                    latency_per_iteration=latency_per_iteration,
                    output_path=self.output_path,
                    compact=compact,
                    filename=fname,
                )
            logger.info("Steady-state schedule trace: %s", trace_path)
        except Exception as exc:  # never let a visualisation failure abort the run
            logger.warning("Failed to export steady-state trace (%s): %s", fname, exc)
        # Check that all nodes in the steady state workload have a chosen resource allocation
        # self.check_steady_state_workload_allocations(self.ssw)
        self.update_tensor_steady_state_iteration_spaces(tensor_reuse_levels)
        self.update_mapping_with_allocations(transfer_allocations, memory_allocations)
        self.ssw.visualize(os.path.join(self.output_path, "steady_state_workload_final.png"), self.mapping, self.ssis)
        # tla = TensorLifetimeAnalyzer(self.ssw)
        self.steady_state_workload = self.ssw
        return self.ssw

    def update_tensor_steady_state_iteration_spaces(self, tensor_reuse_levels: TensorReuseLevels):
        for t, ssis in self.ssis.items():
            if isinstance(t, Tensor):
                assert t in tensor_reuse_levels, f"Tensor {t.name} does not have a reuse level assigned."
                reuse_level = tensor_reuse_levels[t]
                for i, iv in enumerate(ssis.get_applicable_temporal_variables()):
                    if i <= reuse_level:
                        iv.reuse = Reuse.REUSE
                    else:
                        iv.reuse = Reuse.NO_REUSE
                for iv in ssis.variables:
                    if iv.type == IterationVariableType.SPATIAL and iv.applicable:
                        iv.reuse = Reuse.REUSE
        # Propagate spatial reuse across transfer boundaries: when one side of a
        # transfer has a SPATIAL variable that is represented as a TEMPORAL on the
        # other side (same dimension and size), mark that temporal as REUSE so that
        # both endpoints display the same reuse boundary.
        for node in self.ssw.get_transfer_nodes():
            for src in node.inputs:
                for dst in node.outputs:
                    self._propagate_spatial_reuse(src, dst)
                    self._propagate_spatial_reuse(dst, src)

    def _propagate_spatial_reuse(self, spatial_side: Tensor, temporal_side: Tensor) -> None:
        """Mark (spatio)temporal variables on ``temporal_side`` as REUSE when they
        match (dimension, size) of an applicable spatial variable on
        ``spatial_side``. Codegen renders both TEMPORAL and SPATIOTEMPORAL as ``t``,
        so either may stand in for a spatial distribution on the other endpoint."""
        if spatial_side not in self.ssis or temporal_side not in self.ssis:
            return
        spatial_keys_not_in_temporal = {
            (iv.dimension, iv.size)
            for iv in self.ssis[spatial_side].variables
            if iv.type == IterationVariableType.SPATIAL
            and iv.applicable
            and iv not in self.ssis[temporal_side].variables  # only look at temporal side vars that are not spatial
        }
        if not spatial_keys_not_in_temporal:
            return
        seen_spatial_keys = set()
        for iv in self.ssis[temporal_side].variables:
            if iv.applicable and iv.type in (
                IterationVariableType.TEMPORAL,
                IterationVariableType.SPATIOTEMPORAL,
            ):
                if (iv.dimension, iv.size) in spatial_keys_not_in_temporal and (
                    iv.dimension,
                    iv.size,
                ) not in seen_spatial_keys:
                    iv.reuse = Reuse.REUSE
                    seen_spatial_keys.add((iv.dimension, iv.size))

    def build_transfer_graph(self) -> Workload:
        new_nodes: dict[str, Node] = {node.name: node for node in self.workload.nodes}
        # Go through the tensors of the workload to find sources and destinations of the tensor
        for tensor in self.workload.tensors:
            srcs = [n for n in self.workload.nodes if isinstance(n, HasOutputs) and tensor in n.outputs]
            assert len(srcs) == 1, f"Expected exactly one source for tensor {tensor}, found {len(srcs)}"
            src = new_nodes[srcs[0].name]
            dsts = [new_nodes[n.name] for n in self.workload.nodes if isinstance(n, HasInputs) and tensor in n.inputs]
            is_constant_i_transfer = isinstance(src, InEdge)
            is_constant_o_transfer = any(isinstance(dst, OutEdge) for dst in dsts)
            if is_constant_i_transfer:
                self.add_two_transfer_nodes_for_constant_input_transfer(tensor, src, dsts, new_nodes)
            elif is_constant_o_transfer:
                self.add_two_transfer_nodes_for_constant_output_transfer(tensor, src, dsts, new_nodes)
            else:
                self.add_single_transfer_node_for_non_constant_transfer(tensor, src, dsts, new_nodes)
        new_workload = Workload(new_nodes.values())
        return new_workload

    def add_two_transfer_nodes_for_constant_input_transfer(
        self, tensor: Tensor, src: HasOutputs, dsts: list[HasInputs], new_nodes: dict[str, Node]
    ):
        """
        For constant transfers, we add two transfer nodes:
        - one from the source to the on-chip memory buffer,
        - a second one from the on-chip memory buffer to the destination.
        This is to ensure that the constant tensor is properly allocated in memory and can be reused across iterations.
        """
        assert isinstance(src, InEdge), f"Expected source of constant transfer to be an InEdge, found {type(src)}"
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

    def update_destination_node_inputs(self, tensor, src, new_nodes, dst, updated_tensor):
        # Find corresponding node in new_nodes as it might have already been updated
        dst_new = new_nodes[dst.name]
        # Update the dst input to the second transfer node
        assert len(src.outputs) == 1, "Src must have exactly one output tensor for index below."
        input_idx = dst_new.inputs.index(tensor)
        new_inputs = dst_new.inputs[:input_idx] + (updated_tensor,) + dst_new.inputs[input_idx + 1 :]
        if isinstance(dst_new, ComputationNode):
            new_dst = ComputationNode(
                type=dst_new.type,
                name=dst_new.name,
                inputs=new_inputs,
                outputs=dst_new.outputs,
                operand_mapping=dst_new.operand_mapping,
            )
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
        """
        assert len(dsts) == 1, "Currently only support single destination for constant output transfer."
        dst = dsts[0]
        assert isinstance(dst, OutEdge), (
            f"Expected destination of constant transfer to be an OutEdge, found {type(dst)}"
        )
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
        new_src = ComputationNode(
            type=src.type,
            name=src.name,
            inputs=src.inputs,
            outputs=new_outputs,
            operand_mapping=src.operand_mapping,
        )
        new_nodes[new_src.name] = new_src
        # Update the mapping entry for this new_src node to be the same as the original src node
        self.mapping.set(new_src, self.mapping.get(src))
        # Remove the original src node from the mapping as it has been updated with new outputs
        self.mapping.remove(src)
        return new_src

    def add_single_transfer_node_for_non_constant_transfer(
        self, tensor: Tensor, src: HasOutputs, dsts: list[HasInputs], new_nodes: dict[str, Node]
    ):
        """
        For non-constant transfers, we add a single transfer node from the source to the destinations.
        """
        transfer_type = self.determine_transfer_type(src, dsts)
        if transfer_type == TransferType.NONE:
            return
        transfer_node, updated_tensors = self.generate_transfer_node(
            dsts, tensor, transfer_type, out_name=f"{tensor.name}_1"
        )
        new_nodes[transfer_node.name] = transfer_node
        for dst, updated_tensor in zip(dsts, updated_tensors, strict=True):
            # Find corresponding node in new_nodes as it might have already been updated
            dst_new = new_nodes[dst.name]
            # Update the dst input to the transfer node
            assert len(src.outputs) == 1, "Src must have exactly one output tensor for index below."
            input_idx = dst_new.inputs.index(tensor)
            new_inputs = dst_new.inputs[:input_idx] + (updated_tensor,) + dst_new.inputs[input_idx + 1 :]
            if isinstance(dst_new, ComputationNode):
                new_dst = ComputationNode(
                    type=dst_new.type,
                    name=dst_new.name,
                    inputs=new_inputs,
                    outputs=dst_new.outputs,
                    operand_mapping=dst_new.operand_mapping,
                )
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

    def update_fusion_splits(self) -> dict[LayerDim, int]:
        # Update the fusion_splits based on the new workload with transfer nodes
        updated_fusion_splits = {}
        for dim, size in self.fusion_splits.items():
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

    def update_mapping_with_allocations(
        self,
        transfer_allocations: TransferAlloc,
        memory_allocations: MemoryAlloc,
    ):
        for tr, alloc in transfer_allocations.items():
            if tr in memory_allocations:
                assert isinstance(memory_allocations[tr], tuple)
                memory_allocation = memory_allocations[tr]
            else:
                memory_allocation = tuple()
            self.mapping.set_for_node(
                tr,
                resource_allocation=(alloc,),
                inter_core_tiling=tuple(),
                memory_allocation=memory_allocation,
            )
        for tr in self.ssw.get_transfer_nodes():
            assert len(self.mapping.get(tr).resource_allocation) == 1, (
                f"Transfer node {tr.name} should have exactly one resource allocation after update."
            )

    def update_cost_lut(self):
        # The new workload contains same computation node names but with different input tensors
        for new_node in self.ssw.get_computation_nodes():
            old_node = next(n for n in self.cost_lut.get_nodes() if n.name == new_node.name)
            self.cost_lut.replace_node(old_node, new_node)
        return self.cost_lut

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
                tensor_ssis = self.generate_tensor_ssis(workload, tensor, succ, ssis)
                ssis[tensor] = tensor_ssis
        # Generate the new tensor SSIS of node outputs
        for node in workload.get_iteration_space_nodes():
            for tensor in node.outputs:
                assert tensor not in ssis, (
                    f"Tensor {tensor.name} already has an SSIS, cannot assign the same tensor multiple SSIS."
                )
                tensor_ssis = self.generate_tensor_ssis(workload, tensor, node, ssis)
                ssis[tensor] = tensor_ssis
        return ssis

    def generate_tensor_ssis(
        self,
        workload: Workload,
        tensor: Tensor,
        node: HasIterationSpace,
        ssis: dict[HasIterationSpace | Tensor, SteadyStateIterationSpace],
    ) -> SteadyStateIterationSpace:
        producer_ssis = ssis.get(node, None)
        if producer_ssis is None:
            raise KeyError(f"Node {node.name} does not have a valid producer SSIS.")
        tensor_dims = workload.get_tensor_dimensions(tensor)
        tensor_ivs = []
        for prod_iv in producer_ssis.variables:
            prod_iv_dim = prod_iv.dimension
            if prod_iv_dim in tensor_dims:
                tensor_effect = LoopEffect.VARYING
            if prod_iv_dim not in tensor_dims:
                tensor_effect = LoopEffect.ABSENT if prod_iv.effect == LoopEffect.ABSENT else LoopEffect.INVARIANT
            tensor_ivs.append(
                IterationVariable(
                    dimension=prod_iv_dim,
                    size=prod_iv.size,
                    type=prod_iv.type,
                    effect=tensor_effect,
                )
            )
        tensor_ssis = SteadyStateIterationSpace(variables=tuple(tensor_ivs))
        return tensor_ssis

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
        else:
            possible_memory_cores_set: set[Core] = set()
            for dst in dsts:
                assert len(self._retrieve_core_allocation(dst)) == 1, "TODO: Support multiple compute allocations."
                possible_memory_cores_set.update(self._retrieve_core_allocation(dst)[0])
            possible_memory_cores = (tuple(sorted(possible_memory_cores_set, key=lambda x: x.id)),)
        return possible_memory_cores

    def determine_possible_inter_core_tiling(
        self, node: TransferNode, possible_dst_allocs: tuple[tuple[Core, ...], ...], dsts: tuple[HasInputs, ...]
    ) -> tuple[InterCoreTiling, ...]:
        possible_inter_core_tiling = []
        for dst_allocs in possible_dst_allocs:
            nb_cores = len(dst_allocs)
            if nb_cores == 1:
                possible_inter_core_tiling.append(tuple())
            else:
                # For now, we only support a single destination with one tiling possibility
                dst = dsts[0]
                try:
                    dst_tiling = tuple(self.ssw.get_unique_dims_inter_core_tiling(dst, self.mapping))
                except KeyError:
                    dst_tiling = self.get_inter_core_tiling_for_mem_allocations(node, dst_allocs)
                possible_inter_core_tiling.append(dst_tiling)
        return tuple(possible_inter_core_tiling)

    def get_inter_core_tiling_for_mem_allocations(
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
        for tiling_loop in compute_tiling:
            _, size = tiling_loop
            if size == len(dst_allocs):
                return tiling_loop
        # No size with exact match found, try to find one that is a multiple of the number of dst allocs
        for tiling_loop in compute_tiling:
            dim, size = tiling_loop
            if size % len(dst_allocs) == 0:
                return (dim, len(dst_allocs))
        raise ValueError(f"No matching tiling found for compute tiling {compute_tiling} and dst allocs {dst_allocs}")

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

    def calculate_multiplicities(self) -> dict[ComputationNode, int]:
        """Calculate the multiplicity of each computation node in the steady state workload."""
        multiplicities = {}
        for node, ssis in self.ssis.items():
            total_iterations = prod(ssis.get_temporal_sizes())
            multiplicities[node] = total_iterations // self.iterations
        return multiplicities

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
            src_type = self._determine_allocation_type(src_allocation[0])
        if dst_type is None:
            dst_allocations = [self._retrieve_core_allocation(dst)[0] for dst in dsts]
            dst_type = self._determine_allocation_type(
                [alloc for dst_alloc in dst_allocations for alloc in dst_alloc]
            )  # flatten the list of dst allocations
        if src_type == "compute" and dst_type == "compute":
            return TransferType.COMPUTE_TO_COMPUTE
        elif src_type == "compute" and dst_type in ("memory", "shim", "offchip"):
            return TransferType.COMPUTE_TO_MEM
        elif src_type in ("memory", "shim", "offchip") and dst_type == "compute":
            return TransferType.MEM_TO_COMPUTE
        elif src_type in ("memory", "shim", "offchip") and dst_type in ("memory", "shim", "offchip"):
            return TransferType.MEM_TO_MEM
        raise ValueError(f"Unsupported transfer type from {src_type} to {dst_type}")

    def _determine_allocation_type(self, allocs: list[Core]) -> str:
        alloc_types = set(alloc.type for alloc in allocs)
        if len(alloc_types) != 1:
            raise ValueError(f"Expected all allocations to be of the same type, found {alloc_types}")
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
        required_nb_memory_cores = ceil(
            total_relevant_unrolling / MAX_RELEVANT_FACTOR_PER_TRANSFER_TYPE[node.transfer_type]
        )
        all_mem_cores = self._get_accelerator_memory_cores()
        candidates = [tuple(combo) for combo in combinations(all_mem_cores, required_nb_memory_cores)]
        return tuple(candidates)
