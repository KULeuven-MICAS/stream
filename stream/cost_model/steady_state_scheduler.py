import logging
import os
from itertools import combinations
from math import ceil, prod
from typing import cast

from xdsl.ir.affine import AffineMap

# if TYPE_CHECKING:
from stream.cost_model.communication_manager import MulticastPathPlan
from stream.cost_model.core_cost_lut import CoreCostLUT
from stream.datatypes import LayerDim
from stream.hardware.architecture.accelerator import Accelerator
from stream.hardware.architecture.core import Core
from stream.mapping.mapping import Mapping
from stream.opt.allocation.constraint_optimization.transfer_and_tensor_allocation import (
    MemoryAlloc,
    TensorAlloc,
    TensorReuseLevels,
    TransferAlloc,
    TransferAndTensorAllocator,
)
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
from stream.workload.steady_state.iteration_space import Reuse, SteadyStateIterationSpace
from stream.workload.utils import generate_steady_state_iteration_spaces, get_equivalent_dimension
from stream.workload.workload import Workload

logger = logging.getLogger(__name__)


class SteadyStateScheduler:
    def __init__(
        self,
        workload: Workload,
        accelerator: "Accelerator",
        mapping: Mapping,
        fusion_splits: dict[LayerDim, int],
        cost_lut: CoreCostLUT,
        nb_cols_to_use: int = 4,
        output_path: str = "",
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

        # Cost model parameters
        self.latency_total = -1
        self.latency_per_iteration = -1
        self.overlap_between_iterations = -1

        self.nb_cols_to_use = nb_cols_to_use

        self.output_path = output_path
        if self.output_path:
            os.makedirs(self.output_path, exist_ok=True)

    def run(self) -> Workload:
        """
        Run the steady state scheduler on the given workload.

        Returns:
            TimeSlotAllocation: The scheduled workload.
        """
        # Update the workload graph to include transfer nodes
        ssw = self.build_transfer_graph()
        # Update the fusion_splits based on the new workload with transfer nodes
        self.fusion_splits = self.update_fusion_splits(ssw)
        # Save the new workload with transfers
        ssw.visualize(os.path.join(self.output_path, "tiled_workload_with_transfers.png"))
        # Update the mapping for the new workload graph
        self.mapping = self.update_mapping(ssw)
        # Update the cost lut for the new workload graph
        self.cost_lut = self.update_cost_lut(ssw)
        # Update the steady state iteration spaces to include transfer nodes and tensors
        self.ssis = self.generate_ssis(ssw)
        # Calculate the number of iterations based on the steady state iteration spaces
        self.iterations = self.calculate_iterations()
        # Calculate the multiplicity of each node's execution in the steady state workload
        multiplicities = self.calculate_multiplicities()
        # Get the timeslots for all nodes
        timeslots = ssw.get_timeslots()
        # At this point, the only nodes without an allocation are the transfer nodes
        tta = TransferAndTensorAllocator(
            ssw,
            timeslots,
            accelerator=self.accelerator,
            iterations=self.iterations,
            ssis=self.ssis,
            multiplicities=multiplicities,
            mapping=self.mapping,
            cost_lut=self.cost_lut,
            nb_cols_to_use=self.nb_cols_to_use,
            output_path=self.output_path,
        )
        (
            tensor_reuse_levels,
            tensor_allocations,
            transfer_allocations,
            memory_allocations,
            total_latency,
            overlap,
            latency_per_iteration,
        ) = tta.solve()
        # total, per_iter, ov = tsa_upd.compute_latency(iterations=self.iterations, offchip_core_id=offchip_core_id)
        # assert total == total_latency_solver, (
        #     f"Calculated total latency {total} does not match total latency from solver {total_latency_solver}."
        # )
        print(
            f"Total latency: {total_latency} "
            f"Latency per iteration: {latency_per_iteration} "
            f"Overlap: {overlap} "
            f" Iterations: {self.iterations}"
        )
        self.latency_total, self.latency_per_iteration, self.overlap_between_iterations = (
            total_latency,
            latency_per_iteration,
            overlap,
        )
        # Export Perfetto-compatible JSON trace of the solved schedule
        try:
            trace_path = export_steady_state_trace(
                tta=tta,
                iterations=self.iterations,
                overlap=overlap,
                latency_per_iteration=latency_per_iteration,
                output_path=self.output_path,
            )
            logger.info("Steady-state schedule trace: %s", trace_path)
        except Exception as exc:  # never let a visualisation failure abort the run
            logger.warning("Failed to export steady-state trace: %s", exc)
        # Check that all nodes in the steady state workload have a chosen resource allocation
        # self.check_steady_state_workload_allocations(ssw)
        self.update_tensor_steady_state_iteration_spaces(ssw, tensor_reuse_levels)
        self.update_mapping_with_allocations(ssw, tensor_allocations, transfer_allocations, memory_allocations)
        ssw.visualize(os.path.join(self.output_path, "steady_state_workload_final.png"), self.mapping, self.ssis)
        # tla = TensorLifetimeAnalyzer(ssw)
        self.steady_state_workload = ssw
        return ssw

    def update_tensor_steady_state_iteration_spaces(self, ssw: Workload, tensor_reuse_levels: TensorReuseLevels):
        for node in ssw.nodes:
            if isinstance(node, TransferNode):
                for tensor in node.outputs:
                    reuse_level = tensor_reuse_levels[tensor]
                    ssis = self.ssis.get(tensor, None)
                    if ssis is None:
                        raise KeyError(f"{tensor} not found in ssis.")
                    for i, iv in enumerate(ssis.get_applicable_temporal_variables()):
                        if i <= reuse_level:
                            iv.reuse = Reuse.REUSE
                        else:
                            iv.reuse = Reuse.NO_REUSE

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
        pass
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
        # Update the mapping entry for this new_dst node to be the same as the original dst node and remove original dst node
        self.mapping.set(new_dst, self.mapping.get(dst))
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
        # Update the mapping entry for this new_src node to be the same as the original src node and remove original src node
        self.mapping.set(new_src, self.mapping.get(src))
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
            # Update the mapping entry for this new_dst node to be the same as the original dst node and remove original dst node
            self.mapping.set(new_dst, self.mapping.get(dst))
            self.mapping.remove(dst)

    def update_fusion_splits(self, new_workload: Workload) -> dict[LayerDim, int]:
        # Update the fusion_splits based on the new workload with transfer nodes
        updated_fusion_splits = {}
        for dim, size in self.fusion_splits.items():
            new_dim = get_equivalent_dimension(self.workload, new_workload, dim)
            updated_fusion_splits[new_dim] = size
        return updated_fusion_splits

    def update_mapping(self, new_workload: Workload):
        # Add transfer node mappings
        for node in new_workload.get_transfer_nodes():
            assert len(node.inputs) == 1, "Transfer node must have exactly one input tensor."
            src = cast(HasOutputs, list(new_workload.predecessors(node))[0])
            dsts = tuple(cast(HasInputs, n) for n in new_workload.successors(node))
            self.update_mapping_for_transfer(node, src, dsts)
        return self.mapping.with_updated_workload(new_workload, self.workload)  # updates FusedGroups

    def update_mapping_with_allocations(
        self,
        ssw: Workload,
        tensor_allocations: TensorAlloc,
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
        for tr in ssw.get_transfer_nodes():
            assert len(self.mapping.get(tr).resource_allocation) == 1, (
                f"Transfer node {tr.name} should have exactly one resource allocation after update."
            )

    def update_cost_lut(self, new_workload: Workload):
        # The new workload contains same computation node names but with different input tensors
        for new_node in new_workload.get_computation_nodes():
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

    def generate_ssis(self, workload: Workload) -> dict[HasIterationSpace | Tensor, SteadyStateIterationSpace]:
        ssis = generate_steady_state_iteration_spaces(
            workload,
            self.mapping,
            self.fusion_splits,
        )
        ssis = self.update_tensor_ssis(workload, ssis)
        return ssis

    def update_tensor_ssis(
        self, workload: Workload, ssis: dict[HasIterationSpace | Tensor, SteadyStateIterationSpace]
    ) -> dict[HasIterationSpace | Tensor, SteadyStateIterationSpace]:
        # Generate the new tensors
        for transfer in workload.get_transfer_nodes():
            transfer_ssis = ssis[transfer]
            for tensor in transfer.tensors:
                # TODO: Change the transfer ssis depending on input/output tensor differences
                ssis[tensor] = transfer_ssis
        return ssis

    def update_mapping_for_transfer(self, node: TransferNode, src: HasOutputs, dsts: tuple[HasInputs, ...]) -> None:
        possible_dst_allocs = self.determine_possible_memory_allocations(node, src, dsts)
        possible_allocations = self.determine_possible_transfer_plans(src, possible_dst_allocs)
        self.mapping.set_for_node(
            node,
            resource_allocation=possible_allocations,
            inter_core_tiling=((),),
            memory_allocation=possible_dst_allocs,
        )

    def determine_possible_memory_allocations(
        self, node: TransferNode, src: HasOutputs, dsts: tuple[HasInputs, ...]
    ) -> tuple[tuple[Core, ...], ...]:
        """
        Determine the memory allocation of the transfer node.
        If the transfer is a X-to-memory transfer, then the possible memory allocations are the possible memory cores.
        Else, the possible memory allocations are determined by the destination nodes' allocations.
        """
        if node.transfer_type in (TransferType.MEM_TO_MEM,) and isinstance(src, InEdge):
            possible_memory_cores = self._get_possible_memory_core_allocations(src)
        elif node.transfer_type in (TransferType.MEM_TO_MEM,) and any(isinstance(dst, OutEdge) for dst in dsts):
            assert len(dsts) == 1, "Currently only support single destination for constant output transfer."
            dst = dsts[0]
            possible_memory_cores = self._retrieve_core_allocation(dst)
        elif node.transfer_type in (TransferType.COMPUTE_TO_MEM,):
            possible_memory_cores = self._get_possible_memory_core_allocations(src)
        else:
            possible_memory_cores_set: set[Core] = set()
            for dst in dsts:
                assert len(self._retrieve_core_allocation(dst)) == 1, "TODO: Support multiple compute allocations."
                possible_memory_cores_set.update(self._retrieve_core_allocation(dst)[0])
            possible_memory_cores = (tuple(sorted(possible_memory_cores_set, key=lambda x: x.id)),)
        return possible_memory_cores

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
        elif src_type == "compute" and dst_type == "memory":
            return TransferType.COMPUTE_TO_MEM
        elif src_type == "memory" and dst_type == "compute":
            return TransferType.MEM_TO_COMPUTE
        elif src_type == "memory" and dst_type == "memory":
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

    def _get_possible_memory_core_allocations(self, src: HasOutputs) -> tuple[tuple[Core, ...], ...]:
        MAX_NB_SRC_ALLOCATION_PER_MEM_CORE = 4
        nb_source_allocations = len(self._retrieve_core_allocation(src)[0])
        required_nb_memory_cores = ceil(nb_source_allocations / MAX_NB_SRC_ALLOCATION_PER_MEM_CORE)
        all_mem_cores = self._get_accelerator_memory_cores()
        candidates = [tuple(combo) for combo in combinations(all_mem_cores, required_nb_memory_cores)]
        return tuple(candidates)
