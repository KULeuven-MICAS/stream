import os
from functools import reduce
from math import prod
from typing import cast

from xdsl.ir.affine import AffineMap

# if TYPE_CHECKING:
from stream.cost_model.core_cost_lut import CoreCostLUT
from stream.datatypes import LayerDim
from stream.hardware.architecture.accelerator import Accelerator
from stream.hardware.architecture.core import Core
from stream.hardware.architecture.noc.communication_link import CommunicationLink
from stream.mapping.mapping import Mapping
from stream.opt.allocation.constraint_optimization.transfer_and_tensor_allocation import TransferAndTensorAllocator
from stream.workload.steady_state.computation import SteadyStateComputation
from stream.workload.steady_state.iteration_space import IterationVariable, SteadyStateIterationSpace
from stream.workload.steady_state.node import Node
from stream.workload.utils import generate_steady_state_iteration_spaces
from stream.workload.workload import (
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
    Workload,
)


class SteadyStateScheduler:
    def __init__(
        self,
        workload: Workload,
        accelerator: "Accelerator",
        mapping: Mapping,
        tiled_dimensions: dict[str, tuple[int, int]],
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
        self.tiled_dimensions = tiled_dimensions
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

    def run(self):
        """
        Run the steady state scheduler on the given workload.

        Returns:
            TimeSlotAllocation: The scheduled workload.
        """
        # Update the workload graph to include transfer nodes
        ssw = self.build_transfer_graph()
        # Save the new workload with transfers
        ssw.visualize_to_file(os.path.join(self.output_path, "tiled_workload_with_transfers.png"))
        # Update the mapping for the new workload graph
        self.mapping = self.update_mapping(ssw)
        # Update the cost lut for the new workload graph
        self.cost_lut = self.update_cost_lut(ssw)
        # Update the steady state iteration spaces to include transfer nodes
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
        )
        tensor_allocations, transfer_allocations, memory_allocations, total_latency_solver = tta.solve()
        print(tsa_upd)
        offchip_core_id = self.accelerator.offchip_core_id
        total, per_iter, ov = tsa_upd.compute_latency(iterations=self.iterations, offchip_core_id=offchip_core_id)
        # assert total == total_latency_solver, (
        #     f"Calculated total latency {total} does not match total latency from solver {total_latency_solver}."
        # )
        print(f"Total latency: {total}, per iteration: {per_iter}, overlap: {ov}")
        self.latency_total, self.latency_per_iteration, self.overlap_between_iterations = total, per_iter, ov
        # Check that all nodes in the steady state workload have a chosen resource allocation
        self.check_steady_state_workload_allocations(ssw)
        ssw.visualize_to_file(os.path.join(self.output_path, "steady_state_workload_final.png"))
        # tla = TensorLifetimeAnalyzer(ssw)
        # tla.summary()
        # tla.visualize()
        self.steady_state_workload = ssw_upd
        return self

    def build_transfer_graph(self) -> Workload:
        new_nodes: dict[str, Node] = {node.name: node for node in self.workload.nodes}
        # Go through the tensors of the workload to find sources and destinations of the tensor
        for tensor in self.workload.tensors:
            srcs = [n for n in self.workload.nodes if isinstance(n, HasOutputs) and tensor in n.outputs]
            assert len(srcs) == 1, f"Expected exactly one source for tensor {tensor}, found {len(srcs)}"
            src = srcs[0]
            dsts = [n for n in self.workload.nodes if isinstance(n, HasInputs) and tensor in n.inputs]
            transfer_type = self.determine_transfer_type(src, dsts)
            if transfer_type != TransferType.NONE:
                transfer_node, updated_tensors = self.generate_transfer_node(dsts, tensor, transfer_type)
                new_nodes[transfer_node.name] = transfer_node
                new_dsts = []
                for dst, updated_tensor in zip(dsts, updated_tensors, strict=True):
                    # Find corresponding node in new_nodes as it might have already been updated
                    dst = new_nodes[dst.name]
                    # Update the dst input to the transfer node
                    assert len(src.outputs) == 1, "Src must have exactly one output tensor for index below."
                    input_idx = dst.inputs.index(src.outputs[0])
                    new_inputs = dst.inputs[:input_idx] + (updated_tensor,) + dst.inputs[input_idx + 1 :]
                    if isinstance(dst, ComputationNode):
                        new_dst = ComputationNode(
                            name=dst.name,
                            inputs=new_inputs,
                            outputs=dst.outputs,
                            operand_mapping=dst.operand_mapping,
                        )
                    elif isinstance(dst, OutEdge):
                        new_dst = OutEdge(
                            name=dst.name,
                            inputs=new_inputs,
                        )
                    else:
                        raise ValueError(f"Unexpected dst node type: {type(dst)}")
                    new_nodes[dst.name] = new_dst
                    new_dsts.append(new_dst)
                # Update the transfer dsts to the new dsts
                transfer_node = self.generate_transfer_node(new_dsts, tensor, transfer_type)
        new_workload = Workload(new_nodes.values())
        return new_workload

    def update_mapping(self, new_workload: Workload):
        # Computation node replacement with new computation nodes
        old_nodes = {node.name: node for node in self.workload.get_computation_nodes()}
        for new_node in new_workload.get_computation_nodes():
            if new_node.name in old_nodes:
                old_node = old_nodes[new_node.name]
                self.mapping.set(new_node, self.mapping.get(old_node))
                self.mapping.remove(old_node)
        # Transfer node mappings
        for node in new_workload.get_transfer_nodes():
            assert len(node.inputs) == 1, "Transfer node must have exactly one input tensor."
            src = cast(HasOutputs, list(new_workload.predecessors(node))[0])
            dsts = tuple(cast(HasInputs, n) for n in new_workload.successors(node))
            self.update_mapping_for_transfer(node, src, dsts)
        return self.mapping

    def update_cost_lut(self, new_workload: Workload):
        # The new workload contains same computation node names but with different input tensors
        for new_node in new_workload.get_computation_nodes():
            old_node = next(n for n in self.cost_lut.get_nodes() if n.name == new_node.name)
            self.cost_lut.replace_node(old_node, new_node)
        return self.cost_lut

    def generate_transfer_node(
        self, dsts: list[HasInputs], tensor: Tensor, transfer_type: TransferType
    ) -> tuple[TransferNode, list[Tensor]]:
        transfer_outputs = self.generate_transfer_output_tensors(tensor, dsts)
        operand_mapping = tuple(AffineMap.identity(len(tensor.shape)) for _ in range(1 + len(dsts)))
        transfer_node = TransferNode(
            name=f"Transfer({tensor.name})",
            inputs=(tensor,),
            outputs=tuple(transfer_outputs),
            transfer_type=transfer_type,
            operand_mapping=operand_mapping,
        )
        return transfer_node, transfer_outputs

    def generate_transfer_output_tensors(self, tensor: Tensor, dsts: list[HasInputs]) -> list[Tensor]:
        transfer_outputs = []
        for i, _ in enumerate(dsts):
            transfer_output = Tensor(
                name=f"{tensor.name}.{i}",
                operand_type=tensor.operand_type,
                shape=tensor.shape,
                subview=tensor.subview,
            )
            transfer_outputs.append(transfer_output)
        return transfer_outputs

    def generate_ssis(self, workload: Workload) -> dict[HasIterationSpace, SteadyStateIterationSpace]:
        fuse_dimensions = self.generate_fuse_dimensions(workload)
        ssis = generate_steady_state_iteration_spaces(
            workload,
            self.mapping,
            fuse_dimensions,
        )
        return ssis

    def generate_fuse_dimensions(self, workload: Workload):
        fuse_dimensions: dict[LayerDim, int] = {}
        for node_name, (dim_idx, size) in self.tiled_dimensions.items():
            node = next(n for n in workload.get_computation_nodes() if n.name == node_name)
            dim = workload.get_dims(node)[dim_idx]
            fuse_dimensions[dim] = size
        return fuse_dimensions

    def get_updated_ssis(
        self, ssis: tuple[IterationVariable, ...], relevant_tensor_dims: list[int]
    ) -> SteadyStateIterationSpace:
        updated_ivs = []
        for iv in ssis:
            updated_ivs.append(
                IterationVariable(
                    dimension=iv.dimension,
                    size=iv.size,
                    relevant=iv.dimension in relevant_tensor_dims,
                )
            )
        return SteadyStateIterationSpace(updated_ivs)

    def update_mapping_for_transfer(self, node: TransferNode, src: HasOutputs, dsts: tuple[HasInputs, ...]) -> None:
        possible_memory_cores = self.determine_possible_memory_allocations(src, dsts)
        possible_allocations = self.determine_possible_transfer_allocations(src, dsts, possible_memory_cores)
        self.mapping.set_for_node(
            node,
            core_allocation=possible_allocations,
            inter_core_tiling=tuple(),
            memory_allocation=possible_memory_cores,
        )

    def determine_possible_memory_allocations(self, src: HasOutputs, dsts: tuple[HasInputs, ...]) -> tuple[Core, ...]:
        is_constant_transfer = isinstance(src, InEdge) or any(isinstance(dst, OutEdge) for dst in dsts)
        if is_constant_transfer:
            possible_memory_cores = self._get_possible_memory_core_allocations()
        else:
            possible_memory_cores = tuple()
        return possible_memory_cores

    def determine_possible_transfer_allocations(
        self, src: HasOutputs, dsts: tuple[HasInputs, ...], possible_memory_cores: tuple[Core, ...]
    ) -> tuple[tuple[CommunicationLink, ...], ...]:
        src_allocs = self.retrieve_node_allocation(src)
        dst_allocs = {x for dst_allocs in [(self.retrieve_node_allocation(dst)) for dst in dsts] for x in dst_allocs}

        possible_resource_allocations = self.accelerator.communication_manager.get_possible_resource_allocations(
            src_allocs=src_allocs,
            dst_allocs=dst_allocs,
            possible_memory_cores=possible_memory_cores,
        )
        return possible_resource_allocations

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

    def retrieve_node_allocation(self, node: Node) -> list[Core]:
        if isinstance(node, InEdge):
            return [
                self.accelerator.get_core(self.accelerator.offchip_core_id),
            ]
        if isinstance(node, OutEdge):
            return [
                self.accelerator.get_core(self.accelerator.offchip_core_id),
            ]
        if isinstance(node, HasOutputs):
            return self.mapping.get(node).core_allocation
        raise ValueError(f"Unexpected source node type: {type(node)}")

    def determine_transfer_type(self, src: HasOutputs, dsts: tuple[HasInputs, ...]) -> TransferType:
        """TODO: Fix the transfer type determination logic to handle all dsts together at the same time."""
        src_allocation = self.retrieve_node_allocation(src)
        dst_allocations = [self.retrieve_node_allocation(dst) for dst in dsts]
        if len(dsts) == 1 and src_allocation == dst_allocations[0]:
            return TransferType.NONE  # No transfer needed if src and dst are the same
        _, all_dims = self.workload.unique_dimensions()
        transfer_types = []
        for dst in dsts:
            if isinstance(src, ComputationNode):
                tensor_operand_mapping = src.operand_mapping[-1]
                global_mapping = self.workload.global_mapping(src, tensor_operand_mapping)
            else:
                assert isinstance(dst, ComputationNode), "Either src or dst must be a ComputationNode"
                assert len(src.outputs) == 1, "Src must have exactly one output tensor for index below"
                tensor_operand_mapping = dst.operand_mapping[dst.inputs.index(src.outputs[0])]
                global_mapping = self.workload.global_mapping(dst, tensor_operand_mapping)
            # Find the unique dimensions of the tensor being transfered
            tensor_dims = [all_dims[expr.position] for expr in global_mapping.results]
            if isinstance(src, InEdge):
                src_inter_core_tiling = []
            else:
                assert isinstance(src, ComputationNode), "Src should be ComputationNode or InEdge"
                src_inter_core_tiling = self.workload.get_unique_dims_inter_core_tiling(src, self.mapping)
            src_inter_core_tiling_dims = [dim for dim, _ in src_inter_core_tiling]
            if isinstance(dst, OutEdge):
                dst_inter_core_tiling = []
            else:
                assert isinstance(dst, ComputationNode), "Dst should be ComputationNode or OutEdge"
                dst_inter_core_tiling = self.workload.get_unique_dims_inter_core_tiling(dst, self.mapping)
            dst_inter_core_tiling_dims = [dim for dim, _ in dst_inter_core_tiling]
            # If the src has an inter_core_tiling with a dim not in the tensor dim, we need to reduce it
            if any(dim not in tensor_dims for dim in src_inter_core_tiling_dims):
                transfer_types.append(TransferType.REDUCE)
            # If dst tiling is equal to src tiling, it's a unicast
            elif dst_inter_core_tiling == src_inter_core_tiling:
                transfer_types.append(TransferType.UNICAST)
            elif len(src_allocation) > 1:
                transfer_types.append(TransferType.JOIN)

            # If the dst has an inter_core_tiling with all dims in the tensor dim,
            if (
                any(dim in tensor_dims for dim in dst_inter_core_tiling_dims)
                and TransferType.UNICAST not in transfer_types
            ):
                transfer_types.append(TransferType.DISTRIBUTE)
            if (
                any(dim not in tensor_dims for dim in dst_inter_core_tiling_dims)
                and TransferType.UNICAST not in transfer_types
            ):
                transfer_types.append(TransferType.BROADCAST)
        return reduce(lambda a, b: a | b, transfer_types)

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

    def _get_possible_memory_core_allocations(self) -> tuple[Core, ...]:
        all_mem_cores = self._get_accelerator_memory_cores()
        candidates = [mem_core for mem_core in all_mem_cores if not mem_core.id == self.accelerator.offchip_core_id]
        return tuple(candidates)
