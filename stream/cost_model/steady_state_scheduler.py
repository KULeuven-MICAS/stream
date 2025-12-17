import os
from collections import defaultdict

import networkx as nx

# if TYPE_CHECKING:
from stream.cost_model.communication_manager import MulticastRequest
from stream.hardware.architecture.accelerator import Accelerator
from stream.hardware.architecture.core import Core
from stream.hardware.architecture.noc.communication_link import CommunicationLink
from stream.opt.allocation.constraint_optimization.timeslot_allocation import TimeSlotAllocation
from stream.opt.allocation.constraint_optimization.transfer_and_tensor_allocation import TransferAndTensorAllocator
from stream.utils import CostModelEvaluationLUT
from stream.workload.computation.computation_node import ComputationNode
from stream.workload.onnx_workload import ComputationNodeWorkload
from stream.workload.steady_state.computation import SteadyStateComputation
from stream.workload.steady_state.iteration_space import SteadyStateIterationSpace
from stream.workload.steady_state.node import SteadyStateNode
from stream.workload.steady_state.rolling_buffer import SteadyStateRollingBuffer
from stream.workload.steady_state.tensor import SteadyStateTensor, TensorFlag
from stream.workload.steady_state.transfer import SteadyStateTransfer, TransferType
from stream.workload.steady_state.workload import SteadyStateWorkload
from stream.workload.utils import get_real_in_edges, get_real_out_edges


class SteadyStateScheduler:
    def __init__(
        self,
        workload: "ComputationNodeWorkload",
        accelerator: "Accelerator",
        original_workload: "ComputationNodeWorkload",
        cost_lut: CostModelEvaluationLUT,
        iterations: int,
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
        self.original_workload = original_workload
        self.cost_lut = cost_lut
        self.iterations = iterations
        self.current_node_id = 0
        self.partitioned_nodes: dict[ComputationNode, list[SteadyStateComputation]] = {}
        self.constant_tensors: dict[int, SteadyStateTensor] = {}

        # Steady state workload that will be set after running the scheduler
        self.steady_state_workload: SteadyStateWorkload | None = None

        # Cost model parameters
        self.latency_total = -1
        self.latency_per_iteration = -1
        self.overlap_between_iterations = -1

        self.allow_constant_tensors_on_mem_core = False
        self.allow_constant_tensors_on_compute_core = False

        self.nb_cols_to_use = nb_cols_to_use

        self.output_path = output_path

    def run(self, allocation: "TimeSlotAllocation"):
        """
        Run the steady state scheduler on the given allocation.

        Args:
            allocation (TimeSlotAllocation): The allocation to be scheduled.

        Returns:
            TimeSlotAllocation: The scheduled allocation.
        """
        # Get the subgraph of the workload that is relevant for the steady state allocation
        ssw = self.prepare_graph(allocation)
        # Convert to TimeSlotAllocation with fixed timeslots for all nodes
        tsa = ssw.to_timeslotallocation()
        # At this point, the only nodes without an allocation are the transfer nodes
        tta = TransferAndTensorAllocator(
            ssw, tsa, accelerator=self.accelerator, iterations=self.iterations, nb_cols_to_use=self.nb_cols_to_use
        )
        tsa_upd, ssw_upd, total_latency_solver = tta.solve()
        print(tsa_upd)
        offchip_core_id = self.accelerator.offchip_core_id
        total, per_iter, ov = tsa_upd.compute_latency(iterations=self.iterations, offchip_core_id=offchip_core_id)
        assert total == total_latency_solver, (
            f"Calculated total latency {total} does not match total latency from solver {total_latency_solver}."
        )
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

    def prepare_graph(self, allocation: "TimeSlotAllocation") -> SteadyStateWorkload:
        steady_state_subgraph = self.get_workload_subgraph(allocation)
        # Create a new SteadyStateWorkload to hold the scheduled nodes, tensors and transfers
        ssw = SteadyStateWorkload()
        # Add all computation nodes from the subgraph to the SteadyStateWorkload
        ssw = self.add_computation_nodes(ssw, steady_state_subgraph, allocation)
        ssw.visualize_to_file(os.path.join(self.output_path, "steady_state_workload_0.png"))
        # Add the ConstantTensorNodes to the SteadyStateWorkload
        ssw = self.add_constant_tensor_nodes(ssw, steady_state_subgraph)
        ssw.visualize_to_file(os.path.join(self.output_path, "steady_state_workload_1.png"))
        # Add the non-constant tensors from this iteration to the SteadyStateWorkload
        ssw = self.add_this_iteration_nonconstant_tensor_nodes(ssw)
        ssw.visualize_to_file(os.path.join(self.output_path, "steady_state_workload_2.png"))
        # Add the TransferNodes to the SteadyStateWorkload
        ssw = self.add_transfer_nodes(ssw)
        ssw.visualize_to_file(os.path.join(self.output_path, "steady_state_workload_3.png"))
        # Bufferize the non-constant steady state tensors
        ssw = self.bufferize_nonconstant_tensors(ssw)
        ssw.visualize_to_file(os.path.join(self.output_path, "steady_state_workload_4.png"))
        return ssw

    def get_workload_subgraph(self, allocation: "TimeSlotAllocation") -> ComputationNodeWorkload:
        """
        Get the subgraph of the workload that is relevant for the steady state allocation.
        This subgraph contains only the computation nodes that are part of the current fusion stack.
        """
        # Get the nodes that are part of the current fusion stack
        assert all(isinstance(node, SteadyStateComputation) for node in allocation.nodes)
        sscs = allocation.get_computation_nodes()
        subgraph_nodes = [
            next(n for n in self.workload if n.id == node.id and n.sub_id == node.sub_id) for node in sscs
        ]
        return self.workload.get_subgraph(subgraph_nodes)

    def add_computation_nodes(
        self,
        steady_state_workload: SteadyStateWorkload,
        subgraph: ComputationNodeWorkload,
        allocation: "TimeSlotAllocation",
    ) -> SteadyStateWorkload:
        """
        Add the computation nodes to the steady state workload.
        This creates new ComputationNode objects in case a node is partitioned across multiple cores.
        """
        for node in reversed(list(nx.topological_sort(subgraph))):  # type: ignore
            if not isinstance(node, ComputationNode):
                continue
            # Get the SteadyStateComputation nodes corresponding to this node
            sscns = [
                n
                for n in allocation.nodes
                if isinstance(n, SteadyStateComputation) and n.id == node.id and n.sub_id == node.sub_id
            ]
            resource_allocations = [node.chosen_resource_allocation for node in sscns]
            self.partitioned_nodes[node] = sscns
            for sscn in sscns:
                steady_state_workload.add(sscn)
                self.current_node_id += 1
                for edge in get_real_out_edges(node, subgraph):
                    _, dst, attrs = edge
                    # Update the 'bits' attribute if it exists
                    # For now we assume the new node will generate original divided by number of core_allocation bits
                    if "bits" in attrs:
                        attrs = attrs.copy()
                        attrs["bits"] = attrs["bits"] / len(resource_allocations)
                    # Add the edge from new_node to destination node
                    dst_sscns = self.partitioned_nodes[dst]
                    for dst_sscn in dst_sscns:
                        # If the destination node is partitioned, we add an edge to each partitioned node
                        steady_state_workload.add_edge(sscn, dst_sscn, **attrs)
        return steady_state_workload

    def add_constant_tensor_nodes(
        self, steady_state_workload: SteadyStateWorkload, subgraph: ComputationNodeWorkload
    ) -> SteadyStateWorkload:
        """
        Add the constant tensor nodes to the steady state workload.
        The constant tensors are tensors needed for computation nodes in the subgraph,
        which are not generated by any of the nodes in the subgraph.
        """
        assert self.accelerator.offchip_core_id is not None, "Off-chip core ID must be set in the accelerator."
        offchip_core = self.accelerator.get_core(self.accelerator.offchip_core_id)
        for node in subgraph.node_list:
            original_node = next(n for n in self.original_workload.node_list if n.id == node.id)
            loop_relevancy_info = original_node.loop_relevancy_info
            # Filter out fake input ops (like W in ReLU)
            real_input_ops = [op for op in node.input_operands if original_node.operand_precision[op] != 0]
            # Create a ConstantTensorNode for each constant tensor in the computation node
            for input_op, input_name in zip(real_input_ops, node.input_names, strict=True):
                ssis = SteadyStateIterationSpace.from_loop_info(
                    loop_relevancy=loop_relevancy_info,
                    intra_core_tiling=[],  # make tensor for entire layer
                    operand=input_op,
                )
                if input_op in node.constant_operands:
                    # This is a constant tensor, add it to the steady state workload
                    tensor = original_node.operand_tensors[input_op]
                    tensor_precision = original_node.operand_precision[input_op]
                    tensor_inputs = tensor.get_inputs()
                    if tensor.equality_hash() not in self.constant_tensors:
                        possible_resource_allocation = self.get_constant_tensor_resource_allocation(offchip_core)
                        full_shape = [ub - lb for lb, ub in tensor.loop_ranges]
                        slices_per_full = ssis.slices_per_full
                        constant_node = SteadyStateTensor(
                            type=TensorFlag.INPUT | TensorFlag.CONSTANT,
                            id=node.id,
                            node_name=f"{input_name}",
                            size=tensor.size * tensor_precision,
                            operand=input_op,
                            steady_state_iteration_space=ssis,
                            possible_resource_allocation=possible_resource_allocation,  # type: ignore
                            subviewtensor_inputs=tensor_inputs,
                            full_shape=full_shape,
                            slices_per_full=slices_per_full,
                        )
                        self.constant_tensors[tensor.equality_hash()] = constant_node
                        steady_state_workload.add(constant_node)
                        self.current_node_id += 1
                    else:
                        constant_node = self.constant_tensors[tensor.equality_hash()]
                    # Add edges from the constant tensor to the partitioned nodes
                    for new_node in self.partitioned_nodes[node]:
                        steady_state_workload.add_edge(constant_node, new_node)

            # Add the constant outputs, i.e. outputs of the last sink nodes of the stack
            out_edges = get_real_out_edges(node, self.workload)
            if len(out_edges) == 0:
                # This is a sink node, add its output tensor as a constant tensor
                output_operand = original_node.output_operand
                output_tensor = original_node.operand_tensors[output_operand]
                output_tensor_precision = original_node.operand_precision[output_operand]
                output_tensor_inputs = output_tensor.get_inputs()
                loop_relevancy_info = original_node.loop_relevancy_info
                if output_tensor not in self.constant_tensors:
                    ssis = SteadyStateIterationSpace.from_loop_info(
                        loop_relevancy=loop_relevancy_info,
                        intra_core_tiling=[],  # make tensor for entire layer
                        operand=output_operand,
                    )
                    possible_resource_allocation = self.get_constant_tensor_resource_allocation(offchip_core)
                    full_shape = [ub - lb for lb, ub in output_tensor.loop_ranges]
                    slices_per_full = ssis.slices_per_full
                    constant_node = SteadyStateTensor(
                        type=TensorFlag.OUTPUT | TensorFlag.CONSTANT,
                        id=node.id,
                        node_name="output",  # TODO: get from actual output name of onnx
                        size=output_tensor.size * output_tensor_precision,
                        operand=output_operand,
                        steady_state_iteration_space=ssis,
                        possible_resource_allocation=possible_resource_allocation,  # type: ignore
                        subviewtensor_inputs=output_tensor_inputs,
                        full_shape=full_shape,
                        slices_per_full=slices_per_full,
                    )
                    steady_state_workload.add(constant_node)
                    self.current_node_id += 1
                else:
                    constant_node = self.constant_tensors[output_tensor]
                # Add edges from the constant tensor to the partitioned nodes
                for new_node in self.partitioned_nodes[node]:
                    steady_state_workload.add_edge(new_node, constant_node)
        return steady_state_workload

    def get_constant_tensor_resource_allocation(self, offchip_core: Core) -> list[Core]:
        """
        Get the resource allocation for the constant tensor.
        The constant tensor is allocated on the off-chip core, and optionally on the compute cores.
        """
        # Initialize with offchip memory core
        possible_resource_allocation = [offchip_core]
        # Add other cores if you want constant tensors to be allocated on memory/compute cores directly
        return possible_resource_allocation

    def add_transfer_nodes(self, steady_state_workload: SteadyStateWorkload) -> SteadyStateWorkload:
        """
        Add the transfer nodes to the steady state workload.
        The transfer nodes are nodes that transfer data between cores or between off-chip and on-chip memory.
        """
        for tensor in steady_state_workload.tensor_nodes:
            if self.is_constant_input_tensor(tensor):
                self.add_transfers_and_post_transfer_tensor_nodes_for_constant_input(tensor, steady_state_workload)
            elif self.is_constant_output_tensor(tensor):
                self.add_transfers_and_pre_transfer_tensor_nodes_for_constant_output(tensor, steady_state_workload)
            elif self.is_nonconstant_output_tensor(tensor):
                self.add_transfers_and_post_transfer_tensor_nodes_for_nonconstant_output(tensor, steady_state_workload)
        # Calculate the optimal paths for the transfer nodes and set the possible_resource_allocation
        self.assign_transfer_paths(steady_state_workload)
        return steady_state_workload

    def is_constant_input_tensor(self, tensor: SteadyStateTensor) -> bool:
        return (TensorFlag.INPUT | TensorFlag.CONSTANT) in tensor.tensor_flag

    def is_nonconstant_output_tensor(self, tensor: SteadyStateTensor) -> bool:
        return (TensorFlag.OUTPUT | TensorFlag.NONCONSTANT) in tensor.tensor_flag

    def is_constant_output_tensor(self, tensor: SteadyStateTensor) -> bool:
        return (TensorFlag.OUTPUT | TensorFlag.CONSTANT) in tensor.tensor_flag

    def add_transfers_and_post_transfer_tensor_nodes_for_constant_input(
        self, tensor: SteadyStateTensor, ssw: SteadyStateWorkload
    ):
        out_edges = list(ssw.out_edges(tensor, data=True))
        all_successors = [i for _, i, _ in out_edges]
        assert all(isinstance(s, (SteadyStateTensor | SteadyStateComputation)) for s in all_successors), (
            "All successors of a tensor node should be either SteadyStateTensor or SteadyStateComputation nodes."
        )
        if not all_successors:
            raise ValueError(f"No valid successors found for constant input {tensor}.")
        # Get correct steady state iteration space for the intra core tilling
        loop_relevancy_info = tensor.origin.loop_relevancy_info
        intra_core_tiling = tensor.origin.intra_core_tiling
        ssis = SteadyStateIterationSpace.from_loop_info(
            loop_relevancy=loop_relevancy_info,
            intra_core_tiling=intra_core_tiling,
            operand=tensor.operand,
            inter_core_tiling=tensor.origin.inter_core_tiling,
        )
        # Get the post transfer tensor node(s)
        grouped_post_transfer_tensor_nodes, grouped_successors = (
            self.get_grouped_post_transfer_tensor_nodes_and_successors(
                all_successors,
                pre_transfer_tensor=tensor,
            )
        )
        for post_transfer_tensor_nodes, successors in zip(
            grouped_post_transfer_tensor_nodes.values(), grouped_successors.values(), strict=False
        ):
            transfer_type, size = self.get_transfer_type_and_size_for_input(post_transfer_tensor_nodes)
            post_transfer_tensor_node_names = [ptn.node_name for ptn in post_transfer_tensor_nodes]
            # Insert a transfer node after the node and connect it to all the successors
            transfer_node = SteadyStateTransfer(
                transfer_type=transfer_type,
                id=tensor.id,
                node_name=f"Transfer({tensor.node_name} -> {post_transfer_tensor_node_names[0]}, ...)",
                srcs=(tensor,),
                dsts=post_transfer_tensor_nodes,  # type: ignore
                size=size,
                tensor=tensor,
                possible_resource_allocation=tuple(),  # will be set later by 'set_transfer_paths'
                possible_memory_core_allocation=tuple(),  # will be set later by 'set_transfer_paths'
                steady_state_iteration_space=ssis,
            )
            ssw.add(transfer_node)
            # Add edge from the original node to the transfer node
            edge_data = {"operand": tensor.operand}
            ssw.add_edge(tensor, transfer_node, **edge_data)
            for ptn, succ in zip(post_transfer_tensor_nodes, successors, strict=True):
                if ptn not in ssw:  # happens for the constant tensors that were already in the graph
                    # Add the post transfer tensor node to the steady state workload
                    ssw.add(ptn)
                # Remove original edge between node and successor
                ssw.remove_edge(tensor, succ)
                # Add edge from transfer node to post transfer tensor node
                ssw.add_edge(transfer_node, ptn, **edge_data)
                # Add edge from post transfer node to successor
                ssw.add_edge(ptn, succ, **edge_data)

    def add_transfers_and_pre_transfer_tensor_nodes_for_constant_output(
        self, tensor: SteadyStateTensor, ssw: SteadyStateWorkload
    ):
        in_edges = list(ssw.in_edges(tensor, data=True))
        all_predecessors = [i for i, _, _ in in_edges]
        assert all(isinstance(s, (SteadyStateTensor | SteadyStateComputation)) for s in all_predecessors), (
            "All predecessors of a tensor node should be either SteadyStateTensor or SteadyStateComputation nodes."
        )
        if not all_predecessors:
            raise ValueError(f"No valid predecessors found for constant output {tensor}.")
        # Get correct steady state iteration space for the intra core tilling
        loop_relevancy_info = tensor.origin.loop_relevancy_info
        intra_core_tiling = tensor.origin.intra_core_tiling
        ssis = SteadyStateIterationSpace.from_loop_info(
            loop_relevancy=loop_relevancy_info,
            intra_core_tiling=intra_core_tiling,
            operand=tensor.operand,
            inter_core_tiling=tensor.origin.inter_core_tiling,
        )
        # Get the post transfer tensor node(s)
        grouped_pre_transfer_tensor_nodes, grouped_predecessors = (
            self.get_grouped_pre_transfer_tensor_nodes_and_predecessors(
                all_predecessors,
                post_transfer_tensor=tensor,
            )
        )
        for pre_transfer_tensor_nodes, predecessors in zip(
            grouped_pre_transfer_tensor_nodes.values(), grouped_predecessors.values(), strict=False
        ):
            transfer_type, size = self.get_transfer_type_and_size_for_output(pre_transfer_tensor_nodes)
            pre_transfer_tensor_node_names = [ptn.node_name for ptn in pre_transfer_tensor_nodes]
            # Insert a transfer node after the node and connect it to all the successors
            transfer_node = SteadyStateTransfer(
                transfer_type=transfer_type,
                id=tensor.id,
                node_name=f"Transfer({pre_transfer_tensor_node_names} -> {tensor.node_name})",
                srcs=pre_transfer_tensor_nodes,
                dsts=(tensor,),  # type: ignore
                size=size,
                tensor=tensor,
                possible_resource_allocation=tuple(),  # will be set later by 'set_transfer_paths'
                possible_memory_core_allocation=tuple(),  # will be set later by 'set_transfer_paths'
                steady_state_iteration_space=ssis,
            )
            ssw.add(transfer_node)
            # Add edge from transfer node to output tensor node
            edge_data = {"operand": tensor.operand}
            ssw.add_edge(transfer_node, tensor, **edge_data)
            for ptn, pred in zip(pre_transfer_tensor_nodes, predecessors, strict=True):
                if ptn not in ssw:  # happens for the constant tensors that were already in the graph
                    # Add the post transfer tensor node to the steady state workload
                    ssw.add(ptn)
                # Remove original edge between predecessor and output tensor
                ssw.remove_edge(pred, tensor)
                # Add edge from predecessor to pre transfer tensor nodes
                ssw.add_edge(pred, ptn, **edge_data)
                # Add edge from pre transfer tensor node to transfer node
                ssw.add_edge(ptn, transfer_node, **edge_data)

    def add_transfers_and_post_transfer_tensor_nodes_for_nonconstant_output(
        self, tensor: SteadyStateTensor, ssw: SteadyStateWorkload
    ):
        out_edges = list(ssw.out_edges(tensor, data=True))
        if len(out_edges) != 1:
            raise NotImplementedError(f"edge data propagation not implemented for multiple outputs yet: {out_edges}")
        edge_data = out_edges[0][2]
        successor = out_edges[0][1]
        assert "operand" in edge_data
        operand = edge_data["operand"]
        # Get the number of tensors that will need to be buffered for this successor CN
        num_tensors = self.calculate_operand_in_degree(successor, operand)
        edge_data["num_tensors"] = num_tensors
        all_successors = [i for _, i, _ in out_edges]
        assert all(isinstance(s, (SteadyStateTensor | SteadyStateComputation)) for s in all_successors), (
            "All successors of a tensor node should be either SteadyStateTensor or SteadyStateComputation nodes."
        )
        if not all_successors:
            raise ValueError(f"No valid successors found for constant input {tensor}.")
        # Get correct steady state iteration space for the intra core tilling
        original_node = next(n for n in self.original_workload.node_list if n.id == tensor.origin.id)
        loop_relevancy_info = original_node.loop_relevancy_info
        intra_core_tiling = original_node.intra_core_tiling
        inter_core_tiling = original_node.inter_core_tiling
        ssis = SteadyStateIterationSpace.from_loop_info(
            loop_relevancy=loop_relevancy_info,
            intra_core_tiling=intra_core_tiling,
            operand=tensor.operand,
            inter_core_tiling=inter_core_tiling,
        )
        # Get the post transfer tensor node(s)
        grouped_post_transfer_tensor_nodes, grouped_successors = (
            self.get_grouped_post_transfer_tensor_nodes_and_successors(
                all_successors,
                pre_transfer_tensor=tensor,
            )
        )
        for post_transfer_tensor_nodes, successors in zip(
            grouped_post_transfer_tensor_nodes.values(), grouped_successors.values(), strict=False
        ):
            transfer_type, size = self.get_transfer_type_and_size_for_input(post_transfer_tensor_nodes)
            post_transfer_tensor_node_names = [ptn.node_name for ptn in post_transfer_tensor_nodes]
            # Insert a transfer node after the node and connect it to all the successors
            transfer_node = SteadyStateTransfer(
                transfer_type=transfer_type,
                id=tensor.id,
                node_name=f"Transfer({tensor.node_name} -> {post_transfer_tensor_node_names})",
                srcs=(tensor,),
                dsts=post_transfer_tensor_nodes,  # type: ignore
                size=size,
                tensor=tensor,
                possible_resource_allocation=tuple(),  # will be set later by 'set_transfer_paths'
                possible_memory_core_allocation=tuple(),  # will be set later by 'set_transfer_paths'
                steady_state_iteration_space=ssis,
            )
            ssw.add(transfer_node)
            # Add edge from the original node to the transfer node
            ssw.add_edge(tensor, transfer_node, **edge_data)
            for ptn, succ in zip(post_transfer_tensor_nodes, successors, strict=True):
                if ptn not in ssw:  # happens for the constant tensors that were already in the graph
                    # Add the post transfer tensor node to the steady state workload
                    ssw.add(ptn)
                # Remove original edge between node and successor
                ssw.remove_edge(tensor, succ)
                # Add edge from transfer node to post transfer tensor node
                ssw.add_edge(
                    transfer_node,
                    ptn,
                    **edge_data,
                )
                # Add edge from post transfer node to successor
                ssw.add_edge(ptn, succ, **edge_data)

    def get_transfer_type_and_size_for_input(
        self, post_transfer_tensor_nodes: tuple[SteadyStateTensor, ...]
    ) -> tuple[TransferType, int]:
        """
        Determine the transfer type based on the post transfer tensor nodes.
        This assumes that the post transfer tensor nodes are all the slice if more than one (broadcast)
        """
        if len(post_transfer_tensor_nodes) > 1:
            loop_ranges = [ptn.loop_ranges for ptn in post_transfer_tensor_nodes]
            if all(lr == loop_ranges[0] for lr in loop_ranges):
                # All loop ranges are the same, this means we are broadcasting the same data to multiple cores
                size = max(ptn.size for ptn in post_transfer_tensor_nodes)
                return TransferType.BROADCAST, size
            else:
                # Different loop ranges, this means we are joining different data from multiple cores
                size = sum(ptn.size for ptn in post_transfer_tensor_nodes)
                return TransferType.DISTRIBUTE, size
        else:
            assert len(post_transfer_tensor_nodes) == 1
            size = post_transfer_tensor_nodes[0].size
            return TransferType.UNICAST, size

    def get_transfer_type_and_size_for_output(
        self, pre_transfer_tensor_nodes: tuple[SteadyStateTensor, ...]
    ) -> tuple[TransferType, int]:
        """
        Determine the transfer type and size based on the pre transfer tensor nodes.
        This assumes that the pre transfer tensor nodes are all mapped to the same column and different.
        """
        size = sum(ptn.size for ptn in pre_transfer_tensor_nodes)
        if len(pre_transfer_tensor_nodes) > 1:
            return TransferType.JOIN, size
        else:
            return TransferType.UNICAST, size

    def get_grouped_post_transfer_tensor_nodes_and_successors(
        self,
        successors: list[SteadyStateNode],
        pre_transfer_tensor: SteadyStateTensor,
    ) -> tuple[
        dict[tuple[tuple[int, int], ...], tuple[SteadyStateTensor, ...]],
        dict[tuple[tuple[int, int], ...], tuple[SteadyStateNode, ...]],
    ]:
        "Grouped by loop ranges to get one joint transfer per broadcastable input."
        post_transfer_tensor_nodes: dict[tuple[tuple[int, int], ...], list[SteadyStateTensor]] = defaultdict(list)
        grouped_successors: dict[tuple[tuple[int, int], ...], list[SteadyStateNode]] = defaultdict(list)
        # # Extra check that groups all successors and post transfers in one in case all are on the same col id
        # first_succ_col_id = next(iter(successors)).chosen_resource_allocation.col_id if successors else None
        # if first_succ_col_id is not None and all(
        #     isinstance(s, SteadyStateComputation) and s.chosen_resource_allocation.col_id == first_succ_col_id
        #     for s in successors
        # ):
        #     all_to_same = True
        # else:
        #     all_to_same = False
        all_to_same = True  # Always group them all together as this is now handled in codegen
        for i, successor in enumerate(successors):
            # Create the tensor node that comes after the transfer node we just created
            if isinstance(successor, SteadyStateTensor):
                # If the successor is a tensor node, we check that it is a sink node output
                assert TensorFlag.CONSTANT in successor.tensor_flag and TensorFlag.OUTPUT in successor.tensor_flag
                assert len(successors) == 1, "There should only be one final output tensor node."
                post_transfer_tensor_nodes[pre_transfer_tensor.loop_ranges] = [successor]
                grouped_successors[pre_transfer_tensor.loop_ranges].append(successor)
            else:
                # Else, it's a SteadyStateComputation and we create a new tensor node for it as post transfer tensor
                assert isinstance(successor, SteadyStateComputation), "Successor should be SteadyStateComputation."
                loop_relevancy_info = pre_transfer_tensor.origin.loop_relevancy_info
                intra_core_tiling = pre_transfer_tensor.origin.intra_core_tiling
                input_operand = pre_transfer_tensor.operand
                ssis = SteadyStateIterationSpace.from_loop_info(
                    loop_relevancy=loop_relevancy_info,
                    intra_core_tiling=intra_core_tiling,
                    operand=input_operand,
                )
                post_transfer_node_name = f"{pre_transfer_tensor.node_name}{'*' * (i + 1)}"
                full_shape = pre_transfer_tensor.full_shape
                slices_per_full = ssis.slices_per_full
                input_tensor = successor.operand_tensors[input_operand]
                input_tensor_precision = successor.operand_precision[input_operand]
                input_subviewtensor_inputs = input_tensor.get_inputs()
                post_transfer_tensor_node = SteadyStateTensor(
                    type=pre_transfer_tensor.tensor_flag,
                    id=pre_transfer_tensor.id,
                    node_name=post_transfer_node_name,
                    size=input_tensor.size * input_tensor_precision,
                    operand=pre_transfer_tensor.operand,
                    steady_state_iteration_space=ssis,
                    possible_resource_allocation=successor.possible_resource_allocation,  # type: ignore
                    subviewtensor_inputs=input_subviewtensor_inputs,
                    full_shape=full_shape,
                    slices_per_full=slices_per_full,
                )
                if all_to_same:
                    key = pre_transfer_tensor.loop_ranges
                else:
                    key = post_transfer_tensor_node.loop_ranges
                post_transfer_tensor_nodes[key].append(post_transfer_tensor_node)
                grouped_successors[key].append(successor)
        post_transfer_tensor_nodes_tuple = {k: tuple(v) for k, v in post_transfer_tensor_nodes.items()}
        grouped_successors_tuple = {k: tuple(v) for k, v in grouped_successors.items()}
        return post_transfer_tensor_nodes_tuple, grouped_successors_tuple

    def get_grouped_pre_transfer_tensor_nodes_and_predecessors(
        self,
        predecessors: list[SteadyStateNode],
        post_transfer_tensor: SteadyStateTensor,
    ) -> tuple[
        dict[int, tuple[SteadyStateTensor, ...]],
        dict[int, tuple[SteadyStateNode, ...]],
    ]:
        """Grouped by the predecessor's col id to get one joint transfer per outputs mapped to the same column."""
        pre_transfer_tensor_nodes: dict[int, list[SteadyStateTensor]] = defaultdict(list)
        grouped_predecessors: dict[int, list[SteadyStateNode]] = defaultdict(list)
        for i, predecessor in enumerate(predecessors):
            assert isinstance(predecessor, SteadyStateComputation), "Predecessor should be SteadyStateComputation."
            loop_relevancy_info = post_transfer_tensor.origin.loop_relevancy_info
            intra_core_tiling = post_transfer_tensor.origin.intra_core_tiling
            output_operand = post_transfer_tensor.operand
            ssis = SteadyStateIterationSpace.from_loop_info(
                loop_relevancy=loop_relevancy_info,
                intra_core_tiling=intra_core_tiling,
                operand=output_operand,
            )
            pre_transfer_node_name = f"{post_transfer_tensor.node_name}{'*' * (i + 1)}"
            full_shape = post_transfer_tensor.full_shape
            slices_per_full = ssis.slices_per_full
            output_tensor = predecessor.operand_tensors[output_operand]
            output_tensor_precision = predecessor.operand_precision[output_operand]
            output_subviewtensor_inputs = output_tensor.get_inputs()
            pre_transfer_tensor_node = SteadyStateTensor(
                type=post_transfer_tensor.tensor_flag,
                id=post_transfer_tensor.id,
                node_name=pre_transfer_node_name,
                size=output_tensor.size * output_tensor_precision,
                operand=post_transfer_tensor.operand,
                steady_state_iteration_space=ssis,
                possible_resource_allocation=predecessor.possible_resource_allocation,  # type: ignore
                subviewtensor_inputs=output_subviewtensor_inputs,
                full_shape=full_shape,
                slices_per_full=slices_per_full,
            )
            assert predecessor.chosen_resource_allocation is not None, (
                "Expected predecessor to have chosen resource allocation."
            )
            col_id = predecessor.chosen_resource_allocation.col_id
            assert col_id is not None
            pre_transfer_tensor_nodes[col_id].append(pre_transfer_tensor_node)
            grouped_predecessors[col_id].append(predecessor)
        pre_transfer_tensor_nodes_tuple = {k: tuple(v) for k, v in pre_transfer_tensor_nodes.items()}
        grouped_predecessors_tuple = {k: tuple(v) for k, v in grouped_predecessors.items()}
        return pre_transfer_tensor_nodes_tuple, grouped_predecessors_tuple

    def assign_transfer_paths(self, steady_state_workload: SteadyStateWorkload) -> None:
        self.process_nonconstant_transfers(steady_state_workload)
        self.process_constant_transfers(steady_state_workload)

    def process_nonconstant_transfers(self, steady_state_workload: SteadyStateWorkload) -> None:
        """
        Process the transfer paths for nonconstant transfers.
        """
        nonconstant_transfers = [
            tr for tr in steady_state_workload.transfer_nodes if self.is_nonconstant_output_tensor(tr.tensor)
        ]
        for transfer_node in nonconstant_transfers:
            src_allocs = tuple([src.chosen_resource_allocation for src in transfer_node.srcs])
            dst_allocs = tuple([dst.chosen_resource_allocation for dst in transfer_node.dsts])
            assert transfer_node.transfer_type == TransferType.UNICAST
            assert len(src_allocs) == 1
            assert len(dst_allocs) == 1
            src_core = src_allocs[0]
            dst_core = dst_allocs[0]
            assert src_core is not None
            assert dst_core is not None
            plan = self.accelerator.communication_manager.get_unicast_plan_no_memory_core(src_core, dst_core)
            links_used = self.accelerator.communication_manager.get_links_for_unicast_plan(plan)
            transfer_node.set_possible_resource_allocation((links_used,))

    def process_constant_transfers(self, steady_state_workload: SteadyStateWorkload) -> None:
        """
        Process the transfer paths for constant transfers.
        """
        # MAX_TRANSFERS_PER_MEM_CORE = 3
        constant_transfers = [
            tr
            for tr in steady_state_workload.transfer_nodes
            if self.is_constant_input_tensor(tr.tensor) or self.is_constant_output_tensor(tr.tensor)
        ]
        # nb_constant_transfers = len(constant_transfers)
        nb_mem_cores_to_use = self.nb_cols_to_use
        cols_to_use_for_constant_transfers = tuple(range(0, nb_mem_cores_to_use))
        for transfer_node in constant_transfers:
            src_allocs = tuple([src.chosen_resource_allocation for src in transfer_node.srcs])
            dst_allocs = tuple([dst.chosen_resource_allocation for dst in transfer_node.dsts])
            # Create a multicast request for the transfer node
            request = MulticastRequest(
                sources=src_allocs,  # type: ignore
                destinations=dst_allocs,  # type: ignore
            )
            multicast_plans = self.accelerator.communication_manager.enumerate_multicast_plans(
                request.sources, request.destinations, cols_to_use_for_constant_transfers
            )
            possible_paths = []
            possible_memory_cores = set()
            for multicast_plan in multicast_plans:
                links_used = self.accelerator.communication_manager.get_links_for_multicast_plan(multicast_plan)
                possible_memory_cores_this_path = self._get_possible_memory_core_allocations(links_used)
                possible_paths.append(links_used)
                possible_memory_cores.update(possible_memory_cores_this_path)
            # Set the possible resource allocation for the transfer node (this also sets the chosen resource allocation)
            transfer_node.set_possible_resource_allocation(tuple(possible_paths))
            # Set the possible memory core allocation for the transfer node
            transfer_node.set_possible_memory_core_allocation(tuple(possible_memory_cores))

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

    def _get_possible_memory_core_allocations(self, links_used: tuple[CommunicationLink, ...]) -> set[Core]:
        all_mem_cores = self._get_accelerator_memory_cores()
        seen_mem_cores = set()
        for link in links_used:
            sender = link.sender
            receiver = link.receiver
            if sender in all_mem_cores:
                seen_mem_cores.add(sender)
            if receiver in all_mem_cores:
                seen_mem_cores.add(receiver)
        return seen_mem_cores & all_mem_cores

    def add_this_iteration_nonconstant_tensor_nodes(
        self, steady_state_workload: SteadyStateWorkload
    ) -> SteadyStateWorkload:
        """
        Add the variable tensor nodes to the steady state workload.
        The variable tensors are tensors that are not constant.
        They are generated after every computation node, and every transfer node, if the successor is not a TensorNode.
        This represents all the tensors explicitly for later memory and data transfer analysis.
        """
        seen_tensors: set[SteadyStateTensor] = set()
        for node in list(steady_state_workload.node_list):
            if not (isinstance(node, SteadyStateComputation) or isinstance(node, SteadyStateTransfer)):
                continue
            # Go through the non-tensor successors
            for _, successor, data in list(steady_state_workload.out_edges(node, data=True)):
                if isinstance(successor, SteadyStateTensor):
                    # If the successor is a tensor node, we check that it is a sink node output
                    assert TensorFlag.CONSTANT in successor.tensor_flag and TensorFlag.OUTPUT in successor.tensor_flag
                    continue  # This is handled in add_transfer_nodes
                if isinstance(node, SteadyStateComputation):
                    output_tensor = node.operand_tensors[node.output_operand]
                    output_tensor_precision = node.operand_precision[node.output_operand]
                    tensor_inputs = output_tensor.get_inputs()
                    tensor_size = output_tensor.size * output_tensor_precision
                    tensor_name = f"{node.name}.{node.output_operand}"
                    resource = node.chosen_resource_allocation
                    original_node = next(n for n in self.original_workload.node_list if n.id == node.id)
                    loop_relevancy_info = original_node.loop_relevancy_info
                    intra_core_tiling = original_node.intra_core_tiling
                    ssis = SteadyStateIterationSpace.from_loop_info(
                        loop_relevancy=loop_relevancy_info,
                        intra_core_tiling=intra_core_tiling,
                        operand=node.output_operand,
                    )
                    output_operand = node.output_operand
                    possible_resource_allocation = [resource]
                    full_shape = [ub - lb for lb, ub in original_node.operand_tensors[output_operand].loop_ranges]
                    slices_per_full = ssis.slices_per_full
                elif isinstance(node, SteadyStateTransfer):  # type: ignore
                    tensor_size = node.size
                    tensor_name = f"{node.node_name}*"
                    resource = successor.chosen_resource_allocation
                    ssis = node.steady_state_iteration_space
                    output_operand = node.tensor.operand
                    tensor_inputs = node.tensor.get_inputs()
                    possible_resource_allocation = [resource]
                    full_shape = node.tensor.full_shape
                    slices_per_full = ssis.slices_per_full
                else:
                    raise ValueError(f"Unexpected node type: {type(node)}")
                variable_tensor_node = SteadyStateTensor(
                    type=TensorFlag.OUTPUT | TensorFlag.NONCONSTANT,
                    id=node.id,
                    node_name=tensor_name,
                    size=tensor_size,
                    operand=output_operand,
                    steady_state_iteration_space=ssis,
                    possible_resource_allocation=possible_resource_allocation,  # type: ignore
                    subviewtensor_inputs=tensor_inputs,
                    full_shape=full_shape,
                    slices_per_full=slices_per_full,
                )
                if variable_tensor_node in seen_tensors:
                    variable_tensor_node = next(t for t in seen_tensors if variable_tensor_node == t)
                else:
                    seen_tensors.add(variable_tensor_node)
                    steady_state_workload.add(variable_tensor_node)
                    # Add edge from the computation/transfer node to the variable tensor node
                    steady_state_workload.add_edge(node, variable_tensor_node)
                # Remove original edge between node and successor
                steady_state_workload.remove_edge(node, successor)  # type: ignore
                # Add edge from the variable tensor node to the successor
                attrs = data.copy()
                steady_state_workload.add_edge(variable_tensor_node, successor, **attrs)
        return steady_state_workload

    def bufferize_nonconstant_tensors(self, steady_state_workload: SteadyStateWorkload) -> SteadyStateWorkload:
        """
        Convert the nonconstant steady state tensors into rolling buffer tensors.
        This is used to represent multiple logically consecutive versions of a single tensor across steady-state iters.
        """
        # Go through each tensor in the steady state workload and check to convert it to a rolling buffer tensor
        for tensor in reversed(steady_state_workload.tensor_nodes):
            # If the tensor is already a rolling buffer tensor, skip it
            if isinstance(tensor, SteadyStateRollingBuffer) or TensorFlag.CONSTANT in tensor.tensor_flag:
                continue
            # Check if there's a successor that is a computation node, and get the input operand associated with edge
            for _, successor, data in list(steady_state_workload.out_edges(tensor, data=True)):
                if isinstance(
                    successor,
                    SteadyStateComputation,
                ) or isinstance(successor, SteadyStateTransfer):
                    assert "num_tensors" in data, (
                        f"Expected 'num_tensors' in edge data between {tensor.node_name} and {successor.node_name}."
                    )
                    # Grab the number of tensors to be cached from the edge data
                    num_tensors = data["num_tensors"]
                    # TODO: The rolling buffer should only be created once, not for each successor!
                    # # Create a rolling buffer tensor to replace the current tensor
                    ssrb = SteadyStateRollingBuffer(
                        base_tensor=tensor,
                        num_tensors=num_tensors,
                    )
                    ssrb.chosen_resource_allocation = tensor.chosen_resource_allocation
                    # Replace the tensor with the rolling buffer, keeping all edges intact
                    steady_state_workload.add(ssrb)
                    # Update the incoming edges
                    for pred, _, in_data in steady_state_workload.in_edges(tensor, data=True):
                        # Add an edge from the predecessor to the rolling buffer tensor
                        steady_state_workload.add_edge(pred, ssrb, **in_data)
                    # # Update the outgoing edges
                    for _, succ, out_data in steady_state_workload.out_edges(tensor, data=True):
                        # Update the out_data to include the increased size of the rolling buffer tensor
                        out_data_new = out_data.copy()
                        out_data_new["bits"] = ssrb.size
                        # Add an edge from the rolling buffer tensor to the successor
                        steady_state_workload.add_edge(ssrb, succ, **out_data_new)
                    # Remove the original tensor node. This also removes edges to/from it.
                    steady_state_workload.remove_node(tensor)  # type: ignore
        return steady_state_workload

    def calculate_operand_in_degree(self, successor, operand):
        eq_node = next(n for n in self.workload.node_list if n.id == successor.id and n.sub_id == successor.sub_id)
        # Get the number of in edges of this equivalent node that have the same operand
        # TODO: If there are multiple successors, the total rolling buffer size is calculated.
        in_edges = get_real_in_edges(eq_node, self.workload)
        num_tensors = sum(1 for _, _, data in in_edges if "operand" in data and data["operand"] == operand)
        # if num_tensors != len(in_edges):
        #     raise NotImplementedError(
        #         f"Expected all in edges of {eq_node} to have the same operand {operand}, but got {in_edges}"
        #     )

        return num_tensors

    def check_steady_state_workload_allocations(self, steady_state_workload: SteadyStateWorkload) -> None:
        """
        Check if all nodes in the steady state workload have a chosen resource allocation.
        """
        for node in steady_state_workload.node_list:
            alloc = node.chosen_resource_allocation
            assert alloc is not None, (
                f"Node {node.node_name} has chosen resource allocation {alloc}. "
                "This should not happen after the TransferAndTensorAllocator has run."
            )
