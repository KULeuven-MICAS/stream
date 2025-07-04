import networkx as nx

# if TYPE_CHECKING:
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
from stream.workload.tensor import Tensor
from stream.workload.utils import get_real_in_edges, get_real_out_edges


class SteadyStateScheduler:
    def __init__(
        self,
        workload: "ComputationNodeWorkload",
        accelerator: "Accelerator",
        original_workload: "ComputationNodeWorkload",
        cost_lut: CostModelEvaluationLUT,
        iterations: int,
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
        offchip_core_id = self.accelerator.offchip_core_id
        tta = TransferAndTensorAllocator(ssw, tsa, offchip_core_id=offchip_core_id, iterations=self.iterations)
        tsa_upd, total_latency = tta.solve()
        print(tsa_upd)
        tot, per_iter, ov = tsa_upd.compute_latency(iterations=self.iterations)
        print(f"Total latency: {tot}, per iteration: {per_iter}, overlap: {ov}")
        # tla = TensorLifetimeAnalyzer(ssw)
        # tla.summary()
        # tla.visualize()
        return tsa_upd

    def prepare_graph(self, allocation: "TimeSlotAllocation") -> SteadyStateWorkload:
        steady_state_subgraph = self.get_workload_subgraph(allocation)
        # Create a new SteadyStateWorkload to hold the scheduled nodes, tensors and transfers
        ssw = SteadyStateWorkload()
        # Add all computation nodes from the subgraph to the SteadyStateWorkload
        ssw = self.add_computation_nodes(ssw, steady_state_subgraph, allocation)
        ssw.visualize_to_file("steady_state_workload_0.png")
        # Add the ConstantTensorNodes to the SteadyStateWorkload
        ssw = self.add_constant_tensor_nodes(ssw, steady_state_subgraph)
        ssw.visualize_to_file("steady_state_workload_1.png")
        # Add the non-constant tensors from this iteration to the SteadyStateWorkload
        ssw = self.add_this_iteration_nonconstant_tensor_nodes(ssw)
        ssw.visualize_to_file("steady_state_workload_2.png")
        # Add the TransferNodes to the SteadyStateWorkload
        ssw = self.add_transfer_nodes(ssw)
        ssw.visualize_to_file("steady_state_workload_3.png")
        # Bufferize the non-constant steady state tensors
        ssw = self.bufferize_nonconstant_tensors(ssw)
        ssw.visualize_to_file("steady_state_workload_4.png")
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
            core_allocations = [node.chosen_core_allocation for node in sscns]
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
                        attrs["bits"] = attrs["bits"] / len(core_allocations)
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
        seen_tensors: set[Tensor] = set()
        for node in subgraph.node_list:
            original_node = next(n for n in self.original_workload.node_list if n.id == node.id)
            loop_relevancy_info = original_node.loop_relevancy_info
            intra_core_tiling = original_node.intra_core_tiling
            # Create a ConstantTensorNode for each constant tensor in the computation node
            for input_op in node.input_operands:
                ssis = SteadyStateIterationSpace.from_loop_info(
                    loop_relevancy=loop_relevancy_info,
                    intra_core_tiling=intra_core_tiling,
                    operand=input_op,
                )
                if input_op in node.constant_operands:
                    # This is a constant tensor, add it to the steady state workload
                    tensor = node.operand_tensors[input_op]
                    original_tensor = original_node.operand_tensors[input_op]
                    if tensor not in seen_tensors:
                        seen_tensors.add(tensor)
                        compute_allocations = set(n.chosen_resource_allocation for n in self.partitioned_nodes[node])
                        possible_resource_allocation = [
                            offchip_core,
                        ] + list(compute_allocations)
                        full_shape = [ub - lb for lb, ub in original_tensor.loop_ranges]
                        slices_per_full = ssis.slices_per_full
                        constant_node = SteadyStateTensor(
                            type=TensorFlag.INPUT | TensorFlag.CONSTANT,
                            id=node.id,
                            node_name=f"{node.name}.{input_op}",
                            size=tensor.size,
                            operand=input_op,
                            steady_state_iteration_space=ssis,
                            possible_resource_allocation=possible_resource_allocation,  # type: ignore
                            full_shape=full_shape,
                            slices_per_full=slices_per_full,
                        )
                        steady_state_workload.add(constant_node)
                        self.current_node_id += 1
                        for new_node in self.partitioned_nodes[node]:
                            steady_state_workload.add_edge(constant_node, new_node)

            # Add the constant outputs, i.e. outputs of the last sink nodes of the stack
            out_edges = get_real_out_edges(node, self.workload)
            if len(out_edges) == 0:
                # This is a sink node, add its output tensor as a constant tensor
                output_operand = node.output_operand
                output_tensor = node.operand_tensors[output_operand]
                original_node = next(n for n in self.original_workload.node_list if n.id == node.id)
                original_output_tensor = original_node.operand_tensors[output_operand]
                loop_relevancy_info = original_node.loop_relevancy_info
                intra_core_tiling = original_node.intra_core_tiling
                if output_tensor not in seen_tensors:
                    seen_tensors.add(output_tensor)
                    ssis = SteadyStateIterationSpace.from_loop_info(
                        loop_relevancy=loop_relevancy_info,
                        intra_core_tiling=intra_core_tiling,
                        operand=output_operand,
                    )
                    compute_allocations = set(n.chosen_resource_allocation for n in self.partitioned_nodes[node])
                    possible_resource_allocation = [
                        offchip_core,
                    ] + list(compute_allocations)
                    full_shape = [ub - lb for lb, ub in original_output_tensor.loop_ranges]
                    slices_per_full = ssis.slices_per_full
                    constant_node = SteadyStateTensor(
                        type=TensorFlag.OUTPUT | TensorFlag.CONSTANT,
                        id=node.id,
                        node_name=f"{node.name}.{output_operand}",
                        size=output_tensor.size,
                        operand=output_operand,
                        steady_state_iteration_space=ssis,
                        possible_resource_allocation=possible_resource_allocation,  # type: ignore
                        full_shape=full_shape,
                        slices_per_full=slices_per_full,
                    )
                    steady_state_workload.add(constant_node)
                    for new_node in self.partitioned_nodes[node]:
                        steady_state_workload.add_edge(new_node, constant_node)
        return steady_state_workload

    def add_transfer_nodes(self, steady_state_workload: SteadyStateWorkload) -> SteadyStateWorkload:
        """
        Add the transfer nodes to the steady state workload.
        The transfer nodes are nodes that transfer data between cores or between off-chip and on-chip memory.
        """
        for tensor in steady_state_workload.tensor_nodes:
            out_edges = list(steady_state_workload.out_edges(tensor, data=True))
            successors = [i for _, i, _ in out_edges]
            assert all(isinstance(s, SteadyStateComputation) for s in successors), (
                "All successors of a tensor node should be either SteadyStateTensor or SteadyStateComputation nodes."
            )
            if not successors:
                continue
            if len(successors) > 1:
                transfer_type = TransferType.BROADCAST
            else:
                transfer_type = TransferType.UNICAST
            # Insert a transfer node after the node and connect it to all the successors
            ssis = tensor.steady_state_iteration_space
            possible_resource_allocation = self.get_transfer_paths(tensor, successors)
            transfer_node = SteadyStateTransfer(
                transfer_type=transfer_type,
                id=tensor.id,
                node_name=f"Transfer({tensor.node_name} -> {[succ.node_name for succ in successors]})",
                src=tensor,
                dst=successors,  # type: ignore
                tensor=tensor,
                possible_resource_allocation=possible_resource_allocation,  # type: ignore
                steady_state_iteration_space=ssis,
            )
            steady_state_workload.add(transfer_node)
            # Add edge from the original node to the transfer node
            steady_state_workload.add_edge(tensor, transfer_node)
            for i, (_, successor, data) in enumerate(out_edges):
                # Create the tensor node that comes after the transfer node we just created
                if isinstance(successor, SteadyStateTensor):
                    # If the successor is a tensor node, we check that it is a sink node output
                    assert TensorFlag.CONSTANT in successor.tensor_flag and TensorFlag.OUTPUT in successor.tensor_flag
                    # If this is the case instead just connect the transfer node to the successor
                    steady_state_workload.add_edge(transfer_node, successor, **data)
                    # Remove original edge between node and successor
                    steady_state_workload.remove_edge(tensor, successor)
                else:
                    post_transfer_node_name = f"{tensor.node_name}{'*' * (i + 1)}"
                    ssis = transfer_node.steady_state_iteration_space
                    full_shape = tensor.full_shape
                    slices_per_full = ssis.slices_per_full
                    assert isinstance(successor, SteadyStateComputation), (
                        "Successor should be a SteadyStateComputation."
                    )
                    post_transfer_tensor_node = SteadyStateTensor(
                        type=tensor.tensor_flag,
                        id=tensor.id,
                        node_name=post_transfer_node_name,
                        size=tensor.size,
                        operand=tensor.operand,
                        steady_state_iteration_space=ssis,
                        possible_resource_allocation=successor.possible_resource_allocation,  # type: ignore
                        full_shape=full_shape,
                        slices_per_full=slices_per_full,
                    )
                    attrs = data.copy()
                    # Add the post transfer tensor node to the steady state workload
                    steady_state_workload.add(post_transfer_tensor_node)
                    # Remove original edge between node and successor
                    steady_state_workload.remove_edge(tensor, successor)
                    # Add edge from transfer node to post transfer tensor node
                    steady_state_workload.add_edge(transfer_node, post_transfer_tensor_node, **attrs)
                    # Add edge from post transfer node to successor
                    steady_state_workload.add_edge(post_transfer_tensor_node, successor, **attrs)
        return steady_state_workload

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
                if isinstance(node, SteadyStateComputation):
                    tensor_size = node.operand_tensors[node.output_operand].size
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
        for tensor in steady_state_workload.tensor_nodes:
            # If the tensor is already a rolling buffer tensor, skip it
            if isinstance(tensor, SteadyStateRollingBuffer) or TensorFlag.CONSTANT in tensor.tensor_flag:
                continue
            # Check if there's a successor that is a computation node, and get the input operand associated with edge
            for _, successor, data in list(steady_state_workload.out_edges(tensor, data=True)):
                if isinstance(successor, SteadyStateComputation):
                    assert "operand" in data, (
                        f"Expected operand in edge data between {tensor.node_name} and {successor.node_name}."
                    )
                    operand = data["operand"]
                    # Find the equivalent computation node in the entire workload that contains all steady state iters
                    eq_node = next(
                        n for n in self.workload.node_list if n.id == successor.id and n.sub_id == successor.sub_id
                    )
                    # Get the number of in edges of this equivalent node that have the same operand
                    # TODO: If there are multiple successors, the total rolling buffer size is calculated.
                    in_edges = get_real_in_edges(eq_node, self.workload)
                    num_tensors = sum(1 for _, _, data in in_edges if "operand" in data and data["operand"] == operand)
                    if num_tensors != len(in_edges):
                        raise NotImplementedError(
                            f"Expected all in edges of {eq_node} to have the same operand {operand}, but got {in_edges}"
                        )
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

    def get_transfer_paths(
        self, predecessor: SteadyStateTensor, successors: list[SteadyStateNode]
    ) -> tuple[tuple[CommunicationLink, ...], ...]:
        """
        Get the possible resource allocations for the transfer node based on the predecessor and successors.
        The possible allocations are all the paths that can be taken from the predecessor to each successor.
        If there's no overlap between paths from predecessor to each successor, the first path for each is combined.
        """
        all_common_paths: set[tuple[CommunicationLink, ...]] = set()
        for pred_alloc in predecessor.possible_resource_allocation:
            assert pred_alloc is not None, f"Predecessor {predecessor.node_name} has no chosen resource allocation."
            paths_per_successor: dict[SteadyStateNode, set[tuple[CommunicationLink]]] = {}
            for successor in successors:
                all_succ_paths = set()
                succ_allocs = (
                    [
                        successor.chosen_resource_allocation,
                    ]
                    if successor.chosen_resource_allocation
                    else successor.possible_resource_allocation
                )
                assert succ_allocs is not None, f"{successor.node_name} has no chosen or possible allocation."
                for succ_alloc in succ_allocs:
                    if isinstance(successor, SteadyStateTensor):
                        # If the successor is a tensor node, we check that it is a sink node output
                        assert (
                            TensorFlag.CONSTANT in successor.tensor_flag and TensorFlag.OUTPUT in successor.tensor_flag
                        )
                    assert isinstance(succ_alloc, Core), (
                        f"Successor {successor.node_name} allocation {succ_alloc} is not a Core."
                    )
                    # Get the paths from the predecessor to the successor
                    paths = self.accelerator.communication_manager.get_all_links_for_pair(pred_alloc, succ_alloc)
                    all_succ_paths.update(paths)
                    if not paths:
                        raise ValueError(
                            f"No communication paths found from {predecessor.node_name} to {successor.node_name}."
                        )
                paths_per_successor[successor] = all_succ_paths
            # Find all overlapping paths for all successors
            # Flatten all paths into a set of unique paths (as tuples for hashability)
            all_paths = set(tuple(path) for paths in paths_per_successor.values() for path in paths)
            # Find intersection: paths that are supersets of a path present in every successor's path list
            common_paths = [
                tuple(path)
                for path in all_paths
                if all(any(all(y in list(path) for y in z) for z in paths) for paths in paths_per_successor.values())
            ]
            if not common_paths:
                succ_allocs = [succ.chosen_resource_allocation for succ in successors]
                raise ValueError(f"No common paths found for preds alloc {pred_alloc} and succs allocs {succ_allocs}.")
            # # If there are no common paths, we take the first path for each successor and join them into a single path
            # if not common_paths:
            #     unique_links: list[CommunicationLink] = []
            #     seen_links: set[CommunicationLink] = set()
            #     for paths in paths_per_successor.values():
            #         if paths:
            #             for link in paths[0]:
            #                 if link not in seen_links:
            #                     unique_links.append(link)
            #                     seen_links.add(link)
            #     common_paths = [unique_links]
            all_common_paths.update(common_paths)
        return tuple(all_common_paths)
