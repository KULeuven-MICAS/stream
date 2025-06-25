from math import ceil

import networkx as nx
from zigzag.datatypes import LayerDim, LayerOperand
from zigzag.workload.layer_node import LoopRelevancyInfo

from stream.cost_model.steady_state_tensor_lifetime import TensorLifetimeAnalyzer

# if TYPE_CHECKING:
from stream.hardware.architecture.accelerator import Accelerator
from stream.hardware.architecture.core import Core
from stream.opt.allocation.constraint_optimization.allocation import TimeSlotAllocation
from stream.workload.computation.computation_node import ComputationNode
from stream.workload.onnx_workload import ComputationNodeWorkload
from stream.workload.steady_state_computation import SteadyStateComputation
from stream.workload.steady_state_iteration_space import IterationVariable, SteadyStateIterationSpace
from stream.workload.steady_state_rolling_buffer import SteadyStateRollingBuffer
from stream.workload.steady_state_tensor import SteadyStateTensor, TensorFlag
from stream.workload.steady_state_transfer import SteadyStateTransfer, TransferType
from stream.workload.steady_state_workload import SteadyStateWorkload
from stream.workload.tensor import Tensor
from stream.workload.utils import get_real_in_edges, get_real_out_edges


class SteadyStateScheduler:
    def __init__(
        self,
        workload: "ComputationNodeWorkload",
        accelerator: "Accelerator",
        original_workload: "ComputationNodeWorkload",
    ):
        """
        Initialize the SteadyStateScheduler with the allocation and accelerator.

        Args:
            workload (ComputationNodeWorkload): The workload to be scheduled.
        """
        self.workload = workload  # Only contains nodes that are part of the current fusion stack
        self.accelerator = accelerator
        self.current_node_id = 0
        self.partitioned_nodes: dict[ComputationNode, list[SteadyStateComputation]] = {}
        self.original_workload = original_workload

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
        tla = TensorLifetimeAnalyzer(ssw)
        tla.summary()
        tla.visualize()
        return allocation

    def prepare_graph(self, allocation: "TimeSlotAllocation") -> SteadyStateWorkload:
        steady_state_subgraph = self.workload.get_subgraph(allocation.nodes)
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
            core_allocations = allocation.get_cores_for_node(node)
            new_nodes = self.get_partitioned_nodes(node, core_allocations)
            self.partitioned_nodes[node] = new_nodes
            for new_node in new_nodes:
                steady_state_workload.add(new_node)
                self.current_node_id += 1
                for edge in get_real_out_edges(node, subgraph):
                    _, dst, attrs = edge
                    # Update the 'bits' attribute if it exists
                    # For now we assume the new node will generate original divided by number of core_allocation bits
                    if "bits" in attrs:
                        attrs = attrs.copy()
                        attrs["bits"] = attrs["bits"] / len(core_allocations)
                    # Add the edge from new_node to destination node
                    dst_new_nodes = self.partitioned_nodes[dst]
                    for dst_new_node in dst_new_nodes:
                        # If the destination node is partitioned, we add an edge to each partitioned node
                        steady_state_workload.add_edge(new_node, dst_new_node, **attrs)
        return steady_state_workload

    def get_partitioned_nodes(self, node: ComputationNode, core_allocations: set[Core]) -> list[SteadyStateComputation]:
        """
        Get the partitioned nodes for a given computation node based on the core allocations.
        This creates new ComputationNode objects for each core allocation.
        """
        # If the node is not partitioned, return it as is
        if len(core_allocations) == 1:
            new_node = SteadyStateComputation(
                id=node.id,
                sub_id=node.sub_id,
                node_name=node.name,
                node_attr=node.extract_node_attr(),
                mapping_attr=node.extract_inter_core_mapping_attr(),
                operand_tensor_reshape=node.operand_tensor_reshape,
                produces_final_output=node.produces_final_output,
                group_id=node.group,
                input_names=node.input_names,
                partially_constant_operands=node.partially_constant_operands,
            )
            new_node.chosen_resource_allocation = next(iter(core_allocations))
            return [new_node]
        # Get the inter-core-tiling
        inter_core_tiling = node.inter_core_tiling
        if len(inter_core_tiling) > 1:
            raise NotImplementedError(
                f"Partitioning of nodes with inter-core tiling {inter_core_tiling} is not supported yet."
            )
        tiling_dim = inter_core_tiling[0][0]
        original_size = node.layer_dim_sizes[tiling_dim]
        nb_tiles = len(core_allocations)
        # Get the original loop range of the node for the tiling dimension
        original_loop_range = node.loop_ranges[tiling_dim]
        # Calculate the new loop range for each partitioned node
        partitioned_loop_ranges = [
            (
                original_loop_range[0] + i * (original_loop_range[1] // nb_tiles),
                original_loop_range[0] + (i + 1) * (original_loop_range[1] // nb_tiles),
            )
            for i in range(nb_tiles)
        ]
        size_per_tile = ceil(original_size / nb_tiles)
        partitioned_nodes: list[SteadyStateComputation] = []
        for i, core in enumerate(core_allocations):
            # Update the layer_dim_sizes for the smaller partitioned tile
            node_attr = node.extract_node_attr()
            node_attr.layer_dim_sizes[tiling_dim] = size_per_tile
            inter_core_mapping_attr = node.extract_inter_core_mapping_attr()
            inter_core_mapping_attr.inter_core_tiling = [(tiling_dim, nb_tiles)]
            # Create a new ComputationNode for each core allocation
            partitioned_node = SteadyStateComputation(
                id=node.id,
                sub_id=node.sub_id,
                node_name=node.name + f".part{i}",
                node_attr=node_attr,
                mapping_attr=inter_core_mapping_attr,
                operand_tensor_reshape=node.operand_tensor_reshape,
                produces_final_output=node.produces_final_output,
                group_id=node.group,
                input_names=node.input_names,
                partially_constant_operands=node.partially_constant_operands,
            )
            partitioned_node.chosen_resource_allocation = core
            partitioned_node.loop_ranges[tiling_dim] = partitioned_loop_ranges[i]

            partitioned_node.set_chosen_core_allocation(core.id)
            partitioned_nodes.append(partitioned_node)
        return partitioned_nodes

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
            intra_core_tiling_dims = [dim for dim, _ in intra_core_tiling]
            # Create a ConstantTensorNode for each constant tensor in the computation node
            for input_op in node.input_operands:
                ssis = self.extract_steady_state_iteration_space(loop_relevancy_info, intra_core_tiling_dims, input_op)
                if input_op in node.constant_operands:
                    # This is a constant tensor, add it to the steady state workload
                    tensor = node.operand_tensors[input_op]
                    if tensor not in seen_tensors:
                        seen_tensors.add(tensor)
                        constant_node = SteadyStateTensor(
                            type=TensorFlag.INPUT | TensorFlag.CONSTANT,
                            id=node.id,
                            node_name=f"{node.name}.{input_op}",
                            size=tensor.size,
                            operand=input_op,
                            steady_state_iteration_space=ssis,
                            possible_resource_allocation=[
                                offchip_core,
                            ],
                        )
                        constant_node.chosen_resource_allocation = offchip_core
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
                loop_relevancy_info = original_node.loop_relevancy_info
                intra_core_tiling = original_node.intra_core_tiling
                intra_core_tiling_dims = [dim for dim, _ in intra_core_tiling]
                if output_tensor not in seen_tensors:
                    seen_tensors.add(output_tensor)
                    ssis = self.extract_steady_state_iteration_space(
                        loop_relevancy_info, intra_core_tiling_dims, output_operand
                    )
                    constant_node = SteadyStateTensor(
                        type=TensorFlag.OUTPUT | TensorFlag.CONSTANT,
                        id=node.id,
                        node_name=f"{node.name}.{output_operand}",
                        size=output_tensor.size,
                        operand=output_operand,
                        steady_state_iteration_space=ssis,
                        possible_resource_allocation=[
                            offchip_core,
                        ],
                    )
                    constant_node.chosen_resource_allocation = offchip_core
                    steady_state_workload.add(constant_node)
                    for new_node in self.partitioned_nodes[node]:
                        steady_state_workload.add_edge(new_node, constant_node)
        return steady_state_workload

    def extract_steady_state_iteration_space(
        self, loop_relevancy_info: LoopRelevancyInfo, intra_core_tiling_dims: list[LayerDim], input_op: LayerOperand
    ):
        ssis_list: list[IterationVariable] = []
        seen_relevant = False
        relevant_dims = loop_relevancy_info.get_r_or_pr_layer_dims(input_op)
        # Append dimensions that are partially relevant
        for relevant_dim in list(relevant_dims):
            if relevant_dim in loop_relevancy_info.pr_dims[input_op]:
                for pr_dim in loop_relevancy_info.pr_dims[input_op][relevant_dim]:
                    relevant_dims.append(pr_dim)
        for dim in intra_core_tiling_dims:
            is_relevant = dim in relevant_dims
            if seen_relevant:
                is_relevant = True
            if is_relevant:
                seen_relevant = True
            ssis_list.append(IterationVariable(dim, is_relevant))
        ssis = SteadyStateIterationSpace(ssis_list)
        return ssis

    def add_transfer_nodes(self, steady_state_workload: SteadyStateWorkload) -> SteadyStateWorkload:
        """
        Add the transfer nodes to the steady state workload.
        The transfer nodes are nodes that transfer data between cores or between off-chip and on-chip memory.
        """
        for node in steady_state_workload.tensor_nodes:
            out_edges = list(steady_state_workload.out_edges(node, data=True))
            successors = [i for _, i, _ in out_edges]
            if not successors:
                continue
            if len(successors) > 1:
                transfer_type = TransferType.BROADCAST
            else:
                transfer_type = TransferType.UNICAST
            tensor = node
            # Insert a transfer node after the node and connect it to all the successors
            ssis = node.steady_state_iteration_space
            transfer_node = SteadyStateTransfer(
                transfer_type=transfer_type,
                id=node.id,
                node_name=f"Transfer({node.node_name} -> {[succ.node_name for succ in successors]})",
                src=node,
                dst=successors,
                tensor=tensor,
                possible_resource_allocation=[
                    node.chosen_resource_allocation,
                ],
                steady_state_iteration_space=ssis,
            )
            transfer_node.chosen_resource_allocation = node.chosen_resource_allocation
            steady_state_workload.add(transfer_node)
            # Add edge from the original node to the transfer node
            steady_state_workload.add_edge(node, transfer_node)
            for i, (_, successor, data) in enumerate(out_edges):
                # Create the tensor node that comes after the transfer node we just created
                if isinstance(successor, SteadyStateTensor):
                    # If the successor is a tensor node, we check that it is a sink node output (viewed as constant output)
                    assert TensorFlag.CONSTANT in successor.tensor_flag and TensorFlag.OUTPUT in successor.tensor_flag
                post_transfer_node_name = f"{tensor.node_name}{'*' * (i + 1)}"
                ssis = transfer_node.steady_state_iteration_space
                post_transfer_tensor_node = SteadyStateTensor(
                    type=tensor.tensor_flag,
                    id=node.id,
                    node_name=post_transfer_node_name,
                    size=tensor.size,
                    operand=tensor.operand,
                    steady_state_iteration_space=ssis,
                    possible_resource_allocation=[
                        successor.chosen_resource_allocation,
                    ],
                )
                post_transfer_tensor_node.chosen_resource_allocation = successor.chosen_resource_allocation
                attrs = data.copy()
                # Add the post transfer tensor node to the steady state workload
                steady_state_workload.add(post_transfer_tensor_node)
                # Remove original edge between node and successor
                steady_state_workload.remove_edge(node, successor)  # type: ignore
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
                    # If the successor is a tensor node, we check that it is a sink node output (viewed as constant output)
                    assert TensorFlag.CONSTANT in successor.tensor_flag and TensorFlag.OUTPUT in successor.tensor_flag
                if isinstance(node, SteadyStateComputation):
                    tensor_size = node.operand_tensors[node.output_operand].size
                    tensor_name = f"{node.name}.{node.output_operand}"
                    resource = node.chosen_resource_allocation
                    original_node = next(n for n in self.original_workload.node_list if n.id == node.id)
                    loop_relevancy_info = original_node.loop_relevancy_info
                    intra_core_tiling = original_node.intra_core_tiling
                    intra_core_tiling_dims = [dim for dim, _ in intra_core_tiling]
                    ssis = self.extract_steady_state_iteration_space(
                        loop_relevancy_info, intra_core_tiling_dims, node.output_operand
                    )
                    output_operand = node.output_operand
                elif isinstance(node, SteadyStateTransfer):  # type: ignore
                    tensor_size = node.size
                    tensor_name = f"{node.node_name}*"
                    resource = successor.chosen_resource_allocation
                    ssis = node.steady_state_iteration_space
                    output_operand = node.tensor.operand
                else:
                    raise ValueError(f"Unexpected successor type: {type(successor)}")
                variable_tensor_node = SteadyStateTensor(
                    type=TensorFlag.OUTPUT | TensorFlag.NONCONSTANT,
                    id=node.id,
                    node_name=tensor_name,
                    size=tensor_size,
                    operand=output_operand,
                    steady_state_iteration_space=ssis,
                    possible_resource_allocation=[
                        node.chosen_resource_allocation,
                    ],
                )
                variable_tensor_node.chosen_resource_allocation = resource
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
        This is used to represent multiple logically consecutive versions of a single tensor across steady-state iterations.
        """
        # Go through each tensor in the steady state workload and check if it needs to be converted to a rolling buffer tensor
        for tensor in steady_state_workload.tensor_nodes:
            # If the tensor is already a rolling buffer tensor, skip it
            if isinstance(tensor, SteadyStateRollingBuffer) or TensorFlag.CONSTANT in tensor.tensor_flag:
                continue
            # Check if there's a successor that is a computation node, and get the input operand associated with the edge
            for _, successor, data in list(steady_state_workload.out_edges(tensor, data=True)):
                if isinstance(successor, SteadyStateComputation):
                    assert (
                        "operand" in data
                    ), f"Expected operand in edge data for tensor {tensor.node_name} to computation {successor.node_name}."
                    operand = data["operand"]
                    # Find the equivalent computation node in the entire workload that contains all steady state iterations
                    eq_node = next(
                        n for n in self.workload.node_list if n.id == successor.id and n.sub_id == successor.sub_id
                    )
                    # Get the number of in edges of this equivalent node that have the same operand
                    # TODO: If there are multiple successors, the total rolling buffer size should be determined based on workload grpah.
                    in_edges = get_real_in_edges(eq_node, self.workload)
                    num_tensors = sum(1 for _, _, data in in_edges if "operand" in data and data["operand"] == operand)
                    if num_tensors != len(in_edges):
                        raise NotImplementedError(
                            f"Expected all in edges of {eq_node} to have the same operand {operand}, but found {in_edges}."
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
                        out_data = out_data.copy()
                        out_data["bits"] = ssrb.size
                        # Add an edge from the rolling buffer tensor to the successor
                        steady_state_workload.add_edge(ssrb, succ, **out_data)
                    # Remove the original tensor node. This also removes edges to/from it.
                    steady_state_workload.remove_node(tensor)  # type: ignore
        return steady_state_workload
