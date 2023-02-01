from math import prod
from typing import Dict

import numpy as np

from stream.classes.workload.node import Node
from stream.classes.workload.tensor import Tensor
from zigzag.classes.workload.layer_node import LayerNode


class ComputationNode(LayerNode, Node):
    """Extension of ZigZag's concept of a "LayerNode" into a more general concept
    called "ComputationNode", which is not necessarily an entire layer, 
    but can represent a smaller chunk of a layer.
    This object also inherits from the "Node" class, which is an abstract baseclass to represent
    different types of onnx nodes needed to accurately schedule the fine-grained graph.
    On top of that, some new information is added for correct dependency generation
    for the finer graph that is built when a layer is split into one and is a
    producer/consumer of another layer.

    Args:
        LayerNode (_type_): _description_
    """

    def __init__(self, node_id, node_attrs, node_name, input_names, output_names, produces_final_output=False, add_missing_node_attrs=False):

        assert isinstance(node_id, tuple), "node_id of ComputationNode initialization should be a tuple: (Layer number, Node number of that layer)"

        if isinstance(node_attrs["core_allocation"], int):
            node_attrs["core_allocation"] = [node_attrs["core_allocation"]]

        LayerNode.__init__(self, node_id, node_attrs, node_name)
        Node.__init__(self, type='computation', energy=None, runtime=None, core_allocation=node_attrs.get('core_allocation', None), input_names=input_names, output_names=output_names)

        # Save whether this ComputationNode produces a final output
        self.produces_final_output = produces_final_output

        # Save the loop ranges of this ComputationNode
        self.loop_ranges: Dict[str, tuple] = node_attrs.get('loop_ranges', {dim: (0, size) for dim, size in self.loop_dim_size.items()})
        self.calculate_pr_loop_ranges()  # adds pr dimensions loop ranges to self.loop_ranges
        # Rename methods mentioning layer to node
        self.attrs = self.layer_attrs
        self.extract_node_info = self.extract_layer_info
        self.get_node_operand = self.get_layer_operand

        # Each ComputationNode will save a tensor for all its defined operands.
        # For example, a conv layer will have an I tensor, W tensor and O tensor.
        self.operand_tensors = {}
        for op in self.operand_list:
            if op == 'O':
                precision = self.operand_precision['O_final']
            else:
                precision = self.operand_precision[op]
            op_dimensionality_order = self.operand_dimensionality_order[op]
            ranges = tuple([self.loop_ranges[dim] for dim in op_dimensionality_order])
            size = prod([upper_bound - lower_bound for (lower_bound, upper_bound) in ranges]) * precision
            self.operand_tensors[op] = Tensor(size=size, origin=self, layer_operand=op, loop_dimensions=op_dimensionality_order, loop_ranges=ranges)

        self.too_large_operands = None  # Will be set by the InterCoreMappingStage or by the FitnessEvaluator
        self.nb_real_predecessors = None

        ''' Add missing layer attr info: operand_tensor_reshape and pr_loop_dim_size for customized workload parser '''
        if add_missing_node_attrs and node_attrs['operator_type'] in ['Conv', 'Conv_downsample', 'MaxPool', 'AveragePool']:
            B = node_attrs['loop_dim_size']['B']
            OX = node_attrs['loop_dim_size']['OX']
            OY = node_attrs['loop_dim_size']['OY']
            IX = self.pr_loop_dim_size['IX']
            IY = self.pr_loop_dim_size['IY']
            node_attrs["pr_loop_dim_size"] = {'IX': IX, 'IY': IY}
            node_attrs["operand_tensor_reshape"] = {'I': (B, -1, IX, IY), 'O': (B, -1, OX, OY)}

    def __str__(self):
        return f"ComputationNode({self.id})"

    def __repr__(self):
        return str(self)

    def __hash__(self) -> int:
        """The hash operator of a node depending on its id. The id is a tuple that can be of variable depth.

        Returns:
            int: the computed hash
        """
        return hash(self.id)

    def __eq__(self, __o: object) -> bool:
        """Compare the equality beween two nodes.
        Two nodes are considered equal if they have equal hardware performance, which happens following attributes are equal:
        - loop_dim_size: The size of the loops.
        - dimension_relations: The partial relevancy between a dimension and two others.
        - operand_precision: The precision at which the operands are stored, which means the operand identifiers should be equal.
        - memory_operand_links: The link between memory operand (paths in mem hierarchy) and this node's operands
        - nb_real_predecessors: The number of real predecessors this node has in the graph. This is required for accurate knowledge of the number of unique nodes.

        Args:
            __o (Node): The other node to compare this node with

        Returns:
            bool: Whether the nodes are equal or not
        """

        if not isinstance(__o, ComputationNode):
            return False
        return self.loop_dim_size == __o.loop_dim_size and self.dimension_relations == __o.dimension_relations and \
               self.operand_precision == __o.operand_precision and self.memory_operand_links == __o.memory_operand_links and \
               self.id[0] == __o.id[0] and self.nb_real_predecessors == __o.nb_real_predecessors

    def __lt__(self, other):
        """Compare two ComputationNodes for the 'less than (<)' operator.

            Args:
                other (ComputationNode): The other ComputationNode.

            Returns:
                bool: self < other
            """
        return self.id < other.id

    def get_operand_for_dim(self, dim) -> str:
        """Return the first operand in the operand_list that has this dim as one of is dimensions

        Args:
            dim (str): The dimension for which to find the operand

        Returns:
            str: The operand that has dim as one of its dimensions
        """
        for op in self.operand_list:
            if dim in self.operand_dimensionality_order[op]:
                return op
        raise ValueError(f"The given dim {dim} doesn't appear in any operand's dimensionality order")

    def calculate_pr_loop_ranges(self):
        """Add the loop ranges of the partially revelant dimensions for this node to self.loop_ranges
        """
        for pr_dim, related_dims_and_scalings in self.pr_scaling_factors.items():
            dim_padding = self.padding.get(pr_dim, (0, 0))
            padding_begin = dim_padding[0]
            # Assume that there is always 2 dimensions involved in the calculation of a pr dimension
            pr_dim_val_min = -padding_begin
            pr_dim_val_max = -padding_begin
            for related_dimension, scaling_factor in related_dims_and_scalings.items():
                pr_dim_val_min += scaling_factor * self.loop_ranges[related_dimension.upper()][0]
                pr_dim_val_max += scaling_factor * (self.loop_ranges[related_dimension.upper()][1] - 1)  # convert to inclusive upper limit
            pr_dim_val_max += 1  # convert to exclusive upper range
            self.loop_ranges[pr_dim] = (pr_dim_val_min, pr_dim_val_max)

    def reshape_operand_tensor(self, tensor, operand):
        """Reshape the tensor back to the representation needed for producer/consumer.
        """
        new_shape = self.operand_tensor_reshape[operand]
        if not new_shape:
            new_shape = tensor.shape
        return np.reshape(tensor, new_shape)

    def set_too_large_operands(self, too_large_operands):
        self.too_large_operands = too_large_operands

    def set_nb_real_predecessors(self, nb_real_predecessors):
        self.nb_real_predecessors = nb_real_predecessors
