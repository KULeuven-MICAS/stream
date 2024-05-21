from typing import Any, TypeAlias
from math import prod

import numpy as np

from stream.classes.workload.node import Node
from stream.classes.workload.tensor import Tensor
from zigzag.datatypes import Constants, LayerDim, LayerOperand, MemoryOperand
from zigzag.workload.layer_attributes import LayerPadding
from zigzag.workload.layer_node import LayerNode, LayerNodeAttributes

OperandTensorReshape: TypeAlias = dict[LayerOperand, tuple[int, int, int, int]]
LoopRanges: TypeAlias = dict[LayerDim, tuple[int, int]]


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

    def __init__(
        self,
        node_id: int,
        node_name: str,
        node_attr: LayerNodeAttributes,
        input_names: list[str],
        output_names: list[str],
        op_type: str = "computation",
        operand_tensor_reshape: OperandTensorReshape | None = None,
        produces_final_output: bool = False,
        group_id: int = 0,
        # To distinguish alternative versions of this node
        sub_id: int = -1,
    ):

        LayerNode.__init__(self, layer_id=node_id, node_name=node_name, node_attr=node_attr)
        Node.__init__(
            self,
            node_id=node_id,
            node_name=node_name,
            type=op_type,
            onchip_energy=0,
            offchip_energy=0,
            runtime=0,
            possible_core_allocation=node_attr.core_allocation,
            input_names=input_names,
            output_names=output_names,
        )

        self.sub_id = sub_id
        self.group = group_id
        self.operand_tensor_reshape = (
            operand_tensor_reshape if operand_tensor_reshape is not None else self.get_operand_tensor_reshape_default()
        )
        # Whether this ComputationNode produces a final output
        self.produces_final_output = produces_final_output

        # self.loop_ranges: dict[str, tuple] = node_attrs.get(
        #     "loop_ranges", {dim: (0, size) for dim, size in self.loop_dim_size.items()}
        # )
        self.loop_ranges: LoopRanges = {layer_dim: (0, size) for layer_dim, size in self.layer_dim_sizes.items()}

        # adds pr dimensions loop ranges to self.loop_ranges
        self.calculate_pr_loop_ranges()
        # Rename methods mentioning layer to node
        self.extract_node_info = self.extract_layer_info
        # Rename function
        self.get_node_operand = self.memory_operand_links.mem_to_layer_op
        self.operand_dimensionality_order: dict[LayerOperand, list[LayerDim]] = {
            layer_op: self.equation.get_r_layer_dims(layer_op) for layer_op in self.equation.get_contained_operands()
        }

        # Each ComputationNode will save a tensor for all its defined operands.
        # For example, a conv layer will have an I tensor, W tensor and O tensor.
        self.operand_tensors: dict[LayerOperand, Tensor] = {}
        self.set_operand_tensors()

        # Will be set by the InterCoreMappingStage or by the FitnessEvaluator
        self.too_large_operands = None
        self.nb_real_predecessors = None

    def set_operand_tensors(self):
        for op in self.layer_operands:
            if op == Constants.OUTPUT_LAYER_OP:
                precision = self.operand_precision.final_output_precision
            else:
                precision = self.operand_precision[op]

            op_dimensionality_order = self.operand_dimensionality_order[op]
            ranges = tuple([self.loop_ranges[dim] for dim in op_dimensionality_order])
            size = prod([upper_bound - lower_bound for (lower_bound, upper_bound) in ranges]) * precision
            self.operand_tensors[op] = Tensor(
                size=size,
                origin=self,
                layer_operand=op,
                loop_dimensions=op_dimensionality_order,
                loop_ranges=ranges,
            )

    def get_operand_tensor_reshape_default(self) -> OperandTensorReshape | None:
        try:
            size_B = self.layer_dim_sizes[LayerDim("B")]
            size_OX = self.layer_dim_sizes[LayerDim("OX")]
            size_OY = self.layer_dim_sizes[LayerDim("OY")]
            size_IX = self.pr_layer_dim_sizes[LayerDim("IX")]
            size_IY = self.pr_layer_dim_sizes[LayerDim("IY")]
            return {
                LayerOperand("I"): (size_B, -1, size_IX, size_IY),
                LayerOperand("O"): (size_B, -1, size_OX, size_OY),
            }
        except KeyError:
            return None

    def __str__(self):
        return f"ComputationNode{self.id}_{self.sub_id}"

    def __hash__(self) -> int:
        """The hash operator of a node depending on its id. The id is a tuple that can be of variable depth.

        Returns:
            int: the computed hash
        """
        return hash((self.id, self.sub_id))

    def __eq__(self, other: object) -> bool:
        """Compare the equality between two nodes.
        Two nodes are considered equal if they have equal hardware performance, which happens following attributes are
        equal:
        - loop_dim_size: The size of the loops.
        - dimension_relations: The partial relevancy between a dimension and two others.
        - operand_precision: The precision at which the operands are stored, which means the operand identifiers should
          be equal.
        - memory_operand_links: The link between memory operand (paths in mem hierarchy) and this node's operands
        - nb_real_predecessors: The number of real predecessors this node has in the graph. This is required for
          accurate knowledge of the number of unique nodes.

        Args:
            other (Node): The other node to compare this node with

        Returns:
            bool: Whether the nodes are equal or not
        """

        return (
            isinstance(other, ComputationNode)
            and self.layer_dim_sizes == other.layer_dim_sizes
            and self.dimension_relations == other.dimension_relations
            and self.operand_precision == other.operand_precision
            and self.memory_operand_links == other.memory_operand_links
            and self.id == other.id
            # and self.nb_real_predecessors == other.nb_real_predecessors
        )

    def __lt__(self, other: "ComputationNode"):
        """Compare two ComputationNodes for the 'less than (<)' operator.

        Args:
            other (ComputationNode): The other ComputationNode.

        Returns:
            bool: self < other
        """
        return self.id < other.id

    def get_operand_for_dim(self, dim: LayerDim) -> LayerOperand:
        """Return the first operand in the operand_list that has this dim as one of is dimensions

        Args:
            dim (str): The dimension for which to find the operand

        Returns:
            str: The operand that has dim as one of its dimensions
        """
        for op in self.layer_operands:
            if dim in self.operand_dimensionality_order[op]:
                return op
        raise ValueError(f"The given dim {dim} doesn't appear in any operand's dimensionality order")

    def calculate_pr_loop_ranges(self):
        """Add the loop ranges of the partially revelant dimensions for this node to self.loop_ranges"""
        for pr_dim, related_dims_and_scalings in self.pr_scaling_factors.items():
            dim_padding = self.padding[pr_dim] if pr_dim in self.padding else LayerPadding.DEFAULT
            padding_begin = dim_padding[0]
            # Assume that there is always 2 dimensions involved in the calculation of a pr dimension
            pr_dim_val_min = -padding_begin
            pr_dim_val_max = -padding_begin
            for related_dimension, scaling_factor in related_dims_and_scalings.items():
                pr_dim_val_min += scaling_factor * self.loop_ranges[related_dimension][0]
                pr_dim_val_max += scaling_factor * (
                    self.loop_ranges[related_dimension][1] - 1
                )  # convert to inclusive upper limit
            pr_dim_val_max += 1  # convert to exclusive upper range
            self.loop_ranges[pr_dim] = (pr_dim_val_min, pr_dim_val_max)

    def reshape_operand_tensor(self, tensor: np.ndarray[Any, Any], operand: LayerOperand):
        """Reshape the tensor back to the representation needed for producer/consumer."""
        if self.operand_tensor_reshape is None or operand not in self.operand_tensor_reshape:
            new_shape = tensor.shape
        else:
            new_shape = self.operand_tensor_reshape[operand]
        return np.reshape(tensor, new_shape)

    def set_too_large_operands(self, too_large_operands: list[MemoryOperand]):
        self.too_large_operands = too_large_operands

    def set_nb_real_predecessors(self, nb_real_predecessors: int):
        self.nb_real_predecessors = nb_real_predecessors

    def update_loop_ranges(self, new_ranges: LoopRanges):
        """Override the loop ranges with a new value for each of the given LayerDims. Keep the old range for the LayerDims not defined in `new_ranges`"""
        for layer_dim in new_ranges:
            self.loop_ranges[layer_dim] = new_ranges[layer_dim]
