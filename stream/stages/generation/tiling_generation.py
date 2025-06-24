import logging
from collections import defaultdict
from typing import Any, Literal

import numpy as np
from onnx import ModelProto, helper, numpy_helper
from zigzag.datatypes import LayerDim

from stream.hardware.architecture.accelerator import Accelerator
from stream.stages.stage import Stage, StageCallable
from stream.workload.computation.computation_node import ComputationNode
from stream.workload.mapping import TILING_T
from stream.workload.onnx_workload import ONNXWorkload

logger = logging.getLogger(__name__)


class TilingGenerationStage(Stage):

    # Split the node in this dimension to enable fusion within core
    FUSION_PARTITION_DIM_DEFAULT: defaultdict[str, LayerDim] = defaultdict(
        lambda: LayerDim("K"),
        {
            "conv": LayerDim("OY"),
            "matmul": LayerDim("D"),
            "gemm": LayerDim("D"),
            "pooling": LayerDim("OY"),
            "add": LayerDim("D"),
            "mul": LayerDim("D"),
            "softmax": LayerDim("K"),
            "max": LayerDim("K"),
            "div": LayerDim("K"),
            "exp": LayerDim("K"),
            "sum": LayerDim("K"),
            "relu": LayerDim("K"),
            "gelu": LayerDim("K"),
            "silu": LayerDim("K"),
        },
    )
    FUSION_PARTITION_SIZE_DEFAULT = 2

    # Split node in this dimension to partition layer over cores. NOTE this list is ordered
    INTER_CORE_PARTITION_DIM_DEFAULT = [LayerDim("G"), LayerDim("H"), LayerDim("K")]

    def __init__(
        self,
        list_of_callables: list[StageCallable],
        *,
        accelerator: Accelerator,
        workload: ONNXWorkload,
        layer_stacks: list[tuple[int, ...]],
        **kwargs: Any,
    ):
        super().__init__(list_of_callables, **kwargs)
        self.accelerator = accelerator
        self.workload = workload

        assert layer_stacks is not None
        self.layer_stacks = layer_stacks
        self.mode = kwargs.get("mode")

    def run(self):
        for node in self.workload.node_list:
            if not isinstance(node, ComputationNode):
                continue
            nb_nodes_in_stack = len(next(stack for stack in self.layer_stacks if node.id in stack))
            self.set_valid_intra_core_tiling(node, nb_nodes_in_stack)
            self.set_valid_inter_core_tiling(node)

        self.kwargs["accelerator"] = self.accelerator
        self.kwargs["workload"] = self.workload
        self.kwargs["layer_stacks"] = self.layer_stacks

        sub_stage = self.list_of_callables[0](
            self.list_of_callables[1:],
            **self.kwargs,
        )
        for cme, extra_info in sub_stage.run():
            yield cme, extra_info

    def set_valid_intra_core_tiling(self, node: ComputationNode, stack_size: int):
        self.remove_invalid_entries_from_intra_core_tiling(node)

        if not node.intra_core_tiling:
            match self.mode:
                case "lbl":
                    node.intra_core_tiling = []
                case "fused":
                    node.intra_core_tiling = self.generate_intra_core_tiling(node)
                case _:
                    raise ValueError("Unsupported mode for hint loops determination.")

        # Override the intra_core_tiling in case the node is alone in a stack
        if stack_size == 1:
            node.intra_core_tiling = []

    def set_valid_inter_core_tiling(self, node: ComputationNode):
        self.remove_invalid_entries_from_inter_core_tiling(node)

        if not node.inter_core_tiling:
            # Add default tiling if not defined by user
            node.inter_core_tiling = self.generate_inter_core_tiling(node)

    def remove_invalid_entries_from_intra_core_tiling(self, node: ComputationNode):
        """If the user-defined intra core tiling is not valid w.r.t. the node's layer dimensions and sizes, change
        the tiling to a valid one"""
        given_tiling = node.intra_core_tiling
        valid_tiling: TILING_T = []

        for layer_dim, factor in given_tiling:
            if layer_dim not in node.layer_dim_sizes:
                # Invalid layer dim: don't include in valid tiling
                logger.warning(
                    f"Given intra core tiling {layer_dim, factor} for node {node} invalid: node does not contain "
                    f"{layer_dim}. Removing {layer_dim}",
                )
            else:
                layer_dim_size = node.layer_dim_sizes[layer_dim]
                if factor == "all":
                    factor = layer_dim_size
                elif isinstance(factor, str):
                    raise ValueError("intra core tiling should be an integer or `all`")
                elif layer_dim_size < factor:
                    # Given tiling factor too large -> use max allowed
                    factor = layer_dim_size
                elif layer_dim_size % factor != 0:
                    # Layer size is not a multiple of the tiling factor -> increase loop size of the layer
                    new_layer_dim_size = node.layer_dim_sizes[layer_dim] + 1
                    while new_layer_dim_size % factor != 0:
                        new_layer_dim_size += 1
                    logger.warning(f"Rounding {node}: {layer_dim} {layer_dim_size} -> {new_layer_dim_size}")
                    node.layer_dim_sizes[layer_dim] = new_layer_dim_size

                valid_tiling.append((layer_dim, factor))

        node.intra_core_tiling = valid_tiling

    def get_fusion_partition_dim(self, node: ComputationNode) -> LayerDim:
        partition_dim = TilingGenerationStage.FUSION_PARTITION_DIM_DEFAULT[node.type]

        # Default partition dim is not present in this node -> take some arbitrary other dim
        if partition_dim not in node.layer_dim_sizes:
            partition_dim: LayerDim = next(
                dim for dim in node.layer_dim_sizes if dim != LayerDim("B") and dim != LayerDim("G")
            )

        return partition_dim

    def generate_intra_core_tiling(self, node: ComputationNode) -> TILING_T:
        partition_dim = self.get_fusion_partition_dim(node)
        size = min(TilingGenerationStage.FUSION_PARTITION_SIZE_DEFAULT, node.layer_dim_sizes[partition_dim])
        tiling = [(partition_dim, size)]
        return tiling

    def remove_invalid_entries_from_inter_core_tiling(self, node: ComputationNode):
        """Check wether this node's inter core tiling has invalid entries: non-existent layer dimension for this node
        or too large tiling size. Remove entry if this is the case
        #TODO it would be more logical to put this code somewhere else
        """
        valid_tiling: TILING_T = []
        for layer_dim, factor in node.inter_core_tiling:
            # Tiling layer dim doesn't exist -> remove
            if layer_dim not in node.layer_dim_sizes:
                logger.warning(
                    f"Inter core tiling ({layer_dim}, {factor}) of {node} invalid: {layer_dim} not in "
                    f"this node's layer dim sizes. Removing {layer_dim}"
                )
                continue

            # Tiling size too high -> reduce
            layer_size = node.layer_dim_sizes[layer_dim]
            if isinstance(factor, int) and factor > layer_size:
                logger.warning(
                    f"Inter core tiling ({layer_dim}, {factor}) of {node} invalid: {factor} exceeds the layer size of "
                    f"{layer_size}. Reducing tiling size to {layer_size}"
                )
                factor = layer_size

            valid_tiling.append((layer_dim, factor))
        node.inter_core_tiling = valid_tiling

    def generate_inter_core_tiling(
        self, node: ComputationNode, default_tiling_size: int | Literal["*"] = 1
    ) -> TILING_T:
        """Give some valid inter-core tiling for the given node: either coming from the default inter-core partitions,
        or any arbitrary layer dimension.
        #TODO will the default size 1 (instead of `*`) work with ConstraintOptimization?
        #"""
        for dim in TilingGenerationStage.INTER_CORE_PARTITION_DIM_DEFAULT:
            if dim in node.layer_dim_sizes and node.layer_dim_sizes[dim] > 1:
                return [(dim, default_tiling_size)]

        # No valid dim found -> just take something
        return [(next(iter(node.layer_dim_sizes)), default_tiling_size)]

    @staticmethod
    def split_operator(model: ModelProto, node_name: str, num_splits: int):
        """
        Replaces an ONNX Conv or Gemm operator in an ONNX model with a sequence of Conv operators with smaller kernel
        sizes
        that are concatenated together. The output channels of each new operator are equal to the output channels
        of the original operator divided by num_splits. Returns the names of the output tensors of the new
        operators and the name of the output tensor of the new Concat operator.

        Arguments:
        model -- the ONNX model to modify
        node_name -- the name of the Conv node to replace
        num_splits -- the number of Conv nodes to replace the original Conv node with

        Returns:
        new_output_names -- a list of names of the output tensors of the new Conv operators
        concat_output_name -- the name of the output tensor of the new Concat operator
        """
        graph = model.graph

        # Find the node to replace
        original_node = None
        for i, node in enumerate(model.graph.node):
            if node.name == node_name:
                original_node = node
                original_node_idx = i
                if node.op_type == "Conv":
                    node_weight_input_idx = 1
                    node_bias_input_idx = 2 if len(node.input) >= 3 else None
                    weight_output_channel_idx = 0
                elif node.op_type == "Gemm":
                    transB = 0
                    for attr in node.attribute:
                        if attr.name == "transB":
                            transB = attr.i
                    node_weight_input_idx = 1
                    node_bias_input_idx = 2 if len(node.input) >= 3 else None
                    weight_output_channel_idx = 0 if transB else 1
                elif node.op_type == "QLinearConv":
                    node_weight_input_idx = 3
                    node_bias_input_idx = 8 if len(node.input) >= 9 else None
                    weight_output_channel_idx = 0
                else:
                    raise ValueError(f"Unsupported operator type {node.op_type}.")
                break

        # Get the shape of the weight of the operator
        weight_input_shape = None
        assert original_node is not None
        original_weight_name = original_node.input[node_weight_input_idx]
        for original_weight in graph.initializer:
            if original_weight.name == original_weight_name:
                weight_input_shape = list(original_weight.dims)
                break
        if weight_input_shape is None:
            raise ValueError(f"Could not determine shape of weight input of operator {node_name}.")

        # Get the shape of the bias of the operator (if it has a bias)
        has_bias = False
        original_bias_name = None
        bias_input_shape = None
        if node_bias_input_idx is not None:
            has_bias = True
            original_bias_name = node.input[node_bias_input_idx]
            for original_bias in graph.initializer:
                if original_bias.name == original_bias_name:
                    bias_input_shape = list(original_bias.dims)
                    break

        # Find the original node's output in value_info
        node_output_is_graph_output = False
        original_node_output_name = original_node.output[0]
        original_node_output_tensor = None
        for value_info in graph.value_info:
            if value_info.name == original_node_output_name:
                original_node_output_tensor = value_info

        if original_node_output_tensor is None:
            for output in graph.output:
                if output.name == original_node_output_name:
                    original_node_output_tensor = output
                    node_output_is_graph_output = True

        if original_node_output_tensor is None:
            raise ValueError(f"Couldn't find {original_node_output_name} in value info.")

        # Add the new output tensors to the value_info
        original_output_elem_type = original_node_output_tensor.type.tensor_type.elem_type
        original_output_shape = [d.dim_value for d in original_node_output_tensor.type.tensor_type.shape.dim]

        output_channels = weight_input_shape[weight_output_channel_idx]

        # Check if num_splits is a divisor of output_channels
        if output_channels % num_splits != 0:
            raise ValueError("num_splits must be a divisor of the output channels.")
        split_size = output_channels // num_splits

        # Get the dim position that encodes the output channels and modify that based on the split output channels
        shape_index_for_split = original_output_shape.index(output_channels)

        # Split the original node into n nodes
        new_nodes = []
        new_weights = []
        new_biases = []
        new_output_names = []
        for i in range(num_splits):
            node_inputs = original_node.input
            # Create new weight tensor
            weight_name = f"{original_weight_name}_split_{i}"
            weight_shape = weight_input_shape
            weight_shape[weight_output_channel_idx] = split_size
            weight_data = numpy_helper.from_array(np.zeros(weight_shape), name=weight_name)
            # Remove the zeros. Right now the new weight tensors have no actual values.
            weight_data.ClearField("raw_data")
            new_weights.append(weight_data)
            # Create new bias tensor if the original node has one
            if has_bias:
                bias_name = f"{original_bias_name}_split_{i}"
                bias_shape = bias_input_shape
                assert len(bias_shape) == 1, "Correct dim idx not implemented for > 1 bias dim."
                bias_shape[0] = split_size
                bias_data = numpy_helper.from_array(np.zeros(bias_shape), name=bias_name)
                bias_data.ClearField("raw_data")
                new_biases.append(bias_data)

            node_inputs[node_weight_input_idx] = weight_name

            new_node = helper.make_node(
                original_node.op_type,
                inputs=node_inputs,
                outputs=[f"{original_node.output[0]}_split_{i}"],
                name=f"{original_node.name}_{i}",
            )
            # Set the new node attributes to the original_node attributes
            new_node.attribute.extend(original_node.attribute)

            new_nodes.append(new_node)
            # Insert the new conv node into the graph
            graph.node.insert(original_node_idx, new_node)
            original_node_idx += 1

            new_output_names.append(new_node.output[0])

        # Create the Concat node
        concat_output_name = f"{original_node.output[0]}_split_concat"
        concat_node = helper.make_node(
            "Concat",
            inputs=new_output_names,
            outputs=[concat_output_name],
            name=f"{original_node.name}_split_concat",
            axis=shape_index_for_split,
        )

        # Insert the Concat node into the node graph
        graph.node.insert(original_node_idx, concat_node)

        # Update connections to original nodes
        for node in model.graph.node:
            if node != original_node and original_node_output_name in node.input:
                output_name_idx = list(node.input).index(original_node_output_name)
                node.input.remove(original_node_output_name)
                node.input.insert(output_name_idx, concat_output_name)

        # If the original node is a graph output, replace it with the Concat output name
        if node_output_is_graph_output:
            for output in graph.output:
                if output.name == original_node_output_name:
                    output.name = concat_output_name

        # Add new weights to the graph
        graph.initializer.extend(new_weights)
        # Add new biases to the graph
        graph.initializer.extend(new_biases)
        # Add new outputs to the graph
        new_output_shape = original_output_shape
        new_output_shape[shape_index_for_split] = split_size
        new_output_tensors = []
        for new_output_name in new_output_names:
            new_output_tensor = helper.make_tensor_value_info(
                new_output_name, original_output_elem_type, original_output_shape
            )
            new_output_tensors.append(new_output_tensor)
        graph.value_info.extend(new_output_tensors)

        # Remove original node from graph
        model.graph.node.remove(original_node)

        return tuple((new_node.name for new_node in new_nodes)), concat_node.name
