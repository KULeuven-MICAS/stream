import os
from math import ceil
import numpy as np
import onnx
from onnx import helper, numpy_helper
from onnx.shape_inference import infer_shapes

from zigzag.classes.stages.Stage import Stage
from stream.classes.workload.onnx_workload import ONNXWorkload
from stream.classes.workload.node import Node
from stream.classes.hardware.architecture.communication_link import CommunicationLink
from stream.classes.hardware.architecture.accelerator import Accelerator
from zigzag.utils import pickle_deepcopy
from stream.classes.stages import utils

from stream.classes.workload.dummy_node import DummyNode

import logging

logger = logging.getLogger(__name__)


class LayerSplittingStage(Stage):
    def __init__(
        self, list_of_callables, *, accelerator, onnx_model, workload, **kwargs
    ):
        super().__init__(list_of_callables, **kwargs)
        self.accelerator = accelerator
        self.onnx_model = onnx_model
        self.workload = workload

        # Set the required kwarg attributes
        self.split_onnx_model_path = None
        self.split_W_double_buffered = None
        required_kwargs = ["split_onnx_model_path", "split_W_double_buffered"]
        utils.set_required_kwargs(required_kwargs, self)

        # Create subfolders for split model save path if they don't exist
        dir_name = os.path.dirname(os.path.abspath(self.split_onnx_model_path))
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name)

        if self.split_W_double_buffered:
            self.split_W_percentage = 0.5
        else:
            self.split_W_percentage = 1

        # Get the weight capacity of all cores
        weight_capacities = {}
        for core in self.accelerator.cores.nodes():
            if core.id == self.accelerator.offchip_core_id:
                continue  # skip offchip core
            core_weight_capacity = core.memory_hierarchy.get_operand_top_level(
                "I2"
            ).memory_instance.size
            weight_capacities[core.id] = core_weight_capacity

        # Get for each layer the split factor we need to be able to fit weights on possible cores
        split_factors = {}
        for node in self.workload.nodes():
            # Get the weight capacity of all possible core allocations of this node
            core_allocations = node.core_allocation
            if isinstance(node, DummyNode) or not isinstance(core_allocations, list):
                continue
            core_capacities = [
                weight_capacities[core_id] for core_id in core_allocations
            ]
            min_core_capacity = min(core_capacities)
            # Get the weight size of this layer
            constant_operands = node.constant_operands
            if not constant_operands:
                continue
            if "W" in node.constant_operands:
                constant_operand = "W"
            elif "B" in node.constant_operands:
                constant_operand = "B"
            else:
                raise NotImplementedError(
                    f"Layer splitting not implemented for {node} with constant operands= {node.constant_operands}."
                )
            weight_size = node.operand_size_bit[constant_operand]
            if weight_size == 0:
                continue
            split_factor = ceil(
                weight_size / (self.split_W_percentage * min_core_capacity)
            )  # 0.5 for double buffering
            if split_factor == 1:
                continue
            # Check if the split_factor is a divisor of the number of output channels
            try:
                output_channels = node.loop_dim_size["K"]
            except KeyError:
                raise NotImplementedError(
                    f"Splitting on output channels requires 'K' loop."
                )
            while divmod(output_channels, split_factor)[1] != 0:
                split_factor += 1
                if split_factor > output_channels:
                    raise ValueError(f"Something went wrong.")
            split_factors[node] = split_factor
        self.split_factors = split_factors

        memory_hierarchy = self.accelerator.get_core(0).memory_hierarchy
        top_level = memory_hierarchy.get_operand_top_level("I2")
        self.weight_size_bits = top_level.memory_instance.size

    def run(self):
        for workload_node in self.workload.nodes():
            if workload_node.type == "conv" or workload_node.type == "gemm":
                try:
                    corresponding_onnx_operator = next(
                        (
                            n
                            for n in self.onnx_model.graph.node
                            if n.name == workload_node.name
                        )
                    )
                except StopIteration:
                    input_names = workload_node.input_names
                    corresponding_onnx_operator = next(
                        (
                            n
                            for n in self.onnx_model.graph.node
                            if n.input == input_names
                        )
                    )
                operator_name = corresponding_onnx_operator.name
                # print(workload_node.name)
                if "W" in workload_node.constant_operands:
                    constant_operand = "W"
                elif "B" in workload_node.constant_operands:
                    constant_operand = "B"
                else:
                    raise NotImplementedError(
                        f"Layer splitting not implemented for {workload_node} with constant operands= {workload_node.constant_operands}."
                    )
                if workload_node in self.split_factors:
                    split_factor = self.split_factors[workload_node]
                    if not split_factor > 1:
                        continue
                    (
                        split_node_names,
                        concat_name,
                    ) = self.split_operator(
                        self.onnx_model, operator_name, split_factor
                    )

                    logger.info(
                        f"Split {workload_node.name} into {split_factor} Conv nodes: {split_node_names} and Concat node: {concat_name}."
                    )

        # Infer the model tensor shapes
        self.onnx_model = infer_shapes(self.onnx_model)
        onnx.save(self.onnx_model, self.split_onnx_model_path)

        self.kwargs["accelerator"] = self.accelerator
        sub_stage = self.list_of_callables[0](
            self.list_of_callables[1:],
            workload_path=self.split_onnx_model_path,
            **self.kwargs,
        )
        for cme, extra_info in sub_stage.run():
            yield cme, extra_info
        # yield None, None

    # Temporarily make this stage a leaf stage such that we can experiment only the pattern matching stage with other stages commented out
    # def is_leaf(self) -> bool:
    #     return True

    @staticmethod
    def split_operator(model, node_name, num_splits):
        """
        Replaces an ONNX Conv or Gemm operator in an ONNX model with a sequence of Conv operators with smaller kernel sizes
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
        original_weight_name = original_node.input[node_weight_input_idx]
        for original_weight in graph.initializer:
            if original_weight.name == original_weight_name:
                weight_input_shape = list(original_weight.dims)
                break
        if weight_input_shape is None:
            raise ValueError(
                f"Could not determine shape of weight input of operator {node_name}."
            )

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
            raise ValueError(
                f"Couldn't find {original_node_output_name} in value info."
            )

        # Add the new output tensors to the value_info
        original_output_elem_type = (
            original_node_output_tensor.type.tensor_type.elem_type
        )
        original_output_shape = [
            d.dim_value for d in original_node_output_tensor.type.tensor_type.shape.dim
        ]

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
            weight_data = numpy_helper.from_array(
                np.zeros(weight_shape), name=weight_name
            )
            # Remove the zeros. Right now the new weight tensors have no actual values.
            weight_data.ClearField("raw_data")
            new_weights.append(weight_data)
            # Create new bias tensor if the original node has one
            if has_bias:
                bias_name = f"{original_bias_name}_split_{i}"
                bias_shape = bias_input_shape
                assert (
                    len(bias_shape) == 1
                ), "Correct dim idx not implemented for > 1 bias dim."
                bias_shape[0] = split_size
                bias_data = numpy_helper.from_array(
                    np.zeros(bias_shape), name=bias_name
                )
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
        # TODO: Check if the concat output shape is equal to the original node's output shape
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
