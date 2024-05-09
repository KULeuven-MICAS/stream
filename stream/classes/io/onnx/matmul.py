import onnx

from zigzag.parser.onnx.ONNXOperatorParser import ONNXOperatorParser
from zigzag.parser.onnx.utils import get_node_input_output_dimension_shapes
from stream.classes.workload.computation_node import ComputationNode

import logging

logger = logging.getLogger(__name__)


class MatMulParser(ONNXOperatorParser):
    """Parses an ONNX MatMul operator into a LayerNode"""

    def __init__(
        self, node_id, node, nodes_outputs, mapping, onnx_model, accelerator
    ) -> None:
        super().__init__(node_id, node, nodes_outputs, mapping, onnx_model, accelerator)

    def run(self):
        """Run the parser"""
        layer_node = self.generate_layer_node_for_matmul()
        return layer_node

    def generate_layer_node_for_matmul(self):
        def get_layer_node_input_format(B, C, K, node_mapping, nodes_outputs):
            """
            Generate the necessary dictionary items required for the Node creation.
            """
            # convert the data types to precisions based on the onnx definition

            # Equation
            d = {}
            d["equation"] = "O[b][k]+=A[b][c]*B[c][k]"

            # Get dimension sizes from input parameters
            K = K
            C = C
            B = B  # Not to be confused with operand 'B' which is the weights
            d["loop_dim_size"] = {"K": K, "C": C, "B": B}
            d["dimension_relations"] = []
            d["operand_precision"] = {"O": 16, "O_final": 8, "B": 8, "A": 8}

            core_allocation = node_mapping["core_allocation"]
            d["core_allocation"] = core_allocation

            spatial_mapping = self.get_spatial_mappings(
                self.accelerator, core_allocation
            )
            d["spatial_mapping"] = spatial_mapping

            d["memory_operand_links"] = {"O": "O", "B": "I2", "A": "I1"}

            # Find the previous layer(s) that should be this node's parent(s)
            node_inputs = self.node.input
            assert (
                len(node_inputs) >= 2
            ), f"MatMul should have atleast two input names, but has: {node_inputs}."
            (first_input_name, second_input_name) = node_inputs[:2]
            d["operand_source"] = {
                "A": [
                    src
                    for (src, src_output_names) in nodes_outputs.items()
                    if first_input_name in src_output_names
                ],
                "B": [
                    src
                    for (src, src_output_names) in nodes_outputs.items()
                    if second_input_name in src_output_names
                ],
            }
            d["constant_operands"] = [
                op for (op, preds) in d["operand_source"].items() if not preds
            ]

            return d

        ia_dimension_shape, oa_dimension_shape = get_node_input_output_dimension_shapes(
            self.node, self.onnx_model
        )

        assert (
            len(ia_dimension_shape) == len(oa_dimension_shape) == 2
        )  # First element is batch size, second is input/output channel
        assert (
            ia_dimension_shape[0] == oa_dimension_shape[0]
        )  # Batch size should be the same for input and output
        # If the batch size is 0, we discard it by setting it to 1 internally inside ZigZag
        batch_size = ia_dimension_shape[0]
        if batch_size == 0:
            B = 1
        else:
            B = batch_size
        C = ia_dimension_shape[1]
        K = oa_dimension_shape[1]

        # Get the hw mapping of this node.
        if self.node.name in self.mapping:
            node_mapping = self.mapping[self.node.name]
        else:
            try:
                node_mapping = self.mapping[self.node.op_type]
            except:
                try:
                    node_mapping = self.mapping["default"]
                except:
                    raise ValueError(
                        f"There is no mapping provided for node {self.node.name}, nor for {self.node.op_type} nor a default one."
                    )

        node_attrs = get_layer_node_input_format(
            B, C, K, node_mapping, self.nodes_outputs
        )
        # Get the node's input(s) and output(s) tensor names
        node_input_names = list(self.node.input)
        node_output_names = list(self.node.output)
        op_type = "matmul"
        node_obj = ComputationNode(
            self.node_id,
            node_attrs,
            self.node.name,
            node_input_names,
            node_output_names,
            op_type,
        )

        return node_obj
