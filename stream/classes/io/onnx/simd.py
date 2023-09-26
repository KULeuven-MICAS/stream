from stream.classes.workload.simd_node import SimdNode
from zigzag.classes.io.onnx.parser import Parser
from zigzag.classes.io.onnx.utils import (
    get_attribute_ints_with_name,
    get_node_input_output_dimension_shapes,
)


class SimdParser(Parser):
    """Parses an onnx operator representing an elementwise operation (simd) into a SimdNode.
    e.g. Add, etc.
    """

    def __init__(self, node_id, node, nodes_outputs, mapping, onnx_model, accelerator):
        super().__init__(node_id, node, nodes_outputs, mapping, onnx_model, accelerator)

    def run(self):
        layer_node = self.generate_layer_node_for_simd()
        return layer_node

    def generate_layer_node_for_simd(self):
        def get_layer_node_input_format(ia_shape, oa_shape, node_mapping):
            """
            Generate the necessary dictionary items required for the LayerNode creation.
            For the pooling node, we pick K as the "channel" dimension. It should be equal to C anyways.
            """
            # convert the data types to precisions based on the onnx definition

            # Equation
            d = {}
            d["equation"] = "O[b][k][oy][ox]+=A[b][k][oy][ox]*B[b][k][oy][ox]"

            # Get dimension sizes from input parameters
            assert (
                ia_shape == oa_shape
            ), "Input and output of simd operation should be identical."
            B = oa_shape[0]
            K = oa_shape[1]
            OX = oa_shape[2]
            OY = oa_shape[3]
            d["loop_dim_size"] = {"B": B, "K": K, "OX": OX, "OY": OY}
            d["operand_precision"] = {"O": 8, "O_final": 8, "A": 8, "B": 8}

            core_allocation = node_mapping["core_allocation"]
            d["core_allocation"] = core_allocation

            spatial_mapping = self.get_spatial_mappings(
                self.accelerator, core_allocation
            )
            d["spatial_mapping"] = spatial_mapping

            # Find the previous layer(s) that should be this node's parent(s)
            node_inputs = self.node.input
            assert (
                len(node_inputs) == 2
            ), f"Simd layer {self.node.name} doesn't have 2 inputs: {node_inputs}."
            (input_name_A, input_name_B) = node_inputs
            constant_operands = []
            if any(
                (
                    input_name_A in output_names
                    for output_names in self.nodes_outputs.values()
                )
            ):
                memory_operand_A = "I1"
            else:
                memory_operand_A = "I2"
                constant_operands.append("A")
            if any(
                (
                    input_name_B in output_names
                    for output_names in self.nodes_outputs.values()
                )
            ):
                memory_operand_B = (
                    "I2"  # TODO: Change this to I1 and fix subsequent uses
                )
            else:
                memory_operand_B = "I2"
                constant_operands.append("B")
            d["operand_source"] = {
                "A": [
                    src
                    for (src, src_output_names) in self.nodes_outputs.items()
                    if input_name_A
                    in src_output_names  # only add if it has a previous layer output
                ],
                "B": [
                    src
                    for (src, src_output_names) in self.nodes_outputs.items()
                    if input_name_B in src_output_names
                ],
            }
            d["memory_operand_links"] = {
                "O": "O",
                "B": memory_operand_B,
                "A": memory_operand_A,
            }
            d["constant_operands"] = constant_operands

            return d

        # Get the input and output activation shapes
        ia_dimension_shape, oa_dimension_shape = get_node_input_output_dimension_shapes(
            self.node, self.onnx_model
        )

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
            ia_dimension_shape, oa_dimension_shape, node_mapping
        )
        # Get the node's input(s) and output(s) tensor names
        node_input_names = list(self.node.input)
        node_output_names = list(self.node.output)
        op_type = self.node.op_type.lower()
        node_obj = SimdNode(
            self.node_id,
            node_attrs,
            self.node.name,
            node_input_names,
            node_output_names,
            op_type,
        )

        return node_obj
