from stream.classes.workload.simd_node import SimdNode
from zigzag.classes.io.onnx.parser import Parser
from zigzag.classes.io.onnx.utils import get_attribute_ints_with_name, get_node_input_output_dimension_shapes


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
            d["equation"] = 'O[b][k][oy][ox]+=A[b][k][oy][ox]*B[b][k][oy][ox]'

            # Get dimension sizes from input parameters
            assert ia_shape == oa_shape, "Input and output of simd operation should be identical."
            B = oa_shape[0]
            K = oa_shape[1]
            OX = oa_shape[2]
            OY = oa_shape[3]
            d["loop_dim_size"] = {'B': B, 'K': K, "OX": OX, "OY": OY}
            d["operand_precision"] = {'O': 8, 'O_final': 8, 'A': 8, 'B': 8}

            core_allocation = node_mapping["core_allocation"]
            d["core_allocation"] =  core_allocation

            spatial_mapping = self.get_spatial_mappings(self.accelerator, core_allocation)
            d["spatial_mapping"] =  spatial_mapping

            d["memory_operand_links"] =  {'O': 'O', 'B': 'I2', 'A': 'I1'}

            # Find the previous layer(s) that should be this node's parent(s)
            node_inputs = self.node.input
            preds = []
            for node_input in node_inputs:
                for n in self.nodes_outputs:
                    if node_input in self.nodes_outputs[n]:
                        preds.append(n)
            assert len(preds) == 2, "Simd layer has more than 2 inputs."
            d["operand_source"] = {'A': [preds[0]], 'B': [preds[1]]}

            return d

        # Get the input and output activation shapes
        ia_dimension_shape, oa_dimension_shape = get_node_input_output_dimension_shapes(self.node, self.onnx_model)

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
                    raise ValueError(f"There is no mapping provided for node {self.node.name}, nor for {self.node.op_type} nor a default one.")

        node_attrs = get_layer_node_input_format(ia_dimension_shape, oa_dimension_shape, node_mapping)
        # Get the node's input(s) and output(s) tensor names
        node_input_names = list(self.node.input)
        node_output_names = list(self.node.output)

        node_obj = SimdNode(self.node_id, node_attrs, self.node.name, node_input_names, node_output_names)

        return node_obj