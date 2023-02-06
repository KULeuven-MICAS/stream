from stream.classes.workload.pooling_node import PoolingNode
from zigzag.classes.io.onnx.parser import Parser
from zigzag.classes.io.onnx.utils import get_attribute_ints_with_name, get_node_input_output_dimension_shapes


class PoolingParser(Parser):
    """Parses an onnx pooling operator into a PoolingNode.
    e.g. MaxPool, AveragePool, etc.
    """
    def __init__(self, node_id, node, nodes_outputs, mapping, onnx_model, accelerator):
        super().__init__(node_id, node, nodes_outputs, mapping, onnx_model, accelerator)

    def run(self):
        layer_node = self.generate_layer_node_for_pooling()
        return layer_node

    def generate_layer_node_for_pooling(self):

        def get_kernel_shape(attrs, ia_dimension_shape):
            """Return the kernel shape of the pooling operator depending on the type of node

            Args:
                attrs (_type_): _description_
            """
            if self.node.op_type in ["MaxPool", "AveragePool"]:
                # Find kernel shape in attrs
                kernel_shape = get_attribute_ints_with_name("kernel_shape", attrs, default=None)
            elif self.node.op_type in ["GlobalMaxPool", "GlobalAveragePool"]:
                assert len(ia_dimension_shape) == 4  # assume the last two dimensions are the pooling kernel dimensions
                kernel_shape = (ia_dimension_shape[2], ia_dimension_shape[3])
            else:
                raise NotImplementedError(f"Pooling node kernel shape extraction not implemented for operand type {self.node.op_type}.")
            return kernel_shape

        def get_layer_node_input_format(kernel_shape, strides, dilations, padding, ia_shape, oa_shape, node_mapping):
            """
            Generate the necessary dictionary items required for the LayerNode creation.
            For the pooling node, we pick K as the "channel" dimension. It should be equal to C anyways.
            """
            # convert the data types to precisions based on the onnx definition


            # Equation
            d = {}
            d["equation"] = 'O[b][k][oy][ox]+=W[fy][fx]*I[b][k][iy][ix]'

            # Get dimension sizes from input parameters
            assert ia_shape[0] == oa_shape[0], "Batch size is different for input and output activations."
            B = oa_shape[0]
            K = oa_shape[1]
            OX = oa_shape[2]
            OY = oa_shape[3]
            C = ia_shape[1]
            IX = ia_shape[2]
            IY = ia_shape[3]
            FX = kernel_shape[0]
            FY = kernel_shape[1]
            assert K == C, f"Input and output channels not equal for pooling node {self.node.name}."
            d["loop_dim_size"] = {'B': B, 'K': K, "OX": OX, "OY": OY, "FX": FX, "FY": FY}
            d["pr_loop_dim_size"] = {"IX": IX, "IY": IY}
            d["dimension_relations"] = [f'ix={strides[0]}*ox+{dilations[0]}*fx', f'iy={strides[1]}*oy+{dilations[1]}*fy']
            d["operand_precision"] = {'O': 8, 'O_final': 8, 'W': 0, 'I': 8}
            d["constant_operands"] = ['W']

            core_allocation = node_mapping["core_allocation"]
            d["core_allocation"] =  core_allocation

            spatial_mapping = self.get_spatial_mappings(self.accelerator, core_allocation)
            d["spatial_mapping"] =  spatial_mapping

            d["memory_operand_links"] =  {'O': 'O', 'W': 'I2', 'I': 'I1'}

            # Find the previous layer(s) that should be this node's parent(s)
            node_inputs = self.node.input
            preds = []
            for node_input in node_inputs:
                for n in self.nodes_outputs:
                    if node_input in self.nodes_outputs[n]:
                        preds.append(n)
            d["operand_source"] = {'I': preds}

            d["padding"] = {'IY': (padding[0], padding[2]), 'IX': (padding[1], padding[3])}

            return d

        # Get the input and output activation shapes
        ia_dimension_shape, oa_dimension_shape = get_node_input_output_dimension_shapes(self.node, self.onnx_model)

        attrs = self.node.attribute

        kernel_shape = get_kernel_shape(attrs, ia_dimension_shape)

        # Find strides in attrs
        strides = get_attribute_ints_with_name("strides", attrs, default=[1, 1])
        # Find dilations in attrs
        dilations = get_attribute_ints_with_name("dilations", attrs, default=[1, 1])
        # Find pads in attrs
        padding = get_attribute_ints_with_name("pads", attrs, default=[0,0,0,0])

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

        node_attrs = get_layer_node_input_format(kernel_shape, strides, dilations, padding, ia_dimension_shape, oa_dimension_shape, node_mapping)
        # Get the node's input(s) and output(s) tensor names
        node_input_names = list(self.node.input)
        node_output_names = list(self.node.output)

        node_obj = PoolingNode(self.node_id, node_attrs, self.node.name, node_input_names, node_output_names)

        return node_obj