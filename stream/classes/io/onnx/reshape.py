from stream.classes.workload.reshape_node import ReshapeNode
from zigzag.classes.io.onnx.parser import Parser
from zigzag.classes.io.onnx.utils import get_node_input_output_dimension_shapes

class ReshapeParser(Parser):
    """Parses an onnx reshape operator into a ReshapeNode.
    """
    def __init__(self, node_id, node, nodes_outputs, mapping, onnx_model) -> None:
        super().__init__(node_id, node, nodes_outputs, mapping, onnx_model)
    
    def run(self):
        return self.generate_reshape_node()

    def generate_reshape_node(self):
        # Get the predecessors of this node
        predecessors = []
        for node_input in self.node.input:
            for n in self.nodes_outputs:
                if node_input in self.nodes_outputs[n]:
                    predecessors.append(n)

        # Get the shape of the operator (this is saved as the second input of the reshape operator, so we need to get the input's dimension shape)
        shape = get_node_input_output_dimension_shapes(self.node, self.onnx_model)[1]
        # Get the input names of the operator
        input_names = [self.node.input[0]]
        # Get the output names of the operator
        output_names = [self.node.output[0]]
        node_obj = ReshapeNode(predecessors, shape, input_names, output_names)
        return node_obj