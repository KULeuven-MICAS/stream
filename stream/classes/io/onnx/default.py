from zigzag.parser.onnx.ONNXOperatorParser import ONNXOperatorParser
from stream.classes.workload.dummy_node import DummyNode


class DefaultNodeParser(ONNXOperatorParser):
    """Parse an ONNX node into a DummyNode."""

    def run(self):
        """Run the parser"""
        return self.generate_dummy_node()

    def generate_dummy_node(self):
        node_name = f"DummyNode({self.node_id})"

        preds: list[int] = []
        for node_input in self.node.input:
            for n in self.nodes_outputs:
                if node_input in self.nodes_outputs[n]:
                    preds.append(n)
        assert len(preds) <= 1
        predecessor = preds[0] if len(preds) == 1 else None

        # Get the input names of this operator
        input_names = list(self.node.input)
        output_names = list(self.node.output)
        op_type = self.node.op_type.lower()
        return DummyNode(
            node_id=self.node_id,
            node_name=node_name,
            predecessor=predecessor,
            input_names=input_names,
            output_names=output_names,
            op_type=op_type,
        )
