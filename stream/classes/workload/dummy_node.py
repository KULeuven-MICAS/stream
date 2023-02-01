from typing import List
from stream.classes.workload.node import Node


class DummyNode(Node):
    """DummyNode of an onnx operator that is not import for finer graph generation or for cost estimation,
    but plays a role because of the passing of the input and output tensors.
    """
    def __init__(self, id: int, predecessors, node_name, input_names: List[str], output_names: List[str]) -> None:
        super().__init__(type="dummy", energy=0, runtime=0, core_allocation=-1, input_names=input_names, output_names=output_names)

        self.id = id
        self.input_operand_source = {'I': predecessors}
        self.name = node_name
