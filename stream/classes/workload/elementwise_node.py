from stream.classes.workload.node import Node
from zigzag.datatypes import LayerOperand


class ElementwiseNode(Node):
    """Class that represents an onnx Reshape node."""

    def __init__(
        self, node_id: int, node_name: str, predecessor: int, input_names: list[str], output_names: list[str]
    ) -> None:
        super().__init__(
            node_id=node_id,
            node_name=node_name,
            type="elementwise",
            onchip_energy=0,
            offchip_energy=0,
            runtime=0,
            core_allocation=[-1],
            input_names=input_names,
            output_names=output_names,
        )
        self.input_operand_source = {LayerOperand("I"): predecessor}

    def join(self, tensor1, tensor2):
        """Join each position in the two tensors to propagate the dependencies (each position should contain a set).

        Args:
            tensor1 (np.ndarray): The first input tensor
            tensor2 (np.ndarray): The second input tensor
        """
        return tensor1 | tensor2
