from zigzag.datatypes import LayerOperand

from stream.workload.node import Node


class ElementwiseNode(Node):

    def __init__(
        self,
        node_id: int,
        node_name: str,
        predecessor: int,
    ) -> None:
        super().__init__(
            node_id=node_id,
            node_name=node_name,
            type="elementwise",
            onchip_energy=0,
            offchip_energy=0,
            runtime=0,
            possible_core_allocation=[-1],
        )
        self.input_operand_source = {LayerOperand("I"): predecessor}

    def join(self, tensor1, tensor2):
        """Join each position in the two tensors to propagate the dependencies (each position should contain a set).

        Args:
            tensor1 (np.ndarray): The first input tensor
            tensor2 (np.ndarray): The second input tensor
        """
        return tensor1 | tensor2
