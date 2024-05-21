from stream.classes.workload.node import Node
from zigzag.datatypes import LayerOperand


class DummyNode(Node):
    """DummyNode of an onnx operator that is not import for finer graph generation or for cost estimation,
    but plays a role because of the passing of the input and output tensors.
    """

    def __init__(
        self,
        node_id: int,
        node_name: str,
        predecessor: int | None,
        input_names: list[str],
        output_names: list[str],
        op_type: str = "dummy",
    ) -> None:
        super().__init__(
            node_id=node_id,
            node_name=node_name,
            type=op_type,
            onchip_energy=0,
            offchip_energy=0,
            runtime=0,
            possible_core_allocation=[-1],
            input_names=input_names,
            output_names=output_names,
        )
        if predecessor is not None:
            self.input_operand_source = {LayerOperand("I"): predecessor}
