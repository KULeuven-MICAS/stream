from xdsl.ir.affine import AffineMap

from stream.parser.onnx.operator_parser import OnnxOperatorParser
from stream.workload.workload import ComputationNode, Tensor

_IDENTITY_MAPS: dict[int, tuple[AffineMap, AffineMap]] = {
    2: (
        AffineMap.from_callable(lambda m, n: (m, n)),
        AffineMap.from_callable(lambda m, n: (m, n)),
    ),
    4: (
        AffineMap.from_callable(lambda b, c, h, w: (b, c, h, w)),
        AffineMap.from_callable(lambda b, c, h, w: (b, c, h, w)),
    ),
}


class ReluParser(OnnxOperatorParser):
    """Parses an ONNX Relu operator into a ComputationNode.

    Uses identity AffineMaps whose rank matches the actual tensor dimensionality.
    Supports 2D (SwiGLU-style) and 4D (ResNet-style) tensors.
    """

    def generate_node(self, name_to_tensor_dict: dict[str, Tensor]) -> ComputationNode:
        inputs = tuple(name_to_tensor_dict[inp] for inp in self.node.input)
        assert len(inputs) == 1, f"Relu must have exactly 1 input, got {len(inputs)}."
        ndim = len(inputs[0].shape)
        if ndim not in _IDENTITY_MAPS:
            raise NotImplementedError(f"ReluParser does not support {ndim}D tensors (only 2D and 4D).")
        mappings = _IDENTITY_MAPS[ndim]

        return ComputationNode(
            type=self.node.op_type,
            name=self.node.name,
            inputs=inputs,
            outputs=self.get_output_tensors(),
            operand_mapping=mappings,
        )
