from xdsl.ir.affine import AffineExpr, AffineMap

from stream.parser.onnx.operator_parser import OnnxOperatorParser
from stream.workload.workload import ComputationNode, Tensor


def einsum_terms(equation: str) -> tuple[list[str], str]:
    """The index letters of each operand and of the result of an explicit or implicit ``equation``; an implicit
    result holds, in alphabetical order, the letters that appear once."""
    equation = equation.replace(" ", "")
    if "..." in equation:
        raise NotImplementedError(f"Einsum with an ellipsis is not supported: {equation!r}")
    operands, _, result = equation.partition("->")
    terms = operands.split(",")
    if "->" not in equation:
        letters = "".join(terms)
        result = "".join(sorted(c for c in set(letters) if letters.count(c) == 1))
    return terms, result


def einsum_permutation(equation: str) -> list[int] | None:
    """The axis permutation of a one-operand ``equation`` that only reorders its axes, else None."""
    terms, result = einsum_terms(equation)
    if len(terms) != 1 or len(set(terms[0])) != len(terms[0]) or sorted(terms[0]) != sorted(result):
        return None
    return [terms[0].index(c) for c in result]


class EinsumParser(OnnxOperatorParser):
    """Parses an ONNX Einsum into a ``ComputationNode`` with one loop per index letter, the result's letters first
    and the contracted ones after them in order of appearance: each operand is indexed by its letters' loops, as a
    tensor core runs it, multiplying its two operands and accumulating over the contracted loops. One operand summed
    over letters it drops is a ``ReduceSum``; one only reordering its axes is a transpose, folded as a layout."""

    def generate_node(self, name_to_tensor_dict: dict[str, Tensor]) -> ComputationNode:
        equation = next(a.s.decode() for a in self.node.attribute if a.name == "equation")
        terms, result = einsum_terms(equation)
        if len(terms) > 2:  # noqa: PLR2004
            raise NotImplementedError(f"Einsum over more than two operands is not supported: {equation!r}")
        inputs = tuple(name_to_tensor_dict[name] for name in self.node.input)
        assert len(inputs) == len(terms), f"Einsum {equation!r} names {len(terms)} operands, got {len(inputs)}."
        for term, tensor in zip(terms, inputs, strict=True):
            assert len(term) == len(tensor.shape), f"Einsum term {term!r} does not match {tensor.name}{tensor.shape}."
        outputs = self.get_output_tensors()
        assert len(outputs) == 1, "Einsum must have exactly 1 output."

        loops = list(result) + [c for c in dict.fromkeys("".join(terms)) if c not in result]

        def access(term: str) -> AffineMap:
            return AffineMap(len(loops), 0, tuple(AffineExpr.dimension(loops.index(c)) for c in term))

        return ComputationNode(
            type="Einsum" if len(terms) == 2 else "ReduceSum",  # noqa: PLR2004
            name=self.node.name,
            inputs=inputs,
            outputs=outputs,
            operand_mapping=(*(access(term) for term in terms), access(result)),
        )
