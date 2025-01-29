from typing import Sequence

from xdsl.dialects.builtin import IndexType, IntegerAttr, StringAttr
from xdsl.ir import Attribute, Block, Dialect, Operation, SSAValue
from xdsl.irdl import (
    AttrSizedOperandSegments,
    IRDLOperation,
    irdl_op_definition,
    opt_operand_def,
    prop_def,
    var_operand_def,
    var_result_def,
)


class EmptySSAValue(SSAValue):
    @property
    def owner(self) -> Operation | Block:
        raise RuntimeError()


@irdl_op_definition
class ComputationNodeOp(IRDLOperation):
    name = "stream.computation_node"

    inputs = var_operand_def()
    outputs = opt_operand_def()

    kernel = prop_def(StringAttr)
    core_allocation = prop_def(StringAttr)
    repeat = prop_def(IntegerAttr[IndexType])

    irdl_options = [AttrSizedOperandSegments()]

    def __init__(
        self,
        inputs: Sequence[Operation | SSAValue],
        output: Operation | SSAValue | None,
        kernel: str,
        core_allocation: str,
        repeat: int,
    ) -> None:
        super().__init__(
            operands=[inputs, output],
            properties={
                "kernel": StringAttr(kernel),
                "core_allocation": StringAttr(core_allocation),
                "repeat": IntegerAttr(repeat, IndexType()),
            },
        )


@irdl_op_definition
class TransferOp(IRDLOperation):
    name = "stream.transfer"

    input = opt_operand_def()
    output = var_result_def()

    tensor = prop_def(StringAttr)

    source = prop_def(StringAttr)
    dest = prop_def(StringAttr)
    repeat = prop_def(IntegerAttr[IndexType])

    def __init__(
        self,
        input: SSAValue | Operation | None,
        result_types: Sequence[Attribute],
        source: str,
        dest: str,
        tensor: str,
        repeat: int,
    ) -> None:
        super().__init__(
            operands=[input],
            result_types=[result_types],
            properties={
                "source": StringAttr(source),
                "dest": StringAttr(dest),
                "tensor": StringAttr(tensor),
                "repeat": IntegerAttr(repeat, IndexType()),
            },
        )


Stream = Dialect(
    "stream",
    [
        ComputationNodeOp,
        TransferOp,
    ],
    [],
)
