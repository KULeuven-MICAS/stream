from typing import Sequence

from xdsl.dialects.builtin import IndexType, IntegerAttr, StringAttr, i64
from xdsl.ir import Attribute, Block, Dialect, Operation, SSAValue
from xdsl.irdl import (
    AttrSizedOperandSegments,
    IRDLOperation,
    irdl_op_definition,
    opt_operand_def,
    prop_def,
    result_def,
    var_operand_def,
    var_result_def,
)
from xdsl.parser import DenseArrayBase, MemRefType


class EmptySSAValue(SSAValue):
    @property
    def owner(self) -> Operation | Block:
        raise RuntimeError()


@irdl_op_definition
class EdgeOp(IRDLOperation):
    name = "stream.edge"

    output = result_def(MemRefType)

    tensor = prop_def(StringAttr)

    def __init__(self, memref_type: MemRefType, tensor: str | StringAttr):
        if isinstance(tensor, str):
            tensor = StringAttr(tensor)
        super().__init__(properties={'tensor': tensor}, result_types=(memref_type,))


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

    offsets = prop_def(DenseArrayBase)
    sizes = prop_def(DenseArrayBase)
    strides = prop_def(DenseArrayBase)

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
        offsets: DenseArrayBase | Sequence[int],
        sizes: DenseArrayBase | Sequence[int],
        strides: DenseArrayBase | Sequence[int],
    ) -> None:
        if not isinstance(offsets, DenseArrayBase):
            offsets = DenseArrayBase.create_dense_int(i64, offsets)
        if not isinstance(sizes, DenseArrayBase):
            sizes = DenseArrayBase.create_dense_int(i64, sizes)
        if not isinstance(strides, DenseArrayBase):
            strides = DenseArrayBase.create_dense_int(i64, strides)
        super().__init__(
            operands=[input],
            result_types=[result_types],
            properties={
                "source": StringAttr(source),
                "dest": StringAttr(dest),
                "tensor": StringAttr(tensor),
                "repeat": IntegerAttr(repeat, IndexType()),
                "offsets": offsets,
                "sizes": sizes,
                "strides": strides,
            },
        )


Stream = Dialect(
    "stream",
    [
        ComputationNodeOp,
        EdgeOp,
        TransferOp,
    ],
    [],
)
