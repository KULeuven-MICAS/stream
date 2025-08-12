from collections.abc import Sequence

from xdsl.dialects.builtin import ArrayAttr, StringAttr, i64
from xdsl.ir import Attribute, Data, Dialect, Operation, ParametrizedAttribute, SSAValue, TypeAttribute
from xdsl.irdl import (
    AttrSizedOperandSegments,
    IRDLOperation,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    opt_operand_def,
    opt_result_def,
    prop_def,
    result_def,
    var_operand_def,
    var_result_def,
)
from xdsl.parser import AttrParser, DenseArrayBase, GenericParser, MemRefType
from xdsl.printer import Printer

from stream.workload.steady_state.iteration_space import IterationVariable, SteadyStateIterationSpace


@irdl_attr_definition
class Channel(ParametrizedAttribute, TypeAttribute):
    name = "channel"


@irdl_attr_definition
class SteadyStateIterationSpaceAttr(Data[SteadyStateIterationSpace]):
    name = "ssis"

    @classmethod
    def parse_parameter(cls, parser: AttrParser) -> SteadyStateIterationSpace:
        def parse_iter_var() -> IterationVariable:
            # TODO: when parsing becomes relevant
            raise NotImplementedError()

        with parser.in_angle_brackets():
            iter_vars = parser.parse_comma_separated_list(GenericParser.Delimiter.NONE, parse_iter_var)
            return SteadyStateIterationSpace(iter_vars)

    def print_parameter(self, printer: Printer) -> None:
        printer.print_string(str(self.data))
        return

        def print_iter_var(var: IterationVariable) -> str:
            return f"{str(var.dimension)}-{var.size}-{'r' if var.relevant else 'i'}"

        val = ", ".join(print_iter_var(v) for v in self.data.variables)
        printer.print_string(f"<{val}>")

    pass


@irdl_op_definition
class EdgeOp(IRDLOperation):
    name = "stream.edge"

    inputs = var_operand_def(MemRefType)
    output = opt_result_def(MemRefType)

    tensor = prop_def(StringAttr)

    irdl_options = [AttrSizedOperandSegments()]

    def __init__(
        self, result_type: MemRefType | None, tensor: str | StringAttr, inputs: Sequence[SSAValue] | None = None
    ) -> None:
        if isinstance(tensor, str):
            tensor = StringAttr(tensor)
        operands = [inputs] if inputs is not None else [[]]
        result_types = [result_type] if result_type is not None else [[]]
        super().__init__(
            operands=operands,
            properties={"tensor": tensor},
            result_types=result_types,
        )


@irdl_op_definition
class ComputationNodeOp(IRDLOperation):
    name = "stream.computation_node"

    inputs = var_operand_def()
    outputs = opt_operand_def()
    result = opt_result_def()

    kernel = prop_def(StringAttr)
    core_allocation = prop_def(StringAttr)
    ssis = prop_def(SteadyStateIterationSpaceAttr)

    irdl_options = [AttrSizedOperandSegments()]

    def __init__(
        self,
        inputs: Sequence[Operation | SSAValue],
        output: Operation | SSAValue | None,
        kernel: str,
        core_allocation: str,
        ssis: SteadyStateIterationSpace,
        result_types: Sequence[Attribute] | None = None,
    ) -> None:
        super().__init__(
            operands=[inputs, output],
            properties={
                "kernel": StringAttr(kernel),
                "core_allocation": StringAttr(core_allocation),
                "ssis": SteadyStateIterationSpaceAttr(ssis),
            },
            result_types=result_types,
        )


@irdl_op_definition
class TransferOp(IRDLOperation):
    name = "stream.transfer"

    inputs = var_operand_def()
    outputs = var_result_def()

    tensor = prop_def(StringAttr)

    offsets = prop_def(DenseArrayBase)
    sizes = prop_def(DenseArrayBase)
    strides = prop_def(DenseArrayBase)
    spatial_strides = prop_def(DenseArrayBase)
    loop_dimensions = prop_def(ArrayAttr[StringAttr])
    memtile = prop_def(StringAttr)

    source = prop_def(StringAttr)
    dest = prop_def(StringAttr)
    ssis = prop_def(SteadyStateIterationSpaceAttr)

    def __init__(  # noqa: PLR0913
        self,
        input: Sequence[SSAValue | Operation],
        result_types: Sequence[Attribute],
        source: str,
        dest: str,
        tensor: str,
        ssis: SteadyStateIterationSpace,
        offsets: DenseArrayBase | Sequence[int],
        sizes: DenseArrayBase | Sequence[int],
        strides: DenseArrayBase | Sequence[int],
        spatial_strides: DenseArrayBase | Sequence[int],
        loop_dimensions: ArrayAttr[StringAttr] | Sequence[str],
        memtile: str | StringAttr,
    ) -> None:
        if not isinstance(offsets, DenseArrayBase):
            offsets = DenseArrayBase.create_dense_int(i64, offsets)
        if not isinstance(sizes, DenseArrayBase):
            sizes = DenseArrayBase.create_dense_int(i64, sizes)
        if not isinstance(strides, DenseArrayBase):
            strides = DenseArrayBase.create_dense_int(i64, strides)
        if not isinstance(spatial_strides, DenseArrayBase):
            spatial_strides = DenseArrayBase.create_dense_int(i64, spatial_strides)
        if not isinstance(loop_dimensions, ArrayAttr):
            loop_dimensions = ArrayAttr([StringAttr(dim) for dim in loop_dimensions])
        if not isinstance(memtile, StringAttr):
            memtile = StringAttr(memtile)
        super().__init__(
            operands=[input],
            result_types=[result_types],
            properties={
                "source": StringAttr(source),
                "dest": StringAttr(dest),
                "tensor": StringAttr(tensor),
                "ssis": SteadyStateIterationSpaceAttr(ssis),
                "offsets": offsets,
                "sizes": sizes,
                "strides": strides,
                "spatial_strides": spatial_strides,
                "loop_dimensions": loop_dimensions,
                "memtile": memtile,
            },
        )


@irdl_op_definition
class ChannelOp(IRDLOperation):
    name = "stream.channel"

    channel = result_def(Channel)

    def __init__(
        self,
    ) -> None:
        super().__init__(
            result_types=[Channel()],
        )


@irdl_op_definition
class PushOp(IRDLOperation):
    name = "stream.push"

    input = operand_def()
    channel = operand_def()

    offsets = prop_def(DenseArrayBase)
    sizes = prop_def(DenseArrayBase)
    strides = prop_def(DenseArrayBase)
    spatial_strides = prop_def(DenseArrayBase)
    loop_dimensions = prop_def(ArrayAttr[StringAttr])
    memtile = prop_def(StringAttr)

    ssis = prop_def(SteadyStateIterationSpaceAttr)

    def __init__(
        self,
        input: SSAValue | Operation,
        channel: SSAValue | Operation,
        ssis: SteadyStateIterationSpace,
        offsets: DenseArrayBase | Sequence[int],
        sizes: DenseArrayBase | Sequence[int],
        strides: DenseArrayBase | Sequence[int],
        spatial_strides: DenseArrayBase | Sequence[int],
        loop_dimensions: ArrayAttr[StringAttr] | Sequence[str],
        memtile: str | StringAttr,
    ) -> None:
        if not isinstance(offsets, DenseArrayBase):
            offsets = DenseArrayBase.create_dense_int(i64, offsets)
        if not isinstance(sizes, DenseArrayBase):
            sizes = DenseArrayBase.create_dense_int(i64, sizes)
        if not isinstance(strides, DenseArrayBase):
            strides = DenseArrayBase.create_dense_int(i64, strides)
        if not isinstance(spatial_strides, DenseArrayBase):
            spatial_strides = DenseArrayBase.create_dense_int(i64, spatial_strides)
        if not isinstance(loop_dimensions, ArrayAttr):
            loop_dimensions = ArrayAttr([StringAttr(dim) for dim in loop_dimensions])
        if not isinstance(memtile, StringAttr):
            memtile = StringAttr(memtile)
        super().__init__(
            operands=[
                input,
                channel,
            ],
            properties={
                "ssis": SteadyStateIterationSpaceAttr(ssis),
                "offsets": offsets,
                "sizes": sizes,
                "strides": strides,
                "spatial_strides": spatial_strides,
                "loop_dimensions": loop_dimensions,
                "memtile": memtile,
            },
        )


@irdl_op_definition
class PullOp(IRDLOperation):
    name = "stream.pull"

    channel = operand_def()
    output = result_def()

    offsets = prop_def(DenseArrayBase)
    sizes = prop_def(DenseArrayBase)
    strides = prop_def(DenseArrayBase)
    spatial_strides = prop_def(DenseArrayBase)
    loop_dimensions = prop_def(ArrayAttr[StringAttr])
    memtile = prop_def(StringAttr)

    ssis = prop_def(SteadyStateIterationSpaceAttr)

    def __init__(
        self,
        result_type: Attribute,
        channel: SSAValue | Operation,
        ssis: SteadyStateIterationSpace,
        offsets: DenseArrayBase | Sequence[int],
        sizes: DenseArrayBase | Sequence[int],
        strides: DenseArrayBase | Sequence[int],
        spatial_strides: DenseArrayBase | Sequence[int],
        loop_dimensions: ArrayAttr[StringAttr] | Sequence[str],
        memtile: str | StringAttr,
    ) -> None:
        if not isinstance(offsets, DenseArrayBase):
            offsets = DenseArrayBase.create_dense_int(i64, offsets)
        if not isinstance(sizes, DenseArrayBase):
            sizes = DenseArrayBase.create_dense_int(i64, sizes)
        if not isinstance(strides, DenseArrayBase):
            strides = DenseArrayBase.create_dense_int(i64, strides)
        if not isinstance(spatial_strides, DenseArrayBase):
            spatial_strides = DenseArrayBase.create_dense_int(i64, spatial_strides)
        if not isinstance(loop_dimensions, ArrayAttr):
            loop_dimensions = ArrayAttr([StringAttr(dim) for dim in loop_dimensions])
        if not isinstance(memtile, StringAttr):
            memtile = StringAttr(memtile)
        super().__init__(
            operands=[channel],
            result_types=[result_type],
            properties={
                "ssis": SteadyStateIterationSpaceAttr(ssis),
                "offsets": offsets,
                "sizes": sizes,
                "strides": strides,
                "spatial_strides": spatial_strides,
                "loop_dimensions": loop_dimensions,
                "memtile": memtile,
            },
        )


Stream = Dialect(
    "stream",
    [
        ComputationNodeOp,
        EdgeOp,
        TransferOp,
        ChannelOp,
        PushOp,
        PullOp,
    ],
    [
        Channel,
        SteadyStateIterationSpaceAttr,
    ],
)
