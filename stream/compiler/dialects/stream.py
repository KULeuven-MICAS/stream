from collections.abc import Sequence

from xdsl.dialects.builtin import IndexType, IntegerAttr, StringAttr, i64
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
from xdsl.parser import ArrayAttr, AttrParser, DenseArrayBase, MemRefType
from xdsl.printer import Printer

from stream.datatypes import LayerDim
from stream.workload.steady_state.iteration_space import SteadyStateIterationSpace
from stream.workload.workload import InEdge, OutEdge


@irdl_attr_definition
class Channel(ParametrizedAttribute, TypeAttribute):
    name = "channel"


@irdl_attr_definition
class LayerDimAttr(Data[LayerDim]):
    name = "layer_dim"

    @classmethod
    def parse_parameter(cls, parser: AttrParser) -> LayerDim:
        raise NotImplementedError()

    def print_parameter(self, printer: Printer) -> None:
        printer.print_string(str(self.data))


@irdl_attr_definition
class SteadyStateIterationSpaceAttr(Data[SteadyStateIterationSpace]):
    name = "ssis"

    @classmethod
    def parse_parameter(cls, parser: AttrParser) -> SteadyStateIterationSpace:
        raise NotImplementedError()

    def print_parameter(self, printer: Printer) -> None:
        printer.print_string(str(self.data))


@irdl_op_definition
class InEdgeOp(IRDLOperation):
    name = "stream.in_edge"
    output = result_def(MemRefType)
    tensor = prop_def(StringAttr)

    def __init__(self, node: InEdge):
        result_type = node.output.subview.source.type
        super().__init__(
            result_types=(result_type,),
            properties={"tensor": StringAttr(node.name)},
        )


@irdl_op_definition
class OutEdgeOp(IRDLOperation):
    name = "stream.out_edge"
    inputs = var_operand_def(MemRefType)
    tensor = prop_def(StringAttr)

    def __init__(self, node: OutEdge, inputs: Sequence[SSAValue | Operation]):
        super().__init__(
            operands=(inputs,),
            properties={"tensor": StringAttr(node.name)},
        )


@irdl_op_definition
class ComputationNodeOp(IRDLOperation):
    name = "stream.computation_node"

    inputs = var_operand_def()
    output = opt_operand_def()
    result = opt_result_def()

    kernel = prop_def(StringAttr)
    core_allocation = prop_def(Attribute)
    ssis = prop_def(SteadyStateIterationSpaceAttr)

    irdl_options = [AttrSizedOperandSegments()]

    def __init__(
        self,
        result_types: Sequence[Attribute],
        kernel: str,
        inputs: Sequence[SSAValue | Operation],
        core_allocation: Attribute,
        ssis: SteadyStateIterationSpace,
        outputs: Sequence[SSAValue | Operation] = [],
    ):
        super().__init__(
            operands=(inputs, outputs),
            result_types=result_types,
            properties={
                "kernel": StringAttr(kernel),  # TODO: what kernel?
                "core_allocation": core_allocation,
                "ssis": SteadyStateIterationSpaceAttr(ssis),
            },
        )


@irdl_op_definition
class TransferOp(IRDLOperation):
    name = "stream.transfer"

    inputs = var_operand_def()
    outputs = var_result_def()

    offsets = prop_def(DenseArrayBase)
    sizes = prop_def(DenseArrayBase)
    strides = prop_def(DenseArrayBase)
    spatial_strides = prop_def(DenseArrayBase)
    operand_indeces = prop_def(ArrayAttr[LayerDimAttr])
    memtile = prop_def(Attribute)

    ssis = prop_def(SteadyStateIterationSpaceAttr)

    def __init__(  # noqa: PLR0913
        self,
        input: Sequence[SSAValue | Operation],
        result_types: Sequence[Attribute],
        ssis: SteadyStateIterationSpace,
        offsets: DenseArrayBase | Sequence[int],
        sizes: DenseArrayBase | Sequence[int],
        strides: DenseArrayBase | Sequence[int],
        spatial_strides: DenseArrayBase | Sequence[int],
        memtile: Attribute,
        operand_indeces: ArrayAttr[LayerDimAttr],
    ) -> None:
        if not isinstance(offsets, DenseArrayBase):
            offsets = DenseArrayBase.create_dense_int(i64, offsets)
        if not isinstance(sizes, DenseArrayBase):
            sizes = DenseArrayBase.create_dense_int(i64, sizes)
        if not isinstance(strides, DenseArrayBase):
            strides = DenseArrayBase.create_dense_int(i64, strides)
        if not isinstance(spatial_strides, DenseArrayBase):
            spatial_strides = DenseArrayBase.create_dense_int(i64, spatial_strides)
        super().__init__(
            operands=[input],
            result_types=[result_types],
            properties={
                "ssis": SteadyStateIterationSpaceAttr(ssis),
                "offsets": offsets,
                "sizes": sizes,
                "strides": strides,
                "spatial_strides": spatial_strides,
                "memtile": memtile,
                "operand_indeces": operand_indeces,
            },
        )

    def get_relevant_output(
        self, spatial_vars: Sequence[tuple[LayerDim, int]], spatio_temporal_vars: Sequence[LayerDim]
    ) -> SSAValue:
        result = 0
        mult = 1
        for spat_var in self.ssis.data.get_spatial_variables():
            if spat_var.applicable and spat_var.dimension not in spatio_temporal_vars:
                for dim, dim_val in spatial_vars:
                    if dim == spat_var.dimension:
                        result += dim_val * mult
                mult *= spat_var.size
        return self.outputs[result]


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
    memtile = prop_def(Attribute)
    operand_indeces = prop_def(ArrayAttr[LayerDimAttr])

    ssis = prop_def(SteadyStateIterationSpaceAttr)

    def __init__(  # noqa: PLR0913
        self,
        input: SSAValue | Operation,
        channel: SSAValue | Operation,
        ssis: SteadyStateIterationSpace,
        offsets: DenseArrayBase | Sequence[int],
        sizes: DenseArrayBase | Sequence[int],
        strides: DenseArrayBase | Sequence[int],
        memtile: Attribute,
        operand_indeces: ArrayAttr[LayerDimAttr],
    ) -> None:
        if not isinstance(offsets, DenseArrayBase):
            offsets = DenseArrayBase.create_dense_int(i64, offsets)
        if not isinstance(sizes, DenseArrayBase):
            sizes = DenseArrayBase.create_dense_int(i64, sizes)
        if not isinstance(strides, DenseArrayBase):
            strides = DenseArrayBase.create_dense_int(i64, strides)
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
                "memtile": memtile,
                "operand_indeces": operand_indeces,
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
    memtile = prop_def(Attribute)
    operand = prop_def(IntegerAttr[IndexType])
    operand_indeces = prop_def(ArrayAttr[LayerDimAttr])

    ssis = prop_def(SteadyStateIterationSpaceAttr)

    def __init__(  # noqa: PLR0913
        self,
        result_type: Attribute,
        channel: SSAValue | Operation,
        ssis: SteadyStateIterationSpace,
        offsets: DenseArrayBase | Sequence[int],
        sizes: DenseArrayBase | Sequence[int],
        strides: DenseArrayBase | Sequence[int],
        memtile: Attribute,
        operand_indeces: ArrayAttr[LayerDimAttr],
    ) -> None:
        if not isinstance(offsets, DenseArrayBase):
            offsets = DenseArrayBase.create_dense_int(i64, offsets)
        if not isinstance(sizes, DenseArrayBase):
            sizes = DenseArrayBase.create_dense_int(i64, sizes)
        if not isinstance(strides, DenseArrayBase):
            strides = DenseArrayBase.create_dense_int(i64, strides)
        super().__init__(
            operands=[channel],
            result_types=[result_type],
            properties={
                "ssis": SteadyStateIterationSpaceAttr(ssis),
                "offsets": offsets,
                "sizes": sizes,
                "strides": strides,
                "memtile": memtile,
                "operand_indeces": operand_indeces,
            },
        )


Stream = Dialect(
    "stream",
    [
        ComputationNodeOp,
        InEdgeOp,
        OutEdgeOp,
        TransferOp,
        ChannelOp,
        PushOp,
        PullOp,
    ],
    [
        Channel,
        SteadyStateIterationSpaceAttr,
        LayerDimAttr,
    ],
)
