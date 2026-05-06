from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from math import prod
from typing import Self, TypeVar

from xdsl.dialects.builtin import (
    ContainerType,
    FixedBitwidthType,
    IntAttr,
    NoneAttr,
    StringAttr,
)
from xdsl.dialects.utils import AbstractYieldOperation
from xdsl.ir import (
    Attribute,
    Data,
    Dialect,
    IsTerminator,
    Operation,
    ParametrizedAttribute,
    Region,
    SSAValue,
    StrEnum,
    TypeAttribute,
    VerifyException,
)
from xdsl.irdl import (
    IRDLOperation,
    ParameterDef,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    opt_prop_def,
    prop_def,
    region_def,
    result_def,
    traits_def,
    var_operand_def,
    var_result_def,
)
from xdsl.parser import ArrayAttr, AttrParser, MLIRTokenKind
from xdsl.printer import Printer
from xdsl.traits import IsolatedFromAbove, Pure

from stream.datatypes import LayerDim
from stream.workload.steady_state.iteration_space import (
    SteadyStateIterationSpace,
)
from stream.workload.workload import OutEdge


@irdl_attr_definition
class Channel(ParametrizedAttribute, TypeAttribute):
    name = "stream.channel"


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


@irdl_attr_definition
class CoreAttr(ParametrizedAttribute):
    name = "core"

    col: ParameterDef[IntAttr]
    row: ParameterDef[IntAttr]

    def __init__(self, col: int | IntAttr, row: int | IntAttr):
        if isinstance(col, int):
            col = IntAttr(col)
        if isinstance(row, int):
            row = IntAttr(row)
        super().__init__([col, row])

    def print_parameters(self, printer: Printer) -> None:
        printer.print_string(f"t{self.col.data}-{self.row.data}")


StrensorCovT_co = TypeVar("StrensorCovT_co", bound=FixedBitwidthType, covariant=True)


class StrensorVarType(StrEnum):
    KERNEL = "k"
    SPATIAL = "s"
    TEMPORAL = "t"
    ABSENT = "a"
    POINT = "p"
    CONSTANT = "c"


@dataclass(frozen=True)
class StrensorVar:
    type: StrensorVarType
    size: int
    dim: LayerDim

    @staticmethod
    def parse_dim(parser: AttrParser) -> tuple[StrensorVarType, LayerDim]:
        assert parser._current_token.kind == MLIRTokenKind.BARE_IDENT
        val = parser._current_token.text
        parser._consume_token(MLIRTokenKind.BARE_IDENT)
        return StrensorVarType(val[0]), LayerDim(int(val[1:]))

    @classmethod
    def parse_optional_var(cls, parser: AttrParser) -> Self | None:
        size = parser.parse_optional_integer()
        if size is None:
            return
        parser.parse_characters("(")
        _ = parser._current_token
        var_type, dim = cls.parse_dim(parser)
        parser.parse_characters(")")
        return cls(var_type, size, dim)

    def __str__(self) -> str:
        return f"{self.size}({str(self.type)}{self.dim.position})"


@dataclass(frozen=True)
class StrensorSpace:
    """
    Outermost to innermost collection of StrensorVars
    """

    vars: tuple[StrensorVar, ...]

    def __str__(self) -> str:
        return ", ".join(map(str, self.vars))

    def get_kernel_variables(self) -> Iterable[StrensorVar]:
        for var in self.vars:
            if var.type == StrensorVarType.KERNEL:
                yield var

    def get_spatial_variables(self) -> Iterable[StrensorVar]:
        for var in self.vars:
            if var.type == StrensorVarType.SPATIAL:
                yield var

    def get_temporal_variables(self) -> Iterable[StrensorVar]:
        for var in self.vars:
            if var.type == StrensorVarType.TEMPORAL:
                yield var


@irdl_attr_definition
class StrensorSpaceAttr(Data[StrensorSpace]):
    name = "stream.ss"

    def print_parameter(self, printer: Printer) -> None:
        """Print the attribute parameter."""
        printer.print_string("<" + str(self.data) + ">")

    @classmethod
    def parse_parameter(cls, parser: AttrParser) -> StrensorSpace:
        """Parse the attribute parameter."""
        parser.parse_characters("<")
        vars = []
        while (var := StrensorVar.parse_optional_var(parser)) is not None:
            vars.append(var)
            parser.parse_optional_characters(",")
        parser.parse_characters(">")
        return StrensorSpace(tuple(vars))


@irdl_attr_definition
class StrensorVarAttr(Data[StrensorVar]):
    name = "stream.sv"

    def print_parameter(self, printer: Printer) -> None:
        """Print the attribute parameter."""
        with printer.in_angle_brackets():
            printer.print_string(str(self.data))

    @classmethod
    def parse_parameter(cls, parser: AttrParser) -> StrensorVar:
        """Parse the attribute parameter."""
        parser.parse_characters("<")
        var = StrensorVar.parse_optional_var(parser)
        assert var is not None
        parser.parse_characters(">")
        return var


@irdl_attr_definition
class StrensorType(
    ParametrizedAttribute,
    TypeAttribute,
    ContainerType[FixedBitwidthType],
):
    name = "stream.strensor"

    # shape: ParameterDef[ArrayAttr[IntAttr]]
    element_type: ParameterDef[FixedBitwidthType]
    ssis: ParameterDef[StrensorSpaceAttr]
    # ssis: ParameterDef[SteadyStateIterationSpaceAttr]
    core_allocation: ParameterDef[ArrayAttr[StringAttr]]
    # how many vars are kept local?
    reuse_index: ParameterDef[IntAttr]

    def __init__(
        self,
        element_type: FixedBitwidthType,
        ssis: StrensorSpace | StrensorSpaceAttr,
        core_allocation: Sequence[StringAttr] | ArrayAttr[StringAttr] = tuple(),
        reuse_index: int | IntAttr = 0,
    ):
        if not isinstance(ssis, StrensorSpaceAttr):
            ssis = StrensorSpaceAttr(ssis)
        if not isinstance(core_allocation, ArrayAttr):
            core_allocation = ArrayAttr(core_allocation)
        if not isinstance(reuse_index, IntAttr):
            reuse_index = IntAttr(reuse_index)
        super().__init__([element_type, ssis, ArrayAttr(core_allocation), reuse_index])

    @property
    def num_reuse_vars(self) -> int:
        return len(self.ssis.data.vars) - self.reuse_index.data

    # def get_num_dims(self) -> int:
    #     return len(self.shape.data)
    #
    def get_relevant_reuse_vars(self) -> Iterable[StrensorVar]:
        # only get reuse vars:
        vars = self.ssis.data.vars[len(self.ssis.data.vars) - self.reuse_index.data :]
        # only get relevant temporal vars
        relevant_dims = {var.dim for var in self.ssis.data.get_kernel_variables()}
        for var in vars:
            if var.type == StrensorVarType.TEMPORAL and var.dim in relevant_dims:
                yield var

    def get_local_shape(self) -> tuple[int, ...]:
        """
        Get the shape of all tensors kept local.
        This does not include the kernel dimensions.
        """
        return tuple(var.size for var in self.get_relevant_reuse_vars())

    def get_kernel_shape(self) -> tuple[int, ...]:
        return tuple(var.size for var in self.ssis.data.get_kernel_variables())

    def get_element_type(self) -> FixedBitwidthType:
        return self.element_type

    def print_parameters(self, printer: Printer) -> None:
        vars = self.ssis.data.vars
        if self.reuse_index.data > 0:
            non_reuse_vars = map(str, vars[: len(vars) - self.reuse_index.data])
            reuse_vars = map(str, vars[len(vars) - self.reuse_index.data :])
            shape = "x".join(non_reuse_vars) + "|" + "x".join(reuse_vars)
        else:
            shape = "x".join(map(str, vars))
        with printer.in_angle_brackets():
            printer.print_string(shape)
            printer.print_string("x")
            printer.print_attribute(self.element_type)
            if isinstance(self.core_allocation, ArrayAttr):
                printer.print_string(", ")
                printer.print_attribute(self.core_allocation)

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> Sequence[Attribute]:
        parser.parse_characters("<")
        vars: list[StrensorVar] = []
        non_reuse_idx = None
        while (var := StrensorVar.parse_optional_var(parser)) is not None:
            vars.append(var)
            if parser.parse_optional_characters("|"):
                non_reuse_idx = len(vars)
            else:
                parser.parse_shape_delimiter()
        ss = StrensorSpaceAttr(StrensorSpace(tuple(vars)))
        element_type = parser.parse_type()
        if parser.parse_optional_characters(",") is not None:
            cores = parser.parse_attribute()
        else:
            cores = NoneAttr()
        parser.parse_characters(">")
        reuse_idx = IntAttr(len(vars) - non_reuse_idx if non_reuse_idx else 0)
        return element_type, ss, cores, reuse_idx


@irdl_op_definition
class InEdgeOp(IRDLOperation):
    name = "stream.in_edge"
    output = result_def(StrensorType)
    tensor = prop_def(StringAttr)

    def __init__(self, name: str, result_type: Attribute):
        super().__init__(
            result_types=(result_type,),
            properties={"tensor": StringAttr(name)},
        )


@irdl_op_definition
class OutEdgeOp(IRDLOperation):
    name = "stream.out_edge"
    inputs = var_operand_def(StrensorType)
    tensor = prop_def(StringAttr)

    def __init__(self, node: OutEdge, inputs: Sequence[SSAValue | Operation]):
        super().__init__(
            operands=(inputs,),
            properties={"tensor": StringAttr(node.name)},
        )


@irdl_op_definition
class GatherOp(IRDLOperation):
    name = "stream.gather"
    inputs = var_operand_def(StrensorType)
    output = result_def(StrensorType)

    traits = traits_def(Pure())

    def __init__(self, inputs: Sequence[SSAValue | Operation], result_type: Attribute):
        super().__init__(operands=(inputs,), result_types=(result_type,))


@irdl_op_definition
class ComputationNodeOp(IRDLOperation):
    name = "stream.computation_node"

    inputs = var_operand_def()
    # output = opt_operand_def()
    output = result_def()

    kernel = prop_def(StringAttr)
    # optional offset for spatially unrolled graphs:
    # offset = opt_prop_def(IntAttr)
    # core_allocation = prop_def(Attribute)
    # ssis = prop_def(SteadyStateIterationSpaceAttr)
    spatial_index = opt_prop_def(StrensorSpaceAttr)

    # irdl_options = [AttrSizedOperandSegments()]

    def __init__(
        self,
        inputs: Sequence[SSAValue | Operation],
        result_types: Sequence[Attribute],
        kernel: str,
        spatial_index: StrensorSpaceAttr | None = None,
        # offset: int | None | IntAttr = None,
        # core_allocation: Attribute,
        # ssis: SteadyStateIterationSpace,
        # outputs: Sequence[SSAValue | Operation] = [],
    ):
        super().__init__(
            operands=(inputs,),
            result_types=result_types,
            properties={
                "kernel": StringAttr(kernel),
                "spatial_index": spatial_index,
                # "core_allocation": core_allocation,
                # "ssis": SteadyStateIterationSpaceAttr(ssis),
            },
        )


@irdl_op_definition
class TransferOp(IRDLOperation):
    name = "stream.transfer"

    inputs = var_operand_def(StrensorType)
    outputs = var_result_def(StrensorType)

    offset = opt_prop_def(IntAttr)
    # offsets = prop_def(DenseArrayBase)
    # sizes = prop_def(DenseArrayBase)
    # strides = prop_def(DenseArrayBase)
    # spatial_strides = prop_def(DenseArrayBase)
    # operand_indeces = prop_def(ArrayAttr[LayerDimAttr])
    # memtile = prop_def(Attribute)

    # ssis = prop_def(SteadyStateIterationSpaceAttr)

    def __init__(  # noqa: PLR0913
        self,
        input: Sequence[SSAValue | Operation],
        result_types: Sequence[Attribute],
        offset: int | None | IntAttr = None,
        # ssis: SteadyStateIterationSpace,
        # offsets: DenseArrayBase | Sequence[int],
        # sizes: DenseArrayBase | Sequence[int],
        # strides: DenseArrayBase | Sequence[int],
        # spatial_strides: DenseArrayBase | Sequence[int],
        # memtile: Attribute,
        # operand_indeces: ArrayAttr[LayerDimAttr],
    ) -> None:
        # if not isinstance(offsets, DenseArrayBase):
        #     offsets = DenseArrayBase.create_dense_int(i64, offsets)
        # if not isinstance(sizes, DenseArrayBase):
        #     sizes = DenseArrayBase.create_dense_int(i64, sizes)
        # if not isinstance(strides, DenseArrayBase):
        #     strides = DenseArrayBase.create_dense_int(i64, strides)
        # if not isinstance(spatial_strides, DenseArrayBase):
        #     spatial_strides = DenseArrayBase.create_dense_int(i64, spatial_strides)
        #
        if isinstance(offset, int):
            offset = IntAttr(offset)
        super().__init__(
            operands=[input],
            result_types=[result_types],
            properties={
                #     # "ssis": SteadyStateIterationSpaceAttr(ssis),
                "offset": offset,
                #     # "sizes": sizes,
                #     # "strides": strides,
                #     # "spatial_strides": spatial_strides,
                #     # "memtile": memtile,
                #     # "operand_indeces": operand_indeces,
            },
        )

    def get_relevant_output(
        self,
        spatial_vars: Sequence[tuple[LayerDim, int]],
        spatio_temporal_vars: Sequence[LayerDim],
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

    def verify_(self) -> None:
        # number of spat vars must correspond to number of core allocations in output
        for output in self.outputs:
            assert isinstance(output.type, StrensorType)
            num_spat_results = prod(x.size for x in output.type.ssis.data.get_spatial_variables())
            if num_spat_results != len(output.type.core_allocation):
                raise VerifyException(
                    f"number of spatial results ({num_spat_results}) does not correspond "
                    f"with number of core allocations ({len(output.type.core_allocation)})"
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
class FusionGroupOp(IRDLOperation):
    name = "stream.fusion_group"

    # inputs = var_operand_def(StrensorType)
    # output = result_def(StrensorType)
    #
    body = region_def("single_block")

    traits = traits_def(IsolatedFromAbove())

    def __init__(
        self,
        # inputs: Sequence[SSAValue | Operation],
        body: Region,
        # result_type: StrensorType,
    ):
        super().__init__(
            # operands=[inputs],
            regions=[body],
            # result_types=[result_type],
        )


@irdl_op_definition
class YieldOp(AbstractYieldOperation[Attribute]):
    name = "stream.yield"

    traits = traits_def(IsTerminator(), Pure())


@irdl_op_definition
class PushOp(IRDLOperation):
    name = "stream.push"

    input = operand_def()
    channel = operand_def()

    spatial_index = opt_prop_def(StrensorSpaceAttr)
    # sizes = prop_def(DenseArrayBase)
    # strides = prop_def(DenseArrayBase)
    # memtile = prop_def(Attribute)
    # operand_indeces = prop_def(ArrayAttr[LayerDimAttr])
    #
    # ssis = prop_def(SteadyStateIterationSpaceAttr)

    def __init__(  # noqa: PLR0913
        self,
        input: SSAValue | Operation,
        channel: SSAValue | Operation,
        spatial_index: StrensorSpaceAttr | None = None,
        # ssis: SteadyStateIterationSpace,
        # sizes: DenseArrayBase | Sequence[int],
        # strides: DenseArrayBase | Sequence[int],
        # memtile: Attribute,
        # operand_indeces: ArrayAttr[LayerDimAttr],
    ) -> None:
        # if not isinstance(offset, IntAttr):
        #     offset = IntAttr(offset)
        # if not isinstance(sizes, DenseArrayBase):
        #     sizes = DenseArrayBase.create_dense_int(i64, sizes)
        # if not isinstance(strides, DenseArrayBase):
        #     strides = DenseArrayBase.create_dense_int(i64, strides)
        super().__init__(
            operands=[
                input,
                channel,
            ],
            properties={
                # "ssis": SteadyStateIterationSpaceAttr(ssis),
                "spatial_index": spatial_index,
                # "sizes": sizes,
                # "strides": strides,
                # "memtile": memtile,
                # "operand_indeces": operand_indeces,
            },
        )


@irdl_op_definition
class PullOp(IRDLOperation):
    name = "stream.pull"

    channel = operand_def()
    output = result_def()

    spatial_index = opt_prop_def(StrensorSpaceAttr)
    # offset = prop_def(IntAttr)
    # sizes = prop_def(DenseArrayBase)
    # strides = prop_def(DenseArrayBase)
    # memtile = prop_def(Attribute)
    # operand = prop_def(IntegerAttr[IndexType])
    # operand_indeces = prop_def(ArrayAttr[LayerDimAttr])

    # ssis = prop_def(SteadyStateIterationSpaceAttr)

    def __init__(  # noqa: PLR0913
        self,
        result_type: Attribute,
        channel: SSAValue | Operation,
        spatial_index: StrensorSpaceAttr | None = None,
        # ssis: SteadyStateIterationSpace,
        #
        # offset: int | IntAttr,
        # sizes: DenseArrayBase | Sequence[int],
        # strides: DenseArrayBase | Sequence[int],
        # memtile: Attribute,
        # operand_indeces: ArrayAttr[LayerDimAttr],
    ) -> None:
        # if not isinstance(offset, IntAttr):
        #     offset = IntAttr(offset)
        #     offsets = DenseArrayBase.create_dense_int(i64, offsets)
        # if not isinstance(sizes, DenseArrayBase):
        #     sizes = DenseArrayBase.create_dense_int(i64, sizes)
        # if not isinstance(strides, DenseArrayBase):
        #     strides = DenseArrayBase.create_dense_int(i64, strides)
        super().__init__(
            operands=[channel],
            result_types=[result_type],
            properties={
                # "ssis": SteadyStateIterationSpaceAttr(ssis),
                "spatial_index": spatial_index,
                # "offset": offset,
                # "sizes": sizes,
                # "strides": strides,
                # "memtile": memtile,
                # "operand_indeces": operand_indeces,
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
        GatherOp,
        FusionGroupOp,
        YieldOp,
    ],
    [
        StrensorType,
        CoreAttr,
        Channel,
        SteadyStateIterationSpaceAttr,
        LayerDimAttr,
        StrensorSpaceAttr,
        StrensorVarAttr,
    ],
)
