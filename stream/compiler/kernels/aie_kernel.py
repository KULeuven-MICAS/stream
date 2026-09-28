from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import ClassVar

from snaxc.dialects.snax import LayoutCast
from snaxc.ir.tsl import Stride, TiledStride, TiledStridedLayout
from xdsl.dialects.builtin import AnyDenseElement, FunctionType, StringAttr, bf16
from xdsl.dialects.func import CallOp, FuncOp
from xdsl.dialects.scf import ForOp, IndexSwitchOp, YieldOp
from xdsl.ir import Operation, Region, SSAValue
from xdsl.pattern_rewriter import PatternRewriter
from xdsl.rewriter import InsertPoint
from xdsl.traits import SymbolTable
from xdsl_aie.dialects.aie import CoreOp, DeviceOp, ObjectFifoAcquireOp

from stream.compiler.dialects.stream import ComputationNodeOp, StrensorVar, StrensorVarAttr
from stream.compiler.kernels.library import CallDim, KernelLibrary, KernelSpec

MAC_TILED = "default"
CONTIGUOUS = "contiguous"


def acquired_object(value: SSAValue) -> SSAValue:
    """The value behind any layout casts, which is where the object fifo acquires it."""
    while isinstance(cast := value.owner, LayoutCast):
        value = cast.source
    return value


def yielded_value(region: Region) -> SSAValue:
    terminator = region.block.last_op
    assert isinstance(terminator, YieldOp)
    return terminator.arguments[0]


def tiled_layout(rows: int, cols: int, tile_rows: int, tile_cols: int) -> TiledStridedLayout:
    """Row-major tiles of tile_rows x tile_cols, each tile row major."""
    col_tiles = cols // tile_cols
    return TiledStridedLayout(
        [
            TiledStride([Stride(tile_rows * tile_cols * col_tiles, rows // tile_rows), Stride(tile_cols, tile_rows)]),
            TiledStride([Stride(tile_rows * tile_cols, col_tiles), Stride(1, tile_cols)]),
        ]
    )


def row_major_layout(rows: int, cols: int) -> TiledStridedLayout:
    return TiledStridedLayout([TiledStride([Stride(cols, rows)]), TiledStride([Stride(1, cols)])])


def elementwise_operand_layout(m: int, n: int, layout: str, mac: Mapping[str, int]) -> TiledStridedLayout:
    """Row major for ``contiguous``, else the tiling a matmul leaves its output in."""
    return row_major_layout(m, n) if layout == CONTIGUOUS else tiled_layout(m, n, mac["m"], mac["n"])


def induction_variable(op: Operation, var: StrensorVar, occurrence: int = 0) -> SSAValue:
    """The induction variable of the enclosing loop iterating ``var``.

    ``IterationSpaceToFor`` names every loop it creates after the variable it drives,
    which is the only way back from a kernel call to where in the iteration space it sits.
    A dimension split into parts of equal size gives loops that name themselves alike, so
    ``occurrence`` picks among them counting from the innermost out -- the order the parts
    themselves are read in.
    """
    parent, seen = op.parent_op(), 0
    while parent is not None:
        if isinstance(parent, ForOp) and parent.attributes.get("layer_dim") == StrensorVarAttr(var):
            if seen == occurrence:
                return parent.body.block.args[0]
            seen += 1
        parent = parent.parent_op()
    raise ValueError(f"kernel call is not inside the loop iterating {var} ({occurrence})")


@dataclass(frozen=True)
class StateOperand:
    """A buffer a kernel keeps on its core from one step of a loop to the next.

    Read at ``carried_over - 1`` and written at ``carried_over``, which is the recurrence
    :func:`~stream.workload.iterator_type.is_state_operand` recognises and what makes that
    dimension SEQUENTIAL. ``rows`` is the extent kept per step; the node supplies the extent
    of ``indexed_by``, so splitting that dimension divides the state with it. ``handover`` is
    how deep a copy the next step of the computation reads, zero for a state kept private.
    """

    name: str
    rows: int
    carried_over: int
    indexed_by: int
    handover: int = 0


@dataclass(kw_only=True)
class AIEKernel(ABC):
    element_type: AnyDenseElement = bf16
    library: KernelLibrary | None = field(default=None, compare=False, repr=False)
    DIMS: ClassVar[Mapping[str, int]] = {}

    @property
    def unique_name(self) -> str:
        return self.function_name

    @property
    def library_key(self) -> str:
        return self.function_name

    @property
    def spec(self) -> KernelSpec:
        if self.library is None:
            raise ValueError(f"{type(self).__name__} needs a kernel library")
        spec = self.library.spec(self.library_key)
        if spec is None:
            raise ValueError(f"the kernel library does not describe {self.library_key}")
        return spec

    @property
    def mac(self) -> Mapping[str, int]:
        if self.library is None:
            raise ValueError(f"{type(self).__name__} needs a kernel library")
        return self.library.mac

    def call_shape(self) -> dict[str, int]:
        return {d.name: int(getattr(self, d.name)) for d in self.spec.dims}

    def call_tile(self) -> list[tuple[int, int, CallDim]]:
        """Each call dimension the library declares, as (node dimension position, size, declaration)."""
        return [(self.DIMS[d.name], int(getattr(self, d.name)), d) for d in self.spec.dims]

    def validate(self) -> None:
        self.spec.validate(self.call_shape())

    @property
    def linkwith_name(self) -> str:
        return self.spec.object.format(**self.call_shape())

    @property
    @abstractmethod
    def function_name(self) -> str: ...

    @abstractmethod
    def function_type(self, op: ComputationNodeOp) -> FunctionType: ...

    @abstractmethod
    def function_call(self, op: ComputationNodeOp) -> Sequence[Operation]: ...

    def operand_layouts(self) -> Sequence[TiledStridedLayout]:
        return []

    def state_operands(self) -> Sequence[StateOperand]:
        """What this kernel keeps in its core between iterations. Empty for a kernel that
        keeps nothing, which is every kernel that is not carrying a running reduction."""
        return []

    def rewrite(self, op: ComputationNodeOp, rewriter: PatternRewriter) -> None:
        # find device op to insert function call
        device_op = op
        while not isinstance(device_op, DeviceOp):
            assert device_op.parent
            device_op = device_op.parent

        SymbolTable.insert_or_update(
            device_op,
            FuncOp(self.function_name, self.function_type(op), Region(), "private"),
        )

        # find core op to set link_with attribute
        core_op = op
        while not isinstance(core_op, CoreOp):
            assert core_op.parent
            core_op = core_op.parent
        core_op.link_with = StringAttr(self.linkwith_name)

        # replace computation node with func call op
        rewriter.insert_op(self.function_call(op), InsertPoint.after(op))
        rewriter.erase_matched_op()


@dataclass
class AIEKernelWithZeroing(AIEKernel, ABC):
    @property
    @abstractmethod
    def zero_name(self) -> str: ...

    @abstractmethod
    def zero_type(self, op: ComputationNodeOp) -> FunctionType: ...

    def zero_call(self, buffer: SSAValue) -> Operation:
        return CallOp(self.zero_name, [buffer], [])

    def rewrite(self, op: ComputationNodeOp, rewriter: PatternRewriter) -> None:
        # find device op to insert zero call
        device_op = op
        while not isinstance(device_op, DeviceOp):
            assert device_op.parent
            device_op = device_op.parent

        SymbolTable.insert_or_update(device_op, FuncOp(self.zero_name, self.zero_type(op), Region(), "private"))

        output = acquired_object(op.inputs[-1])
        if isinstance(switch := output.owner, IndexSwitchOp):
            outputs = [acquired_object(yielded_value(case)) for case in switch.case_regions]
        else:
            outputs = [output]
        for buffer in outputs:
            assert isinstance(buffer.owner, ObjectFifoAcquireOp)
            rewriter.insert_op(self.zero_call(buffer), InsertPoint.after(buffer.owner))

        # Then, rewrite op as before:
        AIEKernel.rewrite(self, op, rewriter)
