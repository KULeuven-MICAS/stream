from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar, cast

from snaxc.dialects.snax import LayoutCast
from snaxc.ir.tsl import Stride, TiledStride, TiledStridedLayout
from xdsl.dialects.builtin import AnyDenseElement, FunctionType, MemRefType, StringAttr, SymbolRefAttr, bf16
from xdsl.dialects.func import CallOp, FuncOp
from xdsl.dialects.scf import ForOp, IndexSwitchOp, YieldOp
from xdsl.ir import Operation, Region, SSAValue
from xdsl.ir.affine import AffineDimExpr
from xdsl.pattern_rewriter import PatternRewriter
from xdsl.printer import Printer
from xdsl.rewriter import InsertPoint
from xdsl.traits import SymbolTable
from xdsl_aie.dialects.aie import DeviceOp, ObjectFifoAcquireOp

from stream.compiler.dialects.stream import ComputationNodeOp, StrensorType, StrensorVar, StrensorVarAttr
from stream.compiler.kernels.binding import Binding, Bindings, check
from stream.compiler.kernels.library import CallDim, KernelLibrary, KernelSpec

if TYPE_CHECKING:
    from stream.workload.node import ComputationNode

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


def device_of(op: Operation) -> DeviceOp:
    parent = op.parent_op()
    while parent is not None and not isinstance(parent, DeviceOp):
        parent = parent.parent_op()
    assert isinstance(parent, DeviceOp)
    return parent


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


class KernelDeclaration(FuncOp):
    """A kernel's private ``func.func``, its symbol quoted where MLIR needs it, which xDSL 0.29 does not do."""

    def print(self, printer: Printer) -> None:
        printer.print(" private @")
        printer.print_identifier_or_string_literal(self.sym_name.data)
        printer.print_attribute(self.function_type)
        printer.print_op_attributes(self.attributes, print_keyword=True)


@dataclass(frozen=True)
class StateOperand:
    """A buffer a kernel keeps on its core from one step of a loop to the next: read at ``carried_over - 1`` and
    written at ``carried_over``, ``rows`` per step times the extent of ``indexed_by``. ``handover`` is how deep a
    copy the next step of the computation reads, zero for a state kept private."""

    name: str
    rows: int
    carried_over: str
    indexed_by: str
    handover: int = 0


@dataclass(kw_only=True)
class AIEKernel(ABC):
    element_type: AnyDenseElement = bf16
    library: KernelLibrary | None = field(default=None, compare=False, repr=False)
    OPERAND_AXES: ClassVar[Mapping[str, tuple[int, int]]] = {}
    """Each call dimension's operand and axis: the operand indexes the node's operands, inputs
    then output (so ``-1`` is the output), and the axis counts from that operand's last. A
    kernel addresses only the trailing axes, so the node's leading ones are batch axes it is
    called once per index of."""
    FALLBACK: ClassVar[Mapping[str, tuple[str, str | None]]] = {}
    """The symbol and object, ``None`` for the library's, stream declares for a call it names otherwise,
    where no provider binds the kernel."""

    @property
    def unique_name(self) -> str:
        return self.function_name

    @property
    def spec(self) -> KernelSpec:
        if self.library is None:
            raise ValueError(f"{type(self).__name__} needs a kernel library")
        spec = self.library.spec(self.function_name)
        if spec is None:
            raise ValueError(f"the kernel library does not describe {self.function_name}")
        return spec

    @property
    def mac(self) -> Mapping[str, int]:
        if self.library is None:
            raise ValueError(f"{type(self).__name__} needs a kernel library")
        return self.library.mac

    def call_shape(self) -> dict[str, int]:
        return {d.name: int(getattr(self, d.name)) for d in self.spec.dims}

    def dim_positions(self, node: "ComputationNode") -> dict[str, int]:
        """Where each of the kernel's dimensions sits in ``node``'s iteration space."""
        positions = {}
        for name, (operand, axis) in self.OPERAND_AXES.items():
            expr = node.operand_mapping[operand].results[axis]
            if not isinstance(expr, AffineDimExpr):
                raise ValueError(f"{node.name}'s {name} axis is not one iteration dimension")
            positions[name] = expr.position
        return positions

    def output_axes(self, op: ComputationNodeOp) -> list:
        """The output dimensions a call's rows and columns run along, where ``OPERAND_AXES`` places ``m`` and ``n``."""
        kernel = [var.dim for var in cast(StrensorType, op.output.type).ssis.data.get_kernel_variables()]
        return [kernel[self.OPERAND_AXES[name][1]] for name in ("m", "n")]

    def call_tile(self, node: "ComputationNode") -> list[tuple[int, int, CallDim]]:
        """Each call dimension the library declares, as (node dimension position, size, declaration)."""
        positions = self.dim_positions(node)
        return [(positions[d.name], int(getattr(self, d.name)), d) for d in self.spec.dims]

    def validate(self) -> None:
        self.spec.validate(self.call_shape())

    def call_dims(self, op: ComputationNodeOp) -> dict[str, int]:
        """The call's dimensions at the extents its operands have, inputs then output as ``OPERAND_AXES``
        counts them: a runtime dimension's call takes the tile it is handed rather than the declared block."""
        shapes = [cast(MemRefType[AnyDenseElement], operand.type).get_shape() for operand in op.inputs]
        extents = {name: shapes[operand][axis] for name, (operand, axis) in self.OPERAND_AXES.items()}
        return self.call_shape() | extents

    def bind(self, call: str, dims: dict[str, int], bindings: Bindings) -> Binding:
        """What a call to ``call`` resolves to: the provider's binding, or stream's own declaration."""
        spec = self.spec
        if spec.binding is not None:
            entry = {"binding": spec.binding, "args": {**dims, "npu": bindings.npu}}
            return bindings.resolve(entry if call == self.function_name else entry | {"companion": call})
        symbol, object_file = self.FALLBACK.get(call, (call, None))
        if (object_file := object_file or spec.object) is None:
            raise ValueError(f"the kernel library names neither a binding nor an object for {self.function_name}")
        return bindings.resolve({"symbol": symbol, "object": object_file})

    @property
    @abstractmethod
    def function_name(self) -> str: ...

    @abstractmethod
    def function_call(self, op: ComputationNodeOp) -> Sequence[Operation]: ...

    def initialize(self, op: ComputationNodeOp, rewriter: PatternRewriter) -> list[CallOp]:
        """The calls that prepare the operands before the kernel's first, inserted where the operands are acquired."""
        return []

    def operand_layouts(self) -> Sequence[TiledStridedLayout]:
        return []

    def work_share(self, index: int, width: int, steps: int) -> float:
        """Share of a node's work the core at ``index`` of ``width`` does, holding ``steps`` slices."""
        return 1.0 / max(width, 1)

    def state_operands(self) -> Sequence[StateOperand]:
        """What this kernel keeps in its core between iterations. Empty for a kernel that
        keeps nothing, which is every kernel that is not carrying a running reduction."""
        return []

    def rewrite(self, op: ComputationNodeOp, rewriter: PatternRewriter, bindings: Bindings) -> None:
        """Replace ``op`` by its calls, each declared as the binding it resolves to, the kernel's own last."""
        device, dims = device_of(op), self.call_dims(op)
        initial = self.initialize(op, rewriter)
        ops = self.function_call(op)
        rewriter.insert_op(ops, InsertPoint.after(op))
        rewriter.erase_matched_op()
        calls = [call for top in ops for call in top.walk() if isinstance(call, CallOp)]
        for call in sorted([*calls, *initial], key=lambda c: c.callee.string_value() == self.function_name):
            binding = self.bind(name := call.callee.string_value(), dims, bindings)
            types = [argument.type for argument in call.arguments]
            check(name, binding, types)
            declaration = KernelDeclaration(binding.name, FunctionType.from_lists(types, []), Region(), "private")
            declaration.attributes["link_with"] = StringAttr(binding.object_file_name)
            SymbolTable.insert_or_update(device, declaration)
            call.properties["callee"] = SymbolRefAttr(binding.name)


@dataclass
class AIEKernelWithZeroing(AIEKernel, ABC):
    FALLBACK: ClassVar[Mapping[str, tuple[str, str | None]]] = {"zero": ("zero_bf16", None)}

    def initialize(self, op: ComputationNodeOp, rewriter: PatternRewriter) -> list[CallOp]:
        """Zero the output, in every buffer it may be acquired into."""
        output = acquired_object(op.inputs[-1])
        if isinstance(switch := output.owner, IndexSwitchOp):
            outputs = [acquired_object(yielded_value(case)) for case in switch.case_regions]
        else:
            outputs = [output]
        zeros = []
        for buffer in outputs:
            assert isinstance(buffer.owner, ObjectFifoAcquireOp)
            zeros.append(zero := CallOp("zero", [buffer], []))
            rewriter.insert_op(zero, InsertPoint.after(buffer.owner))
        return zeros
