from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

from snaxc.dialects.snax import LayoutCast
from snaxc.ir.tsl import Stride, TiledStride, TiledStridedLayout
from xdsl.dialects.builtin import FunctionType, StringAttr
from xdsl.dialects.func import CallOp, FuncOp
from xdsl.dialects.scf import ForOp, IndexSwitchOp, YieldOp
from xdsl.ir import Operation, Region, SSAValue
from xdsl.pattern_rewriter import PatternRewriter
from xdsl.rewriter import InsertPoint
from xdsl.traits import SymbolTable
from xdsl_aie.dialects.aie import CoreOp, DeviceOp

from stream.compiler.dialects.stream import ComputationNodeOp, StrensorVar, StrensorVarAttr
from stream.compiler.kernels import manifest
from stream.compiler.kernels.manifest import CALL_DIMS

# Intrinsic MAC tile of the AIE2p kernels, and the layouts an operand can take.
# mm.cc takes 8 rows when bf16 matmuls run on the bfp16 MACs and 4 when they do not.
R, T = 4, 8
MAC_ROWS_BFP16 = 8
MAC_TILED = "default"
CONTIGUOUS = "contiguous"
VECTOR_LANES = 16
"""Elements a vectorized elementwise kernel loads per step."""


def acquired_object(value: SSAValue) -> SSAValue:
    """The value behind any layout casts, which is where the object fifo acquires it."""
    while isinstance(cast := value.owner, LayoutCast):
        value = cast.source
    return value


def yielded_value(region: Region) -> SSAValue:
    terminator = region.block.last_op
    assert isinstance(terminator, YieldOp)
    return terminator.arguments[0]


def elementwise_operand_layout(m: int, n: int, layout: str, mac_rows: int = R) -> TiledStridedLayout:
    """Layout of one operand of an elementwise kernel.

    An elementwise kernel walks its operands linearly, so it imposes no layout of
    its own; the layout only has to match what the operands already are.
    ``default`` is the r x t tiling a GEMM writes its output in, so an elementwise
    layer fused behind one needs no transformation. ``contiguous`` is plain row
    major, which is what a layer reading from and writing to memory wants: its
    transfers then run the length of a row instead of one MAC tile at a time.
    """
    if layout == CONTIGUOUS:
        return TiledStridedLayout([TiledStride([Stride(n, m)]), TiledStride([Stride(1, n)])])
    mt, nt = m // mac_rows, n // T
    return TiledStridedLayout(
        [
            TiledStride([Stride(mac_rows * T * nt, mt), Stride(T, mac_rows)]),
            TiledStride([Stride(mac_rows * T, nt), Stride(1, T)]),
        ]
    )


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


@dataclass
class AIEKernel(ABC):
    utilization: float

    @property
    def unique_name(self) -> str:
        return self.function_name

    @property
    @abstractmethod
    def linkwith_name(self) -> str: ...

    @property
    @abstractmethod
    def function_name(self) -> str: ...

    @abstractmethod
    def function_type(self, op: ComputationNodeOp) -> FunctionType: ...

    @abstractmethod
    def function_call(self, op: ComputationNodeOp) -> Sequence[Operation]: ...

    def operand_layouts(self) -> Sequence[TiledStridedLayout]:
        return []

    def granule(self) -> list[tuple[int, int]]:
        """The finest tile one call covers, as (parser dimension position, size),
        innermost loop first.

        A fused group that declares no intra-core tiling is tiled at its kernels'
        granules in this nest order; the reduction or carried dimension leads, because
        the output (or running state) stays put only across the innermost loop. Empty
        means this kernel puts no floor under the tiling."""
        return []

    def growable(self) -> tuple[int, ...]:
        """Granule positions the kernel consumes in a run-time loop; a compiled block's
        own dimensions are fixed, so a larger tile there never reaches the kernel."""
        return ()

    def granule_floor(self) -> tuple[int, ...]:
        """Granule positions that stay in the tiling even at full extent.

        A level whose tile equals its extent looks redundant, but some kernels' hand-out
        rides on it: the elementwise row length is what keeps the transfers whole
        contiguous rows, and dropping it turned the slab distribution row-interleaved
        and corrupted the output. Kernels without such a dependence keep the default and
        the level is dropped, which is what the flash lowering requires."""
        return ()

    @property
    def manifest_key(self) -> str:
        """The kernel library entry declaring this source's shapes and costs."""
        return self.function_name

    def call_shape(self) -> dict[str, int]:
        """The dimensions one call covers, by the name the manifest uses."""
        return {name: size for name in CALL_DIMS if (size := getattr(self, name, None))}

    def block_sizes(self) -> dict[int, tuple[int, ...]]:
        """Sizes the compiled source accepts at each granule position, finest first.

        A position absent from the mapping is fixed at its granule value. Empty until a
        kernel library declares otherwise, which is the right answer for a mapper that
        has not been told what it is compiling against."""
        return manifest.blocks(self.manifest_key, self.call_shape())

    def work_share(self, index: int, width: int, steps: int) -> float:
        """Share of a node's work the core at ``index`` of ``width`` does, holding ``steps``
        slices of the split dimension.

        Uniform unless this kernel's iteration space is not rectangular, in which case the
        cores do unequal amounts and latency is set by the busiest. Declared here because
        the kernel that skips the work is the one that knows the shape of what is left."""
        return 1.0 / max(width, 1)

    def validate_shape(self) -> None:
        """Reject a shape the kernel library does not compile this source at.

        Silent with no library attached, which is the only honest answer then."""
        declared, shape = manifest.entry(self.manifest_key), self.call_shape()
        for name, size in declared.get("fixed", {}).items():
            if shape.get(name) != size:
                raise ValueError(
                    f"{self.manifest_key} is compiled with {name}={size}, not {shape.get(name)}"
                )
        if declared.get("blocks"):
            for position, sizes in self.block_sizes().items():
                name = CALL_DIMS[position]
                if shape.get(name) not in sizes:
                    raise ValueError(
                        f"{self.manifest_key} compiles {name} of {sizes}, not {shape.get(name)}"
                    )

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

        # Zero the output buffer where it is acquired; an index switch acquires one per case.
        output = acquired_object(op.inputs[-1])
        if isinstance(switch := output.owner, IndexSwitchOp):
            outputs = [acquired_object(yielded_value(case)) for case in switch.case_regions]
        else:
            outputs = [output]
        for buffer in outputs:
            rewriter.insert_op(self.zero_call(buffer), InsertPoint.after(buffer.owner))

        # Then, rewrite op as before:
        AIEKernel.rewrite(self, op, rewriter)
