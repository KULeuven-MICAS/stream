from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import ClassVar, cast

from snaxc.dialects.snax import LayoutCast
from snaxc.ir.tsl import TiledStridedLayout
from xdsl.dialects.arith import ConstantOp
from xdsl.dialects.builtin import AnyDenseElement, ArrayAttr, IntAttr, IntegerAttr, MemRefType, StringAttr, f32, i32
from xdsl.dialects.func import CallOp
from xdsl.dialects.scf import IndexSwitchOp, YieldOp
from xdsl.ir import Attribute, Block, Operation, OpResult, Region, SSAValue
from xdsl.pattern_rewriter import PatternRewriter
from xdsl.rewriter import InsertPoint
from xdsl_aie.dialects.aie import BufferOp, CoreOp, ObjectFifoAcquireOp, ObjectFifoOp, ObjectFIFOReleaseOp, TileOp

from stream.compiler.dialects.stream import ComputationNodeOp
from stream.compiler.kernels.aie_kernel import (
    AIEKernelWithZeroing,
    acquired_object,
    device_of,
    tiled_layout,
    yielded_value,
)
from stream.compiler.kernels.binding import Bindings


@dataclass(kw_only=True)
class GemmKernel(AIEKernelWithZeroing):
    OPERAND_AXES: ClassVar[Mapping[str, tuple[int, int]]] = {"m": (-1, -2), "k": (0, -1), "n": (-1, -1)}
    m: int
    k: int
    n: int
    layout: str
    precision: str = "medium"
    """How the multiplies round, as IRON's ``Precision`` names it: ``medium`` takes A and B to bfp16 where the device
    has bfp16 MACs, ``high`` multiplies in bf16. Each is its own library entry, which says how its object is built,
    its MAC tile and its cycles; ``medium`` is the plain one."""

    @property
    def unique_name(self) -> str:
        return f"{self.library_name}_{self.m}_{self.k}_{self.n}"

    @property
    def library_name(self) -> str:
        """The library's entry: ``medium`` takes the entry of the call that sums into a float accumulator where the
        library describes one, as a provider's binding builds it."""
        if self.precision != "medium":
            return f"{self.function_name}_{self.precision}"
        summed = f"matmul_{self.element_type}_f32"
        return summed if self.library is not None and self.library.spec(summed) is not None else self.function_name

    @property
    def function_name(self) -> str:
        return f"matmul_{self.element_type}_{self.element_type}"

    def operand_layouts(self) -> Sequence[TiledStridedLayout]:
        r, s, t = self.mac["m"], self.mac["k"], self.mac["n"]
        return [
            tiled_layout(self.m, self.k, r, s),
            tiled_layout(self.k, self.n, s, t),
            tiled_layout(self.m, self.n, r, t),
        ]

    def check_operands(self, op: ComputationNodeOp) -> None:
        """The object file is compiled for one tile, so a mapping that tiles the loop
        differently would read past its operands."""
        assert op.output is not None
        for operand, expected in zip(op.inputs, ((self.m, self.k), (self.k, self.n), (self.m, self.n)), strict=True):
            shape = tuple(cast(MemRefType[AnyDenseElement], operand.type).get_shape())
            # Leading axes of extent one are the batch index the call is made at.
            while len(shape) > len(expected) and shape[0] == 1:
                shape = shape[1:]
            if shape != expected:
                raise ValueError(
                    f"kernel {self.unique_name} takes a {expected[0]}x{expected[1]} operand "
                    f"but the tiling gives it {shape}"
                )

    @property
    def ACCUMULATOR(self) -> str | None:  # noqa: N802
        """mm.cc as a provider builds it sums its K tiles into a float buffer of the core's own, as IRON's GEMMs
        accumulate, and casts it to the output after the last; an object built ahead adds into the output."""
        return "fp32" if self.library is not None and self.spec.binding is not None else None

    def binding_options(self) -> dict[str, bool]:
        return {"accumulate": self.ACCUMULATOR is not None}

    def initialize(self, op: ComputationNodeOp, rewriter: PatternRewriter) -> list[CallOp]:
        """Zero each output tile's accumulator where the output is acquired, and once the sums are done send the
        tiles on one at a time: acquire one, cast its accumulator into it, release it."""
        if self.ACCUMULATOR is None:
            return super().initialize(op, rewriter)
        output = acquired_object(op.inputs[-1])
        tiles = _acquired(output)
        accumulators = self._accumulators(op, len(tiles), rewriter)
        calls = []
        for tile, accumulator in zip(tiles, accumulators, strict=True):
            calls.append(zero := CallOp("zero", [accumulator], []))
            rewriter.insert_op(zero, InsertPoint.after(cast(Operation, tile.owner)))
        release = _release(output)
        port, one = release.port, IntegerAttr.from_int_and_width(1, 32)
        shape = cast(MemRefType[AnyDenseElement], tiles[0].type)
        laid_out = _layout(op.inputs[-1])
        for accumulator in accumulators:
            acquired = ObjectFifoAcquireOp(port, 1, release.objFifo_name, shape.get_shape(), shape.element_type)
            tile = [LayoutCast(acquired.results[0], laid_out)] if laid_out is not None else []
            target = tile[0].dest if tile else acquired.results[0]
            size = ConstantOp.from_int_and_width(self.m * self.n, i32)
            calls.append(cast_call := CallOp("cast", [accumulator, target, size.result], []))
            rewriter.insert_op(
                [acquired, *tile, size, cast_call, ObjectFIFOReleaseOp(port, one, release.objFifo_name)],
                InsertPoint.before(release),
            )
        rewriter.erase_op(release)
        return calls

    def rewrite(self, op: ComputationNodeOp, rewriter: PatternRewriter, bindings: Bindings) -> None:
        """Replace ``op`` by its calls; summing in accumulators, the output's early acquire goes, and its fifo, where
        the module declares it, holds the one tile the core sends at a time."""
        if self.ACCUMULATOR is None:
            super().rewrite(op, rewriter, bindings)
            return
        device, output = device_of(op), op.inputs[-1]
        acquires = {cast(ObjectFifoAcquireOp, tile.owner) for tile in _acquired(acquired_object(output))}
        super().rewrite(op, rewriter, bindings)
        _erase_unused(output, rewriter)
        for acquire in acquires:
            name = acquire.objFifo_name.string_value()
            casts = {use.operation for result in acquire.results for use in result.uses}
            for unread in [c for c in casts if isinstance(c, LayoutCast) and not c.dest.uses]:
                rewriter.erase_op(unread)
            if not any(result.uses for result in acquire.results):
                rewriter.erase_op(acquire)
            for fifo in device.walk():
                if isinstance(fifo, ObjectFifoOp) and fifo.sym_name.data == name:
                    fifo.elemNumber = _with_producer_depth(fifo.elemNumber, 1)

    def function_call(self, op: ComputationNodeOp) -> Sequence[Operation]:
        self.check_operands(op)
        if self.ACCUMULATOR is None:
            return [CallOp(self.function_name, [op.inputs[0], op.inputs[1], op.inputs[2]], [])]
        output = acquired_object(op.inputs[-1])
        accumulators = self._accumulators(op, len(_acquired(output)))
        if not isinstance(switch := output.owner, IndexSwitchOp):
            return [CallOp(self.function_name, [op.inputs[0], op.inputs[1], accumulators[0]], [])]
        tiles = _acquired(output)
        picked = IndexSwitchOp(
            switch.arg,
            switch.cases,
            Region(Block([YieldOp(accumulators[tiles.index(acquired_object(yielded_value(switch.default_region)))])])),
            [
                Region(Block([YieldOp(accumulators[tiles.index(acquired_object(yielded_value(case)))])]))
                for case in switch.case_regions
            ],
            [accumulators[0].type],
        )
        return [picked, CallOp(self.function_name, [op.inputs[0], op.inputs[1], picked.results[0]], [])]

    def _accumulators(
        self, op: ComputationNodeOp, count: int, rewriter: PatternRewriter | None = None
    ) -> list[SSAValue]:
        """The core's float buffers the kernel sums its output tiles in, one per tile it holds at once, made once."""
        device, tile = device_of(op), _core_tile(op)
        found = []
        for index in range(count):
            name = f"gemm_accumulator_{tile.col.value.data}_{tile.row.value.data}_{index}"
            buffer = next((b for b in device.walk() if isinstance(b, BufferOp) and b.sym_name.data == name), None)
            if buffer is None:
                assert rewriter is not None
                buffer = BufferOp(tile.result, f32, ArrayAttr([IntAttr(self.m * self.n)]), StringAttr(name))
                rewriter.insert_op(buffer, InsertPoint.after(tile))
            found.append(buffer.buffer)
        return found


def _core_tile(op: Operation) -> TileOp:
    parent = op.parent_op()
    while parent is not None and not isinstance(parent, CoreOp):
        parent = parent.parent_op()
    assert isinstance(parent, CoreOp) and isinstance(parent.tile, OpResult) and isinstance(parent.tile.op, TileOp)
    return parent.tile.op


def _acquired(output: SSAValue) -> list[SSAValue]:
    """Every buffer ``output`` may be acquired into: each case of a switch between them, else itself."""
    if isinstance(switch := output.owner, IndexSwitchOp):
        return [acquired_object(yielded_value(case)) for case in switch.case_regions]
    return [output]


def _layout(output: SSAValue) -> Attribute | None:
    """The laid-out type the kernel writes ``output`` in: its cast's, or that of the casts a switch picks between."""
    if isinstance(output.owner, LayoutCast):
        return output.type
    if isinstance(switch := output.owner, IndexSwitchOp):
        cases = [yielded_value(case) for case in switch.case_regions]
        return next((case.type for case in cases if isinstance(case.owner, LayoutCast)), None)
    return None


def _erase_unused(value: SSAValue, rewriter: PatternRewriter) -> None:
    """Erase what defines ``value`` once nothing reads it, and the casts and switch it is read through."""
    while isinstance(owner := value.owner, Operation) and not any(result.uses for result in owner.results):
        if isinstance(owner, ObjectFifoAcquireOp):
            return
        source = owner.operands[0] if isinstance(owner, LayoutCast) else None
        rewriter.erase_op(owner)
        if source is None:
            return
        value = source


def _with_producer_depth(depth: Attribute, producer: int) -> Attribute:
    """A fifo's depths with its producer's set to ``producer``."""
    if isinstance(depth, ArrayAttr):
        rest = [cast(IntegerAttr, d) for d in depth.data[1:]]
        return ArrayAttr([IntegerAttr.from_int_and_width(producer, 32), *rest])
    consumers = cast(IntegerAttr, depth).value.data
    return ArrayAttr([IntegerAttr.from_int_and_width(producer, 32), IntegerAttr.from_int_and_width(consumers, 32)])


def _release(output: SSAValue) -> ObjectFIFOReleaseOp:
    """The release that sends ``output`` on: the first of its fifo after it, in its block or one enclosing it."""
    acquire = _acquired(output)[0].owner
    assert isinstance(acquire, ObjectFifoAcquireOp)
    after: Operation | None = cast(Operation, output.owner)
    while after is not None:
        follower = after.next_op
        while follower is not None:
            for candidate in follower.walk():
                if isinstance(candidate, ObjectFIFOReleaseOp) and candidate.objFifo_name == acquire.objFifo_name:
                    return candidate
            follower = follower.next_op
        after = after.parent_op()
    raise ValueError(f"no release of @{acquire.objFifo_name.string_value()} follows its acquire")
