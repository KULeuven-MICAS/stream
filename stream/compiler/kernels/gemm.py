from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import ceil, prod
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
    MAC_TILED,
    AIEKernel,
    AIEKernelWithZeroing,
    acquired_object,
    device_of,
    tiled_layout,
    yielded_value,
)
from stream.compiler.kernels.binding import Binding, Bindings
from stream.compiler.kernels.library import KernelLibrary


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
    post_ops: tuple[str, ...] = ()
    """The elementwise kernels, by their registry names, the core runs on each output tile once its sum is done, in
    order: the first reads the sum cast to the output's type, each writes the output, and a binary one also reads the
    node's next input after its two operands, at the output tile."""

    @property
    def unique_name(self) -> str:
        return "_".join([f"{self.library_name}_{self.m}_{self.k}_{self.n}", *self.post_ops])

    @property
    def SCRATCH_BITS(self) -> int:  # noqa: N802
        """The bf16 tile the sum is cast into for the post-ops to read, which write the output."""
        return self.m * self.n * self.element_type.bitwidth if self.post_ops else 0

    def post_kernels(self) -> list[AIEKernel]:
        """The post-ops' kernels, called at the output tile."""
        from stream.compiler.kernels.registry import AIE_KERNELS  # noqa: PLC0415

        return [AIE_KERNELS[name](m=self.m, n=self.n, layout=MAC_TILED, library=self.library) for name in self.post_ops]

    def post_op_inputs(self) -> int:
        """The inputs the post-ops read beside the output tile."""
        return sum(len(kernel.operand_layouts()) - 2 for kernel in self.post_kernels())

    def post_op_cycles(self, elements: float) -> int:
        """Cycles the post-ops take over ``elements`` of the output, at their measured calls or their family's rate."""
        total = 0
        for kernel in self.post_kernels():
            spec, library = kernel.spec, cast(KernelLibrary, kernel.library)
            cycles, ops = spec.call_cycles(kernel.call_shape()) or (1, library.families[spec.family].ops_per_cycle)
            total += ceil(cycles * elements / ops)
        return total

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
            *[tiled_layout(self.m, self.n, r, t)] * self.post_op_inputs(),
            tiled_layout(self.m, self.n, r, t),
        ]

    def check_operands(self, op: ComputationNodeOp) -> None:
        """The object file is compiled for one tile, so a mapping that tiles the loop
        differently would read past its operands."""
        assert op.output is not None
        tiles = ((self.m, self.k), (self.k, self.n), *[(self.m, self.n)] * (self.post_op_inputs() + 1))
        for operand, expected in zip(op.inputs, tiles, strict=True):
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

    def bind(self, call: str, dims: dict[str, int], bindings: Bindings) -> Binding:
        """A post-op's call binds as its own kernel at the output tile."""
        for kernel in self.post_kernels():
            if call == kernel.function_name:
                return kernel.bind(call, kernel.call_shape(), bindings)
        return super().bind(call, dims, bindings)

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
            post, cast_into = self._post_op_calls(op, target, size.result, rewriter)
            calls += [cast_call := CallOp("cast", [accumulator, cast_into, size.result], []), *post]
            rewriter.insert_op(
                [acquired, *tile, size, cast_call, *post, ObjectFIFOReleaseOp(port, one, release.objFifo_name)],
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
        extra = list(op.inputs[2:-1])
        super().rewrite(op, rewriter, bindings)
        _erase_unused(output, rewriter)
        for value in extra:
            _read_at_post_ops(value, rewriter)
            name = cast(ObjectFifoAcquireOp, acquired_object(value).owner).objFifo_name.string_value()
            for fifo in device.walk():
                if isinstance(fifo, ObjectFifoOp) and fifo.sym_name.data == name:
                    fifo.elemNumber = ArrayAttr([IntegerAttr.from_int_and_width(1, 32)] * (len(fifo.consumerTiles) + 1))
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
            return [CallOp(self.function_name, [op.inputs[0], op.inputs[1], op.inputs[-1]], [])]
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

    def _post_op_calls(
        self, op: ComputationNodeOp, output: SSAValue, size: SSAValue, rewriter: PatternRewriter
    ) -> tuple[list[CallOp], SSAValue]:
        """The post-ops' calls on one output tile, each reading the tile the one before wrote, the last writing
        ``output``, and the tile the sum is cast into, which the first reads (``output`` with no post-ops). They
        alternate between ``output`` and a tile of the core's own: a kernel may read its operand twice, so none writes
        the tile it reads."""
        kernels = self.post_kernels()
        if not kernels:
            return [], output
        device, tile = device_of(op), _core_tile(op)
        name = f"post_op_tile_{tile.col.value.data}_{tile.row.value.data}"
        buffer = next((b for b in device.walk() if isinstance(b, BufferOp) and b.sym_name.data == name), None)
        if buffer is None:
            buffer = BufferOp(tile.result, self.element_type, ArrayAttr([IntAttr(self.m * self.n)]), StringAttr(name))
            rewriter.insert_op(buffer, InsertPoint.after(tile))
        elif prod(cast(MemRefType[AnyDenseElement], buffer.buffer.type).get_shape()) != self.m * self.n:
            raise ValueError(f"@{name} holds another tile size than kernel {self.unique_name}'s {self.m}x{self.n}")
        extra = iter(op.inputs[2:-1])
        tiles = [buffer.buffer, output] if len(kernels) % 2 else [output, buffer.buffer]
        calls = []
        for index, kernel in enumerate(kernels):
            read, write = tiles[index % 2], tiles[(index + 1) % 2]
            operands = [next(extra) for _ in range(len(kernel.operand_layouts()) - 2)]
            calls.append(CallOp(kernel.function_name, [*operands, read, write, size], []))
        return calls, tiles[0]

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


def _read_at_post_ops(value: SSAValue, rewriter: PatternRewriter) -> None:
    """Acquire a post-op's input, read through ``value``, just before the post-op that reads it and release it right
    after, so the core holds it only once its sum is done, as the core producing it sends it. Producer and reader sum
    in step, so one buffer serves both."""
    (reader,) = {use.operation for use in value.uses}
    acquired = acquired_object(value)
    casts = []
    while isinstance(owner := value.owner, LayoutCast):
        casts.insert(0, owner)
        value = owner.source
    for unread in [use.operation for use in acquired.uses if use.operation not in casts]:
        if isinstance(unread, LayoutCast) and not unread.dest.uses:
            rewriter.erase_op(unread)
    acquire, release = cast(ObjectFifoAcquireOp, acquired.owner), _release(acquired)
    for moved in (acquire, *casts):
        moved.detach()
        rewriter.insert_op(moved, InsertPoint.before(reader))
    release.detach()
    rewriter.insert_op(release, InsertPoint.after(reader))


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
