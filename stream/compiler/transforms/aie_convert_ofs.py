from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import reduce
from itertools import product
from math import isqrt, prod
from typing import Self

from xdsl.context import Context
from xdsl.dialects import scf
from xdsl.dialects.arith import AddiOp, ConstantOp, MuliOp
from xdsl.dialects.builtin import (
    ArrayAttr,
    DenseArrayBase,
    IndexType,
    IntegerAttr,
    IntegerType,
    MemRefType,
    ModuleOp,
    StringAttr,
    SymbolRefAttr,
    i32,
)
from xdsl.dialects.csl import RewritePattern
from xdsl.dialects.scf import ForOp, IndexSwitchOp
from xdsl.ir import Attribute, Block, Operation, OpResult, Region, SSAValue
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint, Rewriter
from xdsl.traits import SymbolTable
from xdsl.utils.hints import isa
from xdsl_aie.dialects.aie import (
    BDDimLayout,
    BDDimLayoutArray,
    BDDimLayoutArrayAttr,
    CoreOp,
    DeviceOp,
    DMABDOp,
    EndOp,
    ObjectFifoAcquireOp,
    ObjectFifoLinkOp,
    ObjectFifoOp,
    ObjectFifoPortEnum,
    ObjectFIFOReleaseOp,
    ObjectFIFOSubviewAccessOp,
    RuntimeSequenceOp,
    TileOp,
)
from xdsl_aie.dialects.aiex import DmaAwaitTaskOp, DmaConfigureTaskForOp, DmaStartTaskOp

from stream.compiler.dialects.stream import (
    ChannelOp,
    ComputationNodeOp,
    PullOp,
    PushOp,
    StrensorType,
    StrensorVar,
    StrensorVarAttr,
    StrensorVarType,
    YieldOp,
)
from stream.compiler.transforms.unroll import iterate_spat_vars
from stream.datatypes import LayerDim


@dataclass(frozen=True)
class Stride:
    size: int
    stride: int
    iteration_t: int
    spatial: bool = False


@dataclass(frozen=True)
class StrideSet:
    strides: tuple[Stride, ...]

    def repeats(self) -> int:
        return prod(var.size for var in self.strides if not var.stride)

    def size(self) -> int:
        return prod(var.size for var in self.strides if var.stride)

    def total_size(self) -> int:
        return self.repeats() * self.size()

    def split(self) -> dict[int, Self]:
        spatial_strides = [s for s in self.strides if s.spatial]
        if len(spatial_strides) == 0:
            return {0: self}
        assert len(spatial_strides) == 1
        spatial_stride = spatial_strides[0]
        idx = self.strides.index(spatial_stride)
        new_strides = self.strides[:idx] + self.strides[idx + 1 :]
        result = {}
        for i in range(spatial_stride.size):
            result[i * spatial_stride.stride] = type(self)(new_strides)
        return result

    def force_squash(self) -> Self:
        # Remove all transormations, reduce to 1D transfer
        total_size = prod(var.size for var in self.strides if var.stride)
        repeat_size = prod(var.size for var in self.strides if not var.stride)
        return type(self)((Stride(total_size, 1, 0), Stride(repeat_size, 0, 0)))

    def canonicalize(self) -> Self:
        if any(s.spatial for s in self.strides):
            raise RuntimeError("cannot canonicalize strideset with spatial strides")
        new_strides: list[Stride] = []
        for var in self.strides:
            assert var.size != 0
            if var.size == 1:
                continue
            if not new_strides:
                new_strides.append(var)
            # check for possible squash
            elif var.stride == new_strides[-1].size * new_strides[-1].stride:
                new_strides[-1] = Stride(
                    var.size * new_strides[-1].size,
                    new_strides[-1].stride,
                    var.iteration_t // new_strides[-1].size,
                )
            else:
                new_strides.append(var)
        return type(self)(tuple(new_strides))

    def legalize(self) -> Self:  # noqa: PLR0912
        if any(s.spatial for s in self.strides):
            raise RuntimeError("cannot legalize strideset with spatial strides")
        new_strides: list[Stride] = []
        # make sure that no bound limits are exceeded
        # FIXME: figure out actual limits
        # these are innermost to outermost:
        bound_limits = (1024, 1024, 16384, 64)
        for i, (stride, bound_limit) in enumerate(zip(self.strides, bound_limits, strict=False)):
            if stride.size >= bound_limit:
                if i < len(bound_limits) - 1:
                    # find largest number under bound that is a divisor of the size:
                    divider = None
                    for d in reversed(range(min(bound_limit, isqrt(stride.size) + 1))):
                        if stride.size % d == 0:
                            divider = d
                            break
                    if divider is None:
                        raise RuntimeError("Could not find legalized transfer for the runtime sequence.")
                else:
                    divider = stride.size // bound_limit
                tiled_size = stride.size // divider
                tiled_stride = Stride(tiled_size, stride.stride, stride.iteration_t)
                tiling_stride = Stride(
                    divider,
                    stride.stride * tiled_size,
                    stride.iteration_t * tiled_size,
                )
                # tile and legalize recursively
                return type(self)(
                    (
                        *self.strides[:i],
                        tiled_stride,
                        tiling_stride,
                        *self.strides[i + 1 :],
                    )
                ).legalize()
        changed = False
        min_nonzero_strides = 3
        min_total_strides = 4
        for var in self.strides:
            if var.stride == 0 and var.size != 1:
                while len(new_strides) < min_nonzero_strides:
                    changed = True
                    new_strides.append(Stride(1, 0, var.iteration_t))
            new_strides.append(var)
        while len(new_strides) < min_total_strides:
            changed = True
            new_strides.append(Stride(1, 0, self.strides[-1].iteration_t))
        new = type(self)(tuple(new_strides))
        if changed:
            return new.legalize()
        else:
            return new


@dataclass
class ChannelToObjectFifoPass(RewritePattern):
    shim_tiles: dict[str, SSAValue]
    of_count: int = 0
    """
    Converts channels to object fifo definitions
    """

    def compute_to_mem(
        self,
        producers: Sequence[PushOp],
        consumers: Sequence[PullOp],
        transforms: Sequence[tuple[StrensorVar, StrensorVar]],
        name_base: str,
    ) -> Sequence[ObjectFifoOp]:
        spatial_dims = [
            x[1] for x in transforms if x[0].type == StrensorVarType.SPATIAL and x[1].type == StrensorVarType.SPATIAL
        ]

        join_dims = [
            x[0] for x in transforms if x[0].type == StrensorVarType.SPATIAL and x[1].type == StrensorVarType.TEMPORAL
        ]

        ofs: list[ObjectFifoOp] = []

        # max one spatial dimension:
        assert len(spatial_dims) <= 1
        for i, spatial in enumerate(iterate_spat_vars(spatial_dims)):
            # Join Patterns:
            if join_dims:
                assert len(join_dims) == 1
                assert len(spatial_dims) + len(join_dims) == len(transforms)

                # find correct target:
                target = next(
                    c
                    for c in consumers
                    if c.spatial_index is not None and set(spatial) <= set(c.spatial_index.data.vars)
                )

                switch_join: list[ObjectFifoOp] = []

                # find correct consumer:
                for j, join in enumerate(iterate_spat_vars(join_dims)):
                    source = next(
                        p
                        for p in producers
                        if p.spatial_index is not None and set(spatial) | set(join) <= set(p.spatial_index.data.vars)
                    )

                    assert isinstance(source_type := source.input.type, StrensorType)

                    # number of elements is the kernel shape
                    local_shape = source_type.get_local_shape()
                    assert len(local_shape) <= 1
                    num_elements = max((2, prod(local_shape)))

                    object_fifo = ObjectFifoOp.from_referenced_type(
                        self.get_tile(source),
                        [self.get_tile(target)],
                        name_base + f"join_{i}_{j}",
                        (num_elements, 2),
                        source_type.get_element_type(),
                        source_type.get_kernel_shape(),
                    )
                    switch_join.append(object_fifo)

                    # annotate source:
                    source.attributes["of"] = object_fifo.sym_name

                # annotate target with all ofs:
                target.attributes["of"] = ArrayAttr(x.sym_name for x in switch_join)
                ofs.extend(switch_join)

            else:
                raise NotImplementedError()

        return ofs

    def compute_to_compute(
        self,
        producers: Sequence[PushOp],
        consumers: Sequence[PullOp],
        transforms: Sequence[tuple[StrensorVar, StrensorVar]],
        name_base: str,
    ) -> Sequence[ObjectFifoOp]:
        spatial_dims = [
            x[1] for x in transforms if x[0].type == StrensorVarType.SPATIAL and x[1].type == StrensorVarType.SPATIAL
        ]

        join_dims = [
            x[0] for x in transforms if x[0].type == StrensorVarType.SPATIAL and x[1].type == StrensorVarType.TEMPORAL
        ]

        broadcast_dims = [
            x[1] for x in transforms if x[0].type == StrensorVarType.ABSENT and x[1].type == StrensorVarType.SPATIAL
        ]

        ofs: list[ObjectFifoOp] = []

        # max one spatial dimension:
        assert len(spatial_dims) <= 1
        for i, spatial in enumerate(iterate_spat_vars(spatial_dims)):
            # Switch Join Patterns:
            if join_dims:
                assert len(join_dims) == 1
                assert len(spatial_dims) + len(join_dims) == len(transforms)

                # find correct target:
                target = next(
                    c
                    for c in consumers
                    if c.spatial_index is not None and set(spatial) <= set(c.spatial_index.data.vars)
                )

                switch_join: list[ObjectFifoOp] = []

                assert isinstance(target_type := target.output.type, StrensorType)

                # find correct consumer:
                for j, join in enumerate(iterate_spat_vars(join_dims)):
                    source = next(
                        p
                        for p in producers
                        if p.spatial_index is not None and set(spatial) | set(join) <= set(p.spatial_index.data.vars)
                    )

                    object_fifo = ObjectFifoOp.from_referenced_type(
                        self.get_tile(source),
                        [self.get_tile(target)],
                        name_base + f"switch_join_{i}_{j}",
                        (2, 2),  # TODO: correct object fifo depth for switch joins
                        target_type.get_element_type(),
                        target_type.get_kernel_shape(),
                    )
                    switch_join.append(object_fifo)

                    # annotate source:
                    source.attributes["of"] = object_fifo.sym_name

                # annotate target with all ofs:
                target.attributes["of"] = ArrayAttr(x.sym_name for x in switch_join)
                ofs.extend(switch_join)

            elif len(join_dims) == 0 and len(broadcast_dims) == 0:
                # simple unicast pattern:
                source = next(
                    p
                    for p in producers
                    if p.spatial_index is not None and set(spatial) <= set(p.spatial_index.data.vars)
                )
                target = next(
                    c
                    for c in consumers
                    if c.spatial_index is not None and set(spatial) <= set(c.spatial_index.data.vars)
                )
                assert isinstance(target_type := target.output.type, StrensorType)
                object_fifo = ObjectFifoOp.from_referenced_type(
                    self.get_tile(source),
                    [self.get_tile(target)],
                    name_base + f"unicast_{i}",
                    (2, 2),  # TODO: correct object fifo depth for unicasts
                    target_type.get_element_type(),
                    target_type.get_kernel_shape(),
                )
                # annotate push / pull:
                source.attributes["of"] = object_fifo.sym_name
                target.attributes["of"] = object_fifo.sym_name
                ofs.append(object_fifo)

            elif len(broadcast_dims) == 1:
                assert len(broadcast_dims) == 1
                assert len(spatial_dims) + len(broadcast_dims) == len(transforms)

                # find correct source
                source = next(
                    p
                    for p in producers
                    if p.spatial_index is not None and set(spatial) <= set(p.spatial_index.data.vars)
                )

                targets = [
                    c
                    for c in consumers
                    for broadcast in iterate_spat_vars(broadcast_dims)
                    if c.spatial_index is not None and set(spatial) | set(broadcast) <= set(c.spatial_index.data.vars)
                ]

                assert isinstance(target_type := targets[0].output.type, StrensorType)

                object_fifo = ObjectFifoOp.from_referenced_type(
                    self.get_tile(source),
                    [self.get_tile(t) for t in targets],
                    name_base + f"broadcast_{i}",
                    (2,) * (1 + len(targets)),  # TODO: correct object fifo depth for broadcasts
                    target_type.get_element_type(),
                    target_type.get_kernel_shape(),
                )

                # annotate source:
                source.attributes["of"] = object_fifo.sym_name
                for target in targets:
                    # annotate target with all ofs:
                    target.attributes["of"] = object_fifo.sym_name
                ofs.append(object_fifo)

            else:
                raise NotImplementedError()

        return ofs

    def mem_to_compute(
        self,
        producers: Sequence[PushOp],
        consumers: Sequence[PullOp],
        transforms: Sequence[tuple[StrensorVar, StrensorVar]],
        name_base: str,
    ) -> Sequence[ObjectFifoOp]:
        assert isinstance(consumers[0].output.type, StrensorType)
        relevant_dims = {var.dim for var in consumers[0].output.type.ssis.data.get_kernel_variables()}
        broadcast_dims = [
            x[1]
            for x in transforms
            if x[0].type != StrensorVarType.SPATIAL
            and x[1].type == StrensorVarType.SPATIAL
            and x[1].dim not in relevant_dims
        ]

        distribute_dims = [
            x[1]
            for x in transforms
            if x[0].type != StrensorVarType.SPATIAL
            and x[1].type == StrensorVarType.SPATIAL
            and x[1].dim in relevant_dims
        ]

        spatial_dims = [
            x[1] for x in transforms if x[0].type == StrensorVarType.SPATIAL and x[1].type == StrensorVarType.SPATIAL
        ]

        if len(broadcast_dims) > 0:
            if len(distribute_dims) > 0:
                name_base += "distribroad_"
            else:
                name_base += "broadcast_"
        elif len(distribute_dims) > 0:
            name_base += "distribute_"
        else:
            name_base += "unicast_"

        ofs: list[ObjectFifoOp] = []

        for s, spatial in enumerate(iterate_spat_vars(spatial_dims)):
            # find correct source:
            source = next(
                p for p in producers if p.spatial_index is not None and set(spatial) <= set(p.spatial_index.data.vars)
            )

            spat_ofs: list[ObjectFifoOp] = []

            producer_tile = self.get_tile(source)

            # max one distribute dimension:
            if len(distribute_dims) > 1:
                raise NotImplementedError()

            assert len(distribute_dims) <= 1
            for i, distribute in enumerate(iterate_spat_vars(distribute_dims)):
                # get all broadcast targets:
                targets: list[PullOp] = []
                # max one broadcast dimension:
                assert len(broadcast_dims) <= 1
                for broadcast in iterate_spat_vars(broadcast_dims):
                    for c in consumers:
                        assert c.spatial_index is not None
                        # match on spatial index:
                        if set(spatial) | set(distribute) | set(broadcast) == set(c.spatial_index.data.vars):
                            targets.append(c)

                # gather all broadcast tiles:
                consumer_tiles = tuple(self.get_tile(x) for x in targets)

                assert isinstance(target_type := targets[0].output.type, StrensorType)

                # number of elements is the kernel shape
                local_shape = target_type.get_local_shape()
                assert len(local_shape) <= 1
                num_elements = max((2, prod(local_shape)))

                object_fifo = ObjectFifoOp.from_referenced_type(
                    producer_tile,
                    consumer_tiles,
                    name_base + f"{s}_{i}",
                    (2,) + (num_elements,) * len(consumer_tiles),
                    target_type.get_element_type(),
                    target_type.get_kernel_shape(),
                )
                spat_ofs.append(object_fifo)

                # annotate targets:
                for target in targets:
                    target.attributes["of"] = object_fifo.sym_name

            # annotate source
            source.attributes["of"] = ArrayAttr(x.sym_name for x in spat_ofs)
            ofs.extend(spat_ofs)
        return ofs

    def get_tile(self, op: PushOp | PullOp, memtile: str = "") -> SSAValue:
        parent = op.parent_op()
        while not isinstance(parent, CoreOp | RuntimeSequenceOp):
            assert parent is not None
            parent = parent.parent_op()
        if isinstance(parent, CoreOp):
            return parent.tile
        else:  # runtime sequence
            return self.shim_tiles[memtile]

    def shim_to_mem(
        self,
        producer: PushOp,
        consumers: Sequence[PullOp],
        transforms: Sequence[tuple[StrensorVar, StrensorVar]],
        name_base: str,
    ) -> Sequence[ObjectFifoOp]:
        distribute_dims = [x[1] for x in transforms if x[1].type == StrensorVarType.SPATIAL]

        # Distribute Patterns:
        if distribute_dims:
            assert len(distribute_dims) == 1
            assert len(distribute_dims) == len(transforms)

            distributes: list[ObjectFifoOp] = []

            # find correct consumer:
            for i, distribute in enumerate(iterate_spat_vars(distribute_dims)):
                target = next(
                    t
                    for t in consumers
                    if t.spatial_index is not None and set(distribute) <= set(t.spatial_index.data.vars)
                )

                assert isinstance(target_type := target.output.type, StrensorType)

                object_fifo = ObjectFifoOp.from_referenced_type(
                    self.get_tile(producer, target_type.core_allocation.data[0].data),
                    [self.get_tile(target)],
                    name_base + f"mem_{i}",
                    (2, 2),
                    target_type.get_element_type(),
                    target_type.get_local_shape() + target_type.get_kernel_shape(),
                )
                distributes.append(object_fifo)

                # annotate source:
                target.attributes["of"] = object_fifo.sym_name

            # annotate target with all ofs:
            producer.attributes["of"] = ArrayAttr(x.sym_name for x in distributes)

            return distributes

        else:
            assert len(consumers) == 1
            assert isinstance(strensor := consumers[0].output.type, StrensorType)
            consumer_tiles = tuple(map(self.get_tile, consumers))
            producer_tile = self.get_tile(producer, strensor.core_allocation.data[0].data)
            object_fifo = ObjectFifoOp.from_referenced_type(
                producerTile=producer_tile,
                consumerTiles=consumer_tiles,
                name=name_base + "mem",
                elemNumber=(2, 2),
                referenced_type=strensor.get_element_type(),
                shape=strensor.get_local_shape() + strensor.get_kernel_shape(),
            )
            producer.attributes["of"] = object_fifo.sym_name
            for consumer in consumers:
                consumer.attributes["of"] = object_fifo.sym_name
            return (object_fifo,)

    def mem_to_shim(
        self,
        producers: Sequence[PushOp],
        consumers: Sequence[PullOp],
        transforms: Sequence[tuple[StrensorVar, StrensorVar]],
        name_base: str,
    ) -> Sequence[ObjectFifoOp]:
        join_dims = [x[0] for x in transforms if x[0].type == StrensorVarType.SPATIAL]

        ofs: list[ObjectFifoOp] = []

        # Join Patterns:
        if join_dims:
            assert len(join_dims) == 1
            assert len(join_dims) == len(transforms)

            # find correct target:
            assert len(consumers) == 1
            target = consumers[0]

            switch_join: list[ObjectFifoOp] = []

            # find correct consumer:
            for j, join in enumerate(iterate_spat_vars(join_dims)):
                source = next(
                    p for p in producers if p.spatial_index is not None and set(join) <= set(p.spatial_index.data.vars)
                )

                assert isinstance(source_type := source.input.type, StrensorType)

                object_fifo = ObjectFifoOp.from_referenced_type(
                    self.get_tile(source),
                    [self.get_tile(target, source_type.core_allocation.data[0].data)],
                    name_base + f"mem_{j}",
                    (2, 2),
                    source_type.get_element_type(),
                    source_type.get_local_shape() + source_type.get_kernel_shape(),
                )
                switch_join.append(object_fifo)

                # annotate source:
                source.attributes["of"] = object_fifo.sym_name

            # annotate target with all ofs:
            target.attributes["of"] = ArrayAttr(x.sym_name for x in switch_join)
            ofs.extend(switch_join)

        else:
            breakpoint()
            raise NotImplementedError()

        return ofs

    @staticmethod
    def is_shim(tile: str):
        return tile[-1] == "0"

    @staticmethod
    def is_mem(tile: str):
        return tile[-1] == "1"

    @staticmethod
    def is_compute(tile: str):
        return int(tile[-1]) > 1

    @op_type_rewrite_pattern
    def match_and_rewrite(self, channel: ChannelOp, rewriter: PatternRewriter):  # noqa: PLR0912
        if "of" in channel.attributes:
            # already converted
            return

        device_op = channel.parent_op()
        assert isinstance(device_op, DeviceOp)
        # calculate the difference between input and output strensor spaces
        producers: list[PushOp] = []
        consumers: list[PullOp] = []
        for use in channel.channel.uses:
            if isinstance(use.operation, PushOp):
                producers.append(use.operation)
            elif isinstance(use.operation, PullOp):
                consumers.append(use.operation)
            else:
                raise RuntimeError("channel used by non-push/pull operation")
        assert isinstance(in_type := producers[0].input.type, StrensorType)
        assert isinstance(out_type := consumers[0].output.type, StrensorType)
        in_ss = in_type.ssis.data
        out_ss = out_type.ssis.data

        # get ssis transformations of transfer:
        if len(in_ss.vars) == len(out_ss.vars):
            transformations = [
                (x, y)
                for x, y in zip(in_ss.vars, out_ss.vars, strict=True)
                if StrensorVarType.SPATIAL in (x.type, y.type)
            ]
        elif all(v.type == StrensorVarType.CONSTANT for v in in_ss.vars):
            transformations = [(x, x) for x in out_ss.vars if x.type == StrensorVarType.SPATIAL]

        elif all(v.type == StrensorVarType.CONSTANT for v in out_ss.vars):
            transformations = [(x, x) for x in in_ss.vars if x.type == StrensorVarType.SPATIAL]
        else:
            raise NotImplementedError()

        # use dispatcher based on object fifo type:
        name_base = f"of_{self.of_count}_"
        if self.is_shim(in_type.core_allocation.data[0].data):
            assert len(producers) == 1
            ops = self.shim_to_mem(producers[0], consumers, transformations, name_base)
        elif self.is_mem(in_type.core_allocation.data[0].data):
            if self.is_compute(out_type.core_allocation.data[0].data):
                ops = self.mem_to_compute(producers, consumers, transformations, name_base)
            elif self.is_shim(out_type.core_allocation.data[0].data):
                ops = self.mem_to_shim(producers, consumers, transformations, name_base)
            else:
                raise NotImplementedError("going from mem tile to unknown")
        elif self.is_compute(in_type.core_allocation.data[0].data):
            if self.is_compute(out_type.core_allocation.data[0].data):
                ops = self.compute_to_compute(producers, consumers, transformations, name_base)
            elif self.is_mem(out_type.core_allocation.data[0].data):
                ops = self.compute_to_mem(producers, consumers, transformations, name_base)
            else:
                raise NotImplementedError("going from compute tile to unknown")
        else:
            raise NotImplementedError()

        for op in ops:
            del op.properties["repeat_count"]
        self.of_count += 1
        end_op = device_op.region.block.last_op
        assert isinstance(end_op, EndOp)
        rewriter.insert_op(ops, InsertPoint.before(end_op))

        channel.attributes["of"] = StringAttr(name_base)


@dataclass
class RealizeLinks(RewritePattern):
    """
    Converts pull-push pairs into object fifo links
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, pull: PullOp, rewriter: PatternRewriter):
        # looking for push <-> pull pairs
        if not any(isinstance(use.operation, PushOp) for use in pull.output.uses):
            return
        assert len(pull.output.uses) == 1
        push = next(iter(pull.output.uses)).operation
        assert isinstance(push, PushOp)

        assert isinstance((strensor := pull.output.type), StrensorType)

        ofs_pull = pull.attributes.get("of")
        assert isa(ofs_pull, StringAttr) or isa(ofs_pull, ArrayAttr[StringAttr])
        ofs_push = push.attributes.get("of")
        assert isa(ofs_push, StringAttr) or isa(ofs_push, ArrayAttr[StringAttr])

        num_elements = prod(strensor.get_kernel_shape()) * prod(strensor.get_local_shape())
        if isinstance(ofs_pull, ArrayAttr):
            # join link
            assert isinstance(ofs_push, StringAttr)
            link = ObjectFifoLinkOp(
                [SymbolRefAttr(o) for o in ofs_pull],
                [SymbolRefAttr(ofs_push)],
                tuple(range(0, num_elements, num_elements // len(ofs_pull))),
                [],
            )
        elif isinstance(ofs_push, ArrayAttr):
            # distribute link
            assert isinstance(ofs_pull, StringAttr)
            link = ObjectFifoLinkOp(
                [SymbolRefAttr(ofs_pull)],
                [SymbolRefAttr(o) for o in ofs_push],
                [],
                tuple(range(0, num_elements, num_elements // len(ofs_push))),
            )
        else:
            # unicast link
            assert isinstance(ofs_pull, StringAttr)
            assert isinstance(ofs_push, StringAttr)
            link = ObjectFifoLinkOp(
                [SymbolRefAttr(ofs_pull)],
                [SymbolRefAttr(ofs_push)],
                [],
                [],
            )

        # insert link near object fifo definition
        last_fifo = ofs_push.data[-1] if isinstance(ofs_push, ArrayAttr) else ofs_push

        assert (device_op := pull.parent_op()) is not None
        while not isinstance(device_op, DeviceOp):
            assert (device_op := device_op.parent_op()) is not None

        last_fifo_op = SymbolTable.lookup_symbol(device_op, last_fifo)
        assert isinstance(last_fifo_op, ObjectFifoOp)
        rewriter.insert_op(link, InsertPoint.after(last_fifo_op))

        rewriter.erase_op(push)
        rewriter.erase_op(pull)


@dataclass
class TransferToRuntimeSequence(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: PushOp | PullOp, rewriter: PatternRewriter):  # noqa: PLR0912, PLR0915
        if not isinstance(runtime_sequence := op.parent_op(), RuntimeSequenceOp):
            return

        # find strensor type in memtile and compute tile:
        if isinstance(op, PushOp):
            mem_op = next(op.operation for op in op.channel.uses if isinstance(op.operation, PullOp))
            mem_strensor = mem_op.output.type
            assert isinstance(mem_strensor, StrensorType)
            mem_push = next(op.operation for op in mem_op.output.uses)
            assert isinstance(mem_push, PushOp)
            compute_op = next(op.operation for op in mem_push.channel.uses if isinstance(op.operation, PullOp))
            compute_strensor = compute_op.output.type
            assert isinstance(compute_strensor, StrensorType)
        else:
            mem_op = next(op.operation for op in op.channel.uses if isinstance(op.operation, PushOp))
            mem_strensor = mem_op.input.type
            assert isinstance(mem_strensor, StrensorType)
            mem_pull = mem_op.input.owner
            assert isinstance(mem_pull, PullOp)
            compute_op = next(op.operation for op in mem_pull.channel.uses if isinstance(op.operation, PushOp))
            compute_strensor = compute_op.input.type
            assert isinstance(compute_strensor, StrensorType)

        # iterate the zipped mem and compute strensors in reverse (innermost -> outermost)
        def iter_strensors() -> Iterable[tuple[StrensorVar, StrensorVar]]:
            yield from zip(
                reversed(mem_strensor.ssis.data.vars),
                reversed(compute_strensor.ssis.data.vars),
                strict=True,
            )

        vars: list[StrensorVar] = []

        strides: list[Stride] = []

        arg = op.input if isinstance(op, PushOp) else op.output
        constant_strensor = arg.type
        assert isinstance(constant_strensor, StrensorType)

        dim_strides: dict[LayerDim, int] = {}
        mult = 1
        for var in reversed(constant_strensor.ssis.data.vars):
            dim_strides[var.dim] = mult
            mult *= var.size

        iteration_mult = 1

        # first kernel vars:
        for _, var in iter_strensors():
            stride = dim_strides[var.dim] if var.dim in dim_strides else 0
            if var.type == StrensorVarType.KERNEL:
                strides.append(Stride(var.size, stride, iteration_mult))
                vars.append(var)
                if var.dim in dim_strides:
                    dim_strides[var.dim] *= var.size

        # then pure spatial vars:
        for mvar, _ in iter_strensors():
            stride = dim_strides[mvar.dim] if mvar.dim in dim_strides else 0
            if mvar.type == StrensorVarType.SPATIAL:
                vars.append(mvar)
                if mvar.dim in dim_strides:
                    strides.append(Stride(mvar.size, stride, iteration_mult, True))
                    dim_strides[mvar.dim] *= mvar.size
                iteration_mult *= mvar.size

        # next, iterate temporal/absent vars kept local in a memtile
        for i, (mvar, cvar) in enumerate(iter_strensors()):
            stride = dim_strides[cvar.dim] if cvar.dim in dim_strides else 0
            if cvar.type == StrensorVarType.ABSENT and i < mem_strensor.reuse_index.data:
                iteration_mult *= cvar.size
            if cvar.type == StrensorVarType.TEMPORAL and i < mem_strensor.reuse_index.data:
                vars.append(mvar)
                if cvar.dim in dim_strides:
                    strides.append(Stride(cvar.size, stride, iteration_mult))
                    dim_strides[cvar.dim] *= cvar.size
                iteration_mult *= cvar.size

        # broadcast vars timing:
        for mvar, cvar in iter_strensors():
            stride = dim_strides[cvar.dim] if cvar.dim in dim_strides else 0
            if mvar.type == StrensorVarType.ABSENT and cvar.type == StrensorVarType.SPATIAL:
                iteration_mult *= cvar.size

        # then, iterate the join / distribute vars:
        for mvar, cvar in iter_strensors():
            stride = dim_strides[cvar.dim] if cvar.dim in dim_strides else 0
            if mvar.type == StrensorVarType.TEMPORAL and cvar.type == StrensorVarType.SPATIAL:
                vars.append(mvar)
                if cvar.dim in dim_strides:
                    # only add relevant
                    strides.append(Stride(cvar.size, stride, iteration_mult))
                    dim_strides[cvar.dim] *= cvar.size
                iteration_mult *= cvar.size

        # then, remaining vars:
        for i, (mvar, cvar) in enumerate(iter_strensors()):
            stride = dim_strides[cvar.dim] if cvar.dim in dim_strides else 0
            if cvar.type == StrensorVarType.ABSENT and i >= mem_strensor.reuse_index.data:
                iteration_mult *= cvar.size
            if cvar.type == StrensorVarType.TEMPORAL and i >= mem_strensor.reuse_index.data:
                # add stride even if irrelevant for repeated transfers
                strides.append(Stride(cvar.size, stride, iteration_mult))
                iteration_mult *= cvar.size
                vars.append(mvar)
                if cvar.dim in dim_strides:
                    dim_strides[cvar.dim] *= cvar.size

        # print(op)
        # print(compute_strensor)
        # print(mem_strensor)
        # pp(strides)
        # breakpoint()

        stride_dict = StrideSet(tuple(strides)).split()
        # squash weight transformations:
        if op.attributes["of"].data in ("of_1_mem", "of_2_mem", "of_3_mem") and False:
            stride_dict = {x: y.force_squash().legalize() for x, y in stride_dict.items()}
        else:
            stride_dict = {x: y.canonicalize().legalize() for x, y in stride_dict.items()}

        for i, (spatial_offset, stride_set) in enumerate(stride_dict.items()):
            ofs = op.attributes.get("of")
            assert isa(ofs, StringAttr) or isa(ofs, ArrayAttr[StringAttr])
            if isinstance(ofs, StringAttr):
                of = ofs
            else:
                of = ofs.data[i]
            hardware_strides = stride_set.strides[:4]
            # Perform software for loop unrolling:
            software_strides = stride_set.strides[4:]
            software_strides_ranges = [
                [Stride(1, var.stride * i, var.iteration_t * i) for i in range(var.size)] for var in software_strides
            ]
            combined_ranges = list(product(*software_strides_ranges))
            reduced_ranges = [
                reduce(
                    lambda x, y: Stride(1, x.stride + y.stride, x.iteration_t + y.iteration_t),
                    x,
                    Stride(1, 0, 0),
                )
                for x in combined_ranges
            ]

            for r in reduced_ranges:
                bd_dimensions = BDDimLayoutArrayAttr(
                    BDDimLayoutArray([BDDimLayout((var.size, var.stride)) for var in hardware_strides[::-1]])
                )

                dma_bd = DMABDOp(
                    arg,
                    offset=spatial_offset + r.stride,
                    len=prod(var.size for var in hardware_strides[:3]),
                    dimensions=bd_dimensions,
                )

                # configure task
                task = DmaConfigureTaskForOp(
                    of.data,
                    Region(Block([dma_bd, EndOp()])),
                    issue_token=False,
                    repeat_count=hardware_strides[3].size - 1,
                )

                task.attributes["iteration_t"] = IntegerAttr.from_index_int_value(r.iteration_t)

                rewriter.insert_op([task], InsertPoint.before(op))

        # remove yields from pull ops:
        if isinstance(op, PullOp):
            yielded = next(use for use in op.output.uses if isinstance(use.operation, YieldOp))
            assert yielded.index == 0
            op.output.replace_by(runtime_sequence.body.block.args[-1])
            rewriter.erase_op(yielded.operation)

        # remove output from edge op operands
        rewriter.erase_matched_op(safe_erase=False)


@dataclass
class TransferToObjectFIFOPattern(RewritePattern):
    def generate_switch_join(
        self,
        op: PullOp,
        ofs: Sequence[StringAttr],
        strensor: StrensorType,
        rewriter: PatternRewriter,
    ):
        *_, t_var = strensor.ssis.data.get_temporal_variables()
        for_op = op.parent_op()
        assert isinstance(for_op, ForOp)
        assert isinstance((layer_dim := for_op.attributes.get("layer_dim")), StrensorVarAttr)
        assert layer_dim.data == t_var
        # one acquire per fifo
        acquires = []
        releases = []
        port = ObjectFifoPortEnum.Consume
        for of in ofs:
            acquire_op = ObjectFifoAcquireOp(
                IntegerAttr.from_int_and_width(port.get_int(), 32),
                IntegerAttr.from_int_and_width(1, 32),
                object_fifo=of.data,
                shape=strensor.get_kernel_shape(),
                element_type=strensor.get_element_type(),
            )
            acquires.append(acquire_op)
            release_op = ObjectFIFOReleaseOp(
                IntegerAttr.from_int_and_width(port.get_int(), 32),
                IntegerAttr.from_int_and_width(1, 32),
                object_fifo=of.data,
            )
            releases.append(release_op)
        access_ops = [ObjectFIFOSubviewAccessOp(IntegerAttr(0, i32), acquire) for acquire in acquires]
        # toggle between acquires with index switch op:
        index_switch = IndexSwitchOp(
            arg=for_op.body.block.args[0],
            cases=DenseArrayBase.from_list(IntegerType(64), list(range(t_var.size))),
            default_region=Region(Block([scf.YieldOp(access_ops[0])])),
            case_regions=[Region(Block([scf.YieldOp(access_ops[i])])) for i in range(t_var.size)],
            result_types=access_ops[0].result_types,
        )
        # put all acquries before for op:
        rewriter.insert_op(acquires, InsertPoint.before(for_op))
        rewriter.insert_op(access_ops, InsertPoint.before(for_op))
        # put selection in for op:
        rewriter.insert_op(index_switch, InsertPoint.at_start(for_op.body.block))
        # put all releases after for op:
        rewriter.insert_op(releases, InsertPoint.after(for_op))
        # replace use
        op.output.replace_by(index_switch.results[0])
        # delete original op
        rewriter.erase_matched_op()

    def generate_reuse_pattern(  # noqa: PLR0912, PLR0915
        self,
        op: PullOp | PushOp,
        of: str,
        strensor: StrensorType,
        rewriter: PatternRewriter,
    ):
        relevant_reuse_vars = tuple(strensor.get_relevant_reuse_vars())

        # select correct port and operand
        if isinstance(op, PushOp):
            port = ObjectFifoPortEnum.Produce
            operand = op.input
        else:  # pull
            operand = op.output
            port = ObjectFifoPortEnum.Consume

        reuse_factor = prod(strensor.get_local_shape())

        # acquire:
        acquire_op = ObjectFifoAcquireOp(
            IntegerAttr.from_int_and_width(port.get_int(), 32),
            IntegerAttr.from_int_and_width(reuse_factor, 32),
            object_fifo=of,
            shape=strensor.get_kernel_shape(),
            element_type=strensor.get_element_type(),
        )

        # accesses:
        access_ops = [ObjectFIFOSubviewAccessOp(IntegerAttr(i, i32), acquire_op) for i in range(reuse_factor)]

        # index op to select correct access:
        index_ops: list[Operation] = [
            mult_val := ConstantOp.from_int_and_width(1, IndexType()),
            add_val := ConstantOp.from_int_and_width(0, IndexType()),
        ]
        for_op = op.parent_op()
        assert isinstance(for_op, ForOp)
        innermost = None
        # innermost to outermost:
        for iter_var in reversed(relevant_reuse_vars):
            assert isinstance((layer_dim := for_op.attributes.get("layer_dim")), StrensorVarAttr)
            while layer_dim.data != iter_var:
                for_op = for_op.parent_op()
                assert isinstance(for_op, ForOp)
                assert isinstance((layer_dim := for_op.attributes.get("layer_dim")), StrensorVarAttr)
            if innermost is None:
                innermost = for_op
            i_arg = MuliOp(mult_val, for_op.body.block.args[0])
            add_val = AddiOp(add_val, i_arg)
            mult_val = MuliOp(mult_val, for_op.ub)
            index_ops.extend([i_arg, add_val, mult_val])
        if relevant_reuse_vars:
            for_op = for_op.parent_op()
            assert isinstance(for_op, ForOp)
            print(of)

        index_switch = IndexSwitchOp(
            arg=add_val,
            cases=DenseArrayBase.from_list(IntegerType(64), list(range(reuse_factor))),
            default_region=Region(Block([scf.YieldOp(access_ops[0])])),
            case_regions=[Region(Block([scf.YieldOp(access_ops[i])])) for i in range(reuse_factor)],
            result_types=access_ops[0].result_types,
        )
        index_ops.append(index_switch)

        # put index switch at innermost relevant for loop
        if innermost is not None:
            rewriter.insert_op(index_ops, InsertPoint.at_start(innermost.body.block))
        # or just before use if no relevant loops exist:
        elif isinstance(op, PullOp):
            use_op = next(use.operation for use in op.output.uses)
            rewriter.insert_op(index_ops, InsertPoint.before(use_op))
        else:
            assert isinstance(op.input, OpResult)
            use_op = op.input.op
            rewriter.insert_op(index_ops, InsertPoint.before(use_op))

        release_op = ObjectFIFOReleaseOp(
            IntegerAttr.from_int_and_width(port.get_int(), 32),
            IntegerAttr.from_int_and_width(reuse_factor, 32),
            object_fifo=of,
        )

        # FIXME: this is mainly necessary because of bad reuse in output stream IR
        # push insertion point higher until next relevant dimension is found
        # if "of_12" in of:
        #     breakpoint()
        relevant_dims = {var.dim for var in strensor.ssis.data.get_kernel_variables()}
        while True:
            assert isinstance((layer_dim := for_op.attributes.get("layer_dim")), StrensorVarAttr)
            if layer_dim.data.dim in relevant_dims:
                break
            for_op = for_op.parent_op()
            if not isinstance(for_op, ForOp):
                break
        # FIXME: end

        assert (for_yield := for_op.body.block.last_op) is not None
        rewriter.insert_op(release_op, InsertPoint.before(for_yield))
        rewriter.insert_op([acquire_op, *access_ops], InsertPoint.at_start(for_op.body.block))

        # set output of computation node op if this was a push op
        if isinstance(op, PushOp):
            assert isinstance(op.input, OpResult)
            assert isinstance(compute := op.input.op, ComputationNodeOp)
            new_compute = ComputationNodeOp(
                (*compute.inputs, index_switch.results[0]),
                compute.result_types,
                compute.kernel.data,
                compute.spatial_index,
            )
            rewriter.replace_op(compute, new_compute)

        operand.replace_by(index_switch.results[0])
        rewriter.erase_matched_op()

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: PushOp | PullOp, rewriter: PatternRewriter):  # noqa: PLR0912, PLR0915
        # Only handle remaining pull/push ops in core ops, which should be converted to object fifos
        # they are assumed to be compute cores (handle links before this step)
        core_op = op
        while not isinstance(core_op, CoreOp):
            assert (core_op := core_op.parent_op()) is not None

        if isinstance(op, PushOp):
            strensor = op.input.type
        else:
            strensor = op.output.type
        assert isinstance(strensor, StrensorType)
        ofs = op.attributes.get("of")
        assert isa(ofs, ArrayAttr[StringAttr]) or isa(ofs, StringAttr)

        if isinstance(ofs, ArrayAttr):
            assert isinstance(op, PullOp)
            # TODO: make sure there is no other temporal reuse happening
            self.generate_switch_join(op, ofs.data, strensor, rewriter)
        else:
            self.generate_reuse_pattern(op, ofs.data, strensor, rewriter)


class StrensorToMemref(RewritePattern):
    """
    Converts a strensor runtime sequence to a memref one.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: RuntimeSequenceOp, rewriter: PatternRewriter):
        block = op.body.block

        if not any(isinstance(arg, StrensorType) for arg in block.arg_types):
            return

        new_arg_types = [
            MemRefType(arg.element_type, (x.size for x in arg.ssis.data.vars))
            for arg in block.arg_types
            if isinstance(arg, StrensorType)
        ]

        new_op = RuntimeSequenceOp(Region(new_block := Block(arg_types=new_arg_types)))

        # rewrite block args:
        for old_arg, new_arg in zip(block.args, new_block.args, strict=True):
            old_arg.replace_by(new_arg)

        # move ops:
        for block_op in tuple(block.ops):
            block_op.detach()
            rewriter.insert_op(block_op, InsertPoint.at_end(new_block))

        # replace op:
        rewriter.replace_matched_op(new_op)


@dataclass
class OrderDMAs(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: RuntimeSequenceOp, rewriter: PatternRewriter) -> None:
        dma_ops = [
            (op, iteration_t.value.data)
            for op in op.body.block.ops
            if isinstance(op, DmaConfigureTaskForOp)
            if isinstance(iteration_t := op.attributes.get("iteration_t"), IntegerAttr)
        ]
        # sort by iteration_t
        dma_ops = tuple(x[0] for x in sorted(dma_ops, key=lambda x: x[1]))
        for dma_op in dma_ops:
            dma_op.detach()
        rewriter.insert_op(dma_ops, InsertPoint.at_start(op.body.block))


@dataclass
class SyncDMAs(RewritePattern):
    """
    This pass will synchronize dma configure taks ops, inserting wait statements where needed.
    We only allocate one bd per object fifo, and will wait for it to finish every time
    a new transfer for that object fifo is initiated.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: RuntimeSequenceOp, rewriter: PatternRewriter):
        active_tasks: dict[Attribute, list[DmaConfigureTaskForOp]] = {}

        # ping ponging between two bds per object fifo, so we can have at most one active task per object fifo at a time
        nb_bds_per_of = 2

        for dma in op.walk():
            if not isinstance(dma, DmaConfigureTaskForOp):
                continue

            # update active tasks list and potentionaly sync on previous one
            if dma.alloc not in active_tasks:
                active_tasks[dma.alloc] = [dma]
            elif len(active_tasks[dma.alloc]) < nb_bds_per_of:
                active_tasks[dma.alloc].append(dma)
            else:
                assert len(active_tasks[dma.alloc]) == nb_bds_per_of
                to_sync = active_tasks[dma.alloc].pop(0)
                to_sync.issue_token = IntegerAttr.from_int_and_width(1, 1)
                rewriter.insert_op(DmaAwaitTaskOp(to_sync), InsertPoint.before(dma))
                active_tasks[dma.alloc].append(dma)

        # at the end, wait for all latest tasks
        for tasklist in active_tasks.values():
            task = tasklist[-1]
            task.issue_token = IntegerAttr.from_int_and_width(1, 1)
            rewriter.insert_op(DmaAwaitTaskOp(task), InsertPoint.at_end(op.body.block))


@dataclass
class StartDMAs(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: DmaConfigureTaskForOp, rewriter: PatternRewriter):
        rewriter.insert_op(DmaStartTaskOp(op), InsertPoint.after(op))


@dataclass
class RemoveChannels(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ChannelOp, rewriter: PatternRewriter):
        rewriter.erase_matched_op()


@dataclass
class RemoveEmptyCores(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: CoreOp, rewriter: PatternRewriter):
        if isinstance(op.region.block.first_op, EndOp):
            rewriter.erase_matched_op()


class AIEConvertOfs(ModulePass):
    """
    Convert stream transfers into object fifo transfer patterns
    """

    name = "aie-convert-ofs"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        # create new shim tile
        device = next(op for op in op.walk() if isinstance(op, DeviceOp))
        shim_tiles = {
            "tile_0_1": TileOp(0, 0),
            "tile_1_1": TileOp(1, 0),
            "tile_2_1": TileOp(2, 0),
            "tile_3_1": TileOp(3, 0),
            "tile_4_1": TileOp(4, 0),
            "tile_5_1": TileOp(5, 0),
            "tile_6_1": TileOp(6, 0),
            "tile_7_1": TileOp(7, 0),
        }
        PatternRewriteWalker(ChannelToObjectFifoPass({x: y.result for x, y in shim_tiles.items()})).rewrite_module(op)
        Rewriter().insert_op(
            [x for x in shim_tiles.values() if x.result.uses], InsertPoint.at_start(device.region.block)
        )
        PatternRewriteWalker(TransferToRuntimeSequence(), apply_recursively=False).rewrite_module(op)
        PatternRewriteWalker(StrensorToMemref()).rewrite_module(op)
        PatternRewriteWalker(OrderDMAs(), apply_recursively=False).rewrite_module(op)
        PatternRewriteWalker(SyncDMAs(), apply_recursively=False).rewrite_module(op)
        PatternRewriteWalker(StartDMAs(), apply_recursively=False).rewrite_module(op)
        PatternRewriteWalker(RealizeLinks()).rewrite_module(op)
        PatternRewriteWalker(TransferToObjectFIFOPattern()).rewrite_module(op)
        PatternRewriteWalker(RemoveChannels()).rewrite_module(op)
        PatternRewriteWalker(RemoveEmptyCores()).rewrite_module(op)
