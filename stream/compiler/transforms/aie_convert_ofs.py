from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from functools import reduce
from itertools import product
from math import isqrt, prod
from typing import Self, cast

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
    CoreOp,
    DeviceOp,
    DMABDOp,
    EndOp,
    ObjectFIFO,
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

# A dimension a memory tile hands out arrives in a spatial part and a temporal one.
SPLIT_PARTS = 2
# The first row of the array holding compute tiles; below it are the shim and memory rows.
COMPUTE_ROW = 2


def _splittable(var: StrensorVar, dim: LayerDim) -> bool:
    """Whether a variable may be folded into a dimension's split.

    A kernel variable never is: it describes the block itself, not how the block is handed
    out. An absent one may, when it carries the split's own dimension -- an operand the
    dimension does not index still has to say how far the split reaches, and says it by
    being absent over that extent.
    """
    return var.dim == dim and var.type is not StrensorVarType.KERNEL


def _align_spaces(
    in_vars: Sequence[StrensorVar], out_vars: Sequence[StrensorVar]
) -> list[tuple[StrensorVar, StrensorVar]] | None:
    """Pair two steady-state spaces variable by variable, splitting an output variable the
    other side expresses as several.

    A memory tile feeding two compute rows of one layer sees the split it hands out as
    spatial by temporal -- one tile a column, serving each row in turn -- where the cores
    see a single spatial variable of the product. The same holds on the way back, with the
    sides reversed. Returns None when the two spaces do not line up this way, which leaves
    the caller to report it.
    """
    pairs: list[tuple[StrensorVar, StrensorVar]] = []
    i = j = 0
    while i < len(in_vars) and j < len(out_vars):
        source, target = in_vars[i], out_vars[j]
        if source.dim == target.dim and source.size == target.size:
            pairs.append((source, target))
            i, j = i + 1, j + 1
            continue
        if source.dim != target.dim:
            return None
        if target.size > source.size and not target.size % source.size:
            group, rest, k = [source], target.size // source.size, i + 1
            while rest > 1 and k < len(in_vars) and _splittable(in_vars[k], target.dim):
                if rest % in_vars[k].size:
                    break
                group.append(in_vars[k])
                rest //= in_vars[k].size
                k += 1
            if rest != 1:
                return None
            pairs.extend((var, StrensorVar(target.type, var.size, target.dim)) for var in group)
            i, j = k, j + 1
        elif source.size > target.size and not source.size % target.size:
            group, rest, k = [target], source.size // target.size, j + 1
            while rest > 1 and k < len(out_vars) and _splittable(out_vars[k], source.dim):
                if rest % out_vars[k].size:
                    break
                group.append(out_vars[k])
                rest //= out_vars[k].size
                k += 1
            if rest != 1:
                return None
            pairs.extend((StrensorVar(source.type, var.size, source.dim), var) for var in group)
            i, j = i + 1, k
        else:
            return None
    return pairs if i == len(in_vars) and j == len(out_vars) else None


def _consumer_point(groups: Sequence[Sequence[StrensorVar]], outer: dict[LayerDim, int]) -> set[StrensorVar]:
    """The spatial index a consumer carries, from the parts the producer hands out over.

    One part per dimension normally, and this is then the union it always was. Where a
    dimension arrives in two parts because the memory tile splits what the cores hold
    whole, the two recombine the way the runtime descriptor lays them out: the spatial
    part runs fastest, since the stride loops take the spatial variables before the
    temporal ones whatever their order in the space.
    """
    merged: dict[LayerDim, int] = {}
    seen: dict[LayerDim, int] = {}
    for group in groups:
        for var in group:
            seen[var.dim] = seen.get(var.dim, 0) + 1
            if var.dim not in merged:
                merged[var.dim] = var.size
                continue
            if seen[var.dim] > SPLIT_PARTS or var.dim not in outer:
                raise NotImplementedError(
                    f"dimension {var.dim} is handed out in {seen[var.dim]} parts, and the "
                    f"index recombines two whose extents are known"
                )
            merged[var.dim] = var.size * outer[var.dim] + merged[var.dim]
    return {StrensorVar(StrensorVarType.POINT, index, dim) for dim, index in merged.items()}


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

    def split(self, into: int = 1) -> list[tuple[int, Self]]:
        """One (offset, strides) pair per destination fifo, replicas included."""
        spatial_strides = [s for s in self.strides if s.spatial]
        if len(spatial_strides) == 0:
            return [(0, self)] * into
        assert len(spatial_strides) == 1
        spatial_stride = spatial_strides[0]
        idx = self.strides.index(spatial_stride)
        new_strides = self.strides[:idx] + self.strides[idx + 1 :]
        return [(i * spatial_stride.stride, type(self)(new_strides)) for i in range(spatial_stride.size)]

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
        # Exclusive bounds, innermost to outermost. A shim BD encodes the d0 and d1 wraps
        # in 10 bits and the iteration wrap in 6 (mlir-aie ShimBdFieldWidths), so those
        # three are exact. d2 has no wrap field of its own, its extent following the
        # transfer length, so its bound is a conservative stand-in.
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
                    # The bound is exclusive, so the tile has to come strictly under it.
                    # size // bound_limit leaves a tile of exactly the bound whenever the
                    # size is a multiple of it, which is no progress and recurses forever,
                    # so take the smallest divisor that does get under.
                    divider = None
                    for d in range(2, stride.size + 1):
                        if stride.size % d == 0 and stride.size // d < bound_limit:
                            divider = d
                            break
                    if divider is None:
                        raise RuntimeError("Could not find legalized transfer for the runtime sequence.")
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


NB_COLUMNS = 8


def column_of(tile: str) -> int:
    """The column of a tile named ``tile_<column>_<row>``."""
    return int(tile.split("_")[1])


@dataclass
class ChannelToObjectFifoPass(RewritePattern):
    shim_tiles: dict[int, SSAValue]
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
                    point = _consumer_point((spatial, join), {v.dim: v.size for v in spatial_dims})
                    source = next(
                        p for p in producers if p.spatial_index is not None and point <= set(p.spatial_index.data.vars)
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
                target.attributes["relay_dim"] = StringAttr(str(join_dims[0].dim))
                ofs.extend(switch_join)

            else:
                # Unicast: nothing to gather, because the layer holds one core per column
                # and each writes to the memory tile of its own column.
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
                assert isinstance(source_type := source.input.type, StrensorType)
                object_fifo = ObjectFifoOp.from_referenced_type(
                    self.get_tile(source),
                    [self.get_tile(target)],
                    name_base + f"unicast_{i}",
                    (2, 2),
                    source_type.get_element_type(),
                    source_type.get_kernel_shape(),
                )
                source.attributes["of"] = object_fifo.sym_name
                target.attributes["of"] = object_fifo.sym_name
                ofs.append(object_fifo)

        return ofs

    def switch_fork(
        self,
        producers: Sequence[PushOp],
        consumers: Sequence[PullOp],
        spatial: Sequence[StrensorVar],
        spatial_dims: Sequence[StrensorVar],
        fork_dims: Sequence[StrensorVar],
        name_base: str,
        i: int,
    ) -> Sequence[ObjectFifoOp]:
        """One core's fifos to the several it hands to, one for each of them to hold."""
        assert len(fork_dims) == 1
        source = next(
            p for p in producers if p.spatial_index is not None and set(spatial) <= set(p.spatial_index.data.vars)
        )
        assert isinstance(source_type := source.input.type, StrensorType)

        fifos: list[ObjectFifoOp] = []
        for j, fork in enumerate(iterate_spat_vars(fork_dims)):
            point = _consumer_point((spatial, fork), {v.dim: v.size for v in spatial_dims})
            target = next(
                c for c in consumers if c.spatial_index is not None and point <= set(c.spatial_index.data.vars)
            )
            object_fifo = ObjectFifoOp.from_referenced_type(
                self.get_tile(source),
                [self.get_tile(target)],
                name_base + f"switch_fork_{i}_{j}",
                # One object apiece: the core holds one of every fifo it hands to across the
                # whole turn, so a second of each buys no overlap and costs the core the room
                # it needs for the operands it is reading.
                (1, 1),
                source_type.get_element_type(),
                source_type.get_kernel_shape(),
            )
            fifos.append(object_fifo)
            target.attributes["of"] = object_fifo.sym_name

        source.attributes["of"] = ArrayAttr(x.sym_name for x in fifos)
        source.attributes["relay_dim"] = StringAttr(str(fork_dims[0].dim))
        return fifos

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

        # The mirror of a join: one core holds over time what several hold at once.
        fork_dims = [
            x[1] for x in transforms if x[0].type == StrensorVarType.TEMPORAL and x[1].type == StrensorVarType.SPATIAL
        ]

        ofs: list[ObjectFifoOp] = []

        # max one spatial dimension:
        assert len(spatial_dims) <= 1
        for i, spatial in enumerate(iterate_spat_vars(spatial_dims)):
            # Switch Fork Patterns:
            if fork_dims:
                assert len(spatial_dims) + len(fork_dims) == len(transforms)
                ofs.extend(self.switch_fork(producers, consumers, spatial, spatial_dims, fork_dims, name_base, i))

            # Switch Join Patterns:
            elif join_dims:
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
                    point = _consumer_point((spatial, join), {v.dim: v.size for v in spatial_dims})
                    source = next(
                        p for p in producers if p.spatial_index is not None and point <= set(p.spatial_index.data.vars)
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
                target.attributes["relay_dim"] = StringAttr(str(join_dims[0].dim))
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

    def mem_to_compute(  # noqa: PLR0912
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
                        point = _consumer_point(
                            (spatial, distribute, broadcast),
                            {v.dim: v.size for v in spatial_dims},
                        )
                        if point == set(c.spatial_index.data.vars):
                            targets.append(c)

                # gather all broadcast tiles:
                consumer_tiles = tuple(self.get_tile(x) for x in targets)

                if not targets:
                    raise NotImplementedError(f"no consumer carries the spatial index {point}")
                assert isinstance(target_type := targets[0].output.type, StrensorType)

                # number of elements is the kernel shape
                local_shape = target_type.get_local_shape()
                assert len(local_shape) <= 1
                num_elements = max((self.held_count(target_type), prod(local_shape)))

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
            if distribute_dims:
                source.attributes["relay_dim"] = StringAttr(str(distribute_dims[0].dim))
            ofs.extend(spat_ofs)
        return ofs

    def get_tile(self, op: PushOp | PullOp, destination: str = "") -> SSAValue:
        parent = op.parent_op()
        while not isinstance(parent, CoreOp | RuntimeSequenceOp):
            assert parent is not None
            parent = parent.parent_op()
        if isinstance(parent, CoreOp):
            return parent.tile
        # In the runtime sequence the tile is a shim, one per column, so it is the
        # destination's column that picks it and not the row it happens to sit on.
        return self.shim_tiles[column_of(destination)]

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
                    self.held_shape(target_type),
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
                shape=self.held_shape(strensor),
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
                    self.held_shape(source_type),
                )
                switch_join.append(object_fifo)

                # annotate source:
                source.attributes["of"] = object_fifo.sym_name

            # annotate target with all ofs:
            target.attributes["of"] = ArrayAttr(x.sym_name for x in switch_join)
            ofs.extend(switch_join)

        else:
            # Straight (non-join) mem -> shim copy. This is the terminal-output path:
            # when a fused region ends in an element-wise op (e.g. SwiGLU without the
            # down-projection GEMM), the inter-core spatial gather already happened on
            # the compute -> mem transfer, so the mem -> shim transfer is a plain 1:1
            # copy of one mem tile to the shim (host). Mirror of shim_to_mem's
            # non-distribute branch, with producer/consumer roles reversed.
            assert len(producers) == 1
            producer = producers[0]
            assert isinstance(strensor := producer.input.type, StrensorType)
            producer_tile = self.get_tile(producer)
            consumer_tiles = tuple(
                self.get_tile(consumer, strensor.core_allocation.data[0].data) for consumer in consumers
            )
            object_fifo = ObjectFifoOp.from_referenced_type(
                producerTile=producer_tile,
                consumerTiles=consumer_tiles,
                name=name_base + "mem",
                elemNumber=(2, 2),
                referenced_type=strensor.get_element_type(),
                shape=self.held_shape(strensor),
            )
            producer.attributes["of"] = object_fifo.sym_name
            for consumer in consumers:
                consumer.attributes["of"] = object_fifo.sym_name
            ofs.append(object_fifo)

        return ofs

    @staticmethod
    def held_count(strensor: StrensorType) -> int:
        """How many buffers of one fifo element a tile needs to keep its consumer fed.

        Two overlaps the next transfer with the current compute. One is enough when nothing
        varies outside the tensor's reuse window: the fifo then hands over the same data
        every iteration, which is also the single copy the allocator costed it at.
        """
        variables = strensor.ssis.data.vars
        outer = variables[: len(variables) - strensor.reuse_index.data]
        return 2 if any(var.type == StrensorVarType.TEMPORAL for var in outer) else 1

    @classmethod
    def held_shape(cls, strensor: StrensorType) -> tuple[int, ...]:
        """The shape the tile this strensor lives on holds of it.

        A memory tile stages a whole block and redistributes it kernel tile by kernel
        tile, so it holds the block; a core holds a single kernel tile.
        """
        if cls.is_mem(strensor.core_allocation.data[0].data):
            return strensor.get_local_shape() + strensor.get_kernel_shape()
        return strensor.get_kernel_shape()

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
    def match_and_rewrite(self, channel: ChannelOp, rewriter: PatternRewriter):  # noqa: PLR0912, PLR0915
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
        elif (aligned := _align_spaces(in_ss.vars, out_ss.vars)) is not None:
            transformations = [(x, y) for x, y in aligned if StrensorVarType.SPATIAL in (x.type, y.type)]
        else:
            raise NotImplementedError(
                f"a transfer between spaces of different rank is only handled when one "
                f"side is constant or the two line up variable by variable, and neither "
                f"holds here: {in_ss} to {out_ss}"
            )

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

        assert isinstance(pull.output.type, StrensorType)

        ofs_pull = pull.attributes.get("of")
        assert isa(ofs_pull, StringAttr) or isa(ofs_pull, ArrayAttr[StringAttr])
        ofs_push = push.attributes.get("of")
        assert isa(ofs_push, StringAttr) or isa(ofs_push, ArrayAttr[StringAttr])

        # A memory tile gathers from the cores above it and hands out to the cores below,
        # and either side may be several fifos: one row a layer makes one of them single,
        # two rows a layer makes both plural.
        pulls = list(ofs_pull) if isinstance(ofs_pull, ArrayAttr) else [ofs_pull]
        pushes = list(ofs_push) if isinstance(ofs_push, ArrayAttr) else [ofs_push]

        assert (device_op := pull.parent_op()) is not None
        while not isinstance(device_op, DeviceOp):
            assert (device_op := device_op.parent_op()) is not None

        # A layer wider than the one it feeds sends the tile several streams and takes one
        # back, or the other way round. The tile can relay them side by side -- converting
        # each one's layout as it goes, which is the whole reason it is there -- once the
        # thin side has a fifo per stream too.
        if len(pulls) != len(pushes) and min(len(pulls), len(pushes)) == 1:
            pulls, pushes = self.widen_relay(device_op, pull, push, pulls, pushes, rewriter)

        self.match_link_objects(device_op, pulls, pushes)

        def offsets(names: Sequence[StringAttr], across: Sequence[StringAttr]) -> Sequence[int]:
            # A side with one fifo carries no offsets: the dialect reads a non-empty pair of
            # offset lists as a join and a distribute at once, which it rejects. The rest
            # divide the object on the other side between them, so each gets a contiguous
            # share of whatever that side stages -- one round of it or several.
            if len(names) == 1:
                return []
            step = self.object_elements(device_op, across[0]) // len(names)
            return tuple(i * step for i in range(len(names)))

        if len(pulls) > 1 and len(pushes) > 1:
            # The tile has no one buffer to gather into and hand out from -- the dialect
            # takes a join or a distribute, not both -- and it needs none: each core above
            # feeds the core below it through a relay of its own. The two sides have to be
            # handing out over the same dimension for the pairing to mean anything, and
            # the fifos are in that dimension's order on both.
            gathered, handed = pull.attributes.get("relay_dim"), push.attributes.get("relay_dim")
            if len(pulls) != len(pushes) or gathered is None or gathered != handed:
                raise NotImplementedError(
                    f"a memory tile relaying {len(pulls)} fifos over {gathered} to "
                    f"{len(pushes)} over {handed} has no pairing between them"
                )
            links = [
                ObjectFifoLinkOp([SymbolRefAttr(a)], [SymbolRefAttr(b)], [], [])
                for a, b in zip(pulls, pushes, strict=True)
            ]
        else:
            links = [
                ObjectFifoLinkOp(
                    [SymbolRefAttr(o) for o in pulls],
                    [SymbolRefAttr(o) for o in pushes],
                    offsets(pulls, pushes),
                    offsets(pushes, pulls),
                )
            ]

        # insert link near object fifo definition
        last_fifo = ofs_push.data[-1] if isinstance(ofs_push, ArrayAttr) else ofs_push

        last_fifo_op = SymbolTable.lookup_symbol(device_op, last_fifo)
        assert isinstance(last_fifo_op, ObjectFifoOp)
        rewriter.insert_op(links, InsertPoint.after(last_fifo_op))

        rewriter.erase_op(push)
        rewriter.erase_op(pull)

    @classmethod
    def widen_relay(
        cls,
        device: DeviceOp,
        pull: PullOp,
        push: PushOp,
        pulls: Sequence[StringAttr],
        pushes: Sequence[StringAttr],
        rewriter: PatternRewriter,
    ) -> tuple[Sequence[StringAttr], Sequence[StringAttr]]:
        """Give the single side of a lopsided link one fifo per stream on the other.

        The core across it then takes turns between them, as it does when wired to several
        cores directly, and the tile forwards each stream on its own rather than cutting one
        up or gluing several together. Both sides come back unchanged if it cannot widen.
        """
        thin_is_pull = len(pulls) < len(pushes)
        thin, wide = (pulls, pushes) if thin_is_pull else (pushes, pulls)
        thin_op, wide_op = (pull, push) if thin_is_pull else (push, pull)
        over = wide_op.attributes.get("relay_dim")
        if thin_op.attributes.get("relay_dim") is not None or over is None:
            return pulls, pushes
        original = cls.fifo(device, thin[0])
        if not cls.faces_core(original):
            return pulls, pushes
        # The streams together carry what the one fifo carried, so each takes a share rather
        # than a copy: the core holds one of every stream at once.
        depth = cls.share_depth(original.elemNumber, len(wide))
        original.properties["elemNumber"] = depth
        clones = [original]
        for j in range(1, len(wide)):
            clone = ObjectFifoOp(
                original.producerTile,
                list(original.consumerTiles),
                depth,
                original.elemType,
                StringAttr(f"{original.sym_name.data}_relay_{j}"),
                original.dimensionsToStream,
                original.dimensionsFromStreamPerConsumer,
            )
            clones.append(clone)
        rewriter.insert_op(clones[1:], InsertPoint.after(original))
        names = ArrayAttr([f.sym_name for f in clones])
        # The core across the thin side is the one that has to take turns. Every column feeds
        # the same channel, so re-annotate the op already carrying this fifo, not the first.
        wanted = PushOp if thin_is_pull else PullOp
        turning = next(
            use.operation
            for use in thin_op.channel.uses
            if isinstance(use.operation, wanted) and use.operation.attributes.get("of") == thin[0]
        )
        for annotated in (turning, thin_op):
            annotated.attributes["of"] = names
            annotated.attributes["relay_dim"] = over
        widened = list(names)
        return (widened, list(pushes)) if thin_is_pull else (list(pulls), widened)

    @staticmethod
    def _rebuild_depth(held: Attribute, per_endpoint: Callable[[int, int], int]) -> Attribute:
        """A fifo's object count, endpoint by endpoint."""
        if isinstance(held, ArrayAttr):
            return ArrayAttr(
                IntegerAttr.from_int_and_width(per_endpoint(i, n.value.data), 32) for i, n in enumerate(held)
            )
        assert isinstance(held, IntegerAttr)
        return IntegerAttr.from_int_and_width(per_endpoint(0, held.value.data), 32)

    @classmethod
    def scale_depth(cls, held: Attribute, factor: int) -> Attribute:
        """More objects of a smaller kind, so the tile keeps as much in flight as before."""
        return cls._rebuild_depth(held, lambda _, n: n * factor)

    @classmethod
    def share_depth(cls, held: Attribute, streams: int) -> Attribute:
        """The producer's share when one fifo becomes several.

        It holds one object of every stream at once and has nowhere to put a second set. The
        tile keeps what it had: it is the side that has to stay ahead.
        """
        return cls._rebuild_depth(held, lambda i, n: max(1, n // streams) if i == 0 else n)

    @staticmethod
    def fifo(device: DeviceOp, name: StringAttr) -> ObjectFifoOp:
        found = SymbolTable.lookup_symbol(device, name)
        assert isinstance(found, ObjectFifoOp)
        return found

    @staticmethod
    def faces_core(fifo: ObjectFifoOp) -> bool:
        """Whether either end of this fifo is a compute tile, which fixes its object."""
        tiles = [fifo.producerTile, *fifo.consumerTiles]
        rows = [t.owner.row.value.data for t in tiles if isinstance(t.owner, TileOp)]
        return any(row >= COMPUTE_ROW for row in rows)

    @classmethod
    def object_elements(cls, device: DeviceOp, name: StringAttr) -> int:
        return prod(cast(ObjectFIFO[Attribute], cls.fifo(device, name).elemType).buffer.get_shape())

    @classmethod
    def match_link_objects(cls, device: DeviceOp, pulls: Sequence[StringAttr], pushes: Sequence[StringAttr]) -> None:
        """Size a link's single side to the whole of its many side.

        A link moves whole objects: gathering, one object below is filled from one of each
        above; handing out, one above is cut into one for each below. A memory tile sizes its
        buffer by the reuse window it stages, which is more than it needs when it only
        forwards. Only the link sees both sides, so the two are reconciled here.
        """
        if len(pulls) > 1 and len(pushes) > 1:
            return
        many, single = (pulls, pushes) if len(pulls) >= len(pushes) else (pushes, pulls)
        # A core acquires the object its kernel is written for, so it is the tile's side that
        # gives way. With one fifo either side that is what decides which of the two moves.
        if len(many) == 1 and not cls.faces_core(cls.fifo(device, many[0])):
            many, single = single, many
        held = cls.fifo(device, single[0])
        if cls.faces_core(held):
            return
        parts = [cls.object_elements(device, name) for name in many]
        if len(set(parts)) != 1:
            raise NotImplementedError(f"a link handing out objects of differing sizes {parts}")
        need = sum(parts)
        shape = tuple(cast(ObjectFIFO[Attribute], held.elemType).buffer.get_shape())
        # A tile may stage several rounds of the same gather or hand-out, which is its own
        # business: the offsets step by one round either way. Only a side that cannot hold a
        # whole number of them has the wrong object.
        if prod(shape) >= need and not prod(shape) % need:
            return
        kernel = shape[-2:] if len(shape) > 1 else shape
        count, rem = divmod(need, prod(kernel))
        if rem:
            raise NotImplementedError(f"{need} elements is not a whole number of {kernel} blocks")
        element = cast(ObjectFIFO[Attribute], held.elemType).buffer.get_element_type()
        held.properties["elemType"] = ObjectFIFO[Attribute].from_element_type_and_shape(
            element, kernel if count == 1 else (count, *kernel)
        )
        # A smaller object leaves the tile less in flight than it was given, so it keeps as
        # much by holding more of them.
        if need < prod(shape) and not prod(shape) % need:
            held.properties["elemNumber"] = cls.scale_depth(held.elemNumber, prod(shape) // need)


def transfer_endpoints(op: PushOp | PullOp) -> tuple[StrensorType, StrensorType]:
    """The strensor where this transfer first lands and where it finally lands.

    A transfer reaches its compute tile either directly or by way of a memory tile, so
    the chain of pushes and pulls is walked to its end. Both ends coincide for a direct
    transfer, and one descriptor then covers the whole movement.
    """
    if isinstance(op, PushOp):
        stops = [next(u.operation for u in op.channel.uses if isinstance(u.operation, PullOp))]
        while onward := next((u.operation for u in stops[-1].output.uses if isinstance(u.operation, PushOp)), None):
            stops.append(next(u.operation for u in onward.channel.uses if isinstance(u.operation, PullOp)))
        first, last = stops[0].output.type, stops[-1].output.type
    else:
        stops = [next(u.operation for u in op.channel.uses if isinstance(u.operation, PushOp))]
        while isinstance(back := stops[-1].input.owner, PullOp):
            stops.append(next(u.operation for u in back.channel.uses if isinstance(u.operation, PushOp)))
        first, last = stops[0].input.type, stops[-1].input.type
    assert isinstance(first, StrensorType) and isinstance(last, StrensorType)
    return first, last


@dataclass
class TransferToRuntimeSequence(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: PushOp | PullOp, rewriter: PatternRewriter):  # noqa: PLR0912, PLR0915
        if not isinstance(runtime_sequence := op.parent_op(), RuntimeSequenceOp):
            return

        mem_strensor, compute_strensor = transfer_endpoints(op)

        # iterate the zipped mem and compute strensors in reverse (innermost -> outermost)
        def iter_strensors() -> Iterable[tuple[StrensorVar, StrensorVar]]:
            mem_vars, compute_vars = mem_strensor.ssis.data.vars, compute_strensor.ssis.data.vars
            if len(mem_vars) == len(compute_vars):
                yield from zip(reversed(mem_vars), reversed(compute_vars), strict=True)
                return
            # A memory tile serving two compute rows of one layer splits a dimension the
            # cores hold whole, so the two sides differ by a variable.
            pairs = _align_spaces(mem_vars, compute_vars)
            if pairs is None:
                raise NotImplementedError(
                    f"a runtime transfer between spaces that do not line up variable by "
                    f"variable: {mem_strensor.ssis.data} to {compute_strensor.ssis.data}"
                )
            yield from reversed(pairs)

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

        ofs = op.attributes.get("of")
        assert isa(ofs, StringAttr) or isa(ofs, ArrayAttr[StringAttr])
        names = [ofs] if isinstance(ofs, StringAttr) else list(ofs.data)
        groups = [(x, y.canonicalize().legalize()) for x, y in StrideSet(tuple(strides)).split(len(names))]

        for of, (spatial_offset, stride_set) in zip(names, groups, strict=True):
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

            outermost_first = hardware_strides[::-1]

            for r in reduced_ranges:
                dma_bd = DMABDOp(
                    arg,
                    offset=spatial_offset + r.stride,
                    len=prod(var.size for var in hardware_strides[:3]),
                    sizes=[var.size for var in outermost_first],
                    strides=[var.stride for var in outermost_first],
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
    def generate_switch(
        self,
        op: PushOp | PullOp,
        ofs: Sequence[StringAttr],
        strensor: StrensorType,
        rewriter: PatternRewriter,
    ):
        """Take turns between several fifos, one per step of the loop that names them.

        Reading it is a join, over the cores the layer behind is spread across; writing it
        is the mirror. Either way one object of each is held across the turn.
        """
        *_, t_var = strensor.ssis.data.get_temporal_variables()
        for_op = op.parent_op()
        assert isinstance(for_op, ForOp)
        assert isinstance((layer_dim := for_op.attributes.get("layer_dim")), StrensorVarAttr)
        assert layer_dim.data == t_var
        # one acquire per fifo
        acquires = []
        releases = []
        pushing = isinstance(op, PushOp)
        port = ObjectFifoPortEnum.Produce if pushing else ObjectFifoPortEnum.Consume
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
        if pushing:
            assert isinstance(op.input, OpResult)
            assert isinstance(compute := op.input.op, ComputationNodeOp)
            rewriter.replace_op(
                compute,
                ComputationNodeOp(
                    (*compute.inputs, index_switch.results[0]),
                    compute.result_types,
                    compute.kernel.data,
                    compute.spatial_index,
                ),
            )
            op.input.replace_by(index_switch.results[0])
        else:
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

        # index op to select correct access, only when there is something to select between:
        # building it unconditionally would register uses on loop arguments that are never
        # inserted anywhere.
        selecting = reuse_factor > 1
        index_ops: list[Operation] = []
        if selecting:
            index_ops = [
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
            if selecting:
                i_arg = MuliOp(mult_val, for_op.body.block.args[0])
                add_val = AddiOp(add_val, i_arg)
                mult_val = MuliOp(mult_val, for_op.ub)
                index_ops.extend([i_arg, add_val, mult_val])
        if relevant_reuse_vars:
            for_op = for_op.parent_op()

        # A single object needs no selection, and the acquire below already dominates
        # every use of it.
        if not selecting:
            selected = access_ops[0].results[0]
        else:
            index_switch = IndexSwitchOp(
                arg=add_val,
                cases=DenseArrayBase.from_list(IntegerType(64), list(range(reuse_factor))),
                default_region=Region(Block([scf.YieldOp(access_ops[0])])),
                case_regions=[Region(Block([scf.YieldOp(access_ops[i])])) for i in range(reuse_factor)],
                result_types=access_ops[0].result_types,
            )
            index_ops.append(index_switch)
            selected = index_switch.results[0]

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
        relevant_dims = {var.dim for var in strensor.ssis.data.get_kernel_variables()}
        while isinstance(for_op, ForOp):
            assert isinstance((layer_dim := for_op.attributes.get("layer_dim")), StrensorVarAttr)
            if layer_dim.data.dim in relevant_dims:
                break
            for_op = for_op.parent_op()
        # FIXME: end

        # No enclosing loop varies the operand, so it is acquired once around the nest.
        block = for_op.body.block if isinstance(for_op, ForOp) else for_op.region.block
        assert (terminator := block.last_op) is not None
        rewriter.insert_op(release_op, InsertPoint.before(terminator))
        rewriter.insert_op([acquire_op, *access_ops], InsertPoint.at_start(block))

        # set output of computation node op if this was a push op
        if isinstance(op, PushOp):
            assert isinstance(op.input, OpResult)
            assert isinstance(compute := op.input.op, ComputationNodeOp)
            new_compute = ComputationNodeOp(
                (*compute.inputs, selected),
                compute.result_types,
                compute.kernel.data,
                compute.spatial_index,
            )
            rewriter.replace_op(compute, new_compute)

        operand.replace_by(selected)
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
            # TODO: make sure there is no other temporal reuse happening
            self.generate_switch(op, ofs.data, strensor, rewriter)
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
    Insert the waits a runtime sequence needs.

    A fifo ping pongs between two buffer descriptors, so a third transfer on that fifo
    waits for the first. That is about the descriptor rather than the data, so it
    applies whichever way the fifo moves.

    Every transfer still outstanding at the end is waited on as well. For a fifo
    carrying data out that is about the data, since the host may only read an output
    once it has landed. For a fifo carrying data in it is about the descriptor: the
    cores take the data through the fifo's own lock protocol and need no wait, but the
    descriptor stays allocated until it is awaited, and a runtime sequence is not
    always run alone. A fused operator concatenates one sequence per runlist entry, so
    descriptors left outstanding accumulate across entries until a shim tile runs out
    of them.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: RuntimeSequenceOp, rewriter: PatternRewriter):
        device = op.parent_op()
        assert isinstance(device, DeviceOp)
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

        # At the end, wait for every transfer still outstanding, whichever way its fifo
        # moves, so no descriptor is left allocated when the sequence ends.
        for tasklist in active_tasks.values():
            for task in tasklist:
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
        shim_tiles = {column: TileOp(column, 0) for column in range(NB_COLUMNS)}
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
        _verify_shim_fifos_are_fed(op)


def _verify_shim_fifos_are_fed(module: ModuleOp) -> None:
    """Fail the build on a shim fifo nothing fills, which would hang on hardware."""
    shim_rows = {tile.result: tile.row.value.data for tile in module.walk() if isinstance(tile, TileOp)}
    fed = {task.alloc.string_value() for task in module.walk() if isinstance(task, DmaConfigureTaskForOp)}
    starved = [
        fifo.sym_name.data
        for fifo in module.walk()
        if isinstance(fifo, ObjectFifoOp) and shim_rows.get(fifo.producerTile) == 0 and fifo.sym_name.data not in fed
    ]
    if starved:
        raise RuntimeError(
            f"objectfifo(s) produced by a shim tile with no runtime transfer to fill them: {', '.join(sorted(starved))}"
        )
