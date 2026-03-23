import string
from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from functools import reduce
from itertools import product
from math import isqrt, prod
from typing import Self, cast

from snaxc.dialects.snax import LayoutCast
from snaxc.dialects.tsl import TiledStridedLayoutAttr
from snaxc.ir.tsl import Stride, TiledStride, TiledStridedLayout
from xdsl.context import MLContext
from xdsl.dialects.arith import AddiOp, ConstantOp, MuliOp
from xdsl.dialects.builtin import (
    ArrayAttr,
    DenseArrayBase,
    FixedBitwidthType,
    IndexType,
    IntegerAttr,
    IntegerType,
    MemRefType,
    ModuleOp,
    NoneAttr,
    ShapedType,
    StringAttr,
    SymbolRefAttr,
    i32,
)
from xdsl.dialects.func import CallOp, FuncOp
from xdsl.dialects.scf import ForOp, IndexSwitchOp, YieldOp
from xdsl.ir import Attribute, Operation, OpResult, Region, SSAValue
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint, Rewriter
from xdsl.utils.hints import isa
from xdsl_aie.dialects.aie import (
    AIEDeviceEnum,
    BDDimLayout,
    BDDimLayoutArray,
    BDDimLayoutArrayAttr,
    Block,
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
    ObjectFIFOSubview,
    ObjectFIFOSubviewAccessOp,
    RuntimeSequenceOp,
    SymbolTable,
    TileOp,
)
from xdsl_aie.dialects.aiex import (
    DmaAwaitTaskOp,
    DmaConfigureTaskForOp,
    DmaMemcpyNdOp,
    DmaStartTaskOp,
    DmaWaitOp,
)

from stream.compiler.dialects.stream import (
    ChannelOp,
    ComputationNodeOp,
    InEdgeOp,
    OutEdgeOp,
    PullOp,
    PushOp,
    SteadyStateIterationSpaceAttr,
    TransferOp,
)
from stream.compiler.kernels.aie_kernel import AIEKernel
from stream.compiler.transforms.convert_aie_kernels import ConvertAIEKernels
from stream.compiler.transforms.iteration_space_to_for import iteration_space_to_for
from stream.workload.steady_state.iteration_space import (
    IterationVariable,
    IterationVariableType,
    Reuse,
    SteadyStateIterationSpace,
)


def get_of_name(source: TileOp, dest: TileOp, operand: str) -> str:
    of_name: str = "of_"
    # compute tile specific objectfifos:
    if source.row.value.data > 1 or dest.row.value.data > 1:
        of_name += f"{source.col.value.data}{source.row.value.data}"
        of_name += f"to{dest.col.value.data}{dest.row.value.data}"
    else:  # shim objectififos
        of_name += f"{source.col.value.data}shim"
    of_name += "_" + operand
    return of_name


@dataclass
class TileOpManager:
    device_op: DeviceOp

    tile_ops: dict[tuple[int, int], TileOp] = field(init=False)

    def __post_init__(self):
        self.tile_ops = {}

        # index existing tile ops
        for op in self.device_op.region.walk():
            if isinstance(op, TileOp):
                self.tile_ops[(op.col.value.data, op.row.value.data)] = op

    def insert_or_update(self, x: int, y: int) -> TileOp:
        # return pre-existing op
        if (x, y) in self.tile_ops:
            return self.tile_ops[(x, y)]

        # create and insert op
        rewriter = Rewriter()
        rewriter.insert_op(tile_op := TileOp(x, y), InsertPoint.at_start(self.device_op.region.block))
        tile_op.result.name_hint = f"tile-{x}-{y}"
        self.tile_ops[(x, y)] = tile_op
        return tile_op

    def get_tile(self, operation: Operation) -> TileOp:
        parent = operation
        while True:
            if isinstance(parent, CoreOp):
                assert isinstance(parent.tile, OpResult) and isinstance(parent.tile.op, TileOp)
                return parent.tile.op
            if isinstance(parent, RuntimeSequenceOp):
                if isinstance(operation, PushOp | PullOp):
                    if isinstance(attr := operation.memtile, ArrayAttr):
                        if isinstance(attr := attr.data[0], ArrayAttr):
                            memtile_idx = cast(tuple[IntegerAttr[IndexType], ...], attr.data)
                            return self.insert_or_update(memtile_idx[0].value.data, 0)
                    return self.insert_or_update(0, 0)
            parent = parent.parent_op()
            if parent is None:
                raise RuntimeError()


def is_shim(tile: TileOp) -> bool:
    return tile.row.value.data == 0


@dataclass
class SortPullPushOp:  # noqa: PLW1641 for no hash
    op: PullOp | PushOp

    def __init__(self, op: PullOp | PushOp, tile_op_manager: TileOpManager):
        self.op = op
        self.tile = tile_op_manager.get_tile(op)

    def __lt__(self, other: "SortPullPushOp") -> bool:
        # TODO: this has changed from spatial strides to offsets, make sure this remains correct
        return self.op.offsets.get_values()[-1] < other.op.offsets.get_values()[-1]

    def __eq__(self, other) -> bool:
        if not isinstance(other, SortPullPushOp):
            return False
        return (
            list(reversed(self.op.offsets.get_values())) == list(reversed(other.op.offsets.get_values()))
            and self.tile == other.tile
        )


@dataclass
class ObjectFifoHop:
    fifos: list[ObjectFifoOp]
    DB_EXTRA: int = 0  # 1 for double buffering, 0 for no DB

    @property
    def fifo(self) -> ObjectFifoOp:
        assert len(self.fifos) == 1, "More than one fifo in this hop"
        return self.fifos[0]

    @property
    def start(self) -> Iterable[TileOp]:
        for fifo in self.fifos:
            producer = fifo.producerTile
            assert isinstance(producer, OpResult)
            assert isinstance(producer.op, TileOp)
            yield producer.op

    @property
    def end(self) -> Iterable[TileOp]:
        for fifo in self.fifos:
            for consumer in fifo.consumerTiles:
                assert isinstance(consumer, OpResult)
                assert isinstance(consumer.op, TileOp)
                yield consumer.op

    @classmethod
    def to_memtile(
        cls, producers: Sequence[PushOp], memtiles: Sequence[TileOp], tile_op_manager: TileOpManager, name_base: str
    ) -> Self:
        # when coming from shim, send to custom handler for memtile reuse
        if is_shim(tile_op_manager.get_tile(producers[0])):
            return cls.shim_to_mem(producers[0], memtiles, tile_op_manager, name_base)
        else:
            return cls.compute_to_mem(producers, memtiles, tile_op_manager, name_base)

    @classmethod
    def from_memtile(
        cls, consumers: Sequence[PullOp], memtiles: Sequence[TileOp], tile_op_manager: TileOpManager, name_base: str
    ) -> Self:
        if is_shim(tile_op_manager.get_tile(consumers[0])):
            return cls.mem_to_shim(consumers[0], memtiles, tile_op_manager, name_base)
        else:
            return cls.mem_to_compute(consumers, memtiles, tile_op_manager, name_base)

    @classmethod
    def compute_to_compute(
        cls,
        producers: Sequence[PushOp],
        consumers: Sequence[PullOp],
        tile_op_manager: TileOpManager,
        name_base: str,
    ) -> Self:
        assert isinstance(memref_type := producers[0].input.type, MemRefType)
        assert isinstance(memref_type_consumer := consumers[0].output.type, MemRefType)
        assert memref_type.get_element_type() == memref_type_consumer.get_element_type()
        assert memref_type.get_shape() == memref_type_consumer.get_shape()
        if len(producers) > 1:
            of_type = "switch_join"
        else:
            of_type = "unicast"
        producers = sorted(producers, key=lambda op: SortPullPushOp(op, tile_op_manager))
        producer_tiles = [tile_op_manager.get_tile(producer) for producer in producers]
        consumers = sorted(consumers, key=lambda op: SortPullPushOp(op, tile_op_manager))
        consumer_tiles = [tile_op_manager.get_tile(consumer) for consumer in consumers]
        object_fifos: list[ObjectFifoOp] = []
        for i, producer_tile in enumerate(producer_tiles):
            if len(producers) > 1:
                of_name = name_base + "_" + of_type + "_" + string.ascii_lowercase[i]
            else:
                of_name = name_base + "_" + of_type
            object_fifo = ObjectFifoOp.from_referenced_type(
                elemNumber=(producers[0].ssis.data.nb_local_tensors() + cls.DB_EXTRA,) * (1 + len(consumer_tiles)),
                producerTile=producer_tile,
                consumerTiles=consumer_tiles,
                referenced_type=memref_type.get_element_type(),
                shape=memref_type.get_shape(),
                name=of_name,
            )
            assert isinstance(object_fifo.repeat_count, IntegerAttr)
            if object_fifo.repeat_count.value.data == 1:
                del object_fifo.properties["repeat_count"]
            object_fifos.append(object_fifo)
        return cls(object_fifos)

    @classmethod
    def compute_to_mem(
        cls, producers: Sequence[PushOp], memtiles: Sequence[TileOp], tile_op_manager: TileOpManager, name_base: str
    ) -> Self:
        assert isinstance(memref_type := producers[0].input.type, MemRefType)
        if len(producers) > 1:
            of_type = "join"
        else:
            of_type = "unicast"
        producers = sorted(producers, key=lambda op: SortPullPushOp(op, tile_op_manager))
        producer_tiles = [tile_op_manager.get_tile(producer) for producer in producers]
        object_fifos: list[ObjectFifoOp] = []

        def memtile_selector(i: int):
            if len(memtiles) == 1:
                return memtiles[0]
            spat_vars = producers[0].ssis.data.get_spatial_variables()
            spat_vars = [x for x in spat_vars if x.applicable]
            used_vars = [var for var in spat_vars if var.size == len(memtiles)]
            assert len(used_vars) == 1
            used_var = used_vars[0]
            other_vars = spat_vars[: spat_vars.index(used_var)]
            div = prod(x.size for x in other_vars)
            mod = len(memtiles)
            return memtiles[(i // div) % mod]

        for i, of_producer in enumerate(producer_tiles):
            if len(producers) > 1:
                of_name = name_base + "_" + of_type + "_" + string.ascii_lowercase[i]
            else:
                of_name = name_base + "_" + of_type
            object_fifo = ObjectFifoOp.from_referenced_type(
                elemNumber=(producers[0].ssis.data.nb_local_tensors() + cls.DB_EXTRA,) * 2,
                producerTile=of_producer,
                consumerTiles=[memtile_selector(i)],
                referenced_type=memref_type.get_element_type(),
                shape=memref_type.get_shape(),
                name=of_name,
            )
            del object_fifo.properties["repeat_count"]
            object_fifos.append(object_fifo)
        return cls(object_fifos)

    @classmethod
    def mem_to_compute(  # noqa: PLR0912
        cls, consumers: Sequence[PullOp], memtiles: Sequence[TileOp], tile_op_manager: TileOpManager, name_base: str
    ) -> Self:
        assert isinstance(memref_type := consumers[0].output.type, MemRefType)
        unique_consumers = len(set(x.offsets for x in consumers))
        if len(consumers) > 1:
            # determine whether to broadcast / distribute
            if unique_consumers == 1:
                of_type = "broadcast"
            elif unique_consumers == len(consumers):
                of_type = "distribute"
            else:
                of_type = "distribroad"
        else:
            of_type = "unicast"
        consumers = sorted(consumers, key=lambda op: SortPullPushOp(op, tile_op_manager))
        consumer_tiles = [tile_op_manager.get_tile(consumer) for consumer in consumers]

        def memtile_selector(i: int):
            if len(memtiles) == 1:
                return memtiles[0]
            spat_vars = consumers[0].ssis.data.get_spatial_variables()
            spat_vars = [x for x in spat_vars if x.applicable]
            used_vars = [var for var in spat_vars if var.size == len(memtiles)]
            assert len(used_vars) == 1
            used_var = used_vars[0]
            other_vars = spat_vars[spat_vars.index(used_var) + 1 :]
            div = prod(x.size for x in other_vars)
            mod = len(memtiles)
            return memtiles[(i // div) % mod]

        if of_type == "distribute":
            fifos = [(memtile_selector(i), [tile]) for i, tile in enumerate(consumer_tiles)]
        elif of_type == "distribroad":
            # gather unique consumer tiles
            unique_consumer_tiles = defaultdict(list)
            for consumer in consumers:
                unique_consumer_tiles[consumer.offsets].append(tile_op_manager.get_tile(consumer))
            fifos = [(memtile_selector(i), tiles) for i, tiles in enumerate(unique_consumer_tiles.values())]
        else:  # broadcast or unicast
            assert len(memtiles) == 1
            fifos = [(memtiles[0], consumer_tiles)]
        object_fifos: list[ObjectFifoOp] = []
        # FIXME: remove this stupid hardcoded factor:
        distribroad_factor = 0 if of_type == "distribroad" else 0
        for i, (of_producer, of_consumers) in enumerate(fifos):
            if of_type in ("distribute", "distribroad"):
                of_name = name_base + "_" + of_type + "_" + string.ascii_lowercase[i]
            else:
                of_name = name_base + "_" + of_type
            object_fifo = ObjectFifoOp.from_referenced_type(
                elemNumber=(consumers[0].ssis.data.nb_local_tensors() + cls.DB_EXTRA + distribroad_factor,)
                * (1 + len(of_consumers)),
                producerTile=of_producer,
                consumerTiles=of_consumers,
                referenced_type=memref_type.get_element_type(),
                shape=memref_type.get_shape(),
                name=of_name,
                repeat_count=consumers[0].ssis.data.reuse_factor_mem(),
            )
            assert isinstance(object_fifo.repeat_count, IntegerAttr)
            if object_fifo.repeat_count.value.data == 1:
                del object_fifo.properties["repeat_count"]
            object_fifos.append(object_fifo)
        return cls(object_fifos)

    @classmethod
    def shim_to_mem(
        cls, producer: PushOp, memtiles: Sequence[TileOp], tile_op_manager: TileOpManager, name_base: str
    ) -> Self:
        assert isinstance(memref_type := producer.input.type, MemRefType)
        fifos = []
        for i, memtile in enumerate(memtiles):
            object_fifo = ObjectFifoOp.from_referenced_type(
                elemNumber=(1 + cls.DB_EXTRA, 1 + cls.DB_EXTRA),
                producerTile=tile_op_manager.insert_or_update(memtile.col.value.data, 0),
                consumerTiles=[memtile],
                referenced_type=memref_type.get_element_type(),
                shape=producer.ssis.data.shape_mem(),
                name=name_base + "mem" + "_" + string.ascii_lowercase[i],
                repeat_count=1,
            )
            del object_fifo.properties["repeat_count"]

            if object_fifo.repeat_count is not None and object_fifo.repeat_count.value.data == 1:
                del object_fifo.properties["repeat_count"]
            fifos.append(object_fifo)
        return cls(fifos)

    @classmethod
    def mem_to_shim(
        cls, consumer: PullOp, memtiles: Sequence[TileOp], tile_op_manager: TileOpManager, name_base: str
    ) -> Self:
        assert isinstance(memref_type := consumer.output.type, MemRefType)
        fifos = []
        for i, memtile in enumerate(memtiles):
            object_fifo = ObjectFifoOp.from_referenced_type(
                elemNumber=(1 + cls.DB_EXTRA, 1 + cls.DB_EXTRA),
                producerTile=memtile,
                consumerTiles=[tile_op_manager.insert_or_update(memtile.col.value.data, 0)],
                referenced_type=memref_type.get_element_type(),
                shape=consumer.ssis.data.shape_mem(len(memtiles)),
                name=name_base + "mem" + "_" + string.ascii_lowercase[i],
            )
            del object_fifo.properties["repeat_count"]
            fifos.append(object_fifo)
        return cls(fifos)


@dataclass
class ObjectFifoChain:
    hops: list[ObjectFifoHop]
    links: list[ObjectFifoLinkOp]

    @property
    def start(self) -> Iterable[TileOp]:
        return self.hops[0].start

    @property
    def end(self) -> Iterable[TileOp]:
        return self.hops[-1].end

    @classmethod
    def from_channel(
        cls, channel: SSAValue, memref_type: MemRefType[Attribute], tile_op_manager: TileOpManager, name_base: str
    ):
        # gather consumers / producers
        producers = list(use.operation for use in channel.uses if isinstance(use.operation, PushOp))
        producer_tiles = [tile_op_manager.get_tile(op) for op in producers]
        consumers = list(use.operation for use in channel.uses if isinstance(use.operation, PullOp))
        consumer_tiles = [tile_op_manager.get_tile(op) for op in consumers]

        # determine hops
        hops: Sequence[ObjectFifoHop]
        if is_shim(consumer_tiles[0]) or is_shim(producer_tiles[0]):
            # pass through the memtile
            assert isa(attr := producers[0].memtile, ArrayAttr[ArrayAttr[IntegerAttr[IndexType]]])
            memtile_idxs = [subattr.data for subattr in attr.data]
            memtiles = [
                tile_op_manager.insert_or_update(memtile_idx[0].value.data, memtile_idx[1].value.data)
                for memtile_idx in memtile_idxs
            ]
            hops = [
                ObjectFifoHop.to_memtile(producers, memtiles, tile_op_manager, name_base),
                ObjectFifoHop.from_memtile(consumers, memtiles, tile_op_manager, name_base),
            ]
        else:
            hops = [ObjectFifoHop.compute_to_compute(producers, consumers, tile_op_manager, name_base)]

        # generate links for every hop
        links: Sequence[ObjectFifoLinkOp] = []
        for i in range(len(hops) - 1):
            links.extend(cls.get_links(hops[i], hops[i + 1]))

        return cls(hops, links)

    @staticmethod
    def get_links(hop_in: ObjectFifoHop, hop_out: ObjectFifoHop) -> Sequence[ObjectFifoLinkOp]:
        if len(hop_in.fifos) > len(hop_out.fifos):
            # determine src offsets
            assert isinstance(memref_out := hop_out.fifos[0].elemType.buffer, MemRefType)
            offset = prod(memref_out.get_shape()) // (len(hop_in.fifos) // len(hop_out.fifos))
            src_offsets = [i * offset for i in range(len(hop_in.fifos) // len(hop_out.fifos))]
            return [
                ObjectFifoLinkOp(
                    [fifin.sym_name.data for fifin in hop_in.fifos if fifout.producerTile in fifin.consumerTiles],
                    [fifout.sym_name.data],
                    src_offsets,
                    [],
                )
                for fifout in hop_out.fifos
            ]
        elif len(hop_out.fifos) > len(hop_in.fifos):
            assert isinstance(memref_in := hop_in.fifos[0].elemType.buffer, MemRefType)
            offset = prod(memref_in.get_shape()) // (len(hop_out.fifos) // len(hop_in.fifos))
            dst_offsets = [i * offset for i in range(len(hop_out.fifos) // len(hop_in.fifos))]
            return [
                ObjectFifoLinkOp(
                    [fifin.sym_name.data],
                    [fifout.sym_name.data for fifout in hop_out.fifos if fifout.producerTile in fifin.consumerTiles],
                    [],
                    dst_offsets,
                )
                for fifin in hop_in.fifos
            ]
        else:
            assert len(hop_in.fifos) == 1
            assert len(hop_out.fifos) == 1
            return [
                ObjectFifoLinkOp(
                    [hop_in.fifos[0].sym_name.data],
                    [hop_out.fifos[0].sym_name.data],
                    [],
                    [],
                )
            ]

    def get_of(self, op: PullOp | PushOp, tile_op_manager: TileOpManager):
        """
        Get the correct of in this chain for the given operation
        """
        tile = tile_op_manager.get_tile(op)
        result = []
        for hop in self.hops:
            for fifo in hop.fifos:
                if isinstance(op, PullOp):
                    if tile.result in fifo.consumerTiles:
                        result.append(fifo)
                if isinstance(op, PushOp):
                    if tile.result == fifo.producerTile:
                        result.append(fifo)
        return result

    def __contains__(self, target: ObjectFifoOp) -> bool:
        return any(target in hop.fifos for hop in self.hops)


@dataclass
class ObjectFifoManager:
    tile_op_manager: TileOpManager
    sequence_op: RuntimeSequenceOp
    device_op: DeviceOp

    counter: int = 0
    channel_to_of: dict[SSAValue, ObjectFifoChain] = field(default_factory=dict)

    def insert_or_update(self, channel: SSAValue, memref_type: MemRefType[Attribute]) -> ObjectFifoChain:  # noqa: PLR0915
        # find previous
        if channel in self.channel_to_of:
            return self.channel_to_of[channel]

        of_chain = ObjectFifoChain.from_channel(channel, memref_type, self.tile_op_manager, f"of_{self.counter}")
        self.counter += 1
        self.channel_to_of[channel] = of_chain

        # insert fifo ops
        for hop in of_chain.hops:
            for fifo in hop.fifos:
                SymbolTable.insert_or_update(self.device_op, fifo)

        # insert link ops
        for link in of_chain.links:
            self.device_op.region.block.add_op(link)

        return of_chain

    def get_of_chain(self, of: ObjectFifoOp | str) -> ObjectFifoChain:
        if isinstance(of, str):
            of = self.of_from_name(of)
        for chain in self.channel_to_of.values():
            if of in chain:
                return chain
        raise RuntimeError(f"ObjectFifoOp {of.sym_name.data} not found in channel_to_of mapping")

    def insert_or_update_of(self, object_fifo: ObjectFifoOp) -> ObjectFifoOp:
        SymbolTable.insert_or_update(self.device_op, object_fifo)
        return object_fifo

    def of_from_name(self, name: str) -> ObjectFifoOp:
        result = SymbolTable.lookup_symbol(self.device_op, name)
        assert isinstance(result, ObjectFifoOp)
        return result

    def update_depths(self):
        current_fifo_depth: dict[str, int] = defaultdict(int)

        for op in self.device_op.region.block.walk():
            if isinstance(op, ObjectFifoAcquireOp):
                of_name = op.objFifo_name.root_reference.data

                # update acquire size
                op.size = IntegerAttr.from_int_and_width(current_fifo_depth[of_name] + 1, 32)

                # update access index for all accesses based on this acquire
                for subview_access in [
                    x.operation for x in op.result.uses if isinstance(x.operation, ObjectFIFOSubviewAccessOp)
                ]:
                    subview_access.index = IntegerAttr.from_int_and_width(current_fifo_depth[of_name], 32)

                # increase current_depth
                current_fifo_depth[of_name] += 1

                # increase the depth of objectfifo if it does not suffice
                of = self.of_from_name(of_name)
                if of.elemNumber.value.data < current_fifo_depth[of_name] + 1:
                    of.elemNumber = IntegerAttr.from_int_and_width(current_fifo_depth[of_name] + 1, 32)

            elif isinstance(op, ObjectFIFOReleaseOp):
                of_name = op.objFifo_name.root_reference.data
                current_fifo_depth[of_name] -= 1

    def all_acquires(self, of_name: str) -> Iterator[ObjectFifoAcquireOp]:
        for op in self.device_op.walk():
            if isinstance(op, ObjectFifoAcquireOp) and op.objFifo_name.string_value() == of_name:
                yield op


def canonicalize_transformation(sizes: Sequence[int], strides: Sequence[int]) -> tuple[list[int], list[int]]:
    """
    Examples:

        Size 1 can be omitted:
        [1, 1], [1, 1] -> [], []
        [4, 1], [1, 1] -> [4], [1]
        [1, 4], [4, 1] -> [4], [1]

        Squash redundancy:
        [4, 4], [4, 1] -> [16], [1]

    """

    resulting_strides: list[int] = []
    resulting_sizes: list[int] = []

    for size, stride in zip(reversed(sizes), reversed(strides), strict=False):
        assert size != 0
        if size == 1:
            continue
        if not resulting_sizes:
            resulting_sizes.insert(0, size)
            resulting_strides.insert(0, stride)
            continue
        # check for squash
        if stride == resulting_sizes[0] * resulting_strides[0]:
            resulting_sizes[0] *= size
        else:
            resulting_sizes.insert(0, size)
            resulting_strides.insert(0, stride)

    return resulting_sizes, resulting_strides


@dataclass
class PutTransfersBeforeFirstUse(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: TransferOp, rewriter: PatternRewriter):
        assert op.parent
        operation_uses = set(x.operation for x in op.results[0].uses)
        try:
            first_use_op: Operation = next(o for o in op.parent.walk() if o in operation_uses)
        except StopIteration:
            # Print descriptive error message with relevant operation uses
            raise RuntimeError(
                f"TransferOp has no uses in the parent region. "
                f"Operation uses: {operation_uses}. "
                f"TransferOp details: {op}."
            ) from None
        while op.parent_op() is not first_use_op.parent_op():
            assert (parent := first_use_op.parent_op()) is not None
            first_use_op = parent

        op.detach()
        rewriter.insert_op(op, InsertPoint.before(first_use_op))


@dataclass
class TransferToRuntimeSequence(RewritePattern):
    object_fifo_manager: ObjectFifoManager

    arg_order: list[str]

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: PushOp | PullOp, rewriter: PatternRewriter):  # noqa: PLR0912, PLR0915
        if not isinstance(runtime_sequence := op.parent_op(), RuntimeSequenceOp):
            return

        if isinstance(op, PushOp):
            memref_type = op.input.type
        else:
            memref_type = op.output.type
        assert isinstance(memref_type, MemRefType)
        of_chain = self.object_fifo_manager.insert_or_update(op.channel, memref_type)

        # get edge
        if isinstance(op, PushOp):
            assert isinstance(op.input, OpResult)
            edge = op.input.op
        else:
            edge = next((use.operation for use in op.output.uses), None)
            if edge is None:
                # TODO: this transfer should not be present anymore
                rewriter.erase_matched_op()
                return
        assert isinstance(edge, OutEdgeOp | InEdgeOp)

        arg_index = self.arg_order.index(edge.tensor.data)
        arg = runtime_sequence.body.block.args[arg_index]
        assert isinstance(arg.type, MemRefType)

        # step 1: calculate sizes / strides
        ssis = op.ssis.data

        # calculate iteration multipliers for every var in the transfer:
        iteration_mults: dict[IterationVariable, int] = defaultdict(lambda: 1)
        iteration_mult = 1

        # gather all vars to iterate in the dma call:
        all_vars: Sequence[IterationVariable] = []

        # First, iterate over kernel dimensions in the order of operand indeces:
        operand_indeces = [x.data for x in op.operand_indeces]
        for index in operand_indeces[::-1]:  # in reverse for row-major fashion
            kernel_var = next(var for var in ssis.get_kernel_variables() if var.dimension == index)
            all_vars.append(kernel_var)

        # Next, iterate temporal vars kept local in a memtile
        reuse_tvars = []
        for var in ssis.get_temporal_variables():
            if var.mem_tile_reuse == MemTileReuse.REUSE:
                iteration_mults[var] = iteration_mult
                iteration_mult *= var.size
                reuse_tvars.append(var)
            else:
                break
        non_reuse_tvars = ssis.get_temporal_variables()[len(reuse_tvars) :]
        all_vars.extend(var for var in reuse_tvars if var.relevant)

        # Then, iterate the relevant spatial vars:
        for var in ssis.get_spatial_variables():
            iteration_mults[var] = iteration_mult
            iteration_mult *= var.size
        all_vars.extend(var for var in ssis.get_spatial_variables() if var.relevant)

        # Finally, remaining applicable temporal dims
        # assume output stationarity:
        for var in non_reuse_tvars:
            iteration_mults[var] = iteration_mult
            iteration_mult *= var.size
        if isinstance(op, PushOp):
            all_vars.extend(var for var in non_reuse_tvars if var.applicable)
        else:  # pull op
            all_vars.extend(var for var in non_reuse_tvars if var.relevant)

        # Calculate strides along with these iteration vars:
        seen_dims = defaultdict(lambda: 1)
        all_strides: dict[IterationVariable, int] = {}

        arg_strides = arg.type.get_strides()
        assert isa(arg_strides, Sequence[int])
        arg_strides = {x: y for x, y in zip(operand_indeces, arg_strides, strict=True)}

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
                total_size = prod(var.size for var in self.strides if var.stride)
                repeat = prod(var.size for var in self.strides if var.stride == 0)
                return type(self)((Stride(total_size, 1, 0), Stride(repeat, 0, 0)))

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

            def legalize(self) -> Self:
                if any(s.spatial for s in self.strides):
                    raise RuntimeError("cannot legalize strideset with spatial strides")
                new_strides: list[Stride] = []
                # make sure that no bound limits are exceeded
                # FIXME: figure out actual limits
                # these are innermost to outermost:
                bound_limits = (1024, 1024, 16384, 64)
                for i, (stride, bound_limit) in enumerate(zip(self.strides, bound_limits, strict=False)):
                    if stride.size > bound_limit:
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
                        tiling_stride = Stride(divider, stride.stride * tiled_size, stride.iteration_t * tiled_size)
                        # tile and legalize recursively
                        return type(self)(
                            (*self.strides[:i], tiled_stride, tiling_stride, *self.strides[i + 1 :])
                        ).legalize()
                changed = False
                # make sure the inner 3 most strides are nonzero
                for var in self.strides:
                    if var.stride == 0 and var.size != 1:
                        while len(new_strides) < 3:
                            changed = True
                            new_strides.append(Stride(1, 0, var.iteration_t))
                    new_strides.append(var)
                # make sure the transform is at least 4 strides long
                while len(new_strides) < 4:
                    changed = True
                    new_strides.append(Stride(1, 0, self.strides[-1].iteration_t))
                new = type(self)(tuple(new_strides))
                if changed:
                    return new.legalize()
                else:
                    return new

            def force_squash(self) -> Self:
                # Remove all transormations, reduce to 1D transfer
                total_size = prod(var.size for var in self.strides if var.stride)
                repeat_size = prod(var.size for var in self.strides if not var.stride)
                return type(self)((Stride(total_size, 1, 0), Stride(repeat_size, 0, 0)))

        strides: list[Stride] = []

        for var in all_vars:
            # multiply the stride by previous iteration vars
            if var.dimension in arg_strides:
                stride = seen_dims[var.dimension] * arg_strides[var.dimension]
                seen_dims[var.dimension] *= var.size
            else:
                stride = 0
            assert isinstance(op.memtile, ArrayAttr)
            spatial = len(op.memtile) > 1 and var.type == IterationVariableType.SPATIAL and var.size == len(op.memtile)
            strides.append(Stride(var.size, stride, iteration_mults[var], spatial))

        stride_dict = StrideSet(tuple(strides)).split()
        if "of_1" in of_chain.hops[1].fifos[0].sym_name.data or "of_20" in of_chain.hops[1].fifos[0].sym_name.data:
            stride_dict = {x: y.canonicalize().legalize() for x, y in stride_dict.items()}
        else:
            stride_dict = {x: y.force_squash().legalize() for x, y in stride_dict.items()}

        # select correct hop for fifo:
        if isinstance(op, PullOp):
            hop = of_chain.hops[1]
        else:
            hop = of_chain.hops[0]

        for i, (spatial_offset, stride_set) in enumerate(stride_dict.items()):
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
                    hop.fifos[i].sym_name.data,
                    Region(Block([dma_bd, EndOp()])),
                    issue_token=False,
                    repeat_count=hardware_strides[3].size - 1,
                )

                task.attributes["iteration_t"] = IntegerAttr.from_index_int_value(r.iteration_t)

                rewriter.insert_op([task], InsertPoint.before(op))

        # remove output from edge op operands
        rewriter.erase_matched_op(safe_erase=False)


@dataclass
class TransferToObjectFIFOPattern(RewritePattern):
    object_fifo_manager: ObjectFifoManager

    release_op: dict[str, Operation | None] = field(default_factory=dict)  # pyright: ignore

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: PushOp | PullOp, rewriter: PatternRewriter):  # noqa: PLR0912, PLR0915
        # Only handle pull/push ops in core ops, which should be converted to object fifos
        parent = op
        while True:
            if isinstance(parent, CoreOp):
                break
            parent = parent.parent_op()
            if parent is None:
                return

        if isinstance(op, PushOp):
            memref_type = op.input.type
        else:
            memref_type = op.output.type
        assert isinstance(memref_type, MemRefType)
        of_chain = self.object_fifo_manager.insert_or_update(op.channel, memref_type)

        # decide whether to consume or produce
        join_ofs: Sequence[ObjectFifoOp] = []
        if isinstance(op, PullOp):
            port = ObjectFifoPortEnum.Consume
            ofs = of_chain.get_of(op, self.object_fifo_manager.tile_op_manager)
            if len(ofs) > 1:
                # programmatic join
                join_ofs = ofs
                of = None
            else:
                assert len(ofs) == 1
                of = ofs[0]
            operand = op.output
        else:
            port = ObjectFifoPortEnum.Produce
            ofs = of_chain.get_of(op, self.object_fifo_manager.tile_op_manager)
            assert len(ofs) == 1
            of = ofs[0]
            operand = op.input

        assert isinstance(memref_type := operand.type, MemRefType)

        # if of is not None and "15" in of.sym_name.data:
        #     breakpoint()

        if len(join_ofs):
            # custom handling for programmatic join here:

            # multiple spatio-temporal variables get very complex, handle just one
            # for now (unfortunately, there is no way to check this anymore)
            use_op = next(x for x in op.results[0].uses).operation
            assert isinstance(use_op, ComputationNodeOp)
            ssis_dest = use_op.ssis.data
            st_var = ssis_dest.get_temporal_variables()[0]
            for_op = op.parent_op()
            assert isinstance(for_op, ForOp)
            assert for_op.attributes.get("layer_dim") == StringAttr(str(st_var.dimension))
            # one acquire per fifo
            acquires = []
            releases = []
            for of in join_ofs:
                of_name = of.sym_name.data
                acquire_op = ObjectFifoAcquireOp(
                    IntegerAttr.from_int_and_width(port.get_int(), 32),
                    IntegerAttr.from_int_and_width(1, 32),
                    object_fifo=of_name,
                    shape=memref_type.get_shape(),
                    element_type=memref_type.get_element_type(),
                )
                acquires.append(acquire_op)
                release_op = ObjectFIFOReleaseOp(
                    IntegerAttr.from_int_and_width(port.get_int(), 32),
                    IntegerAttr.from_int_and_width(1, 32),
                    object_fifo=of_name,
                )
                releases.append(release_op)
            access_ops = [ObjectFIFOSubviewAccessOp(IntegerAttr(0, i32), acquire) for acquire in acquires]
            # toggle between acquires with index switch op:
            index_switch = IndexSwitchOp(
                arg=for_op.body.block.args[0],
                cases=DenseArrayBase.from_list(IntegerType(64), list(range(st_var.size))),
                default_region=Region(Block([YieldOp(access_ops[0])])),
                case_regions=[Region(Block([YieldOp(access_ops[i])])) for i in range(st_var.size)],
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
            operand.replace_by(index_switch.results[0])
            # delete original op
            rewriter.erase_matched_op()
            return

        # otherwise, default flow with one objectfifo:
        # assert of is not None
        # if "of_11" in of.sym_name.data:
        #     print([(str(v), v.compute_tile_reuse) for v in op.ssis.data.variables])
        #     breakpoint()

        first_relevant_iter = next(iv for iv in op.ssis.data.get_temporal_variables() if iv.relevant)
        first_relevant_index = op.ssis.data.get_temporal_variables().index(first_relevant_iter)

        last_reuse = None
        for var in reversed(op.ssis.data.get_temporal_variables()):
            if var.reuse == Reuse.REUSE:
                last_reuse = var
                break
        if last_reuse:
            last_reuse_index = op.ssis.data.get_temporal_variables().index(last_reuse)
            reuse_iters = op.ssis.data.get_temporal_variables()[first_relevant_index : last_reuse_index + 1]
        else:
            reuse_iters = []

        relevant_reuse_iters = [iv for iv in reuse_iters if iv.relevant]

        reuse_factor = prod(iv.size for iv in reuse_iters if iv.relevant)

        # update object fifo depth
        # of.elemNumber = IntegerAttr.from_int_and_width(reuse_factor, 32)

        of_name = of.sym_name.data

        # acquire:
        acquire_op = ObjectFifoAcquireOp(
            IntegerAttr.from_int_and_width(port.get_int(), 32),
            IntegerAttr.from_int_and_width(reuse_factor, 32),
            object_fifo=of_name,
            shape=memref_type.get_shape(),
            element_type=memref_type.get_element_type(),
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
        for iter_var in relevant_reuse_iters:
            assert "layer_dim" in for_op.attributes
            while for_op.attributes["layer_dim"] != StringAttr(str(iter_var.dimension)):
                for_op = for_op.parent_op()
                assert isinstance(for_op, ForOp)
            i_arg = MuliOp(mult_val, for_op.body.block.args[0])
            add_val = AddiOp(add_val, i_arg)
            mult_val = MuliOp(mult_val, for_op.ub)
            index_ops.extend([i_arg, add_val, mult_val])

        index_switch = IndexSwitchOp(
            arg=add_val,
            cases=DenseArrayBase.from_list(IntegerType(64), list(range(reuse_factor))),
            default_region=Region(Block([YieldOp(access_ops[0])])),
            case_regions=[Region(Block([YieldOp(access_ops[i])])) for i in range(reuse_factor)],
            result_types=access_ops[0].result_types,
        )
        index_ops.append(index_switch)

        release_op = ObjectFIFOReleaseOp(
            IntegerAttr.from_int_and_width(port.get_int(), 32),
            IntegerAttr.from_int_and_width(reuse_factor, 32),
            object_fifo=of_name,
        )

        # there should only be one use now
        if isinstance(op, PullOp):
            use_op = next(use.operation for use in op.output.uses)
        else:
            assert isinstance(op.input, OpResult)
            use_op = op.input.op

        # get to same level in the iteration tree:
        while op.parent is not use_op.parent:
            use_op = use_op.parent_op()
            assert use_op is not None

        # put acquire and accesses at last level of reuse:
        use_op_reuse = use_op
        for _ in range(len(reuse_iters)):
            use_op_reuse = use_op_reuse.parent_op()
            assert use_op_reuse is not None

        rewriter.insert_op(release_op, InsertPoint.after(use_op_reuse))
        rewriter.insert_op([acquire_op, *access_ops], InsertPoint.before(use_op_reuse))
        rewriter.insert_op(index_ops, InsertPoint.before(use_op))

        # set output of computation node op if this was a push op
        if isinstance(op, PushOp):
            assert isinstance(op.input, OpResult)
            assert isinstance(compute := op.input.op, ComputationNodeOp)
            new_compute = ComputationNodeOp(
                compute.result_types,
                compute.kernel.data,
                compute.inputs,
                compute.core_allocation,
                compute.ssis.data,
                (index_switch.results[0],),
            )
            rewriter.replace_op(compute, new_compute)

        operand.replace_by(index_switch.results[0])
        rewriter.erase_matched_op()

        return


@dataclass
class MMPattern(RewritePattern):
    tile_op_manager: TileOpManager

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ComputationNodeOp, rewriter: PatternRewriter) -> None:
        if op.kernel.data != "gemm_32x32x32_0_0":
            return

        input_types = [operand.type for operand in op.inputs]
        if op.outputs:
            input_types.append(op.outputs.type)

        function_name = "matmul_bf16_bf16"

        func_op = FuncOp(function_name, (input_types, []), Region(), "private")
        zero_func_op = FuncOp("zero_bf16", (input_types[-1:], []), Region(), "private")

        # find  device op to insert function call
        device_op = op
        while not isinstance(device_op, DeviceOp):
            assert device_op.parent
            device_op = device_op.parent
        device_op = cast(DeviceOp, device_op)

        SymbolTable.insert_or_update(device_op, func_op)
        SymbolTable.insert_or_update(device_op, zero_func_op)

        # find core op to set link_with attribute
        core_op = op
        while not isinstance(core_op, CoreOp):
            assert core_op.parent
            core_op = core_op.parent
        core_op = cast(CoreOp, core_op)

        core_op.link_with = StringAttr(op.kernel.data + ".o")

        inputs: list[SSAValue | Operation] = list(op.inputs)
        if op.outputs:
            inputs.append(op.outputs)

        # insert zero func call for first use
        output = SSAValue.get(inputs[-1])
        assert isinstance(output, OpResult)
        zero_call = CallOp("zero_bf16", inputs[-1:], [])
        rewriter.insert_op(zero_call, InsertPoint.after(output.op))

        func_call = CallOp(function_name, inputs, [])
        rewriter.insert_op(func_call, InsertPoint.after(op))
        rewriter.erase_matched_op()


@dataclass
class MatVecPattern(RewritePattern):
    tile_op_manager: TileOpManager

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ComputationNodeOp, rewriter: PatternRewriter) -> None:
        if op.kernel.data != "matvec_vectorized_bf16_bf16":
            return

        op_inputs = [op.inputs[1], op.inputs[0]]

        input_types = [operand.type for operand in op_inputs]
        if op.outputs:
            input_types.append(op.outputs.type)

        # fist three i32 params: (m, n, row_offset)
        input_types = [i32] * 3 + input_types

        function_name = "matvec_vectorized_bf16_bf16"

        func_op = FuncOp(function_name, (input_types, []), Region(), "private")

        # find  device op to insert function call
        device_op = op
        while not isinstance(device_op, DeviceOp):
            assert device_op.parent
            device_op = device_op.parent
        device_op = cast(DeviceOp, device_op)

        SymbolTable.insert_or_update(device_op, func_op)

        # find core op to set link_with attribute
        core_op = op
        while not isinstance(core_op, CoreOp):
            assert core_op.parent
            core_op = core_op.parent
        core_op = cast(CoreOp, core_op)

        core_op.link_with = StringAttr("mv.o")

        c32 = ConstantOp.from_int_and_width(32, i32)
        c0 = ConstantOp.from_int_and_width(0, i32)

        inputs: list[SSAValue | Operation] = []

        # M
        inputs.append(c32)
        # K
        inputs.append(c32)
        # Row Offset
        inputs.append(c0)

        inputs.extend(list(op_inputs))
        if op.outputs:
            inputs.append(op.outputs)

        func_call = CallOp(function_name, inputs, [])
        rewriter.insert_op((c0, c32, func_call), InsertPoint.after(op))
        rewriter.erase_matched_op()


@dataclass
class ConvPattern(RewritePattern):
    tile_op_manager: TileOpManager

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ComputationNodeOp, rewriter: PatternRewriter) -> None:
        if op.kernel.data != "conv2dk1_i8":
            return

        input_types = [operand.type for operand in op.inputs]
        if op.outputs:
            input_types.append(op.outputs.type)

        # four i32's?
        input_types.extend([i32] * 4)

        func_op = FuncOp(op.kernel.data, (input_types, []), Region(), "private")

        # find  device op to insert function call
        device_op = op
        while not isinstance(device_op, DeviceOp):
            assert device_op.parent
            device_op = device_op.parent
        device_op = cast(DeviceOp, device_op)

        SymbolTable.insert_or_update(device_op, func_op)

        # find core op to set link_with attribute
        core_op = op
        while not isinstance(core_op, CoreOp):
            assert core_op.parent
            core_op = core_op.parent
        core_op = cast(CoreOp, core_op)

        core_op.link_with = StringAttr(op.kernel.data + ".o")

        c32 = ConstantOp.from_int_and_width(32, i32)
        c64 = ConstantOp.from_int_and_width(64, i32)
        c10 = ConstantOp.from_int_and_width(10, i32)

        inputs: list[SSAValue | Operation] = list(op.inputs)
        if op.outputs:
            inputs.append(op.outputs)
        inputs.extend([c32, c64, c64, c10])

        func_call = CallOp(op.kernel.data, inputs, [])

        rewriter.replace_matched_op((c32, c64, c10, func_call))


@dataclass
class SiluPattern(RewritePattern):
    tile_op_manager: TileOpManager

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ComputationNodeOp, rewriter: PatternRewriter) -> None:
        if op.kernel.data != "silu_bf16":
            return

        input_types = [operand.type for operand in op.inputs]
        if op.outputs:
            input_types.append(op.outputs.type)

        # size parameter
        input_types.extend([i32])

        func_op = FuncOp(op.kernel.data, (input_types, []), Region(), "private")

        # find  device op to insert function call
        device_op = op
        while not isinstance(device_op, DeviceOp):
            assert device_op.parent
            device_op = device_op.parent
        device_op = cast(DeviceOp, device_op)

        SymbolTable.insert_or_update(device_op, func_op)

        # find core op to set link_with attribute
        core_op = op
        while not isinstance(core_op, CoreOp):
            assert core_op.parent
            core_op = core_op.parent
        core_op = cast(CoreOp, core_op)

        core_op.link_with = StringAttr("silu.o")

        inputs: list[SSAValue | Operation] = list(op.inputs)
        if op.outputs:
            inputs.append(op.outputs)

        c32 = ConstantOp.from_int_and_width(32, i32)
        inputs.extend([c32])

        func_call = CallOp(op.kernel.data, inputs, [])

        rewriter.insert_op((c32, func_call), InsertPoint.after(op))
        rewriter.erase_matched_op()


@dataclass
class ElementwiseMulPattern(RewritePattern):
    tile_op_manager: TileOpManager

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ComputationNodeOp, rewriter: PatternRewriter) -> None:
        if op.kernel.data != "eltwise_mul_bf16_vector":
            return

        input_types = [operand.type for operand in op.inputs]
        if op.outputs:
            input_types.append(op.outputs.type)

        # size parameter
        input_types.extend([i32])

        func_op = FuncOp(op.kernel.data, (input_types, []), Region(), "private")

        # find  device op to insert function call
        device_op = op
        while not isinstance(device_op, DeviceOp):
            assert device_op.parent
            device_op = device_op.parent
        device_op = cast(DeviceOp, device_op)

        SymbolTable.insert_or_update(device_op, func_op)

        # find core op to set link_with attribute
        core_op = op
        while not isinstance(core_op, CoreOp):
            assert core_op.parent
            core_op = core_op.parent
        core_op = cast(CoreOp, core_op)

        core_op.link_with = StringAttr("mul.o")

        inputs: list[SSAValue | Operation] = list(op.inputs)
        if op.outputs:
            inputs.append(op.outputs)

        c32 = ConstantOp.from_int_and_width(32, i32)
        inputs.extend([c32])

        func_call = CallOp(op.kernel.data, inputs, [])

        rewriter.insert_op((c32, func_call), InsertPoint.after(op))
        rewriter.erase_matched_op()


@dataclass
class PassThroughMemTile(RewritePattern):
    changes: dict[str, str]
    tile_op_manager: TileOpManager
    object_fifo_manager: ObjectFifoManager

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ObjectFifoOp, rewriter: PatternRewriter):
        # not supporting any broadcast yet
        if len(op.consumerTiles) != 1:
            return

        # if connects to shim:
        assert isinstance(producerTile := op.producerTile, OpResult)
        assert isinstance(producerTile.op, TileOp)
        assert isinstance(consumerTile := op.consumerTiles[0], OpResult)
        assert isinstance(consumerTile.op, TileOp)

        # source/dest must be shim
        if producerTile.op.row.value.data == 0:
            # shim = producerTile
            compute = consumerTile
            shim_is_producer = True
        elif consumerTile.op.row.value.data == 0:
            # shim = consumerTile
            compute = producerTile
            shim_is_producer = False
        else:
            return

        # other one must be compute tile
        assert isinstance(compute.op, TileOp)
        if compute.op.row.value.data < 2:  # noqa: PLR2004
            return

        memtile = self.tile_op_manager.insert_or_update(0, 1)

        objectfifo_compute = ObjectFifoOp(
            memtile if shim_is_producer else op.producerTile,
            list(op.consumerTiles) if shim_is_producer else [memtile],
            op.elemNumber,
            op.elemType,
            op.sym_name,
            op.dimensionsToStream,
            op.dimensionsFromStreamPerConsumer,
            op.disable_synchronization,
            op.plio,
            op.via_DMA,
        )

        operand = op.sym_name.data.split("_")[-1]
        if shim_is_producer:
            shim_name = get_of_name(producerTile.op, memtile, operand)
        else:
            shim_name = get_of_name(memtile, consumerTile.op, operand)

        objectfifo_shim = ObjectFifoOp(
            op.producerTile if shim_is_producer else memtile,
            [memtile] if shim_is_producer else list(op.consumerTiles),
            op.elemNumber,
            op.elemType,
            shim_name,
            op.dimensionsToStream,
            op.dimensionsFromStreamPerConsumer,
            op.disable_synchronization,
            op.plio,
            op.via_DMA,
        )

        self.changes[op.sym_name.data] = shim_name

        self.object_fifo_manager.insert_or_update_of(objectfifo_compute)
        self.object_fifo_manager.insert_or_update_of(objectfifo_shim)

        if shim_is_producer:
            link = ObjectFifoLinkOp([shim_name], [op.sym_name.data], [], [])
        else:
            link = ObjectFifoLinkOp([op.sym_name.data], [shim_name], [], [])

        rewriter.insert_op(link, InsertPoint.after(objectfifo_shim))


@dataclass
class SetDistribution(RewritePattern):
    runtime_sequence: RuntimeSequenceOp
    object_fifo_manager: ObjectFifoManager

    @op_type_rewrite_pattern
    def match_and_rewrite(self, device_op: DeviceOp, rewriter: PatternRewriter):
        of_links: dict[SymbolRefAttr, list[SymbolRefAttr]] = defaultdict(list)
        of_link_ops: dict[tuple[SymbolRefAttr, SymbolRefAttr], ObjectFifoLinkOp] = {}

        for op in device_op.region.block.ops:
            if isinstance(op, ObjectFifoLinkOp):
                if len(op.fifoIns) != 1 or len(op.fifoOuts) != 1:
                    continue
                of_links[op.fifoIns.data[0]].append(op.fifoOuts.data[0])
                of_link_ops[(op.fifoIns.data[0], op.fifoOuts.data[0])] = op

        # filter out sets with only one link, no distribute needed
        of_links = {
            source: dests
            for source, dests in of_links.items()
            if len(dests) > 1 and "shim" in source.root_reference.data
        }
        # otherwise, sort values
        of_links = {
            source: sorted(dests, key=lambda dest: dest.root_reference.data) for source, dests in of_links.items()
        }

        # order the copies
        for source, dests in of_links.items():
            # list of copies mapping destination to list of copies
            copies: dict[SymbolRefAttr, list[DmaMemcpyNdOp]] = {}
            for dest in dests:
                copies[dest] = []

            for op in self.runtime_sequence.walk():
                if not isinstance(op, DmaMemcpyNdOp):
                    continue
                if op.metadata in copies:
                    copies[op.metadata].append(op)

            # for a correct distribute pattern, all elements should copy the same number of elements
            lengths = [len(v) for v in copies.values()]
            if len(set(lengths)) != 1:
                raise RuntimeError("distribute pattern detected with differing number of dma copies")

            # reorder memcpys based on the first root reference
            for i in range(lengths[0]):
                op = copies[dests[0]][i]
                for j in range(1, len(dests)):
                    new_op = copies[dests[j]][i]
                    new_op.detach()
                    rewriter.insert_op(new_op, InsertPoint.after(op))
                    op = new_op

            # create link op
            # calculate destination offset
            of_source = self.object_fifo_manager.of_from_name(source.root_reference.data)
            assert isinstance(memref_type := of_source.elemType.buffer, MemRefType)
            nb_elements = prod(memref_type.get_shape())
            dst_offsets = list(range(0, nb_elements * len(dests), nb_elements))

            # update source object fifo shape
            of_source.elemType = ObjectFIFO.from_element_type_and_shape(
                memref_type.get_element_type(), (len(dests),) + memref_type.get_shape()
            )

            # create new link op
            new_link_op = ObjectFifoLinkOp([source], dests, [], dst_offsets)

            # insert after last link
            rewriter.insert_op(new_link_op, InsertPoint.after(of_link_ops[(source, dests[-1])]))

            # erase all the rest:
            for i in range(len(dests)):
                rewriter.erase_op(of_link_ops[(source, dests[i])])


@dataclass
class SetJoin(RewritePattern):
    runtime_sequence: RuntimeSequenceOp
    object_fifo_manager: ObjectFifoManager

    @op_type_rewrite_pattern
    def match_and_rewrite(self, device_op: DeviceOp, rewriter: PatternRewriter):
        of_links: dict[SymbolRefAttr, list[SymbolRefAttr]] = defaultdict(list)
        of_link_ops: dict[tuple[SymbolRefAttr, SymbolRefAttr], ObjectFifoLinkOp] = {}

        for op in device_op.region.block.ops:
            if isinstance(op, ObjectFifoLinkOp):
                if len(op.fifoIns) != 1 or len(op.fifoOuts) != 1:
                    continue
                of_links[op.fifoOuts.data[0]].append(op.fifoIns.data[0])
                of_link_ops[(op.fifoOuts.data[0], op.fifoIns.data[0])] = op

        # filter out sets with only one link, no distribute needed
        of_links = {
            dest: sources
            for dest, sources in of_links.items()
            if len(sources) > 1 and "shim" in dest.root_reference.data
        }
        # otherwise, sort values
        of_links = {
            dest: sorted(sources, key=lambda source: source.root_reference.data) for dest, sources in of_links.items()
        }

        # order the copies
        for dest, sources in of_links.items():
            # list of copies mapping destination to list of copies
            copies: dict[SymbolRefAttr, list[DmaMemcpyNdOp]] = {}
            for source in sources:
                copies[source] = []

            for op in self.runtime_sequence.walk():
                if not isinstance(op, DmaMemcpyNdOp):
                    continue
                if op.metadata in copies:
                    copies[op.metadata].append(op)

            # for a correct distribute pattern, all elements should copy the same number of elements
            lengths = [len(v) for v in copies.values()]
            if len(set(lengths)) != 1:
                raise RuntimeError("join pattern detected with differing number of dma copies")

            # reorder memcpys based on the first root reference
            for i in range(lengths[0]):
                op = copies[sources[0]][i]
                for j in range(1, len(sources)):
                    new_op = copies[sources[j]][i]
                    new_op.detach()
                    rewriter.insert_op(new_op, InsertPoint.after(op))
                    op = new_op

            # create link op
            # calculate destination offset
            of_dest = self.object_fifo_manager.of_from_name(dest.root_reference.data)
            assert isinstance(memref_type := of_dest.elemType.buffer, MemRefType)
            nb_elements = prod(memref_type.get_shape())
            src_offsets = list(range(0, nb_elements * len(sources), nb_elements))

            # update dest object fifo shape
            of_dest.elemType = ObjectFIFO.from_element_type_and_shape(
                memref_type.get_element_type(), (len(sources),) + memref_type.get_shape()
            )

            # create new link op
            new_link_op = ObjectFifoLinkOp(sources, [dest], src_offsets, [])

            # insert after last link
            rewriter.insert_op(new_link_op, InsertPoint.after(of_link_ops[(dest, sources[-1])]))

            # erase all the rest:
            for i in range(len(sources)):
                rewriter.erase_op(of_link_ops[(dest, sources[i])])


def simplify_strides(
    strides: tuple[int, ...], sizes: tuple[int, ...]
) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    """
    Simplify strides. If possible, collapse two dimensions with the same stride into one.
    If not possible, return None.
    """

    if strides[0] == 0 and sizes[0] != 1:
        # special handling for repeat dma copies
        if strides[1] == 0 and sizes[1] != 1:
            strides = (0,) + strides[2:]
            sizes = (sizes[0] * sizes[1],) + sizes[2:]
            return sizes, strides
        else:
            return None
    same_strides = [strides[i] == strides[i + 1] * sizes[i + 1] for i in range(len(strides) - 1)]
    if True in same_strides:
        collapse_idx = same_strides.index(True)
        new_size = sizes[collapse_idx] * sizes[collapse_idx + 1]
        new_stride = strides[collapse_idx + 1]
        sizes = sizes[:collapse_idx] + (new_size,) + sizes[collapse_idx + 2 :]
        strides = strides[:collapse_idx] + (new_stride,) + strides[collapse_idx + 2 :]
        return sizes, strides


@dataclass
class CollapseMemcpys(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: DmaMemcpyNdOp, rewriter: PatternRewriter):
        # gather offsets, sizes and strides
        offset_1 = cast(tuple[int, ...], op.static_offsets.get_values())[-1]
        sizes_1 = cast(tuple[int, ...], op.static_sizes.get_values())
        strides_1 = cast(tuple[int, ...], op.static_strides.get_values())

        first_non_1 = next((i for i, x in enumerate(sizes_1) if x != 1), len(sizes_1))
        sizes_1 = sizes_1[first_non_1:]
        strides_1 = strides_1[first_non_1:]

        # check if we can simplify
        simplified = simplify_strides(strides_1, sizes_1)
        if simplified is not None:
            sizes_1 = (1,) * (4 - len(simplified[0])) + simplified[0]
            strides_1 = (0,) * (4 - len(simplified[1])) + simplified[1]

            new_op = DmaMemcpyNdOp(
                op.memref,
                op.static_offsets,
                sizes_1,
                strides_1,
                op.metadata,
                op.id,
                op.issue_token,
                op.offsets,
                op.strides,
            )
            rewriter.replace_matched_op(new_op)
            return

        # find next memcpy with the same metadata
        next_op = op
        while True:
            next_op = next_op.next_op
            if next_op is None:
                return
            if isinstance(next_op, DmaMemcpyNdOp) and next_op.metadata == op.metadata:
                break

        # strides should fully overlap
        offset_2 = cast(tuple[int, ...], next_op.static_offsets.get_values())[-1]
        sizes_2 = cast(tuple[int, ...], next_op.static_sizes.get_values())
        strides_2 = cast(tuple[int, ...], next_op.static_strides.get_values())

        first_non_1 = next((i for i, x in enumerate(sizes_2) if x != 1), len(sizes_2))
        sizes_2 = sizes_2[first_non_1:]
        strides_2 = strides_2[first_non_1:]

        # full overlap:
        if sizes_1 == sizes_2 and strides_1 == strides_2:
            if (offset_2 - offset_1) == 0:
                # special case as only 4th can be zero
                sizes_1 = (1,) * (3 - len(sizes_1)) + sizes_1
                strides_1 = (0,) * (3 - len(strides_1)) + strides_1
            sizes_1 = (2,) + sizes_1
            strides_1 = (offset_2 - offset_1,) + strides_1

            simplified = simplify_strides(strides_1, sizes_1)
            if simplified is not None:
                sizes_1 = simplified[0]
                strides_1 = simplified[1]

            if len(sizes_1) > 4:  # noqa: PLR2004
                return
            sizes_1 = (1,) * (4 - len(sizes_1)) + sizes_1
            strides_1 = (0,) * (4 - len(strides_1)) + strides_1
            # remove next op
            new_op = DmaMemcpyNdOp(
                op.memref,
                op.static_offsets,
                sizes_1,
                strides_1,
                op.metadata,
                op.id,
                op.issue_token,
                op.offsets,
                op.strides,
            )
            rewriter.replace_matched_op(new_op)
            rewriter.erase_op(next_op)


@dataclass
class OfNameRewriter(RewritePattern):
    changes: dict[str, str]

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: DmaMemcpyNdOp, rewriter: PatternRewriter):
        if op.metadata.root_reference.data in self.changes:
            op.metadata = SymbolRefAttr(self.changes[op.metadata.root_reference.data])


@dataclass
class InsertRuntimeDMAs(RewritePattern):
    sequence_op: RuntimeSequenceOp

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ObjectFifoOp, rewriter: PatternRewriter):
        # Add Block Argument to SequenceOp
        memref_type = op.elemType.buffer

        assert isinstance(memref_type, MemRefType)

        shape = list(memref_type.get_shape())
        memref_type = MemRefType(memref_type.get_element_type(), shape)

        sequence_block = self.sequence_op.body.block

        sequence_block.insert_arg(memref_type, 0)

        # Insert DMA
        memcpy = DmaMemcpyNdOp(
            sequence_block.args[0],
            static_offsets=[0, 0, 0, 0],
            static_sizes=[1, 1, 1, memref_type.get_shape()[0]],
            static_strides=[0, 0, 0, 1],
            metadata=op.sym_name,
            id=0,
            issue_token=True,
        )

        rewriter.insert_op(memcpy, InsertPoint.at_start(sequence_block))

        # wait for it ...

        wait = DmaWaitOp(op.sym_name)

        rewriter.insert_op(wait, InsertPoint.at_end(sequence_block))


@dataclass
class EraseEdges(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: OutEdgeOp | InEdgeOp | ChannelOp, rewriter: PatternRewriter) -> None:
        rewriter.erase_matched_op()


@dataclass
class OrderDMAs(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: RuntimeSequenceOp, rewriter: PatternRewriter) -> None:
        dma_ops = [op for op in op.body.block.ops if isinstance(op, DmaConfigureTaskForOp)]
        dma_ops = sorted(dma_ops, key=lambda op: op.attributes["iteration_t"].value.data)
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
class SetKernelLayouts(RewritePattern):
    kernels: dict[str, AIEKernel]

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ComputationNodeOp, rewriter: PatternRewriter) -> None:
        aie_kernel = self.kernels.get(op.kernel.data)
        assert aie_kernel is not None
        layouts = aie_kernel.operand_layouts()
        if not layouts:
            return
        shaped_operands = [operand for operand in op.operands if isinstance(operand.type, ShapedType)]
        for layout, operand in zip(layouts, shaped_operands, strict=True):
            assert isa(old_type := operand.type, MemRefType[FixedBitwidthType])
            layout_attr = TiledStridedLayoutAttr(layout)
            if old_type.layout == layout_attr:
                continue
            new_type = MemRefType(old_type.element_type, old_type.shape, layout_attr, old_type.memory_space)
            new_operand = LayoutCast(operand, new_type)
            rewriter.insert_op(new_operand, InsertPoint.before(op))
            operand.replace_by_if(new_operand.results[0], lambda use: use.operation is op)


@dataclass
class HoistLayoutCasts(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: LayoutCast, rewriter: PatternRewriter) -> None:
        assert isinstance(op.source, OpResult)
        if isinstance(op.source.op, ObjectFIFOSubviewAccessOp):
            # good, this is what we want
            return
        elif isinstance(switch := op.source.op, IndexSwitchOp):
            # push up layout cast
            for case in (switch.default_region, *switch.case_regions):
                yield_op = case.block.last_op
                assert isinstance(yield_op, YieldOp)
                yielded = yield_op.arguments[0]
                assert isa(op.dest.type, MemRefType[FixedBitwidthType])
                new_cast = LayoutCast(yielded, op.dest.type)
                yield_op.operands[0] = new_cast.dest
                assert isinstance(yielded.owner, Operation)
                rewriter.insert_op(new_cast, InsertPoint.after(yielded.owner))
            switch.results[0].type = op.dest.type
            op.dest.replace_by(switch.output[0])
            rewriter.erase_op(op)


@dataclass
class SquashLayoutCasts(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: LayoutCast, rewriter: PatternRewriter) -> None:
        layout_casts = [use.operation for use in op.source.uses if isinstance(use.operation, LayoutCast)]
        # all dest types must be equal
        assert all(op.dest.type == cast.dest.type for cast in layout_casts)
        # keep only this one
        for cast_to_remove in filter(lambda x: x is not op, layout_casts):
            cast_to_remove.dest.replace_by(op.dest)
            rewriter.erase_op(cast_to_remove)


@dataclass
class SetKernelLayoutsNPU1(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: CallOp, rewriter: PatternRewriter):
        # handle the conv case
        if op.callee.root_reference.data == "conv2dk1_i8":
            input = op.arguments[0]
            output = op.arguments[2]
            input_type = cast(MemRefType[Attribute], op.arguments[0].type)

            if isinstance(input_type.layout, TiledStridedLayoutAttr):
                return

            input_layout = TiledStridedLayout(
                [
                    TiledStride([Stride(32 * 64, 1)]),  # N
                    TiledStride([Stride(32 * 64, 1)]),  # G
                    TiledStride([Stride(32 * 64, 1)]),  # H
                    TiledStride([Stride(8, 32)]),  # W
                    TiledStride([Stride(8 * 32, 8), Stride(1, 8)]),  # C
                ]
            )

            input_type = MemRefType(
                input_type.element_type, input_type.shape, TiledStridedLayoutAttr(input_layout), input_type.memory_space
            )

            new_input = LayoutCast(input, input_type)
            new_output = LayoutCast(output, input_type)

            rewriter.insert_op([new_input, new_output], InsertPoint.before(op))

            op.operands[0] = new_input.results[0]
            op.operands[2] = new_output.results[0]

        if op.callee.root_reference.data == "matmul_bf16_bf16":
            A_operand = op.operands[0]
            A_type = cast(MemRefType[Attribute], op.arguments[0].type)
            if isinstance(A_type.layout, TiledStridedLayoutAttr):
                return
            layout_A = TiledStridedLayout(
                [
                    TiledStride([Stride(16 * 32 // 4, 32 // 4), Stride(4, 4)]),
                    TiledStride([Stride(16, 32 // 4), Stride(1, 4)]),
                ]
            )
            A_type_new = MemRefType(
                A_type.element_type, A_type.shape, TiledStridedLayoutAttr(layout_A), A_type.memory_space
            )
            A_new = LayoutCast(A_operand, A_type_new)

            B_operand = op.operands[1]
            B_type = cast(MemRefType[Attribute], op.arguments[1].type)
            if isinstance(B_type.layout, TiledStridedLayoutAttr):
                return
            layout_B = TiledStridedLayout(
                [
                    TiledStride([Stride(16 * 32 // 4, 32 // 4), Stride(4, 4)]),
                    TiledStride([Stride(16, 32 // 4), Stride(1, 4)]),
                ]
            )
            B_type_new = MemRefType(
                B_type.element_type, B_type.shape, TiledStridedLayoutAttr(layout_B), B_type.memory_space
            )
            B_new = LayoutCast(B_operand, B_type_new)

            D_operand = op.operands[2]
            D_type = cast(MemRefType[Attribute], op.arguments[2].type)
            if isinstance(D_type.layout, TiledStridedLayoutAttr):
                return
            layout_D = TiledStridedLayout(
                [
                    TiledStride([Stride(16 * 32 // 4, 32 // 4), Stride(4, 4)]),
                    TiledStride([Stride(16, 32 // 4), Stride(1, 4)]),
                ]
            )
            D_type_new = MemRefType(
                D_type.element_type, D_type.shape, TiledStridedLayoutAttr(layout_D), D_type.memory_space
            )
            D_new = LayoutCast(D_operand, D_type_new)

            rewriter.insert_op((A_new, B_new, D_new), InsertPoint.before(op))

            op.operands[0] = A_new.results[0]
            op.operands[1] = B_new.results[0]
            op.operands[2] = D_new.results[0]


@dataclass
class SetKernelLayoutsNPU2(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: CallOp, rewriter: PatternRewriter):
        # handle the conv case
        if op.callee.root_reference.data == "conv2dk1_i8":
            input = op.arguments[0]
            output = op.arguments[2]
            input_type = cast(MemRefType[Attribute], op.arguments[0].type)

            if isinstance(input_type.layout, TiledStridedLayoutAttr):
                return

            input_layout = TiledStridedLayout(
                [
                    TiledStride([Stride(32 * 64, 1)]),  # N
                    TiledStride([Stride(32 * 64, 1)]),  # G
                    TiledStride([Stride(32 * 64, 1)]),  # H
                    TiledStride([Stride(8, 32)]),  # W
                    TiledStride([Stride(8 * 32, 8), Stride(1, 8)]),  # C
                ]
            )

            input_type = MemRefType(
                input_type.element_type, input_type.shape, TiledStridedLayoutAttr(input_layout), input_type.memory_space
            )

            new_input = LayoutCast(input, input_type)
            new_output = LayoutCast(output, input_type)

            rewriter.insert_op([new_input, new_output], InsertPoint.before(op))

            op.operands[0] = new_input.results[0]
            op.operands[2] = new_output.results[0]

        if op.callee.root_reference.data == "matmul_bf16_bf16":
            # m=4 k=8 n=8
            A_operand = op.operands[0]
            A_type = cast(MemRefType[Attribute], op.arguments[0].type)
            if isinstance(A_type.layout, TiledStridedLayoutAttr):
                return
            layout_A = TiledStridedLayout(
                [
                    TiledStride([Stride(128, 8), Stride(8, 4)]),
                    TiledStride([Stride(32, 4), Stride(1, 8)]),
                ]
            )
            A_type_new = MemRefType(
                A_type.element_type, A_type.shape, TiledStridedLayoutAttr(layout_A), A_type.memory_space
            )
            A_new = LayoutCast(A_operand, A_type_new)

            B_operand = op.operands[1]
            B_type = cast(MemRefType[Attribute], op.arguments[1].type)
            if isinstance(B_type.layout, TiledStridedLayoutAttr):
                return
            layout_B = TiledStridedLayout(
                [
                    TiledStride([Stride(256, 4), Stride(8, 8)]),
                    TiledStride([Stride(64, 4), Stride(1, 8)]),
                ]
            )
            B_type_new = MemRefType(
                B_type.element_type, B_type.shape, TiledStridedLayoutAttr(layout_B), B_type.memory_space
            )
            B_new = LayoutCast(B_operand, B_type_new)

            D_operand = op.operands[2]
            D_type = cast(MemRefType[Attribute], op.arguments[2].type)
            if isinstance(D_type.layout, TiledStridedLayoutAttr):
                return
            layout_D = TiledStridedLayout(
                [
                    TiledStride([Stride(128, 8), Stride(8, 4)]),
                    TiledStride([Stride(32, 4), Stride(1, 8)]),
                ]
            )
            D_type_new = MemRefType(
                D_type.element_type, D_type.shape, TiledStridedLayoutAttr(layout_D), D_type.memory_space
            )
            D_new = LayoutCast(D_operand, D_type_new)

            rewriter.insert_op((A_new, B_new, D_new), InsertPoint.before(op))

            op.operands[0] = A_new.results[0]
            op.operands[1] = B_new.results[0]
            op.operands[2] = D_new.results[0]


def get_transform(source: TiledStridedLayout, dest: TiledStridedLayout) -> tuple[list[int], list[int]]:
    """
    Returns sizes, strides
    """

    # list of dim, depth
    keys: list[tuple[int, int]] = []

    for dim in range(source.dimension()):
        for depth in range(source.tstrides[dim].depth()):
            keys.append((dim, depth))

    strides: list[dict[str, Stride]] = []

    for key in keys:
        strides.append(
            {
                "stride_src": source.get_stride(*key),
                "stride_dest": dest.get_stride(*key),
            }
        )

    strides.sort(key=lambda x: x["stride_dest"].step or 0, reverse=True)

    sizes_src, strides_src = zip(*[(x["stride_src"].bound, x["stride_src"].step) for x in strides], strict=False)
    sizes_dest, strides_dest = zip(*[(x["stride_dest"].bound, x["stride_dest"].step) for x in strides], strict=False)

    # canonicalize
    sizes_src, strides_src = canonicalize_transformation(sizes_src, strides_src)
    sizes_dest, strides_dest = canonicalize_transformation(sizes_dest, strides_dest)

    # we only consider transformations at the source for now, so no transform should be happening at dest
    if len(sizes_dest) != 1:
        raise RuntimeError("did not expect dest transformation")

    return (sizes_src, strides_src)


@dataclass
class RealizeLayoutCats(RewritePattern):
    of_manager: ObjectFifoManager

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: LayoutCast, rewriter: PatternRewriter):
        # gather some variables
        assert isinstance(op.source, OpResult)
        assert isinstance(subview_access := op.source.op, ObjectFIFOSubviewAccessOp)
        # assert isinstance(subview_access := op.source.op, ObjectFIFOSubviewAccessOp)
        assert isinstance(subview_access.subview, OpResult)
        assert isinstance(of_acquire := subview_access.subview.op, ObjectFifoAcquireOp)

        if op.dest.type == op.source.type:
            op.dest.replace_by(op.source)
            rewriter.erase_matched_op()
            return

        # get the chain:
        chain = self.of_manager.get_of_chain(of_acquire.objFifo_name.root_reference.data)

        # get all acquires and releases
        consumers: list[ObjectFifoAcquireOp] = []
        producers: list[ObjectFifoAcquireOp] = []
        for hop in chain.hops:
            for fifo in hop.fifos:
                for acquire in self.of_manager.all_acquires(fifo.sym_name.data):
                    match ObjectFifoPortEnum.from_int(acquire.port.value.data):
                        case ObjectFifoPortEnum.Consume:
                            consumers.append(acquire)
                        case ObjectFifoPortEnum.Produce:
                            producers.append(acquire)

        def gather_layout(acquires: Sequence[ObjectFifoAcquireOp]) -> MemRefType[FixedBitwidthType] | None:
            result = []
            for acquire in acquires:
                for subview in acquire.result.uses:
                    if isinstance(subview.operation, ObjectFIFOSubviewAccessOp):
                        for cast in subview.operation.output.uses:
                            if isinstance(cast.operation, LayoutCast):
                                dest_type = cast.operation.dest.type
                                assert isa(dest_type, MemRefType[FixedBitwidthType])
                                result.append(dest_type)
            if len(result) == 0:
                return None
            else:
                assert all([x == result[0] for x in result])
                return result[0]

        consumer_type = gather_layout(consumers)
        producer_type = gather_layout(producers)

        # create row-major layouts for those without explicit casts:

        if consumer_type is None:
            assert producer_type is not None
            assert isinstance(producer_type.layout, TiledStridedLayoutAttr)
            producer_layout = producer_type.layout.data
            strides = [1]
            for size in reversed(producer_type.shape.data[1:]):
                strides = [size.data * strides[0]] + strides
            assert isinstance(producer_type.layout, TiledStridedLayoutAttr)
            tile_bounds = producer_type.layout.data.tile_bounds()
            consumer_layout = TiledStridedLayout.from_strides(strides, tile_bounds)  # pyright: ignore
        elif producer_type is None:
            assert consumer_type is not None
            assert isinstance(consumer_type.layout, TiledStridedLayoutAttr)
            consumer_layout = consumer_type.layout.data
            strides = [1]
            for size in reversed(consumer_type.shape.data[1:]):
                strides = [size.data * strides[0]] + strides
            assert isinstance(consumer_type.layout, TiledStridedLayoutAttr)
            tile_bounds = consumer_type.layout.data.tile_bounds()
            producer_layout = TiledStridedLayout.from_strides(strides, tile_bounds)  # pyright: ignore
        else:
            assert isinstance(consumer_type.layout, TiledStridedLayoutAttr)
            consumer_layout = consumer_type.layout.data
            assert isinstance(producer_type.layout, TiledStridedLayoutAttr)
            producer_layout = producer_type.layout.data

        sizes, strides = get_transform(producer_layout, consumer_layout)

        transform_is_null = len(sizes) == 1 and strides == [1]

        # create BDDimlayout
        bd_layout = BDDimLayoutArrayAttr(
            BDDimLayoutArray([BDDimLayout((size, stride)) for size, stride in zip(sizes, strides, strict=True)])
        )

        # take last fifo in the chain (starting form memtile i)
        hop = chain.hops[-1]
        for fifo in hop.fifos:
            # fifo.elemType = ObjectFIFO([MemRefType(element_type.element_type, element_type.shape, dest_type.layout)])
            if not transform_is_null:
                fifo.dimensionsToStream = bd_layout

        if consumer_type is not None:
            for consumer in consumers:
                consumer.result.type = ObjectFIFOSubview([consumer_type])
                for use in consumer.result.uses:
                    if isinstance(use.operation, ObjectFIFOSubviewAccessOp):
                        use.operation.output.type = consumer_type

        if producer_type is not None:
            for producer in producers:
                producer.result.type = ObjectFIFOSubview([producer_type])
                for use in producer.result.uses:
                    if isinstance(use.operation, ObjectFIFOSubviewAccessOp):
                        use.operation.output.type = producer_type


@dataclass
class OrderCoreOps(RewritePattern):
    # Complete a bubble-type sorting of core ops for a more deterministic output
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: CoreOp, rewriter: PatternRewriter):
        def get_tile_idx(op: CoreOp) -> tuple[int, int]:
            assert isinstance(op.tile, OpResult)
            assert isinstance(op.tile.op, TileOp)
            return (op.tile.op.col.value.data, op.tile.op.row.value.data)

        if not isinstance(next_op := op.next_op, CoreOp):
            return

        if get_tile_idx(op) > get_tile_idx(next_op):
            next_op.detach()
            rewriter.insert_op(next_op, InsertPoint.before(op))


@dataclass
class RemoveSpatioTemporality(RewritePattern):
    # Complete a bubble-type sorting of core ops for a more deterministic output
    @op_type_rewrite_pattern
    def match_and_rewrite(self, core_op: CoreOp, rewriter: PatternRewriter):  # noqa: PLR0912
        block = core_op.region.block
        ops: Sequence[PushOp | PullOp | ComputationNodeOp] = []
        ssis = None

        # First, gather ops
        for op in block.ops:
            if isinstance(op, ComputationNodeOp):
                ssis = op.ssis.data
                # if "silu" in op.kernel.data:
                #     breakvar = True
                #     breakpoint()
            if isinstance(op, PushOp | PullOp | ComputationNodeOp):
                # FIXME: hack, upon hack, upon hack:
                # this is for the output, which we fixed a while back
                # this op is already in the clear, we should not change the
                # ssis anymore
                if isinstance(op, PushOp) and not isinstance(op.memtile, NoneAttr):
                    pass
                else:
                    ops.append(op)
            elif isinstance(op, EndOp):
                pass
            else:
                raise RuntimeError("non-steady state op encountered")

        assert ssis is not None

        if not len(ssis.get_spatio_temporal_variables()):
            return

        new_ssis_vars: list[list[IterationVariable]] = [[] for _ in range(len(ops))]
        new_ssis_tvars: list[list[IterationVariable]] = [[] for _ in range(len(ops))]
        for i, var in enumerate(ssis.variables):
            for j, op in enumerate(ops):
                opvar = deepcopy(op.ssis.data.variables[i])
                if var.dimension != opvar.dimension:
                    # FIXME: hare
                    raise RuntimeError()
                assert var.dimension == opvar.dimension
                assert var.size == opvar.size

                if var.type in (IterationVariableType.KERNEL, IterationVariableType.SPATIAL):
                    new_ssis_vars[j].append(opvar)
                elif var.type == IterationVariableType.SPATIOTEMPORAL:
                    opvar.type = IterationVariableType.TEMPORAL
                    new_ssis_tvars[j].append(opvar)
                else:
                    new_ssis_tvars[j].append(opvar)
        # merge vars again:
        new_ssis_vars = [x + y for x, y in zip(new_ssis_vars, new_ssis_tvars, strict=True)]
        new_ssis = [SteadyStateIterationSpace(vars) for vars in new_ssis_vars]

        for op, ssis in zip(ops, new_ssis, strict=True):
            op.properties["ssis"] = SteadyStateIterationSpaceAttr(ssis)


@dataclass
class WrapInCoreOps(RewritePattern):
    tile_op_manager: TileOpManager
    of_manager: ObjectFifoManager
    core_ops: dict[tuple[int, int], CoreOp | RuntimeSequenceOp]

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter):  # noqa: PLR0912
        shim = (0, 0)
        if isinstance(op, ComputationNodeOp):
            assert isinstance(attr := op.core_allocation, ArrayAttr)
            core = cast(tuple[IntegerAttr[IndexType], ...], attr.data)
            core = tuple(x.value.data for x in core)
        elif isinstance(op, InEdgeOp | OutEdgeOp):
            core = shim
        elif isinstance(op, PushOp):
            # what am i pushing?
            assert isinstance(op.input, OpResult)
            if isinstance(op.input.op, InEdgeOp):
                core = shim
            elif isinstance(op.input.op, ComputationNodeOp):
                assert isinstance(attr := op.input.op.core_allocation, ArrayAttr)
                core = cast(tuple[IntegerAttr[IndexType], ...], attr.data)
                core = tuple(x.value.data for x in core)
            else:
                raise NotImplementedError()
        elif isinstance(op, PullOp):
            # where am i pulling to?
            assert len(op.output.uses) == 1
            use = next(iter(op.output.uses))
            if isinstance(use.operation, OutEdgeOp):
                core = shim
            elif isinstance(use.operation, ComputationNodeOp):
                assert isinstance(attr := use.operation.core_allocation, ArrayAttr)
                core = cast(tuple[IntegerAttr[IndexType], ...], attr.data)
                core = tuple(x.value.data for x in core)
            else:
                raise NotImplementedError()
        else:
            return

        # create core op if it doesn't exist yet
        if core not in self.core_ops:
            core_op = CoreOp(None, self.tile_op_manager.insert_or_update(*core), Region(Block([EndOp()])))
            rewriter.insert_op(core_op, InsertPoint.at_end(self.tile_op_manager.device_op.region.block))
            self.core_ops[core] = core_op
        else:
            core_op = self.core_ops[core]

        op.detach()
        if isinstance(core_op, CoreOp):
            assert core_op.region.block.last_op
            insert_point = InsertPoint.before(core_op.region.block.last_op)
        else:
            insert_point = InsertPoint.at_end(core_op.body.block)
        rewriter.insert_op(op, insert_point)


@dataclass
class InfinteLoopCol(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: CoreOp, rewriter: PatternRewriter):
        aie_end = op.region.block.last_op
        assert isinstance(aie_end, EndOp)
        aie_end.detach()
        start = ConstantOp.from_int_and_width(0, IndexType())
        step = ConstantOp.from_int_and_width(1, IndexType())
        end = ConstantOp.from_int_and_width(0xFFFF_FFFF, IndexType())
        body = rewriter.move_region_contents_to_new_regions(op.region)
        body.block.insert_arg(IndexType(), 0)  # add index argument
        for_op = ForOp(start, end, step, [], body)
        op.region.add_block(Block([start, step, end, for_op, aie_end]))


@dataclass(frozen=True)
class ConvertStreamToAIEPass(ModulePass):
    name = "convert-stream-to-aie"

    # The order in which the edge ops should appear in the runtime sequence
    arg_order: list[str]
    aie_kernels: dict[str, AIEKernel]

    def apply(self, ctx: MLContext, op: ModuleOp, npu: str) -> None:
        # wrap everything in a device op
        #

        npu = AIEDeviceEnum.npu2 if npu == "npu2" else AIEDeviceEnum.npu1

        rewriter = Rewriter()
        device_op = DeviceOp(
            IntegerAttr.from_int_and_width(npu.get_int(), 32),
            rewriter.move_region_contents_to_new_regions(op.body),
        )
        op.body.add_block(Block([device_op]))

        # add a runtime sequence operation
        # find all edges
        edges: list[InEdgeOp | OutEdgeOp] = [edge for edge in op.walk() if isinstance(edge, InEdgeOp | OutEdgeOp)]
        # order = ["Op0.I_in", "Op0.W_in", "Op0.O_out"]
        if not self.arg_order:
            arg_order = [edge.tensor.data for edge in edges]
        else:
            arg_order = self.arg_order

        runtime_arg_types = []
        for operand_name in arg_order:
            edge = next(edge for edge in edges if edge.tensor.data == operand_name)
            operand = edge.inputs[0] if isinstance(edge, OutEdgeOp) else edge.output
            assert operand is not None
            runtime_arg_types.append(operand.type)

        runtime_sequence = RuntimeSequenceOp(Region(Block(arg_types=runtime_arg_types)))
        rewriter.insert_op(runtime_sequence, InsertPoint.at_end(device_op.region.block))

        tile_op_manager = TileOpManager(device_op)
        object_fifo_manager = ObjectFifoManager(tile_op_manager, runtime_sequence, device_op)

        # Order all transfers based on first use
        # PatternRewriteWalker(PutTransfersBeforeFirstUse(), apply_recursively=False).rewrite_module(op)
        #

        PatternRewriteWalker(
            WrapInCoreOps(
                tile_op_manager,
                object_fifo_manager,
                {
                    (0, 0): runtime_sequence,
                },
            ),
            apply_recursively=False,
        ).rewrite_module(op)

        PatternRewriteWalker(OrderCoreOps()).rewrite_module(op)
        with open("test5.mlir", "w") as f:
            f.write(str(op))
        PatternRewriteWalker(RemoveSpatioTemporality()).rewrite_module(op)
        with open("test6.mlir", "w") as f:
            f.write(str(op))

        for core_op in device_op.region.block.ops:
            if isinstance(core_op, CoreOp):
                # insert runtime sequence op
                iteration_space_to_for(core_op.region.block, rewriter)

        with open("test7.mlir", "w") as f:
            f.write(str(op))

        PatternRewriteWalker(
            TransferToObjectFIFOPattern(object_fifo_manager),
            apply_recursively=False,
        ).rewrite_module(op)

        with open("test8.mlir", "w") as f:
            f.write(str(op))

        PatternRewriteWalker(
            TransferToRuntimeSequence(object_fifo_manager, arg_order),
            apply_recursively=False,
        ).rewrite_module(op)

        PatternRewriteWalker(OrderDMAs(), apply_recursively=False).rewrite_module(op)
        PatternRewriteWalker(SyncDMAs(), apply_recursively=False).rewrite_module(op)
        PatternRewriteWalker(StartDMAs(), apply_recursively=False).rewrite_module(op)

        ## lower computation node ops for known kernels

        PatternRewriteWalker(ConvPattern(tile_op_manager), apply_recursively=False).rewrite_module(op)
        PatternRewriteWalker(MMPattern(tile_op_manager), apply_recursively=False).rewrite_module(op)

        # Use the new convert aie kernels operation:
        assert npu is AIEDeviceEnum.npu2
        PatternRewriteWalker(SetKernelLayouts(self.aie_kernels)).rewrite_module(op)
        PatternRewriteWalker(HoistLayoutCasts()).rewrite_module(op)
        PatternRewriteWalker(SquashLayoutCasts()).rewrite_module(op)
        with open("test4.mlir", "w") as f:
            f.write(str(op))
        PatternRewriteWalker(ConvertAIEKernels(self.aie_kernels)).rewrite_module(op)

        # handle layouts
        assert npu is AIEDeviceEnum.npu2
        # PatternRewriteWalker(SetKernelLayouts()).rewrite_module(op)
        # match npu:
        #     case AIEDeviceEnum.npu1:
        #         PatternRewriteWalker(SetKernelLayoutsNPU1()).rewrite_module(op)
        #     case AIEDeviceEnum.npu2:
        #         PatternRewriteWalker(SetKernelLayoutsNPU2()).rewrite_module(op)
        PatternRewriteWalker(RealizeLayoutCats(object_fifo_manager)).rewrite_module(op)

        PatternRewriteWalker(InfinteLoopCol(), apply_recursively=False).rewrite_module(op)

        ## cleanup
        PatternRewriteWalker(EraseEdges()).rewrite_module(op)
