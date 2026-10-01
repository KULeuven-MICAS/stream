import pytest

pytest.importorskip("xdsl_aie", reason="the AIE dialects are a separate install, via stream-setup-aie")

from xdsl.dialects.builtin import MemRefType, ModuleOp, bf16  # noqa: E402
from xdsl.ir import Block, Region  # noqa: E402
from xdsl.pattern_rewriter import PatternRewriteWalker  # noqa: E402
from xdsl_aie.dialects.aie import (  # noqa: E402
    DeviceOp,
    DMABDOp,
    EndOp,
    NextBDOp,
    ObjectFifoOp,
    RuntimeSequenceOp,
    TileOp,
)
from xdsl_aie.dialects.aiex import DmaConfigureTaskForOp  # noqa: E402

from stream.compiler.transforms.aie_convert_ofs import ChainRereads  # noqa: E402

HEAD = 64 * 512
# A head's key, re-read once per query block: the iteration dimension at stride zero.
REREAD = ([4, 8, 64, 64], [0, 64, 512, 1])


def _sequence(offsets, sizes_strides=REREAD):
    """A runtime sequence feeding one fifo a task per offset, as a loop unrolls them."""
    shim, mem = TileOp(0, 0), TileOp(0, 1)
    fifo = ObjectFifoOp.from_referenced_type(shim, [mem], "key", 2, bf16, [64, 64])
    block = Block(arg_types=[MemRefType(bf16, [2 * len(offsets), 64, 512])])
    sizes, strides = sizes_strides
    tasks = [
        DmaConfigureTaskForOp(
            "key",
            Region(Block([DMABDOp(block.args[0], offset, HEAD, sizes, strides), EndOp()])),
            repeat_count=sizes[0] - 1,
        )
        for offset in offsets
    ]
    block.add_ops(tasks)
    return ModuleOp([DeviceOp(0, [shim, mem, fifo, RuntimeSequenceOp(Region(block))])])


def _chain(module, descriptors, iterations=64):
    PatternRewriteWalker(ChainRereads(descriptors, iterations), apply_recursively=False).rewrite_module(module)
    return [op for op in module.walk() if isinstance(op, DmaConfigureTaskForOp)]


def test_a_loop_around_a_rereading_transfer_becomes_one_chained_task():
    """Each head's key re-read four times: four descriptors, each stepping through the heads."""
    (task,) = _chain(_sequence([h * HEAD for h in range(3)]), descriptors=16)
    assert task.repeat_count.value.data == 2
    blocks = list(task.body.blocks)
    assert len(blocks) == 4
    for block in blocks:
        bd = block.first_op
        assert list(bd.static_sizes.get_values()) == [3, 8, 64, 64]
        assert list(bd.static_strides.get_values()) == [HEAD, 64, 512, 1]
        assert isinstance(bd.next_op, NextBDOp if block is not blocks[-1] else EndOp)


def test_a_chain_longer_than_the_shim_descriptors_keeps_its_tasks():
    assert len(_chain(_sequence([h * HEAD for h in range(3)]), descriptors=3)) == 3


def test_a_loop_longer_than_a_descriptor_iterates_keeps_its_tasks():
    assert len(_chain(_sequence([h * HEAD for h in range(3)]), descriptors=16, iterations=2)) == 3


@pytest.mark.parametrize(
    "offsets, sizes_strides",
    [
        ([0, HEAD, 3 * HEAD], REREAD),  # no one step between the heads
        ([0, HEAD, 2 * HEAD], ([1, 8, 64, 64], [0, 64, 512, 1])),  # read once a head
    ],
)
def test_only_an_evenly_stepped_rereading_loop_chains(offsets, sizes_strides):
    assert len(_chain(_sequence(offsets, sizes_strides), descriptors=16)) == 3
