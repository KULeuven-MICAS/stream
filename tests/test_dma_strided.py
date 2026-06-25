"""Tests for the DMA-link strided-transfer penalty (Stream-level model of a strided 2-D DMA over
the L1<->L3 link). Covers: the `dma`/`strided_*_penalty` link fields parsing into a
`CommunicationLink`, the strided-vs-contiguous tile classification, and the penalty applied to a
transfer's link latency by direction. The penalty feeds `_transfer_latency_for_path`, which the
constraint-optimization scheduler already consumes (slot_latency -> total_latency)."""

from math import ceil, prod
from types import SimpleNamespace

from zigzag.utils import open_yaml

from stream.hardware.architecture.noc.communication_link import CommunicationLink, get_bidirectional_edges
from stream.opt.allocation.constraint_optimization.transfer_and_tensor_allocation import (
    TransferAndTensorAllocator,
)
from stream.parser.accelerator_factory import AcceleratorFactory
from stream.parser.accelerator_validator import AcceleratorValidator
from stream.workload.node import TransferType

_HW = "stream/inputs/examples/hardware/meta_prototype_dual_core_simd_offchip.yaml"


# ----- duck-typed stubs (the cost fns touch only a few attributes) -----------------------------
def _tensor(tile, parent, bitwidth=8):
    shaped = SimpleNamespace(get_shape=lambda: parent)
    subview = SimpleNamespace(source=SimpleNamespace(type=shaped))
    return SimpleNamespace(
        shape=tuple(tile), subview=subview,
        size_bits=lambda shape=None: bitwidth * prod(tile),
    )


def _tr(tile, parent, ttype=TransferType.COMPUTE_TO_MEM):
    return SimpleNamespace(inputs=[_tensor(tile, parent)], transfer_type=ttype)


def _path(*links):
    # `sources`/`targets` feed upstream's multicast chain split in
    # get_transfer_latency_for_path; a single source/target keeps chains == 1 so these
    # tests isolate the strided-DMA penalty.
    return SimpleNamespace(links_used=list(links), sources=[0], targets=[0])


def _latency(tr, path):
    # _transfer_latency_for_path is an instance method, but it only reaches self for the
    # (static) _is_strided_transfer and for upstream's shared-memory shortcut. A
    # SimpleNamespace carrying both is a sufficient `self`; the shortcut is stubbed False
    # so these tests isolate the strided-DMA penalty rather than the shared-memory path.
    fake_self = SimpleNamespace(
        _is_strided_transfer=TransferAndTensorAllocator._is_strided_transfer,
        _choice_shares_memory=lambda *_: False,
    )
    return TransferAndTensorAllocator._transfer_latency_for_path(fake_self, tr, path)


# ----- strided classification ------------------------------------------------------------------
def test_strided_classification():
    S = TransferAndTensorAllocator._is_strided_transfer
    assert S(_tr((64, 8), (64, 8))) is False          # whole tensor -> contiguous
    assert S(_tr((8, 8), (64, 8))) is False            # tall row-block (inner dim full) -> contiguous
    assert S(_tr((8, 8), (8, 64))) is True             # wide column-block (inner dim partial) -> strided
    assert S(_tr((8, 16), (8, 64))) is True            # partial inner dim -> strided
    assert S(_tr((8,), (64,))) is False                # 1-D -> defensively not strided


def test_strided_classification_is_defensive():
    # missing/garbage subview -> not strided (never raises)
    bad = SimpleNamespace(inputs=[SimpleNamespace(shape=(8, 8))])
    assert TransferAndTensorAllocator._is_strided_transfer(bad) is False


# ----- penalty applied in the transfer latency -------------------------------------------------
def test_no_penalty_without_dma_link():
    plain = CommunicationLink("A", "B", 512, 0)            # dma defaults False
    base = ceil(8 * 8 * 8 / 512)                           # 8x8 int8 tile / 512 bit/cyc
    assert _latency(_tr((8, 8), (8, 64)), _path(plain)) == base   # strided but non-dma link -> base


def test_dma_link_penalises_strided_write_only():
    dma = CommunicationLink("A", "B", 512, 0, dma=True, strided_write_penalty=3.0, strided_read_penalty=2.0)
    base = ceil(8 * 8 * 8 / 512)
    # strided write -> x3
    assert _latency(_tr((8, 8), (8, 64), TransferType.COMPUTE_TO_MEM), _path(dma)) == ceil(base * 3.0)
    # strided read -> x2
    assert _latency(_tr((8, 8), (8, 64), TransferType.MEM_TO_COMPUTE), _path(dma)) == ceil(base * 2.0)
    # contiguous over the same dma link -> no penalty
    assert _latency(_tr((8, 8), (64, 8), TransferType.COMPUTE_TO_MEM), _path(dma)) == base


def test_penalty_scales_with_transfer_size():
    dma = CommunicationLink("A", "B", 512, 0, dma=True, strided_write_penalty=2.0)
    small = _latency(_tr((8, 8), (8, 64)), _path(dma))
    big = _latency(_tr((64, 8), (64, 128)), _path(dma))   # larger strided tile
    assert big > small
    assert _latency(_tr((), ()), _path()) == 0            # no links -> 0


# ----- parsing: dma fields flow YAML -> CommunicationLink --------------------------------------
def test_get_bidirectional_edges_threads_dma():
    a, b = SimpleNamespace(id=0), SimpleNamespace(id=1)
    edges = get_bidirectional_edges(a, b, bandwidth=128, unit_energy_cost=0, link_type="link",
                                    dma=True, strided_write_penalty=4.0)
    links = [d["cl"] for _, _, d in edges]
    assert all(l.dma for l in links)
    assert all(l.strided_write_penalty == 4.0 for l in links)


def test_validator_and_factory_carry_dma_flag():
    data = open_yaml(_HW)
    for conn in data["core_connectivity"]:
        if conn.get("type") == "bus":   # the shared off-chip bus
            conn["dma"] = True
            conn["strided_write_penalty"] = 2.0
    validator = AcceleratorValidator(data, _HW)
    normalized = validator.normalized_data
    assert validator.validate(), "validator rejected the dma link fields"
    acc = AcceleratorFactory(normalized).create()
    dma_links = [d["cl"] for _, _, d in acc.cores.edges(data=True) if getattr(d.get("cl"), "dma", False)]
    assert dma_links, "no dma-flagged link survived into the core graph"
    assert all(l.strided_write_penalty == 2.0 for l in dma_links)


def test_existing_hardware_links_default_to_non_dma():
    # regression: untouched hardware -> every link dma=False, penalties 1.0 (no behaviour change)
    data = open_yaml(_HW)
    validator = AcceleratorValidator(data, _HW)
    assert validator.validate()
    acc = AcceleratorFactory(validator.normalized_data).create()
    links = [d["cl"] for _, _, d in acc.cores.edges(data=True) if d.get("cl") is not None]
    assert links and all(not l.dma and l.strided_write_penalty == 1.0 and l.strided_read_penalty == 1.0
                         for l in links)
