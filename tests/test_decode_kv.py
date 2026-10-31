"""Decode attention over a KV cache laid out over the off-chip memories of a 4-slice ring and of TPU7x with an HBM per
chiplet: a cache split by KV heads as the cores split them streams each quarter from the HBM beside it, a pooled one
waits on the links out of its HBM, and flash-decoding moves only its partials between slices."""

import pytest
import yaml
from zigzag.utils import open_yaml

from stream.api import SolveOptions, evaluate_mapping
from stream.inputs.testing.workload.make_decode_attention import (
    DHEAD,
    KV_HEADS,
    make_decode_attention,
    make_flash_decoding,
)
from stream.parser.accelerator_factory import AcceleratorFactory
from stream.parser.accelerator_validator import AcceleratorValidator

RING = "stream/inputs/testing/hardware/tpu_like_ring_hbm.yaml"
IRONWOOD = "stream/inputs/testing/hardware/tpu_v7_ironwood_hbm_per_chiplet.yaml"
POOLED_IRONWOOD = "stream/inputs/examples/hardware/tpu_v7_ironwood.yaml"
HBMS = ["hbm_0", "hbm_1", "hbm_2", "hbm_3"]
BY_HEADS = {"memories": HBMS, "split": 1}
BY_BATCH = {"memories": HBMS, "split": 0}
BY_SEQUENCE = {"memories": HBMS, "split": 2}
POOLED = {"memories": ["hbm_0"]}
COPIED = {"memories": HBMS}
HBM_BITS_PER_CYCLE = {"ring": 1, "ironwood": 13400}
ICI_BITS_PER_CYCLE = 364
CONTEXT = {"ring": 256, "ironwood": 8192}


def _validated(path):
    validator = AcceleratorValidator(open_yaml(path), path)
    assert validator.validate()
    return validator.normalized_data


def _hardware(name):
    """The ring with its HBM stacks streaming a bit a cycle, slower than its cores attend (their arrays unroll no
    dimension of an einsum), so that its decode waits on the HBMs as Ironwood's does; or the per-chiplet Ironwood."""
    data = _validated(RING if name == "ring" else IRONWOOD)
    for hbm in HBMS if name == "ring" else ():
        data["memories"][hbm]["ports"]["rw_port_1"]["bandwidth"] = HBM_BITS_PER_CYCLE["ring"]
    for connection in data["connections"] if name == "ring" else ():
        if set(connection["ends"]) & set(HBMS):
            connection["bandwidth"] = HBM_BITS_PER_CYCLE["ring"]
    return AcceleratorFactory(data).create()


def _layouts(cache, q=BY_HEADS):
    return {"q": q, "K": cache, "V": cache, "O": BY_HEADS}


def _decode(tmp_path, hardware, layouts, batch=1, context=None, make=make_decode_attention, layers=(), tiling=()):  # noqa: PLR0913
    """``make``'s decode step on ``hardware`` mapped by the generic generator from a mapping file that only lays out
    its edges, or by ``layers`` fused in one group tiled by ``tiling``."""
    tmp_path.mkdir(exist_ok=True)
    workload = make(tmp_path / "decode.onnx", batch, context or CONTEXT[hardware])
    mapping = {"runtime_args": layouts}
    if layers:
        group = {"name": "decode", "layers": [layer["name"] for layer in layers], "intra_core_tiling": list(tiling)}
        mapping |= {"layers": list(layers), "fused_groups": [group]}
    (tmp_path / "mapping.yaml").write_text(yaml.safe_dump(mapping))
    return evaluate_mapping(
        _hardware(hardware),
        workload,
        str(tmp_path),
        str(tmp_path / "mapping.yaml"),
        SolveOptions(artifacts=False),
    )


def _cache_bits(batch, context):
    return 2 * batch * KV_HEADS * context * DHEAD * 16


def _link_bits(estimate):
    """Bits per firing each tensor's transfers carry over each link, by tensor name and the link's memories."""
    allocation = estimate.context.get("allocation")
    bits: dict[str, dict[tuple[str, ...], float]] = {}
    for tr in allocation.problem.workload.get_transfer_nodes():
        route = allocation.mapping.get(tr).resource_allocation[0]
        moved = allocation.space.moved_bits(tr, route)
        per_link = bits.setdefault(tr.inputs[0].name, {})
        for link, share in route.link_shares:
            key = tuple(sorted(memory.name for memory in link.memories))
            per_link[key] = per_link.get(key, 0) + share * moved
    return bits


def _on_chip(per_link):
    return {link: bits for link, bits in per_link.items() if not set(link) & set(HBMS) and bits}


def _cache_on_chip(estimate):
    """Of the cache's bits per firing over the links out of the HBMs, the share each on-chip link carries."""
    bits = _link_bits(estimate)
    from_hbms = sum(bits[t][link] for t in "KV" for link in bits[t] if set(link) & set(HBMS))
    on_chip = [_on_chip(bits[t]) for t in "KV"]
    return {link: sum(part.get(link, 0) for part in on_chip) / from_hbms for link in set().union(*on_chip)}


def _hbm_shares(estimate):
    """The share of the cache each HBM reads over the whole step."""
    rows = [row for group in estimate.context.get("group_memory_accesses").values() for row in group["offchip"]]
    reads = [(row["memory"], row["read"]) for row in rows if row["tensor"] in ("K", "V") and row["read"]]
    total = sum(read for _, read in reads)
    return {memory: sum(read for m, read in reads if m == memory) / total for memory in dict(reads)}


def _hbm_time(name, batch=1, context=None, hbms=4):
    """Cycles ``hbms`` HBMs take to stream the cache, each its share."""
    return _cache_bits(batch, context or CONTEXT[name]) / hbms / HBM_BITS_PER_CYCLE[name]


def test_the_per_chiplet_ironwood_has_an_hbm_and_a_measured_bandwidth_per_chiplet():
    accelerator = AcceleratorFactory(_validated(IRONWOOD)).create()
    assert [memory.name for memory in accelerator.offchip_memories] == HBMS
    assert {memory.capacity_bits for memory in accelerator.offchip_memories} == {96 * 2**30 * 8}
    models = {key.memory: model for key, model in accelerator.port_bandwidth.items()}
    assert sorted(models) == HBMS
    assert {(model.ceiling, model.contiguous) for model in models.values()} == {(13400, 13400)}


@pytest.mark.parametrize("hardware", ["ring", "ironwood"])
def test_a_cache_split_by_kv_heads_streams_each_quarter_from_the_hbm_beside_its_cores(tmp_path, hardware):
    estimate = _decode(tmp_path, hardware, _layouts(BY_HEADS))
    assert _cache_on_chip(estimate) == {}
    assert _hbm_shares(estimate) == dict.fromkeys(HBMS, 0.25)
    assert _hbm_time(hardware) <= estimate.cycles < 1.4 * _hbm_time(hardware)


def test_a_cache_pooled_in_one_slice_s_hbm_crosses_the_ring_and_waits_on_that_hbm(tmp_path):
    split = _decode(tmp_path / "split", "ring", _layouts(BY_HEADS))
    pooled = _decode(tmp_path / "pooled", "ring", _layouts(POOLED))
    assert _hbm_shares(pooled) == {"hbm_0": 1.0}
    assert _cache_on_chip(pooled) == {("sram_0", "sram_1"): 0.5, ("sram_0", "sram_3"): 0.25, ("sram_1", "sram_2"): 0.25}
    assert pooled.cycles >= _hbm_time("ring", hbms=1)
    assert 3.5 * split.cycles < pooled.cycles <= 4 * split.cycles


def test_a_cache_pooled_in_one_chiplet_s_hbm_waits_on_the_ici_link_to_the_other_chip(tmp_path):
    """Half the cache crosses ICI to the other chip, three quarters the die-to-die link out of the first chiplet."""
    split = _decode(tmp_path / "split", "ironwood", _layouts(BY_HEADS))
    pooled = _decode(tmp_path / "pooled", "ironwood", _layouts(POOLED))
    assert _hbm_shares(pooled) == {"hbm_0": 1.0}
    assert _cache_on_chip(pooled) == {("vmem_0", "vmem_1"): 0.75, ("vmem_1", "vmem_3"): 0.5, ("vmem_2", "vmem_3"): 0.25}
    ici_time = _cache_bits(1, CONTEXT["ironwood"]) / 2 / ICI_BITS_PER_CYCLE
    assert max(ici_time, _hbm_time("ironwood", hbms=1)) <= pooled.cycles < 1.02 * ici_time
    assert pooled.cycles / split.cycles > ici_time / (1.4 * _hbm_time("ironwood"))


def test_a_cache_split_on_the_sequence_while_the_cores_split_heads_crosses_the_chiplets(tmp_path):
    split = _decode(tmp_path / "split", "ironwood", _layouts(BY_HEADS))
    wrong = _decode(tmp_path / "wrong", "ironwood", _layouts(BY_SEQUENCE))
    assert _hbm_shares(wrong) == dict.fromkeys(HBMS, 0.25)
    assert {("vmem_0", "vmem_1"), ("vmem_1", "vmem_3"), ("vmem_2", "vmem_3")} <= set(_cache_on_chip(wrong))
    assert wrong.cycles > 10 * split.cycles


def test_a_cache_split_on_the_sequence_the_softmax_keeps_whole_costs_the_ring_only_its_traffic(tmp_path):
    """Every iteration reads all of the sequence, a quarter from each HBM, so no HBM is busier than with the split by
    heads; the ring, far faster than the HBMs, carries the three quarters each slice reads from the others, a hop or
    two each."""
    split = _decode(tmp_path / "split", "ring", _layouts(BY_HEADS))
    wrong = _decode(tmp_path / "wrong", "ring", _layouts(BY_SEQUENCE))
    assert _hbm_shares(wrong) == dict.fromkeys(HBMS, 0.25)
    assert sum(_cache_on_chip(wrong).values()) == 1.0
    assert wrong.cycles == pytest.approx(split.cycles, rel=0.01)


ATTENTION = [
    {"name": name, "core_allocation": [[0, 1, 2, 3]], "inter_core_tiling": [[{"dim": "D1", "split": 4}]]}
    for name in ("scores", "softmax", "context")
]


def test_a_cache_split_on_the_batch_the_iterations_walk_waits_on_one_hbm_at_a_time(tmp_path):
    """Heads split over the cores, one batch row an iteration: each iteration reads its row from the one HBM holding
    it, so each takes four times as long, though over the step every HBM reads its quarter."""
    walk = {"layers": ATTENTION, "tiling": [{"dim": "scores.D0", "tile": 1}], "batch": 4}
    split = _decode(tmp_path / "split", "ring", _layouts(BY_HEADS), **walk)
    wrong = _decode(tmp_path / "wrong", "ring", _layouts(BY_BATCH), **walk)
    assert _hbm_shares(split) == _hbm_shares(wrong) == dict.fromkeys(HBMS, 0.25)
    assert 3.5 * split.cycles < wrong.cycles <= 4 * split.cycles


@pytest.mark.parametrize("hardware", ["ring", "ironwood"])
def test_flash_decoding_with_its_parts_split_over_the_hbms_moves_only_the_combine_between_slices(tmp_path, hardware):
    """Each slice attends to the part of the sequence in its own HBM, every operator on a part following the split of
    what it reads; only the combine's maximum, sum and context cross between slices, and the step costs about what
    the split by heads does."""
    plain = _decode(tmp_path / "plain", hardware, _layouts(BY_HEADS))
    flash = _decode(tmp_path / "flash", hardware, _layouts(BY_SEQUENCE, q=COPIED), make=make_flash_decoding)
    bits = _link_bits(flash)
    crossing = {tensor: sum(_on_chip(per_link).values()) for tensor, per_link in bits.items() if _on_chip(per_link)}
    assert set(crossing) == {"M", "L", "N"}
    from_hbms = sum(bits[t][link] for t in "KV" for link in bits[t] if set(link) & set(HBMS))
    assert sum(crossing.values()) < from_hbms / 32
    assert _hbm_shares(flash) == dict.fromkeys(HBMS, 0.25)
    assert _hbm_time(hardware) <= flash.cycles < 1.1 * plain.cycles


LONG = {"batch": 1024, "context": 32768}
"""A batch whose 128 GiB cache overflows a chiplet's 96 GiB HBM, and whose scores spill at fusion cuts."""


def test_a_long_batched_cache_copied_in_each_chiplet_s_hbm_overflows_it(tmp_path):
    with pytest.raises(ValueError, match=r"^hbm_0 holds .*K \+ .*V \+ .* GiB > 96 GiB$"):
        _decode(tmp_path, "ironwood", _layouts(COPIED), **LONG)


def test_a_long_batched_cache_split_over_the_hbms_fits_and_its_scores_stay_beside_their_cores(tmp_path):
    """The scores spill to the HBM beside the cores that made them, and the next group reads them back from there."""
    estimate = _decode(tmp_path, "ironwood", _layouts(BY_HEADS), **LONG)
    assert _hbm_shares(estimate) == dict.fromkeys(HBMS, 0.25)
    assert estimate.cycles >= _hbm_time("ironwood", **LONG)
    (cut,) = (name for name in estimate.context.get("mapping").edge_layouts if name.endswith("_cut_in"))
    layout = estimate.context.get("mapping").edge_layouts[cut]
    assert [memory.name for memory in layout.memories] == HBMS and layout.axis == 1
    assert not any(_on_chip(per_link) for per_link in _link_bits(estimate).values())


def test_a_fusion_cut_without_a_layout_is_read_back_where_it_landed(tmp_path):
    """16 sequences of 2M: the scores no longer fit beside the cache, so each group spills them to the HBMs."""
    estimate = _decode(tmp_path, "ironwood", _layouts(BY_HEADS), batch=16, context=2**21)
    assert len(estimate.context.get("group_memory_accesses")) == 3
    assert _hbm_shares(estimate) == dict.fromkeys(HBMS, 0.25)


@pytest.mark.parametrize("context", [CONTEXT["ironwood"], pytest.param(32768, marks=pytest.mark.slow)])
def test_a_cache_split_by_heads_over_an_hbm_per_chiplet_costs_what_the_pooled_hbm_does(tmp_path, context):
    """Each chiplet reaches the pooled HBM over its own link at its four stacks' rate, so the split cache, read only
    by the chiplet beside it, streams as the pooled one does."""
    split = _decode(tmp_path / "split", "ironwood", _layouts(BY_HEADS), context=context)
    path = make_decode_attention(tmp_path / "pooled.onnx", 1, context)
    pooled = evaluate_mapping(POOLED_IRONWOOD, path, str(tmp_path), options=SolveOptions(artifacts=False))
    assert split.cycles == pooled.cycles
