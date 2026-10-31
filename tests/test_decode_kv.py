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
MXUS, VPUS = [0, 1, 4, 5, 8, 9, 12, 13], [2, 6, 10, 14]


def _validated(path):
    validator = AcceleratorValidator(open_yaml(path), path)
    assert validator.validate()
    return validator.normalized_data


def _hardware(name, hbm_capacity_bits=None):
    """The ring with its HBM stacks streaming a bit a cycle, slower than its cores attend (their arrays unroll no
    dimension of an einsum), so that its decode waits on the HBMs as Ironwood's does; or the per-chiplet Ironwood."""
    data = _validated(RING if name == "ring" else IRONWOOD)
    for hbm in HBMS:
        if name == "ring":
            data["memories"][hbm]["ports"]["rw_port_1"]["bandwidth"] = HBM_BITS_PER_CYCLE["ring"]
        if hbm_capacity_bits:
            data["memories"][hbm]["capacity_bits"] = hbm_capacity_bits
    for connection in data["connections"] if name == "ring" else ():
        if set(connection["ends"]) & set(HBMS):
            connection["bandwidth"] = HBM_BITS_PER_CYCLE["ring"]
    return AcceleratorFactory(data).create()


def _layouts(cache, q=BY_HEADS):
    return {"q": q, "K": cache, "V": cache, "O": BY_HEADS}


def _decode(tmp_path, hardware, layouts, batch=1, context=None, make=make_decode_attention, layers=(), tiling=()):  # noqa: PLR0913
    """``make``'s decode step on ``hardware``, by name or built, mapped by the generic generator from a mapping file
    that only lays out its edges, or by ``layers`` fused in one group tiled by ``tiling``."""
    tmp_path.mkdir(exist_ok=True)
    workload = make(tmp_path / "decode.onnx", batch, context or CONTEXT[hardware])
    mapping = {"runtime_args": layouts}
    if layers:
        group = {"name": "decode", "layers": [layer["name"] for layer in layers], "intra_core_tiling": list(tiling)}
        mapping |= {"layers": list(layers), "fused_groups": [group]}
    (tmp_path / "mapping.yaml").write_text(yaml.safe_dump(mapping))
    return evaluate_mapping(
        _hardware(hardware) if isinstance(hardware, str) else hardware,
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


def _flash_layers(einsum_cores, vector_cores):
    """Each part attended to in its own slice, the parts split over the slices, and the combine split by heads."""
    per_part = ["scores", "part_max", "part_shift", "part_exp", "part_sum", "part_context"]
    combine = ["max", "shift", "weight", "weigh_sum", "sum", "weigh_context", "combine", "normalize"]

    def layer(name, dim):
        cores = einsum_cores if name in ("scores", "part_context") else vector_cores
        split = [{"dim": dim, "split": 4}] + ([{"dim": "D1", "split": len(cores) // 4}] if len(cores) > 4 else [])
        return {"name": name, "core_allocation": [cores], "inter_core_tiling": [split]}

    return [layer(name, "D2") for name in per_part] + [layer(name, "D1") for name in combine]


@pytest.mark.parametrize(
    ("hardware", "layers"),
    [("ring", _flash_layers([0, 1, 2, 3], [0, 1, 2, 3])), ("ironwood", _flash_layers(MXUS, VPUS))],
)
def test_flash_decoding_with_its_parts_split_over_the_hbms_moves_only_its_partials(tmp_path, hardware, layers):
    """Each slice attends to the part of the sequence in its own HBM; only each part's maximum, sum and context
    cross between slices to be combined, and the step costs about what the split by heads does."""
    plain = _decode(tmp_path / "plain", hardware, _layouts(BY_HEADS))
    flash = _decode(
        tmp_path / "flash", hardware, _layouts(BY_SEQUENCE, q=COPIED), make=make_flash_decoding, layers=layers
    )
    crossing = {tensor for tensor, per_link in _link_bits(flash).items() if _on_chip(per_link)}
    assert crossing == {"m", "l", "o"}
    assert _hbm_shares(flash) == dict.fromkeys(HBMS, 0.25)
    assert _hbm_time(hardware) <= flash.cycles < 1.3 * plain.cycles


def test_a_cache_split_over_the_hbms_fits_where_its_copy_in_each_does_not(tmp_path):
    """The 32 MiB cache whole in each 16 MiB HBM is refused naming the HBM; a quarter in each fits."""
    hardware = _hardware("ironwood", hbm_capacity_bits=16 * 2**20 * 8)
    with pytest.raises(ValueError, match=r"^hbm_0 holds K \+ V \+ q/4 \+ O/4 = 32 MiB > 16 MiB$"):
        _decode(tmp_path / "copy", hardware, _layouts(COPIED), context=8192)
    assert _decode(tmp_path / "split", hardware, _layouts(BY_HEADS), context=8192).cycles


def test_a_long_batched_cache_copied_in_each_chiplet_s_hbm_overflows_it(tmp_path):
    with pytest.raises(ValueError, match=r"^hbm_0 holds .*K \+ .*V \+ .* GiB > 96 GiB$"):
        _decode(tmp_path, "ironwood", _layouts(COPIED), batch=1024, context=32768)


@pytest.mark.parametrize("context", [CONTEXT["ironwood"], pytest.param(32768, marks=pytest.mark.slow)])
def test_a_cache_split_by_heads_over_an_hbm_per_chiplet_costs_what_the_pooled_hbm_does(tmp_path, context):
    """Each chiplet reaches the pooled HBM over its own link at its four stacks' rate, so the split cache, read only
    by the chiplet beside it, streams as the pooled one does."""
    split = _decode(tmp_path / "split", "ironwood", _layouts(BY_HEADS), context=context)
    path = make_decode_attention(tmp_path / "pooled.onnx", 1, context)
    pooled = evaluate_mapping(POOLED_IRONWOOD, path, str(tmp_path), options=SolveOptions(artifacts=False))
    assert split.cycles == pooled.cycles
