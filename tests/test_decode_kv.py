"""Decode attention over a KV cache placed over the off-chip memories of a 4-slice ring and of TPU7x with an HBM per
chiplet: a cache split by KV heads as the cores split them streams each quarter from the HBM beside it, a pooled one
waits on the links out of its HBM, and flash-decoding moves only its partials between slices."""

from functools import partial

import networkx as nx
import pytest
import yaml
from onnx import helper
from zigzag.utils import open_yaml

from stream.api import SolveOptions, evaluate_mapping
from stream.inputs.testing.workload.make_decode_attention import (
    DHEAD,
    DTYPE,
    GROUP,
    KV_HEADS,
    _axes,
    _save,
    make_decode_attention,
    make_flash_decoding,
)
from stream.ir.infeasibility import InfeasibleAllocationError
from stream.mapping.mapping import offchip_holdings, with_cut_landings
from stream.parser.accelerator_factory import AcceleratorFactory
from stream.parser.accelerator_validator import AcceleratorValidator
from stream.workload.iterator_type import IteratorType, derive_iterator_types
from stream.workload.node import ComputationNode
from stream.workload.utils import determine_fusion_splits

RING = "stream/inputs/testing/hardware/tpu_like_ring_hbm.yaml"
IRONWOOD = "stream/inputs/testing/hardware/tpu_v7_ironwood_hbm_per_chiplet.yaml"
POOLED_IRONWOOD = "stream/inputs/examples/hardware/tpu_v7_ironwood.yaml"
HBMS = ["hbm_0", "hbm_1", "hbm_2", "hbm_3"]
BY_HEADS = {"placement": {"memories": HBMS, "split": 1}}
BY_BATCH = {"placement": {"memories": HBMS, "split": 0}}
BY_SEQUENCE = {"placement": {"memories": HBMS, "split": 2}}
POOLED = {"placement": {"memories": ["hbm_0"]}}
COPIED = {"placement": {"memories": HBMS}}
HBM_BITS_PER_CYCLE = {"ring": 1, "ironwood": 13400}
ICI_BITS_PER_CYCLE = 364
CONTEXT = {"ring": 256, "ironwood": 8192}


def _validated(path):
    validator = AcceleratorValidator(open_yaml(path), path)
    assert validator.validate()
    return validator.normalized_data


def _hardware(name, sram_bits=None):
    """The ring with its HBM stacks streaming a bit a cycle, slower than its cores attend (their arrays unroll no
    dimension of an einsum), so that its decode waits on the HBMs as Ironwood's does, its slices' SRAM ``sram_bits``
    if given; or the per-chiplet Ironwood."""
    data = _validated(RING if name == "ring" else IRONWOOD)
    for core in data["cores"].values() if sram_bits else ():
        core["memories"]["sram_2MB"]["size"] = sram_bits
    for hbm in HBMS if name == "ring" else ():
        data["memories"][hbm]["ports"]["rw_port_1"]["bandwidth"] = HBM_BITS_PER_CYCLE["ring"]
    for connection in data["connections"] if name == "ring" else ():
        if set(connection["ends"]) & set(HBMS):
            connection["bandwidth"] = HBM_BITS_PER_CYCLE["ring"]
    return AcceleratorFactory(data).create()


def _ring_of(slices):
    """The ring with ``slices`` slices, each with an HBM stack of its own."""
    link = {"type": "link", "bandwidth": 128}
    rw = {"r_port_1": {"from": "r_port_1"}, "w_port_1": {"from": "w_port_1"}}
    hbm = {"bandwidth": 8, "serves": {"any": ["read", "write"]}, "read_energy": 15.625, "write_energy": 15.625}
    data = {
        "name": f"ring_of_{slices}",
        "cores": dict.fromkeys(range(slices), "tpu_like.yaml"),
        "memories": {
            **{
                f"sram_{s}": {
                    "offchip": False,
                    "views": [f"{s}.sram_2MB"],
                    "ports": {name: port | {"cores": [s]} for name, port in rw.items()},
                }
                for s in range(slices)
            },
            **{
                f"hbm_{s}": {"offchip": True, "views": [], "capacity_bits": 10**10, "ports": {"rw_port_1": hbm}}
                for s in range(slices)
            },
        },
        "connections": [
            link | {"ends": {f"sram_{s}": list(rw), f"sram_{(s + 1) % slices}": list(rw)}} for s in range(slices)
        ]
        + [link | {"bandwidth": 8, "ends": {f"sram_{s}": list(rw), f"hbm_{s}": ["rw_port_1"]}} for s in range(slices)],
    }
    validator = AcceleratorValidator(data, RING)
    assert validator.validate()
    return AcceleratorFactory(validator.normalized_data).create()


def _placements(cache, q=BY_HEADS):
    return {"q": q, "K": cache, "V": cache, "O": BY_HEADS}


def _decode(  # noqa: PLR0913
    tmp_path,
    hardware,
    placements,
    batch=1,
    context=None,
    make=make_decode_attention,
    layers=(),
    tiling=(),
    *,
    parts=None,
):
    """``make``'s decode step on ``hardware`` mapped by the generic generator from a mapping file that only places
    its edges, its attention split into ``parts`` if given, or by ``layers`` fused in one group tiled by ``tiling``."""
    tmp_path.mkdir(exist_ok=True)
    workload = make(tmp_path / "decode.onnx", batch, context or CONTEXT[hardware])
    mapping = {"runtime_args": placements}
    if layers:
        group = {"name": "decode", "layers": [layer["name"] for layer in layers], "intra_core_tiling": list(tiling)}
        mapping |= {"layers": list(layers), "fused_groups": [group]}
    (tmp_path / "mapping.yaml").write_text(yaml.safe_dump(mapping))
    return evaluate_mapping(
        _hardware(hardware) if isinstance(hardware, str) else hardware,
        workload,
        str(tmp_path),
        str(tmp_path / "mapping.yaml"),
        SolveOptions(artifacts=False, stage_options={"attention_parts": parts} if parts else {}),
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
    estimate = _decode(tmp_path, hardware, _placements(BY_HEADS))
    assert _cache_on_chip(estimate) == {}
    assert _hbm_shares(estimate) == dict.fromkeys(HBMS, 0.25)
    assert _hbm_time(hardware) <= estimate.cycles < 1.4 * _hbm_time(hardware)


def test_a_long_cache_split_by_kv_heads_is_tiled_without_walking_the_heads(tmp_path):
    """8k positions no longer fit a slice whole: the tiler cuts the head dimension or the sequence, never the heads the
    placement splits, so each slice still streams its own quarter."""
    estimate = _decode(tmp_path, "ring", _placements(BY_HEADS), context=8192)
    assert _cache_on_chip(estimate) == {}
    assert _hbm_shares(estimate) == dict.fromkeys(HBMS, 0.25)
    assert estimate.cycles < 1.1 * _hbm_time("ring", context=8192)


def test_a_cache_pooled_in_one_slice_s_hbm_crosses_the_ring_and_waits_on_that_hbm(tmp_path):
    split = _decode(tmp_path / "split", "ring", _placements(BY_HEADS))
    pooled = _decode(tmp_path / "pooled", "ring", _placements(POOLED))
    assert _hbm_shares(pooled) == {"hbm_0": 1.0}
    assert _cache_on_chip(pooled) == {("sram_0", "sram_1"): 0.5, ("sram_0", "sram_3"): 0.25, ("sram_1", "sram_2"): 0.25}
    assert pooled.cycles >= _hbm_time("ring", hbms=1)
    assert 3.5 * split.cycles < pooled.cycles <= 4 * split.cycles


def test_a_cache_pooled_in_one_chiplet_s_hbm_waits_on_the_ici_link_to_the_other_chip(tmp_path):
    """Half the cache crosses ICI to the other chip, three quarters the die-to-die link out of the first chiplet."""
    split = _decode(tmp_path / "split", "ironwood", _placements(BY_HEADS))
    pooled = _decode(tmp_path / "pooled", "ironwood", _placements(POOLED))
    assert _hbm_shares(pooled) == {"hbm_0": 1.0}
    assert _cache_on_chip(pooled) == {("vmem_0", "vmem_1"): 0.75, ("vmem_1", "vmem_3"): 0.5, ("vmem_2", "vmem_3"): 0.25}
    ici_time = _cache_bits(1, CONTEXT["ironwood"]) / 2 / ICI_BITS_PER_CYCLE
    assert max(ici_time, _hbm_time("ironwood", hbms=1)) <= pooled.cycles < 1.02 * ici_time
    assert pooled.cycles / split.cycles > ici_time / (1.4 * _hbm_time("ironwood"))


def test_a_cache_split_on_the_sequence_while_the_cores_split_heads_crosses_the_chiplets(tmp_path):
    split = _decode(tmp_path / "split", "ironwood", _placements(BY_HEADS))
    wrong = _decode(tmp_path / "wrong", "ironwood", _placements(BY_SEQUENCE), parts=1)
    assert _hbm_shares(wrong) == dict.fromkeys(HBMS, 0.25)
    assert {("vmem_0", "vmem_1"), ("vmem_1", "vmem_3"), ("vmem_2", "vmem_3")} <= set(_cache_on_chip(wrong))
    assert wrong.cycles > 10 * split.cycles


def test_a_cache_split_on_the_sequence_the_softmax_keeps_whole_costs_the_ring_only_its_traffic(tmp_path):
    """Every iteration reads all of the sequence, a quarter from each HBM, so no HBM is busier than with the split by
    heads; the ring, far faster than the HBMs, carries the three quarters each slice reads from the others, a hop or
    two each."""
    split = _decode(tmp_path / "split", "ring", _placements(BY_HEADS))
    wrong = _decode(tmp_path / "wrong", "ring", _placements(BY_SEQUENCE), parts=1)
    assert _hbm_shares(wrong) == dict.fromkeys(HBMS, 0.25)
    assert sum(_cache_on_chip(wrong).values()) == 1.0
    assert wrong.cycles == pytest.approx(split.cycles, rel=0.01)


def test_fewer_kv_heads_than_chiplets_copied_in_each_hbm_are_read_from_the_hbm_beside_each_reader(tmp_path):
    """Two KV heads over four chiplets, the cache copied in each HBM: every chiplet streams the head its cores attend
    to from its own HBM, at most one head's K and V, and none crosses the die-to-die link or ICI."""
    placements = {"q": COPIED, "K": COPIED, "V": COPIED, "O": {"placement": {"memories": HBMS, "split": 3}}}
    estimate = _decode(tmp_path, "ironwood", placements, make=partial(make_decode_attention, kv_heads=2))
    bits = _link_bits(estimate)
    assert _cache_on_chip(estimate) == {}
    assert _hbm_shares(estimate) == dict.fromkeys(HBMS, 0.25)
    per_hbm = {hbm: sum(bits[t].get((hbm, f"vmem_{hbm[-1]}"), 0) for t in "KV") for hbm in HBMS}
    assert max(per_hbm.values()) <= 2 * CONTEXT["ironwood"] * DHEAD * 16


ATTENTION = [
    {"name": name, "core_allocation": [[0, 1, 2, 3]], "inter_core_tiling": [[{"dim": "D1", "split": 4}]]}
    for name in ("scores", "softmax", "context")
]


def test_a_cache_split_on_the_batch_the_iterations_walk_waits_on_one_hbm_at_a_time(tmp_path):
    """Heads split over the cores, one batch row an iteration: each iteration reads its row from the one HBM holding
    it, so each takes four times as long, though over the step every HBM reads its quarter."""
    walk = {"layers": ATTENTION, "tiling": [{"dim": "scores.D0", "tile": 1}], "batch": 4}
    split = _decode(tmp_path / "split", "ring", _placements(BY_HEADS), **walk)
    wrong = _decode(tmp_path / "wrong", "ring", _placements(BY_BATCH), **walk)
    assert _hbm_shares(split) == _hbm_shares(wrong) == dict.fromkeys(HBMS, 0.25)
    assert 3.5 * split.cycles < wrong.cycles <= 4 * split.cycles


@pytest.mark.parametrize("hardware", ["ring", "ironwood"])
def test_flash_decoding_with_its_parts_split_over_the_hbms_moves_only_the_combine_between_slices(tmp_path, hardware):
    """Each slice attends to the part of the sequence in its own HBM, every operator on a part following the split of
    what it reads; only the combine's maximum, sum and context cross between slices, and the step costs about what
    the split by heads does. Every transfer's steady state is described, those its readers split finer too."""
    plain = _decode(tmp_path / "plain", hardware, _placements(BY_HEADS))
    flash = _decode(tmp_path / "flash", hardware, _placements(BY_SEQUENCE, q=COPIED), make=make_flash_decoding)
    bits = _link_bits(flash)
    crossing = {tensor: sum(_on_chip(per_link).values()) for tensor, per_link in bits.items() if _on_chip(per_link)}
    assert set(crossing) == {"M", "L", "N"}
    from_hbms = sum(bits[t][link] for t in "KV" for link in bits[t] if set(link) & set(HBMS))
    assert sum(crossing.values()) < from_hbms / 32
    assert _hbm_shares(flash) == dict.fromkeys(HBMS, 0.25)
    assert _hbm_time(hardware) <= flash.cycles < 1.1 * plain.cycles
    (group,) = flash.context.get("group_allocations").values()
    transfers = [node for node in group["steady_state"]["tiled_graph"]["nodes"] if node["kind"] == "transfer"]
    assert all("moved" in node for node in transfers)


SMALL_SLICE = {"hardware": _hardware("ring", sram_bits=512 * 1024 * 8), "context": 32768}
"""Slices of 512 KiB, on which 32k positions of a head's scores do not fit whole."""


def _walks_an_unfinished_reduction(estimate):
    """Whether a group tiles a dim a node reduces over while a node computed from that reduction walks it."""
    groups = zip(estimate.context.get("sub_workloads"), estimate.context.get("sub_mappings"), strict=True)
    for workload, mapping in groups:
        splits = determine_fusion_splits(workload, mapping)
        for node in workload.get_computation_nodes():
            dims = workload.get_dims(node)
            reduced = {dims[p] for p, kind in derive_iterator_types(node).items() if kind is IteratorType.REDUCTION}
            later = [n for n in nx.descendants(workload, node) if isinstance(n, ComputationNode)]
            if any(splits.get(d, 1) > 1 and any(d in workload.get_dims(n) for n in later) for d in reduced):
                return True
    return False


def _scaled_statistics_attention(path, batch, context):
    """Plain attention whose softmax scales its maximum and its sum before using them."""
    value = lambda name, shape: helper.make_tensor_value_info(name, DTYPE, shape)  # noqa: E731
    node = helper.make_node
    nodes = [
        node("Einsum", ["q", "K"], ["S"], equation="bgrd,bgsd->bgrs", name="scores"),
        node("ReduceMax", ["S", "last"], ["m"], keepdims=1, name="max"),
        node("Mul", ["m", "c"], ["mc"], name="scale_max"),
        node("Sub", ["S", "mc"], ["Z"], name="shift"),
        node("Exp", ["Z"], ["E"], name="exp"),
        node("ReduceSum", ["E", "last"], ["l"], keepdims=1, name="sum"),
        node("Mul", ["l", "c"], ["lc"], name="scale_sum"),
        node("Div", ["E", "lc"], ["P"], name="normalize"),
        node("Einsum", ["P", "V"], ["O"], equation="bgrs,bgsd->bgrd", name="context"),
    ]
    cache = [batch, KV_HEADS, context, DHEAD]
    inputs = [value("q", [batch, KV_HEADS, GROUP, DHEAD]), value("K", cache), value("V", cache)]
    constants = [_axes("last", -1), helper.make_tensor("c", DTYPE, [1], [1.0])]
    return _save(nodes, "scaled", inputs, [value("O", [batch, KV_HEADS, GROUP, DHEAD])], path, constants)


def test_a_softmax_reading_its_statistics_through_other_ops_is_not_tiled_along_them(tmp_path):
    placements = {**_placements(BY_HEADS), "c": COPIED}
    estimate = _decode(tmp_path, placements=placements, make=_scaled_statistics_attention, **SMALL_SLICE)
    assert not _walks_an_unfinished_reduction(estimate)


def _cache_view(estimate):
    return estimate.context.get("sub_workloads")[0].get_node_by_name("K").outputs[0].shape


def test_attention_on_a_cache_split_on_the_sequence_is_split_into_a_part_per_hbm(tmp_path):
    """Plain attention whose scores do not fit a slice whole, its cache split on the sequence: Stream views the cache as
    a part per HBM, keeps each part's softmax whole, and streams the cache near the HBMs' rate."""
    estimate = _decode(tmp_path, placements=_placements(BY_SEQUENCE, q=COPIED), **SMALL_SLICE)
    assert _cache_view(estimate) == (1, KV_HEADS, 4, 8192, DHEAD)
    assert "context_combine" in {node.name for w in estimate.context.get("sub_workloads") for node in w.nodes}
    assert not _walks_an_unfinished_reduction(estimate)
    assert _hbm_shares(estimate) == dict.fromkeys(HBMS, 0.25)
    assert _hbm_time("ring", context=32768) <= estimate.cycles < 1.05 * _hbm_time("ring", context=32768)


def test_attention_on_a_cache_split_by_heads_is_split_into_the_fewest_parts_that_fit(tmp_path):
    estimate = _decode(tmp_path, placements=_placements(BY_HEADS), **SMALL_SLICE)
    assert _cache_view(estimate) == (1, KV_HEADS, 32, 1024, DHEAD)
    assert not _walks_an_unfinished_reduction(estimate)
    assert estimate.cycles < 1.01 * _hbm_time("ring", context=32768)


def test_flash_decoding_written_by_hand_keeps_each_part_s_softmax_whole(tmp_path):
    """Its maxima are no softmax's, yet the exponent reads each part's maximum along the part: it is not tiled."""
    placements = _placements(BY_SEQUENCE, q=COPIED)
    estimate = _decode(tmp_path, placements=placements, make=make_flash_decoding, **SMALL_SLICE)
    assert not _walks_an_unfinished_reduction(estimate)


def test_a_part_count_the_cache_layout_does_not_split_into_is_refused(tmp_path):
    with pytest.raises(
        ValueError, match="K is split along its axis 2 over 4 memories, which 2 parts of context's sequence"
    ):
        _decode(tmp_path, placements=_placements(BY_SEQUENCE, q=COPIED), parts=2, **SMALL_SLICE)


@pytest.mark.slow
def test_flash_decoding_too_large_for_one_group_is_cut_into_two(tmp_path):
    """16 sequences of 32k: each part's softmax kept whole, the step is two groups, not a group per operator. The first
    streams K at the HBMs' rate; the second streams V but its cores, whose arrays unroll no dimension of an einsum,
    compute each tile's context after its V arrives rather than while the next one does."""
    estimate = _decode(tmp_path, "ring", _placements(BY_HEADS), batch=16, context=32768, make=make_flash_decoding)
    keys, values = estimate.group_cycles
    half = _hbm_time("ring", batch=16, context=32768) / 2
    assert not _walks_an_unfinished_reduction(estimate)
    assert half <= keys < 1.05 * half
    assert half <= values < 1.55 * half


LONG = {"batch": 1024, "context": 32768}
"""A batch whose 128 GiB cache overflows a chiplet's 96 GiB HBM, and whose scores spill at fusion cuts."""


def test_a_long_batched_cache_copied_in_each_chiplet_s_hbm_overflows_it(tmp_path):
    with pytest.raises(InfeasibleAllocationError, match=r"^hbm_0 holds .*K \+ .*V \+ .* GiB > 96 GiB$"):
        _decode(tmp_path, "ironwood", _placements(COPIED), **LONG)


def test_a_long_batched_cache_split_over_the_hbms_fits_and_its_scores_stay_beside_their_cores(tmp_path):
    """The scores spill to the HBM beside the cores that made them, and the next group reads them back from there."""
    estimate = _decode(tmp_path, "ironwood", _placements(BY_HEADS), **LONG)
    assert _hbm_shares(estimate) == dict.fromkeys(HBMS, 0.25)
    assert estimate.cycles >= _hbm_time("ironwood", **LONG)
    (cut,) = (name for name in estimate.context.get("mapping").edge_placements if name.endswith("_cut_in"))
    placement = estimate.context.get("mapping").edge_placements[cut]
    assert [memory.name for memory in placement.memories] == HBMS and placement.axis == 1
    assert not any(_on_chip(per_link) for per_link in _link_bits(estimate).values())


def test_a_fusion_cut_without_a_layout_is_read_back_where_it_landed(tmp_path):
    """16 sequences of 2M: the scores no longer fit beside the cache, so each group spills them to the HBMs."""
    estimate = _decode(tmp_path, "ironwood", _placements(BY_HEADS), batch=16, context=2**21)
    assert len(estimate.context.get("group_memory_accesses")) == 3
    assert _hbm_shares(estimate) == dict.fromkeys(HBMS, 0.25)


def test_a_fusion_cut_whose_cores_split_only_a_reduction_is_read_back_from_the_nearest_hbm(tmp_path):
    """Split on the sequence, 16 sequences of 32k spill their scores at a cut made by cores splitting the sequence the
    scores reduce over, which indexes no axis of what they write: the cut is read back where each core landed it."""
    estimate = _decode(tmp_path, "ironwood", _placements(BY_SEQUENCE, q=COPIED), batch=16, context=32768, parts=1)
    assert len(estimate.context.get("group_memory_accesses")) > 1
    assert _hbm_shares(estimate) == dict.fromkeys(HBMS, 0.25)


def test_a_fusion_cut_split_on_the_sequence_over_eight_slices_is_read_back_from_the_nearest_hbm(tmp_path):
    hbms = [f"hbm_{s}" for s in range(8)]
    by_sequence = {"placement": {"memories": hbms, "split": 2}}
    placements = {
        "q": {"placement": {"memories": hbms}},
        "K": by_sequence,
        "V": by_sequence,
        "O": {"placement": {"memories": hbms, "split": 1}},
    }
    estimate = _decode(tmp_path, _ring_of(8), placements, batch=16, context=8192, parts=1)
    assert len(estimate.context.get("group_memory_accesses")) > 1
    assert _hbm_shares(estimate) == dict.fromkeys(hbms, 1 / 8)


def test_a_reduction_the_cores_split_crossing_to_a_later_group_is_read_back_from_each_hbm_it_landed_in(tmp_path):
    """Flash-decoding over 2M positions, a part in each chiplet's HBM, cut after the maximum over the parts: that
    maximum lands whole beside every chiplet, and the next group reads it there."""
    placements = _placements(BY_SEQUENCE, q=COPIED)
    estimate = _decode(tmp_path, "ironwood", placements, batch=16, context=2**21, make=make_flash_decoding)
    assert _hbm_shares(estimate) == dict.fromkeys(HBMS, 0.25)
    assert estimate.cycles < 1.05 * _hbm_time("ironwood", batch=16, context=2**21)
    groups = list(estimate.context.get("group_allocations").values())
    copies = [
        name for group in groups for name, placement in group["edge_placements"].items() if placement["axis"] is None
    ]
    assert copies == ["q", "max_cut_in"]
    assert groups[1]["edge_placements"]["max_cut_in"]["memories"] == HBMS
    context = estimate.context
    accelerator, workloads = context.get("accelerator"), context.get("sub_workloads")
    held = offchip_holdings(
        accelerator, workloads, with_cut_landings(accelerator, workloads, context.get("sub_mappings"))
    )
    assert {memory.name: next(n for t, n in shares.items() if t.name == "M") for memory, shares in held.items()} == (
        dict.fromkeys(HBMS, 1)
    )


@pytest.mark.parametrize("context", [CONTEXT["ironwood"], pytest.param(32768, marks=pytest.mark.slow)])
def test_a_cache_split_by_heads_over_an_hbm_per_chiplet_costs_what_the_pooled_hbm_does(tmp_path, context):
    """Each chiplet reaches the pooled HBM over its own link at its four stacks' rate, so the split cache, read only
    by the chiplet beside it, streams as the pooled one does."""
    split = _decode(tmp_path / "split", "ironwood", _placements(BY_HEADS), context=context)
    path = make_decode_attention(tmp_path / "pooled.onnx", 1, context)
    pooled = evaluate_mapping(POOLED_IRONWOOD, path, str(tmp_path), options=SolveOptions(artifacts=False))
    assert split.cycles == pooled.cycles
