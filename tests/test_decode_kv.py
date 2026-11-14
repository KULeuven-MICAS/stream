"""Decode attention over a KV cache placed over the off-chip memories of a 4-slice ring and of TPU7x with an HBM per
chiplet: a cache split by KV heads as the cores split them streams each quarter from the HBM beside it, a pooled one
waits on the links out of its HBM, and flash-decoding moves only its partials between slices."""

import json
import os
import subprocess
import sys
from functools import partial

import networkx as nx
import pytest
from onnx import helper
from placements import (
    COPIED,
    HBMS,
    IRONWOOD,
    POOLED,
    TPU_V7,
    hardware,
    held_once,
    offchip_reads,
    ring_of,
    route_of,
    solve,
    split_on,
)

from stream.inputs.testing.workload.make_decode_attention import (
    DHEAD,
    DTYPE,
    KV_HEADS,
    axes_tensor,
    make_decode_attention,
    make_flash_decoding,
)
from stream.ir.infeasibility import InfeasibleAllocationError
from stream.mapping.generic_generator import GenericMappingGenerator
from stream.mapping.mapping import viewed_placements
from stream.opt.allocation.constraint_optimization.families.traffic import link_name
from stream.stages.context import StageContext
from stream.stages.parsing.accelerator_parser import AcceleratorParserStage
from stream.stages.parsing.onnx_model_parser import ONNXModelParserStage
from stream.stages.stage import LeafStage, MainStage
from stream.workload.iterator_type import IteratorType, derive_iterator_types
from stream.workload.node import ComputationNode
from stream.workload.normalization import expand_normalizations
from stream.workload.utils import determine_fusion_splits

BY_HEADS, BY_BATCH, BY_SEQUENCE = split_on(1), split_on(0), split_on(2)
HBM_BITS_PER_CYCLE = {"ring": 1, "ironwood": 13400}
CROSSING_BITS_PER_CYCLE = {"ring": 128, "ironwood": 364}
"""The slowest link a cache read from another slice's HBM crosses: a ring link, or ICI between Ironwood's chips."""
CONTEXT = {"ring": 256, "ironwood": 8192}


def _hardware(name, **changes):
    """The ring with its HBM stacks streaming a bit a cycle, slower than its cores attend (their arrays unroll no
    dimension of an einsum), so that its decode waits on the HBMs as Ironwood's does; or the per-chiplet Ironwood."""
    if name == "ironwood":
        return hardware(IRONWOOD, **changes)
    return hardware(hbm_bandwidth=HBM_BITS_PER_CYCLE["ring"], **changes)


def _placements(cache, q=BY_HEADS):
    return {"q": q, "K": cache, "V": cache, "O": BY_HEADS}


def _decode(  # noqa: PLR0913
    tmp_path,
    hardware,
    placements,
    batch=1,
    context=None,
    make=make_decode_attention,
    mapping=None,
    parts=None,
    *,
    cut_points=None,
):
    """``make``'s decode step on ``hardware`` mapped by the generic generator from a mapping file that only places
    its edges, its attention split into ``parts`` if given and cut at ``cut_points`` if given, or as ``mapping`` maps
    it."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    workload = make(tmp_path / "decode.onnx", batch, context or CONTEXT[hardware])
    accelerator = _hardware(hardware) if isinstance(hardware, str) else hardware
    mapping = {"runtime_args": placements} | (mapping or {})
    return solve(accelerator, workload, tmp_path, mapping, attention_parts=parts, fusion_cut_points=cut_points)


@pytest.fixture(scope="module")
def by_heads(tmp_path_factory):
    """The decode step on the named hardware, its cache split by KV heads as the cores split them, fused in one group
    whose allocation gives the cache's routes, solved once."""
    solved = {}

    def solve_once(name, context=None):
        key = (name, context or CONTEXT[name])
        if key not in solved:
            solved[key] = _decode(
                tmp_path_factory.mktemp(name), name, _placements(BY_HEADS), context=key[1], cut_points=[]
            )
        return solved[key]

    return solve_once


def _fused(layers, tiling=()):
    """``layers`` fused in one group tiled by ``tiling``."""
    group = {"name": "decode", "layers": [layer["name"] for layer in layers], "intra_core_tiling": list(tiling)}
    return {"layers": list(layers), "fused_groups": [group]}


def _cache_bits(batch, context):
    return 2 * batch * KV_HEADS * context * DHEAD * 16


def _link_bits(estimate):
    """Bits per firing each tensor's transfers carry over each link in every group, by tensor name and the link's
    memories, as each group's allocation reports them."""
    links = {data["cl"] for *_, data in estimate.context.get("accelerator").network.edges(data=True)}
    memories = {link_name(link): tuple(sorted(memory.name for memory in link.memories)) for link in links}
    bits: dict[str, dict[tuple[str, ...], float]] = {}
    for group in estimate.context.get("group_allocations").values():
        for node in group["steady_state"]["tiled_graph"]["nodes"]:
            if node["kind"] != "transfer":
                continue
            per_link = bits.setdefault(node["name"].removeprefix("Transfer(").removesuffix(")"), {})
            for link, carried in zip(node["links"], node["link_bits"], strict=True):
                per_link[memories[link]] = per_link.get(memories[link], 0) + carried
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
    reads = {key: read for key, read in offchip_reads(estimate).items() if key[1] in ("K", "V")}
    total = sum(reads.values())
    return {memory: sum(read for (m, _), read in reads.items() if m == memory) / total for memory, _ in reads}


def _hbm_time(name, batch=1, context=None, hbms=4):
    """Cycles ``hbms`` HBMs take to stream the cache, each its share."""
    return _cache_bits(batch, context or CONTEXT[name]) / hbms / HBM_BITS_PER_CYCLE[name]


def test_the_per_chiplet_ironwood_has_an_hbm_and_a_measured_bandwidth_per_chiplet():
    accelerator = _hardware("ironwood")
    assert [memory.name for memory in accelerator.offchip_memories] == HBMS
    assert {memory.capacity_bits for memory in accelerator.offchip_memories} == {96 * 2**30 * 8}
    models = {key.memory: model for key, model in accelerator.port_bandwidth.items()}
    assert sorted(models) == HBMS
    assert {(model.ceiling, model.contiguous) for model in models.values()} == {(13400, 13400)}


@pytest.mark.parametrize("hardware", ["ring", "ironwood"])
def test_a_cache_split_by_kv_heads_streams_each_quarter_from_the_hbm_beside_its_cores(by_heads, hardware):
    estimate = by_heads(hardware)
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


POOLED_CROSSINGS = {
    "ring": {("sram_0", "sram_1"): 0.5, ("sram_0", "sram_3"): 0.25, ("sram_1", "sram_2"): 0.25},
    "ironwood": {("vmem_0", "vmem_1"): 0.75, ("vmem_1", "vmem_3"): 0.5, ("vmem_2", "vmem_3"): 0.25},
}
"""Of the cache pooled in the first slice's HBM, the share each link carries to the slices that read it."""


@pytest.mark.parametrize("hardware", ["ring", "ironwood"])
def test_a_cache_pooled_in_one_slice_s_hbm_waits_on_that_hbm_or_the_slowest_link_out(tmp_path, hardware):
    """Three quarters of the cache leave the first slice, half of it over the slowest link the ring or ICI has, so the
    step takes what that HBM or that link takes, whichever is slower."""
    pooled = _decode(tmp_path, hardware, _placements(POOLED), cut_points=[])
    assert _hbm_shares(pooled) == {"hbm_0": 1.0}
    assert _cache_on_chip(pooled) == POOLED_CROSSINGS[hardware]
    crossing = _cache_bits(1, CONTEXT[hardware]) / 2 / CROSSING_BITS_PER_CYCLE[hardware]
    slowest = max(_hbm_time(hardware, hbms=1), crossing)
    assert slowest <= pooled.cycles < 1.03 * slowest


SEQUENCE_CROSSINGS = {
    "ring": dict.fromkeys(
        [("sram_0", "sram_1"), ("sram_0", "sram_3"), ("sram_1", "sram_2"), ("sram_2", "sram_3")], 0.25
    ),
    "ironwood": {("vmem_0", "vmem_1"): 0.375, ("vmem_1", "vmem_3"): 0.5, ("vmem_2", "vmem_3"): 0.375},
}
"""Of the cache split on the sequence, the share each link carries to the slices whose cores attend to its heads."""


@pytest.mark.parametrize("hardware", ["ring", "ironwood"])
def test_a_cache_split_on_the_sequence_while_the_cores_split_heads_waits_on_the_slowest_link(
    tmp_path, by_heads, hardware
):
    """Every slice reads three quarters of its heads' cache from the other slices' HBMs, a quarter of the cache over
    each direction of the slowest link: on the ring, far faster than the HBMs, the step costs what the split by heads
    does; on Ironwood it waits on ICI."""
    wrong = _decode(tmp_path, hardware, _placements(BY_SEQUENCE), parts=1)
    assert _hbm_shares(wrong) == dict.fromkeys(HBMS, 0.25)
    assert _cache_on_chip(wrong) == SEQUENCE_CROSSINGS[hardware]
    crossing = _cache_bits(1, CONTEXT[hardware]) / 4 / CROSSING_BITS_PER_CYCLE[hardware]
    assert wrong.cycles == pytest.approx(max(by_heads(hardware).cycles, crossing), rel=0.02)


def test_fewer_kv_heads_than_chiplets_copied_in_each_hbm_are_read_from_the_hbm_beside_each_reader(tmp_path):
    """Two KV heads over four chiplets, the cache copied in each HBM: every chiplet streams the head its cores attend
    to from its own HBM, at most one head's K and V, and none crosses the die-to-die link or ICI."""
    placements = {"q": COPIED, "K": COPIED, "V": COPIED, "O": {"placement": {"memories": HBMS, "split": 3}}}
    estimate = _decode(tmp_path, "ironwood", placements, make=partial(make_decode_attention, kv_heads=2), cut_points=[])
    bits = _link_bits(estimate)
    assert _cache_on_chip(estimate) == {}
    assert _hbm_shares(estimate) == dict.fromkeys(HBMS, 0.25)
    per_hbm = {hbm: sum(bits[t].get((hbm, f"vmem_{hbm[-1]}"), 0) for t in "KV") for hbm in HBMS}
    assert max(per_hbm.values()) <= 2 * CONTEXT["ironwood"] * DHEAD * 16


ATTENTION = [
    {"name": name, "core_allocation": [[0, 1, 2, 3]], "inter_core_tiling": [[{"dim": "D1", "split": 4}]]}
    for name in ("scores", "softmax", "context")
]


def _chosen(estimate):
    """The placement each group reports for each workload input and output, by edge name."""
    groups = estimate.context.get("group_allocations").values()
    return {
        name: (tuple(placement["memories"]), placement["split"])
        for group in groups
        for name, placement in group["edge_placements"].items()
    }


@pytest.mark.slow
def test_the_allocation_places_a_cache_no_placement_names_as_well_as_the_best_declared_one(tmp_path):
    """Split by heads, by sequence, pooled in one HBM or copied in each, against no placement at all, the attention
    unsplit in each, as the allocation sees it: a placement split on the sequence would otherwise split the attention
    into flash-decoding parts, another workload. The allocation splits q, K and V by heads as the cores split them, the
    fastest, rather than copy them, which is exactly as fast but holds four times the bits."""
    declared = {
        name: _decode(tmp_path / name, "ring", _placements(cache), parts=1).cycles
        for name, cache in (("heads", BY_HEADS), ("sequence", BY_SEQUENCE), ("pooled", POOLED), ("copied", COPIED))
    }
    chosen = _decode(tmp_path / "chosen", "ring", {})
    assert declared["copied"] == declared["heads"]
    assert chosen.cycles <= min(declared.values())
    assert {name: _chosen(chosen)[name] for name in "qKV"} == dict.fromkeys("qKV", (tuple(HBMS), 1))
    assert _hbm_shares(chosen) == dict.fromkeys(HBMS, 0.25)


def test_an_unplaced_cache_is_tiled_as_a_placement_following_the_cores_would_be(tmp_path):
    """16 sequences of 8k: the generator splits the batch over the cores. The tiler keeps each core's sequences whole,
    as it would for a placement split on the batch, so the allocation splits q, K and V on the batch, each core
    reading its block beside it, and the step costs what that placement does rather than walking the batch."""
    step = {"batch": 16, "context": 8192}
    by_batch = _decode(tmp_path / "batch", "ring", dict.fromkeys(("q", "K", "V", "O"), BY_BATCH), **step)
    chosen = _decode(tmp_path / "chosen", "ring", {}, **step)
    assert {name: _chosen(chosen)[name] for name in "qKV"} == dict.fromkeys("qKV", (tuple(HBMS), 0))
    assert chosen.cycles <= by_batch.cycles


_CHOOSE = """
import json, sys, tempfile
from pathlib import Path
sys.path.insert(0, "tests")
from test_decode_kv import _chosen, _decode
estimate = _decode(Path(tempfile.mkdtemp()), "ring", {})
print(json.dumps([sorted(_chosen(estimate).items()), estimate.cycles]))
"""


def test_the_placements_the_allocation_chooses_do_not_depend_on_the_hash_seed():
    runs = [
        subprocess.Popen(
            [sys.executable, "-c", _CHOOSE],
            env={**os.environ, "PYTHONHASHSEED": str(seed)},
            stdout=subprocess.PIPE,
            text=True,
        )
        for seed in range(2)
    ]
    chosen = [json.loads(run.communicate()[0].splitlines()[-1]) for run in runs]
    assert all(run.returncode == 0 for run in runs) and chosen[0] == chosen[1]


def test_a_declared_placement_is_the_one_its_input_is_read_from(tmp_path):
    """K pooled in the third slice's HBM, which the allocation would not choose, while it places q and V."""
    estimate = _decode(tmp_path, "ring", {"K": {"placement": {"memories": ["hbm_2"]}}})
    assert _chosen(estimate)["K"] == (("hbm_2",), None)
    allocation = estimate.context.get("allocation")
    tr, _ = route_of(allocation, "K")
    assert {tuple(memory.name for memory in route.sources) for route in allocation.space.path_choices[tr]} == {
        ("hbm_2",)
    }
    assert {memory for memory, tensor in offchip_reads(estimate) if tensor == "K"} == {"hbm_2"}


BY_QUERY_HEADS = [
    {"name": name, "core_allocation": [cores], "inter_core_tiling": [[{"dim": "D2", "split": 4}]]}
    for name, cores in (("scores", [0, 4, 8, 12]), ("softmax", [2, 6, 10, 14]), ("context", [0, 4, 8, 12]))
]
"""An MXU and the VPU of each chiplet split the query heads of each KV head, so every chiplet reads all of K and V."""


def test_a_cache_every_chiplet_reads_whole_is_copied_beside_each_unless_a_copy_overflows_its_hbm(tmp_path):
    """A copy in each HBM keeps the cache off the die-to-die links and ICI. In HBMs of 12 MiB a copy of either 16 MiB
    half of the cache no longer fits, nor does pooling it: the allocation splits both, and what each HBM holds fits."""
    capacity = 12 * 2**23
    walk = {"context": CONTEXT["ironwood"], "mapping": _fused(BY_QUERY_HEADS)}
    ample = _decode(tmp_path / "ample", "ironwood", {}, **walk)
    tight = _decode(tmp_path / "tight", _hardware("ironwood", capacity_bits=capacity), {}, **walk)
    assert _chosen(ample)["K"] == _chosen(ample)["V"] == (tuple(HBMS), None)
    assert [_chosen(tight)[name][0] for name in "KV"] == [tuple(HBMS)] * 2
    assert None not in [_chosen(tight)[name][1] for name in "KV"]
    (group,) = tight.context.get("group_allocations").values()
    assert max(sum(held.values()) for held in group["offchip_holdings"].values()) <= capacity
    assert tight.cycles > ample.cycles


def test_a_cache_split_on_the_batch_the_iterations_walk_waits_on_one_hbm_at_a_time(tmp_path):
    """Heads split over the cores, one batch row an iteration: each iteration reads its row from the one HBM holding
    it, so each takes four times as long, though over the step every HBM reads its quarter."""
    walk = {"mapping": _fused(ATTENTION, [{"dim": "scores.D0", "tile": 1}]), "batch": 4}
    split = _decode(tmp_path / "split", "ring", _placements(BY_HEADS), **walk)
    wrong = _decode(tmp_path / "wrong", "ring", _placements(BY_BATCH), **walk)
    assert _hbm_shares(split) == _hbm_shares(wrong) == dict.fromkeys(HBMS, 0.25)
    assert 3.5 * split.cycles < wrong.cycles <= 4 * split.cycles


@pytest.mark.parametrize("hardware", ["ring", pytest.param("ironwood", marks=pytest.mark.slow)])
def test_flash_decoding_with_its_parts_split_over_the_hbms_moves_only_the_combine_between_slices(
    tmp_path, by_heads, hardware
):
    """Each slice attends to the part of the sequence in its own HBM, every operator on a part following the split of
    what it reads; only the combine's maximum, sum and context cross between slices, and the step costs about what
    the split by heads does. Every transfer's steady state is described, those its readers split finer too."""
    flash = _decode(tmp_path, hardware, _placements(BY_SEQUENCE, q=COPIED), make=make_flash_decoding)
    bits = _link_bits(flash)
    crossing = {tensor: sum(_on_chip(per_link).values()) for tensor, per_link in bits.items() if _on_chip(per_link)}
    assert set(crossing) == {"M", "L", "N"}
    from_hbms = sum(bits[t][link] for t in "KV" for link in bits[t] if set(link) & set(HBMS))
    assert sum(crossing.values()) < from_hbms / 32
    assert _hbm_shares(flash) == dict.fromkeys(HBMS, 0.25)
    assert _hbm_time(hardware) <= flash.cycles < 1.1 * by_heads(hardware).cycles
    (group,) = flash.context.get("group_allocations").values()
    transfers = [node for node in group["steady_state"]["tiled_graph"]["nodes"] if node["kind"] == "transfer"]
    assert all("moved" in node for node in transfers)


@pytest.fixture
def small_slice():
    """Slices of 512 KiB, on which 32k positions of a head's scores do not fit whole."""
    return {"hardware": _hardware("ring", sram_bits=512 * 1024 * 8), "context": 32768}


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


SCALED_STATISTICS = (
    [
        helper.make_node("ReduceMax", ["S", "last"], ["m"], keepdims=1, name="max"),
        helper.make_node("Mul", ["m", "c"], ["mc"], name="scale_max"),
        helper.make_node("Sub", ["S", "mc"], ["Z"], name="shift"),
        helper.make_node("Exp", ["Z"], ["E"], name="exp"),
        helper.make_node("ReduceSum", ["E", "last"], ["l"], keepdims=1, name="sum"),
        helper.make_node("Mul", ["l", "c"], ["lc"], name="scale_sum"),
        helper.make_node("Div", ["E", "lc"], ["P"], name="normalize"),
    ],
    [axes_tensor("last", -1), helper.make_tensor("c", DTYPE, [1], [1.0])],
)
"""A softmax that scales its maximum and its sum before using them."""


def test_a_softmax_reading_its_statistics_through_other_ops_is_not_tiled_along_them(tmp_path, small_slice):
    placements = {**_placements(BY_HEADS), "c": COPIED}
    make = partial(make_decode_attention, softmax=SCALED_STATISTICS)
    estimate = _decode(tmp_path, placements=placements, make=make, **small_slice)
    assert not _walks_an_unfinished_reduction(estimate)


def _cache_view(estimate):
    return estimate.context.get("sub_workloads")[0].get_node_by_name("K").outputs[0].shape


def test_attention_on_a_cache_split_on_the_sequence_is_split_into_a_part_per_hbm(tmp_path, small_slice):
    """Plain attention whose scores do not fit a slice whole, its cache split on the sequence: Stream views the cache as
    a part per HBM, keeps each part's softmax whole, and streams the cache near the HBMs' rate."""
    estimate = _decode(tmp_path, placements=_placements(BY_SEQUENCE, q=COPIED), **small_slice)
    assert _cache_view(estimate) == (1, KV_HEADS, 4, 8192, DHEAD)
    assert "context_combine" in {node.name for w in estimate.context.get("sub_workloads") for node in w.nodes}
    assert not _walks_an_unfinished_reduction(estimate)
    assert _hbm_shares(estimate) == dict.fromkeys(HBMS, 0.25)
    assert _hbm_time("ring", context=32768) <= estimate.cycles < 1.05 * _hbm_time("ring", context=32768)


@pytest.mark.slow
def test_attention_on_a_cache_split_by_heads_is_split_into_parts_that_fit(tmp_path, small_slice):
    estimate = _decode(tmp_path, placements=_placements(BY_HEADS), **small_slice)
    parts = _cache_view(estimate)[2]
    assert parts > 1 and _cache_view(estimate) == (1, KV_HEADS, parts, 32768 // parts, DHEAD)
    assert not _walks_an_unfinished_reduction(estimate)
    assert estimate.cycles < 1.01 * _hbm_time("ring", context=32768)


def test_a_long_cache_pooled_in_one_hbm_is_split_into_parts_its_slices_stream(tmp_path):
    """Two sequences of 128k in slices of 128 KiB: each core's half of the cache is thousands of slices' worth, which
    the parts of the sequence and the tiles cut within them stream through, all from the one HBM."""
    hardware = _hardware("ring", sram_bits=128 * 1024 * 8)
    estimate = _decode(tmp_path, hardware, _placements(POOLED), batch=2, context=131072)
    assert _cache_view(estimate)[2] > 1
    assert not _walks_an_unfinished_reduction(estimate)
    assert _hbm_shares(estimate) == {"hbm_0": 1.0}
    assert estimate.cycles >= _hbm_time("ring", batch=2, context=131072, hbms=1)


def _maximum_held(estimate):
    """Per off-chip memory, the copies of the maximum over the parts it holds, each tensor counted once."""
    held = held_once(estimate)
    return {memory.name: next(n for t, n in shares.items() if t.name == "M") for memory, shares in held.items()}


def _copies(groups):
    return [
        name for group in groups for name, placement in group["edge_placements"].items() if placement["split"] is None
    ]


def test_flash_decoding_written_by_hand_and_cut_after_its_maximum_reads_it_beside_each_slice(tmp_path):
    """In slices of 128 KiB the step is cut after the maximum over the parts. Its maxima are no softmax's, yet the
    exponent reads each part's maximum along the part, so it is not tiled; the maximum, which the cores split, lands
    whole beside every slice, and the next group reads it there."""
    hardware = _hardware("ring", sram_bits=128 * 1024 * 8)
    placements = _placements(BY_SEQUENCE, q=COPIED)
    estimate = _decode(tmp_path, hardware, placements, context=32768, make=make_flash_decoding)
    assert not _walks_an_unfinished_reduction(estimate)
    assert _hbm_shares(estimate) == dict.fromkeys(HBMS, 0.25)
    groups = list(estimate.context.get("group_allocations").values())
    assert _copies(groups) == ["q", "max_cut_in"]
    assert groups[1]["edge_placements"]["max_cut_in"]["memories"] == HBMS
    assert _maximum_held(estimate) == dict.fromkeys(HBMS, 1)


def test_attention_that_fits_as_the_generator_cuts_it_is_not_split(tmp_path):
    """16 sequences of 32k by heads over the chiplets' HBMs: the scores and the context fit in groups of their own, the
    softmax's output spilled between them, which costs less than splitting the sequence."""
    split = _decode(tmp_path / "default", "ironwood", _placements(BY_HEADS), batch=16, context=32768)
    whole = _decode(tmp_path / "whole", "ironwood", _placements(BY_HEADS), batch=16, context=32768, parts=1)
    names = {node.name for workload in split.context.get("sub_workloads") for node in workload.nodes}
    assert "scores" in names and "context_combine" not in names
    assert split.cycles == whole.cycles


def _generator(tmp_path, hardware, placements, batch, context):
    """The generic generator of ``make_decode_attention``'s step on ``hardware``, its softmax expanded and its edges
    placed by ``placements``, as the generation stages hand it over for one part."""
    workload = make_decode_attention(tmp_path / "decode.onnx", batch, context)
    ctx = StageContext.from_kwargs(accelerator=_hardware(hardware), workload_path=workload, output_path=str(tmp_path))
    ctx = MainStage([AcceleratorParserStage, ONNXModelParserStage, LeafStage], ctx).run()[0]
    parsed = expand_normalizations(ctx.get("workload"))
    runtime_args = viewed_placements(parsed, placements)
    return GenericMappingGenerator(ctx.get("accelerator"), parsed, str(tmp_path), runtime_args=runtime_args), parsed


def test_a_cache_the_scores_read_moves_once_over_the_tile_loops_only_the_context_steps(tmp_path):
    """One sequence of 8k on the ring by heads, fused and tiled along the scores' head dim, then the context's: the
    scores do not step the context's dim, so they read K once, its 8 x 8192 x 128 bf16, a quarter into each slice's
    SRAM, though no SRAM holds a quarter next to the tiles."""
    generator, workload = _generator(tmp_path, "ring", _placements(BY_HEADS), 1, 8192)
    (fused,) = workload.split_fusion_groups(cut_points=generator._cut_points(None))
    cns = tuple(fused.get_computation_nodes())
    tiler, cores, unroll, protected, node_unroll = generator._tiler(fused, cns)
    unroll = {fused.leading_dim(d)[0]: f for d, f in unroll.items()}
    per_core = tiler._per_core(cns, unroll, {fused.leading_dim(d)[0] for d in protected})
    tiling = [{"dim": "scores.D4", "tile": 8}, {"dim": "context.D3", "tile": 16}]
    resident = per_core | tiler._seed_resident(cns, tiling, per_core)
    order = list(tiler._seed_resident(cns, tiling, per_core))
    args = (cns, cores, unroll, resident, node_unroll)
    capacities, tiles, _ = tiler._transfers(*args, tiler._capacities(cns, cores))
    room = {memory: capacity - tiles[memory] for memory, capacity in capacities.items()}
    (cache,) = (t for t in cns[0].inputs if t.name == "K")
    assert all(free < 8 * 8192 * 128 * 16 / 4 for free in room.values())
    moved = tiler._moved(*args, order, room, spills=True, tensors={cache})
    bits = 8 * 8192 * 128 * 16
    assert moved == {f"sram_{s}": bits / 4 for s in range(4)} | {None: bits}


def test_a_part_count_the_cache_placement_does_not_split_into_is_refused(tmp_path, small_slice):
    with pytest.raises(
        ValueError, match="K is split along its axis 2 over 4 memories, which 2 parts of context's sequence"
    ):
        _decode(tmp_path, placements=_placements(BY_SEQUENCE, q=COPIED), parts=2, **small_slice)


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


@pytest.mark.slow
def test_a_long_batched_cache_split_over_the_hbms_fits_and_its_scores_stay_beside_their_cores(tmp_path):
    """The scores spill to the HBM beside the cores that made them, and the next group reads them back from there."""
    estimate = _decode(tmp_path, "ironwood", _placements(BY_HEADS), **LONG)
    assert _hbm_shares(estimate) == dict.fromkeys(HBMS, 0.25)
    assert estimate.cycles >= _hbm_time("ironwood", **LONG)
    (cut,) = (name for name in estimate.context.get("mapping").edge_placements if name.endswith("_cut_in"))
    placement = estimate.context.get("mapping").edge_placements[cut]
    assert [memory.name for memory in placement.memories] == HBMS and placement.split == 1
    assert not any(_on_chip(per_link) for per_link in _link_bits(estimate).values())


@pytest.mark.parametrize(
    ("placements", "context", "parts"),
    [(_placements(BY_HEADS), 2**21, None), (_placements(BY_SEQUENCE, q=COPIED), 65536, 1)],
    ids=["heads", "sequence"],
)
def test_a_fusion_cut_without_a_placement_is_read_back_where_it_landed(tmp_path, placements, context, parts):
    """Split by heads, 16 sequences of 2M spill their scores, which no longer fit beside the cache, to the HBMs. Split
    on the sequence, 16 sequences of 64k spill them at a cut made by cores splitting the sequence the scores reduce
    over, which indexes no axis of what they write. Either cut is read back where each core landed it."""
    estimate = _decode(tmp_path, "ironwood", placements, batch=16, context=context, parts=parts)
    assert len(estimate.context.get("group_memory_accesses")) > 1
    assert _hbm_shares(estimate) == dict.fromkeys(HBMS, 0.25)


def test_a_fusion_cut_split_on_the_sequence_over_eight_slices_is_read_back_from_the_nearest_hbm(tmp_path):
    hbms = [f"hbm_{s}" for s in range(8)]
    by_sequence = split_on(2, hbms)
    placements = {"q": {"placement": {"memories": hbms}}, "K": by_sequence, "V": by_sequence, "O": split_on(1, hbms)}
    estimate = _decode(tmp_path, ring_of(8), placements, batch=16, context=8192, parts=1)
    assert len(estimate.context.get("group_memory_accesses")) > 1
    assert _hbm_shares(estimate) == dict.fromkeys(hbms, 1 / 8)


@pytest.mark.slow
def test_a_reduction_the_cores_split_crossing_to_a_later_group_is_read_back_from_each_hbm_it_landed_in(tmp_path):
    """Flash-decoding over 2M positions, a part in each chiplet's HBM, cut after the maximum over the parts: that
    maximum lands whole beside every chiplet, and the next group reads it there."""
    placements = _placements(BY_SEQUENCE, q=COPIED)
    estimate = _decode(tmp_path, "ironwood", placements, batch=16, context=2**21, make=make_flash_decoding)
    assert _hbm_shares(estimate) == dict.fromkeys(HBMS, 0.25)
    assert estimate.cycles < 1.05 * _hbm_time("ironwood", batch=16, context=2**21)
    groups = list(estimate.context.get("group_allocations").values())
    assert _copies(groups) == ["q", "max_cut_in"]
    assert groups[1]["edge_placements"]["max_cut_in"]["memories"] == HBMS
    assert _maximum_held(estimate) == dict.fromkeys(HBMS, 1)


@pytest.mark.parametrize("context", [CONTEXT["ironwood"], pytest.param(32768, marks=pytest.mark.slow)])
def test_a_cache_split_by_heads_over_an_hbm_per_chiplet_costs_what_the_pooled_hbm_does(tmp_path, by_heads, context):
    """Each chiplet reaches the pooled HBM over its own link at its four stacks' rate, so the split cache, read only
    by the chiplet beside it, streams as the pooled one does."""
    split = by_heads("ironwood", context)
    pooled = solve(TPU_V7, make_decode_attention(tmp_path / "pooled.onnx", 1, context), tmp_path, fusion_cut_points=[])
    assert split.cycles == pooled.cycles
