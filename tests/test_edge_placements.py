"""A workload input or output placed over the off-chip memories of a 4-slice ring, each slice with an HBM stack of
its own: a split the cores' split follows stays local, a pooled or wrongly split one costs the ring or one HBM."""

from collections import defaultdict
from fractions import Fraction
from functools import cache
from math import prod

import onnx
import pytest
from conv_windows import onnx_graph
from onnx import helper
from placements import (
    COPIED,
    HBMS,
    POOLED,
    RING,
    hardware,
    held_once,
    offchip_reads,
    route_of,
    solve,
    split_on,
)
from zigzag.utils import open_yaml

from stream.hardware.ports import READ
from stream.inputs.testing.workload.make_decode_attention import DHEAD, KV_HEADS, make_decode_attention
from stream.ir.allocation import AllocationIR
from stream.ir.infeasibility import InfeasibleAllocationError
from stream.mapping.mapping import Mapping, held_bits, placement_options
from stream.opt.allocation.constraint_optimization.allocation_model import AllocationModel
from stream.opt.allocation.constraint_optimization.families import load_families
from stream.opt.allocation.constraint_optimization.families.dma import _port_channels
from stream.parser.mapping_validator import MappingValidator

SPLIT_N = split_on(1)
K_TILE = ("D2", 128)


@cache
def _matmul():
    matmul = helper.make_node("MatMul", ["a", "w"], ["out"], name="MatMul")
    return onnx_graph([matmul], {"a": [1, 1024], "w": [1024, 512]}, ["out"])


def _estimate(tmp_path, w, out=SPLIT_N, tiles=(), hardware=RING):
    """A decode step's 1 x 1024 by 1024 x 512 matmul, its n split over the four cores and tiled over time by
    ``tiles``, ``w`` and ``out`` placed as given."""
    layer = {"name": "MatMul", "core_allocation": [[0, 1, 2, 3]], "inter_core_tiling": [[{"dim": "D1", "split": 4}]]}
    tiling = [{"dim": f"MatMul.{dim}", "tile": tile} for dim, tile in tiles]
    group = {"name": "g", "layers": ["MatMul"], "intra_core_tiling": tiling}
    runtime_args = {name: args for name, args in (("a", COPIED), ("w", w), ("out", out)) if args}
    return solve(
        hardware, _matmul(), tmp_path, {"layers": [layer], "fused_groups": [group], "runtime_args": runtime_args}
    )


def _solve(tmp_path, w, **kwargs):
    return _estimate(tmp_path, w, **kwargs).context.get("allocation")


def _ring_share(route):
    return sum(share for link, share in route.link_shares if not any(memory.offchip for memory in link.memories))


def _hbm_shares(allocation, tr, route):
    space = allocation.space
    return {
        memory.name: sum(space.side_ports(tr, route, memory, READ).values())
        for memory in route.sources
        if memory.offchip
    }


@pytest.mark.parametrize(
    ("w", "out", "reported"),
    [
        (SPLIT_N, SPLIT_N, {"w": (HBMS, 1), "out": (HBMS, 1)}),
        (COPIED, SPLIT_N, {"w": (HBMS, None), "out": (HBMS, 1)}),
        ({}, {}, {"w": (HBMS, 1)}),
    ],
    ids=["split", "copied", "unplaced"],
)
def test_a_weight_and_output_split_as_the_cores_split_them_copied_or_unplaced_stay_in_each_slice(
    tmp_path, w, out, reported
):
    """Each core reads its block of w and lands its block of out in its own slice's HBM. Without a placement the
    allocation splits w along n, as the cores do, and the IR reports it as the mapping file would declare it."""
    allocation = _solve(tmp_path, w, out=out)
    for edge, ends in (("w", "sources"), ("out", "targets")):
        _, route = route_of(allocation, edge)
        assert [memory.name for memory in getattr(route, ends)] == HBMS
        assert route.pairs == ((0, 0), (1, 1), (2, 2), (3, 3))
        assert _ring_share(route) == 0
    tr, route = route_of(allocation, "w")
    assert _hbm_shares(allocation, tr, route) == dict.fromkeys(HBMS, 0.25)
    placements = AllocationIR.from_internal(allocation).edge_placements
    assert {name: (p.memories, p.split) for name, p in placements.items() if name != "a"} == reported


def test_a_pooled_weight_crosses_the_ring_costs_more_and_its_hbm_is_charged_what_the_split_s_four_are(tmp_path):
    """Each HBM reads a quarter of a split weight, and all of the input every core reads whole from its copy."""
    split = _estimate(tmp_path / "split", SPLIT_N)
    pooled = _estimate(tmp_path / "pooled", POOLED)
    allocation = pooled.context.get("allocation")
    tr, route = route_of(allocation, "w")
    assert _ring_share(route) >= 0.75
    assert _hbm_shares(allocation, tr, route) == {"hbm_0": 1.0}
    assert allocation.estimated_cycles > 2 * split.context.get("allocation").estimated_cycles
    split_reads, pooled_reads = offchip_reads(split), offchip_reads(pooled)
    assert [split_reads[(hbm, "w")] for hbm in HBMS] == [pooled_reads[("hbm_0", "w")] // 4] * 4
    assert not any(tensor == "w" and memory != "hbm_0" for memory, tensor in pooled_reads)
    port_bits, a_bits = 8, 1 * 1024 * 16
    assert [split_reads[(hbm, "a")] for hbm in HBMS] == [a_bits // port_bits] * 4


def test_a_weight_split_along_the_axis_walked_over_time_waits_on_one_hbm_though_each_reads_a_quarter(tmp_path):
    """Split along k, which each iteration walks a tile of, every core reads its tile from the same HBM stack; over the
    walk each HBM reads a quarter, as split along n."""
    estimates = [
        _estimate(tmp_path / name, w, tiles=[K_TILE]) for name, w in (("local", SPLIT_N), ("wrong", split_on(0)))
    ]
    local, wrong = (estimate.context.get("allocation") for estimate in estimates)
    tr, route = route_of(wrong, "w")
    assert _hbm_shares(wrong, tr, route) == {"hbm_0": 1.0, "hbm_1": 0.0, "hbm_2": 0.0, "hbm_3": 0.0}
    local_tr, local_route = route_of(local, "w")
    busy = wrong.space.moved_bits(tr, route) * route.memory_share(route.sources[0], leaving=True)
    local_bits = local.space.moved_bits(local_tr, local_route)
    assert busy == 4 * local_bits * local_route.memory_share(local_route.sources[0], leaving=True)
    assert wrong.estimated_cycles > 3 * local.estimated_cycles
    local_reads, reads = (offchip_reads(estimate) for estimate in estimates)
    assert [reads[(hbm, "w")] for hbm in HBMS] == [local_reads[(hbm, "w")] for hbm in HBMS]
    assert len({reads[(hbm, "w")] for hbm in HBMS}) == 1


def test_each_hbm_with_a_measured_bandwidth_is_held_for_its_own_block(tmp_path):
    measured = {hbm: {"ceiling": 8, "contiguous": 8, "burst": 32} for hbm in HBMS}
    allocation = _solve(tmp_path, SPLIT_N, hardware=hardware(bandwidth=measured))
    tr, route = route_of(allocation, "w")
    space = allocation.space
    quarter = space.moved_bits(tr, route) / 4
    assert [space.shared_cycles(hbm, tr, route, 8) for hbm in route.sources] == [quarter / 8] * 4
    assert [space.side_bandwidth(tr, route, hbm, READ).contiguous for hbm in route.sources] == [8 / 0.25] * 4
    report = AllocationIR.from_internal(allocation).performance.memory_ports
    rows = [row for row in report if row.kind == "shared_bandwidth"]
    assert sorted(row.resource for row in rows) == HBMS
    assert all(row.bits_per_iteration / row.bw_bits_per_cycle <= row.allowed_cycle for row in rows)


def test_a_weight_copied_in_each_hbm_over_its_capacity_is_refused_by_what_each_holds_but_fits_split(tmp_path):
    """The 1 MiB weight whole in each 512 KiB HBM stack, the output a quarter in each, landing beside its core; a
    quarter of the weight in each fits."""
    with pytest.raises(InfeasibleAllocationError, match=r"^hbm_0 holds a \+ w \+ out/4 = 1 MiB > 512 KiB$") as refused:
        _solve(tmp_path, COPIED, out={}, hardware=hardware(capacity_bits=4 * 2**20))
    report = refused.value.report
    assert report.nature == "capacity"
    (resource,) = report.resources
    unmet = resource.unmet
    assert (resource.resource.id, unmet.family, unmet.unit) == ("hbm_0", "offchip_capacity", "bytes")
    assert {term.label: term.value for term in unmet.terms} == {"w": 2**20, "a": 2**11, "out/4": 2**8}
    assert (unmet.demand_value, unmet.bound_value) == (sum(term.value for term in unmet.terms), 2**19)
    assert _solve(tmp_path / "split", SPLIT_N, out={}, hardware=hardware(capacity_bits=4 * 2**20))


def test_an_axis_the_cores_split_inside_each_time_tile_is_read_from_two_hbms_per_iteration(tmp_path):
    """n walked in two halves, each split over the four cores: core c reads n = 256 t + 64 c + i, so a contiguous block
    per HBM serves two cores per iteration over the ring; over the walk each HBM reads a quarter."""
    estimate = _estimate(tmp_path, SPLIT_N, tiles=[("D1", 64)])
    allocation = estimate.context.get("allocation")
    tr, route = route_of(allocation, "w")
    assert _hbm_shares(allocation, tr, route) == {"hbm_0": 0.5, "hbm_1": 0.5, "hbm_2": 0.0, "hbm_3": 0.0}
    assert _ring_share(route) > 0
    reads = offchip_reads(estimate)
    assert len({reads[(hbm, "w")] for hbm in HBMS}) == 1


N_TILE = ("D1", 16)
"""n walked 16 at a time: each iteration reads from one HBM, a different one as the walk goes on."""


def test_a_walk_holds_dma_channels_on_every_hbm_it_reads(tmp_path):
    allocation = _solve(tmp_path, SPLIT_N, tiles=[N_TILE], hardware=hardware(dma=True))
    tr, route = route_of(allocation, "w")
    assert sum(route.memory_share(hbm, leaving=True) > 0 for hbm in route.sources) == 1
    model = _model(allocation)
    model.space.transfer_nodes, model.space.path_choices = [tr], {tr: (route,)}
    _, out = _port_channels(model.context)
    assert sorted(memory.name for memory, _ in out) == HBMS


def test_the_busiest_iteration_is_the_one_at_the_slowest_hbm(tmp_path):
    slow = hardware(ports={"hbm_2": {"bandwidth": 4}})
    allocation = _solve(tmp_path, split_on(0), tiles=[K_TILE], hardware=slow)
    tr, route = route_of(allocation, "w")
    assert {hbm.name: route.memory_share(hbm, leaving=True) for hbm in route.sources}["hbm_2"] == 1.0


def test_a_walk_is_priced_at_each_hbm_s_energy_for_its_average_share(tmp_path):
    """Each iteration of the walk reads one HBM, but over the walk each reads a quarter, each at its own energy."""
    costly = {"read_energy": 1000.0, "write_energy": 1000.0}
    allocation = _solve(tmp_path, SPLIT_N, tiles=[N_TILE], hardware=hardware(ports=dict.fromkeys(HBMS[1:], costly)))
    energy = AllocationIR.from_internal(allocation).performance.energy
    reads = {row.resource: row.pj_per_iteration for row in energy.components if row.resource.startswith("hbm")}
    assert reads["hbm_1.rw_port_1"] == pytest.approx(reads["hbm_0.rw_port_1"] * 1000.0 / 15.625, rel=0.01)
    assert reads["hbm_1.rw_port_1"] == pytest.approx(reads["hbm_3.rw_port_1"])


def test_the_allocation_chooses_an_input_s_placement_among_one_hbm_a_block_in_each_or_a_copy_in_each(tmp_path):
    allocation = _solve(tmp_path, {})
    tr, _ = route_of(allocation, "w")
    w = allocation.problem.source_workload.get_node_by_name("w")
    options = placement_options(allocation.problem.accelerator, Mapping(), w)
    assert [([memory.name for memory in option.memories], option.split) for option in options] == [
        *(([hbm], None) for hbm in HBMS),
        (HBMS, 0),
        (HBMS, 1),
        (HBMS, None),
    ]
    assert {choice.placement for choice in allocation.space.path_choices[tr]} == set(options)


@pytest.mark.parametrize(
    ("out", "refusal"),
    [(split_on(0), "axis 0 of extent 1 does not split evenly"), (COPIED, "output out cannot be a copy")],
)
def test_an_output_split_along_an_axis_it_cannot_split_or_copied_is_refused(tmp_path, out, refusal):
    with pytest.raises(ValueError, match=refusal):
        _solve(tmp_path, SPLIT_N, out=out)


@pytest.mark.parametrize(
    "args",
    [
        {"placement": {"memories": "hbm_0"}},
        {"placement": {"split": 1}},
        {"placement": {"memories": [], "split": 0}},
        {"placement": {"memories": HBMS, "split": -1}},
        {"placment": {"memories": HBMS, "split": 0}},
    ],
)
def test_a_malformed_placement_is_refused(args):
    validator = MappingValidator({"layers": [], "runtime_args": {"w": args}})
    assert not validator.validate()


def test_other_runtime_args_are_left_to_their_readers():
    validator = MappingValidator({"layers": [], "runtime_args": {"w": {"layout": "(d0, d1) -> (d1, d0)", "depth": 2}}})
    assert validator.validate()


def test_a_placement_names_memories_of_the_accelerator(tmp_path):
    with pytest.raises(ValueError, match=r"w is placed in \['dram'\], but the memories are"):
        _solve(tmp_path, {"placement": {"memories": ["dram"]}})


@pytest.mark.parametrize(("axis", "dim"), [(1, "D1"), (0, "D2")])
def test_the_generic_generator_splits_the_cores_as_a_placement_splits_the_weight(tmp_path, axis, dim):
    """Split along n or k, each core reads or reduces the block of its own slice's HBM, the iterations walking the
    other axis."""
    estimate = solve(RING, _matmul(), tmp_path, {"runtime_args": {"a": COPIED, "w": split_on(axis), "out": SPLIT_N}})
    (layer,) = open_yaml(tmp_path / "group_0" / "mapping.yaml")["layers"]
    assert layer["inter_core_tiling"] == [[{"dim": dim, "split": 4}]]
    allocation = estimate.context.get("allocation")
    tr, route = route_of(allocation, "w")
    assert _ring_share(route) == 0
    assert _hbm_shares(allocation, tr, route) == dict.fromkeys(HBMS, 0.25)


def _mlp(tmp_path, runtime_args, hardware=RING, up=({"dim": "D1", "split": 4},)):
    """Two matmuls in two fused groups, the hidden tensor between them crossing the cut, the first split over the cores
    by ``up``."""
    nodes = [
        helper.make_node("MatMul", ["a", "w1"], ["hidden"], name="Up"),
        helper.make_node("MatMul", ["hidden", "w2"], ["out"], name="Down"),
    ]
    workload = onnx_graph(nodes, {"a": [1, 256], "w1": [256, 512], "w2": [512, 512]}, ["out"])
    layers = [
        {"name": name, "core_allocation": [[0, 1, 2, 3]], "inter_core_tiling": [list(split)]}
        for name, split in (("Up", up), ("Down", [{"dim": "D1", "split": 4}]))
    ]
    groups = [{"name": name, "layers": [name], "intra_core_tiling": []} for name in ("Up", "Down")]
    return solve(hardware, workload, tmp_path, {"layers": layers, "fused_groups": groups, "runtime_args": runtime_args})


PLACED = {name: SPLIT_N for name in ("w1", "w2", "out")} | {"a": COPIED}


def _model(allocation):
    """The allocation model of ``allocation``'s problem, built anew."""
    specs = [{name: options} for name, options in allocation.families]
    return AllocationModel(allocation.problem, families=load_families(specs), backend=allocation.backend)


def test_of_placements_as_fast_the_allocation_takes_the_one_holding_the_fewest_bits_off_chip(tmp_path):
    assert list(_model(_solve(tmp_path, {})).objective)[:2] == ["latency", "offchip_footprint"]


def test_the_placement_of_an_input_the_allocation_chooses_is_the_one_its_route_reads(tmp_path):
    """w held over all four HBMs cannot be read by a route that takes all of it from hbm_0."""
    allocation = _solve(tmp_path, {})
    tr, _ = route_of(allocation, "w")
    tensor, pooled = tr.inputs[0], next(c for c in allocation.space.path_choices[tr] if len(c.sources) == 1)
    model = _model(allocation)
    model.model.add_constr(model.vars.y[(tr, pooled)] == 1, name="pooled_route")
    over_all = model.vars.x[(tensor, allocation.problem.accelerator.offchip_memories)]
    model.model.add_constr(over_all == 1, name="over_all")
    with pytest.raises(InfeasibleAllocationError):
        model.solve(time_limit_s=60)
    solution = allocation.solution
    assert solution.tensor_placements[tensor] == solution.transfer_routes[tr].placement.memories


def _residual_estimate(tmp_path, runtime_args, hardware=RING):
    """x through two matmuls in two fused groups and added back after the second, so both groups read x."""
    nodes = [
        helper.make_node("MatMul", ["x", "w1"], ["hidden"], name="Up"),
        helper.make_node("MatMul", ["hidden", "w2"], ["y"], name="Down"),
        helper.make_node("Add", ["y", "x"], ["out"], name="Residual"),
    ]
    workload = onnx_graph(nodes, {"x": [1, 512], "w1": [512, 512], "w2": [512, 512]}, ["out"])
    layers = [
        {"name": name, "core_allocation": [[0, 1, 2, 3]], "inter_core_tiling": [[{"dim": "D1", "split": 4}]]}
        for name in ("Up", "Down", "Residual")
    ]
    groups = [{"name": "Up", "layers": ["Up"]}, {"name": "Down", "layers": ["Down", "Residual"]}]
    groups = [group | {"intra_core_tiling": []} for group in groups]
    return solve(hardware, workload, tmp_path, {"layers": layers, "fused_groups": groups, "runtime_args": runtime_args})


def test_an_input_two_groups_read_is_placed_by_the_first_and_read_there_by_the_second(tmp_path):
    estimate = _residual_estimate(tmp_path, {})
    first, second = (AllocationIR(**ir) for ir in estimate.context.get("group_allocations").values())
    assert first.edge_tensors["x"].tensor == second.edge_tensors["x"].tensor
    assert first.edge_placements["x"] == second.edge_placements["x"]
    allocation = estimate.context.get("allocation")
    tr, _ = route_of(allocation, "x")
    assert {choice.placement for choice in allocation.space.path_choices[tr]} == {None}
    first_x, second_x = (
        {memory: held["x"] for memory, held in ir.offchip_holdings.items() if "x" in held} for ir in (first, second)
    )
    assert first_x == second_x and sum(first_x.values()) == 512 * 16


def test_an_input_a_later_group_places_keeps_room_beside_what_an_earlier_group_chose(tmp_path):
    """The first group copies x where room allows. In HBMs of 2,105,000 bits, which fit x split beside the rest but not
    copied, it finds a block of x held in each for the second group, so it splits x and both groups fit."""
    capacity = 2_105_000
    ample, tight = (
        _residual_estimate(tmp_path / name, {"w1": SPLIT_N}, hardware(capacity_bits=bits, hbm_bandwidth=4096))
        for name, bits in (("ample", 10**10), ("tight", capacity))
    )
    groups = [list(estimate.context.get("group_allocations").values()) for estimate in (ample, tight)]
    assert [AllocationIR(**first).edge_placements["x"].split for first, _ in groups] == [None, 1]
    once: dict[str, dict[str, float]] = defaultdict(dict)
    for ir in groups[1]:
        for memory, held in ir["offchip_holdings"].items():
            once[memory] |= held
    assert max(sum(held.values()) for held in once.values()) <= capacity


def test_a_weight_placed_on_chip_beside_several_hbms_is_read_there_and_held_in_every_group(tmp_path):
    estimate = _mlp(tmp_path, PLACED | {"w1": {"placement": {"memories": ["sram_0"]}}})
    assert not {tensor for _, tensor in offchip_reads(estimate)} & {"w1"}
    first, second = (AllocationIR(**ir) for ir in estimate.context.get("group_allocations").values())
    assert all("w1" not in held for ir in (first, second) for held in ir.offchip_holdings.values())
    held = estimate.context.get("allocation").problem.held
    assert {memory.name for memory, bits in held.items() if "w1" in {t.name for t in bits}} == {"sram_0"}


def test_a_tensor_crossing_a_fusion_cut_is_held_once(tmp_path):
    with pytest.raises(
        InfeasibleAllocationError, match=r"^hbm_0 holds a \+ w1/4 \+ hidden/4 \+ w2/4 \+ out/4 = 193 KiB > 100 KiB$"
    ):
        _mlp(tmp_path, PLACED | {"Up_cut_out": SPLIT_N}, hardware=hardware(capacity_bits=100 * 8 * 2**10))


def test_a_cut_whose_cores_land_a_block_in_several_memories_is_refused_without_a_placement(tmp_path):
    """Up's cores split n in two and k in two: the two cores of each block of n land their sums in two HBMs."""
    up = ({"dim": "D1", "split": 2}, {"dim": "D2", "split": 2})
    with pytest.raises(ValueError, match="the cores of Up land a block of hidden in several off-chip memories"):
        _mlp(tmp_path, {"out": SPLIT_N}, up=up)


def test_the_two_ends_of_a_fusion_cut_must_agree(tmp_path):
    runtime_args = PLACED | {"Up_cut_in": SPLIT_N, "Up_cut_out": POOLED}
    with pytest.raises(
        ValueError, match="Up_cut_out and Up_cut_in are the two ends of one fusion cut but declare placement"
    ):
        _mlp(tmp_path, runtime_args)


def test_the_ends_of_a_fusion_cut_merge_what_they_declare(tmp_path):
    """Its producer writes the buffer its consumer reads, so a placement declared for one end and a layout for the
    other hold for both."""
    transposed = {"layout": "(d0, d1) -> (d1, d0)"}
    mappings = _mlp(tmp_path, PLACED | {"Up_cut_out": SPLIT_N, "Up_cut_in": transposed}).context.get("sub_mappings")
    assert mappings[0].edge_placements["Up_cut_out"] == mappings[1].edge_placements["Up_cut_in"]
    assert [memory.name for memory in mappings[0].edge_placements["Up_cut_out"].memories] == HBMS
    assert str(mappings[1].runtime_args["Up_cut_in"]) == str(mappings[0].runtime_args["Up_cut_out"])
    assert "(d0, d1) -> (d1, d0)" in str(mappings[0].runtime_args["Up_cut_out"])


def test_the_reports_carry_each_off_chip_memory_s_capacity_and_access_width(tmp_path):
    estimate = _estimate(tmp_path, SPLIT_N)
    (accesses,) = estimate.context.get("group_memory_accesses").values()
    widths = {(row["memory"], row["read_bandwidth"], row["write_bandwidth"]) for row in accesses["offchip"]}
    assert widths == {(hbm, 8, 8) for hbm in HBMS}
    memories = estimate.context.get("accelerator").get_ir()["memories"]
    assert {hbm: memories[hbm]["capacity_bits"] for hbm in HBMS} == dict.fromkeys(HBMS, 10**10)


def _onnx_edges(path):
    """Each graph input and output of the ONNX at ``path`` as the IR's edge tensors give it."""
    graph = onnx.load(path).graph

    def edge(value, direction):
        tensor = value.type.tensor_type
        bits = 8 * helper.tensor_dtype_to_np_dtype(tensor.elem_type).itemsize
        shape = [d.dim_value for d in tensor.shape.dim]
        return {"tensor": value.name, "shape": shape, "element_bits": bits, "direction": direction}

    return {v.name: edge(v, "in") for v in graph.input} | {v.name: edge(v, "out") for v in graph.output}


def _split_extents(ir):
    """The extent of the axis each splitting placement splits, in its edge's whole tensor, by edge name."""
    return {
        name: ir.edge_tensors[name].shape[placement.split]
        for name, placement in ir.edge_placements.items()
        if placement.split is not None
    }


def _edge_tensors(ir):
    return {name: tensor.model_dump(exclude={"axes"}) for name, tensor in ir.edge_tensors.items()}


def _axes(ir):
    return {name: tensor.axes for name, tensor in ir.edge_tensors.items()}


def test_the_ir_gives_each_edge_s_whole_tensor_as_the_onnx_declares_it(tmp_path):
    """Tiled over k in blocks of 128, the IR still gives the whole 1024 x 512 weight that its placement splits on n."""
    ir = AllocationIR.from_internal(_solve(tmp_path, SPLIT_N, tiles=[K_TILE]))
    assert _edge_tensors(ir) == _onnx_edges(_matmul())
    assert _axes(ir) == {"a": [0, 1], "w": [0, 1], "out": [0, 1]}
    assert _split_extents(ir) == {"w": 512, "out": 512}


@pytest.mark.parametrize(("cache", "parts", "extent"), [(1, None, KV_HEADS), (2, 4, 256), (3, 4, DHEAD)])
def test_the_ir_gives_a_decode_step_s_cache_and_its_placement_as_the_mapping_file_declares_them(
    tmp_path, cache, parts, extent
):
    """Placed by heads, or by sequence or head dim over an attention cut into four parts of its sequence, the cache's
    placement and shape are as the mapping file and the ONNX declare them, not in the axes of the cut attention."""
    placements = {"q": COPIED, "K": split_on(cache), "V": split_on(cache), "O": split_on(1)}
    workload = make_decode_attention(tmp_path / "decode.onnx", 1, 256)
    estimate = solve(RING, workload, tmp_path, {"runtime_args": placements}, attention_parts=parts)
    assert any(node.name.endswith("_part") for node in estimate.context.get("workload").nodes) == bool(parts)
    ir = AllocationIR.from_internal(estimate.context.get("allocation"))
    assert _edge_tensors(ir) == _onnx_edges(tmp_path / "decode.onnx")
    cache_axes = [0, 1, 2, 2, 3] if parts else [0, 1, 2, 3]
    assert _axes(ir) == {"q": [0, 1, 2, 3], "K": cache_axes, "V": cache_axes, "O": [0, 1, 2, 3]}
    splits = {name: placement.split for name, placement in ir.edge_placements.items()}
    assert splits == {"q": None, "K": cache, "V": cache, "O": 1}
    assert _split_extents(ir) == {"K": extent, "V": extent, "O": KV_HEADS}


def test_the_groups_hold_off_chip_what_the_capacity_check_counts_each_tensor_counted_once(tmp_path):
    """The two ends of a fusion cut carry one tensor; each tensor counted once, the groups' holdings, and the whole
    tensors and placements the IR gives, are what the capacity check counts."""
    estimate = _mlp(tmp_path, PLACED | {"Up_cut_out": SPLIT_N})
    groups = [AllocationIR(**ir) for ir in estimate.context.get("group_allocations").values()]
    assert groups[0].edge_tensors["Up_cut_out"].tensor == groups[1].edge_tensors["Up_cut_in"].tensor
    once: dict[str, dict[str, float]] = defaultdict(dict)
    for ir in groups:
        for memory, held in ir.offchip_holdings.items():
            once[memory] |= held
    whole = held_once(estimate, held_bits)
    assert once == {memory.name: {tensor.name: bits for tensor, bits in held.items()} for memory, held in whole.items()}
    tensors = {
        tensor.tensor: (tensor, ir.edge_placements[name]) for ir in groups for name, tensor in ir.edge_tensors.items()
    }
    bits: dict[str, Fraction] = defaultdict(Fraction)
    for tensor, placement in tensors.values():
        share = Fraction(1, len(placement.memories)) if placement.split is not None else Fraction(1)
        for memory in placement.memories:
            bits[memory] += share * prod(tensor.shape) * tensor.element_bits
    assert bits == {memory: sum(held.values()) for memory, held in once.items()}
