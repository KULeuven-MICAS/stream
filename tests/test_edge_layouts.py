"""A workload input or output laid out over the off-chip memories of a 4-slice ring, each slice with an HBM stack of
its own: a split the cores' split follows stays local, a pooled or wrongly split one costs the ring or one HBM."""

from types import SimpleNamespace

import onnx
import pytest
import yaml
from onnx import TensorProto, helper
from zigzag.utils import open_yaml

from stream.api import SolveOptions, evaluate_mapping
from stream.hardware.ports import OUTPUT, READ
from stream.ir.allocation import AllocationIR
from stream.ir.infeasibility import InfeasibleAllocationError
from stream.mapping.mapping import Mapping, placements_of
from stream.opt.allocation.constraint_optimization.families.dma import _port_channels
from stream.parser.accelerator_factory import AcceleratorFactory
from stream.parser.accelerator_validator import AcceleratorValidator
from stream.parser.mapping_validator import MappingValidator
from stream.workload.node import OutEdge
from stream.workload.workload import walked_iterations

RING = "stream/inputs/testing/hardware/tpu_like_ring_hbm.yaml"
HBMS = ["hbm_0", "hbm_1", "hbm_2", "hbm_3"]
SPLIT_N = {"memories": HBMS, "split": 1}
K_TILE = ("D2", 128)


def _matmul(tmp_path, k=1024, n=512):
    tmp_path.mkdir(exist_ok=True)
    inputs = [
        helper.make_tensor_value_info("a", TensorProto.BFLOAT16, [1, k]),
        helper.make_tensor_value_info("w", TensorProto.BFLOAT16, [k, n]),
    ]
    result = helper.make_tensor_value_info("out", TensorProto.BFLOAT16, [1, n])
    graph = helper.make_graph([helper.make_node("MatMul", ["a", "w"], ["out"], name="MatMul")], "g", inputs, [result])
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 19)]), tmp_path / "matmul.onnx")


def _estimate(tmp_path, w, out=SPLIT_N, tiles=(), k=1024, n=512, hardware=RING):  # noqa: PLR0913
    """A decode step's 1 x k by k x n matmul, its n split over the four cores and tiled over time by ``tiles``, ``w``
    and ``out`` laid out as given."""
    _matmul(tmp_path, k, n)
    layer = {"name": "MatMul", "core_allocation": [[0, 1, 2, 3]], "inter_core_tiling": [[{"dim": "D1", "split": 4}]]}
    tiling = [{"dim": f"MatMul.{dim}", "tile": tile} for dim, tile in tiles]
    runtime_args = {name: args for name, args in (("a", {"memories": HBMS}), ("w", w), ("out", out)) if args}
    mapping = {
        "layers": [layer],
        "fused_groups": [{"name": "g", "layers": ["MatMul"], "intra_core_tiling": tiling}],
        "runtime_args": runtime_args,
    }
    (tmp_path / "mapping.yaml").write_text(yaml.safe_dump(mapping))
    return evaluate_mapping(
        hardware,
        str(tmp_path / "matmul.onnx"),
        str(tmp_path),
        str(tmp_path / "mapping.yaml"),
        SolveOptions(artifacts=False),
    )


def _solve(tmp_path, w, **kwargs):
    return _estimate(tmp_path, w, **kwargs).context.get("allocation")


def _route(allocation, edge):
    """The transfer moving edge ``edge``'s tensor, and the route the solve chose for it."""
    workload = allocation.problem.workload
    (tr,) = (
        tr
        for tr in workload.get_transfer_nodes()
        if any(n.name == edge for n in (*workload.predecessors(tr), *workload.successors(tr)))
    )
    return tr, allocation.mapping.get(tr).resource_allocation[0]


def _ring_share(route):
    return sum(share for link, share in route.link_shares if not any(memory.offchip for memory in link.memories))


def _hbm_shares(allocation, tr, route):
    space = allocation.space
    return {
        memory.name: sum(space.side_ports(tr, route, memory, READ).values())
        for memory in route.sources
        if memory.offchip
    }


def test_a_weight_split_as_the_cores_split_it_stays_in_each_slice(tmp_path):
    allocation = _solve(tmp_path, SPLIT_N)
    tr, route = _route(allocation, "w")
    assert [memory.name for memory in route.sources] == HBMS
    assert route.pairs == ((0, 0), (1, 1), (2, 2), (3, 3))
    assert _ring_share(route) == 0
    assert _hbm_shares(allocation, tr, route) == dict.fromkeys(HBMS, 0.25)


def test_a_pooled_weight_crosses_the_ring_and_costs_more(tmp_path):
    local = _solve(tmp_path / "local", SPLIT_N)
    pooled = _solve(tmp_path / "pooled", {"memories": ["hbm_0"]})
    tr, route = _route(pooled, "w")
    assert _ring_share(route) >= 0.75
    assert _hbm_shares(pooled, tr, route) == {"hbm_0": 1.0}
    assert pooled.estimated_cycles > 2 * local.estimated_cycles


def test_a_weight_split_along_the_axis_walked_over_time_waits_on_one_hbm(tmp_path):
    """Split along k, which each iteration walks a tile of, every core reads its tile from the same HBM stack."""
    local = _solve(tmp_path / "local", SPLIT_N, tiles=[K_TILE])
    wrong = _solve(tmp_path / "wrong", {"memories": HBMS, "split": 0}, tiles=[K_TILE])
    tr, route = _route(wrong, "w")
    assert _hbm_shares(wrong, tr, route) == {"hbm_0": 1.0, "hbm_1": 0.0, "hbm_2": 0.0, "hbm_3": 0.0}
    local_tr, local_route = _route(local, "w")
    busy = wrong.space.moved_bits(tr, route) * route.memory_share(route.sources[0], leaving=True)
    local_bits = local.space.moved_bits(local_tr, local_route)
    assert busy == 4 * local_bits * local_route.memory_share(local_route.sources[0], leaving=True)
    assert wrong.estimated_cycles > 3 * local.estimated_cycles


def _ring_data():
    validator = AcceleratorValidator(open_yaml(RING), RING)
    assert validator.validate()
    return validator.normalized_data


def test_each_hbm_with_a_measured_bandwidth_is_held_for_its_own_block(tmp_path):
    data = _ring_data()
    data["bandwidth"] = {hbm: {"ceiling": 8, "contiguous": 8, "burst": 32} for hbm in HBMS}
    allocation = _solve(tmp_path, SPLIT_N, hardware=AcceleratorFactory(data).create())
    tr, route = _route(allocation, "w")
    space = allocation.space
    quarter = space.moved_bits(tr, route) / 4
    assert [space.shared_cycles(hbm, tr, route, 8) for hbm in route.sources] == [quarter / 8] * 4
    assert [space.side_bandwidth(tr, route, hbm, READ).contiguous for hbm in route.sources] == [8 / 0.25] * 4
    report = AllocationIR.from_internal(allocation).performance.memory_ports
    rows = [row for row in report if row.kind == "shared_bandwidth"]
    assert sorted(row.resource for row in rows) == HBMS
    assert all(row.bits_per_iteration / row.bw_bits_per_cycle <= row.allowed_cycle for row in rows)


def _ring_with_hbms_of(capacity_bits):
    data = _ring_data()
    for hbm in HBMS:
        data["memories"][hbm]["capacity_bits"] = capacity_bits
    return AcceleratorFactory(data).create()


def test_a_weight_copied_in_each_hbm_over_its_capacity_is_refused_by_what_each_holds(tmp_path):
    """The 1 MiB weight whole in each 512 KiB HBM stack, the output a quarter in each, landing beside its core."""
    with pytest.raises(InfeasibleAllocationError, match=r"^hbm_0 holds a \+ w \+ out/4 = 1 MiB > 512 KiB$") as refused:
        _solve(tmp_path, {"memories": HBMS}, out={}, hardware=_ring_with_hbms_of(4 * 2**20))
    report = refused.value.report
    assert report.nature == "capacity"
    (resource,) = report.resources
    unmet = resource.unmet
    assert (resource.resource.id, unmet.family, unmet.unit) == ("hbm_0", "offchip_capacity", "bytes")
    assert {term.label: term.value for term in unmet.terms} == {"w": 2**20, "a": 2**11, "out/4": 2**8}
    assert (unmet.demand_value, unmet.bound_value) == (sum(term.value for term in unmet.terms), 2**19)


def test_a_weight_split_over_the_hbms_fits_where_a_copy_in_each_does_not(tmp_path):
    assert _solve(tmp_path, SPLIT_N, out={}, hardware=_ring_with_hbms_of(4 * 2**20))


def _offchip_reads(tmp_path, w, tiles=()):
    (accesses,) = _estimate(tmp_path, w, tiles=tiles).context.get("group_memory_accesses").values()
    return {(row["memory"], row["tensor"]): row["read"] for row in accesses["offchip"] if row["read"]}


def test_each_hbm_is_charged_the_accesses_of_what_it_serves(tmp_path):
    """Each HBM reads a quarter of a split weight, and all of the input every core reads whole from its copy."""
    split = _offchip_reads(tmp_path / "split", SPLIT_N)
    pooled = _offchip_reads(tmp_path / "pooled", {"memories": ["hbm_0"]})
    assert [split[(hbm, "w")] for hbm in HBMS] == [pooled[("hbm_0", "w")] // 4] * 4
    assert not any(tensor == "w" and memory != "hbm_0" for memory, tensor in pooled)
    port_bits, a_bits = 8, 1 * 1024 * 16
    assert [split[(hbm, "a")] for hbm in HBMS] == [a_bits // port_bits] * 4


def test_a_wrongly_split_weight_is_read_evenly_over_the_walk_though_one_hbm_serves_each_iteration(tmp_path):
    reads = _offchip_reads(tmp_path / "wrong", {"memories": HBMS, "split": 0}, tiles=[K_TILE])
    local = _offchip_reads(tmp_path / "local", SPLIT_N, tiles=[K_TILE])
    assert [reads[(hbm, "w")] for hbm in HBMS] == [local[(hbm, "w")] for hbm in HBMS]
    assert len({reads[(hbm, "w")] for hbm in HBMS}) == 1


def test_an_axis_the_cores_split_inside_each_time_tile_is_read_from_two_hbms_per_iteration(tmp_path):
    """n walked in two halves, each split over the four cores: core c reads n = 256 t + 64 c + i, so a contiguous block
    per HBM serves two cores per iteration over the ring; over the walk each HBM reads a quarter."""
    tiles = [("D1", 64)]
    allocation = _solve(tmp_path / "solve", SPLIT_N, tiles=tiles)
    tr, route = _route(allocation, "w")
    assert _hbm_shares(allocation, tr, route) == {"hbm_0": 0.5, "hbm_1": 0.5, "hbm_2": 0.0, "hbm_3": 0.0}
    assert _ring_share(route) > 0
    reads = _offchip_reads(tmp_path / "reads", SPLIT_N, tiles=tiles)
    assert len({reads[(hbm, "w")] for hbm in HBMS}) == 1


def _ring(dma=False, **ports):
    """The ring with each named HBM's port changed as given, each HBM's port a DMA port with ``dma``."""
    validator = AcceleratorValidator(open_yaml(RING), RING)
    assert validator.validate()
    data = validator.normalized_data
    for hbm, changes in ports.items():
        data["memories"][hbm]["ports"]["rw_port_1"] |= changes
    for hbm in HBMS if dma else ():
        data["memories"][hbm]["ports"]["rw_port_1"]["channels"] = 2
        data["memories"][hbm]["dma"] = {"buffer_descriptors": 16, "iterations": 64}
    return AcceleratorFactory(data).create()


N_TILE = ("D1", 16)
"""n walked 16 at a time: each iteration reads from one HBM, a different one as the walk goes on."""


def test_a_walk_holds_dma_channels_on_every_hbm_it_reads(tmp_path):
    allocation = _solve(tmp_path, SPLIT_N, tiles=[N_TILE], hardware=_ring(dma=True))
    tr, route = _route(allocation, "w")
    space = allocation.space
    assert sum(route.memory_share(hbm, leaving=True) > 0 for hbm in route.sources) == 1
    space.transfer_nodes, space.path_choices = [tr], {tr: (route,)}
    one = SimpleNamespace(_raw=1)
    _, out = _port_channels(SimpleNamespace(space=space, vars=SimpleNamespace(y={(tr, route): one})))
    assert sorted(memory.name for memory, _ in out) == HBMS


def test_the_busiest_iteration_is_the_one_at_the_slowest_hbm(tmp_path):
    allocation = _solve(
        tmp_path, {"memories": HBMS, "split": 0}, tiles=[K_TILE], hardware=_ring(hbm_2={"bandwidth": 4})
    )
    tr, route = _route(allocation, "w")
    assert {hbm.name: route.memory_share(hbm, leaving=True) for hbm in route.sources}["hbm_2"] == 1.0


def test_a_walk_is_priced_at_each_hbm_s_energy_for_its_average_share(tmp_path):
    """Each iteration of the walk reads one HBM, but over the walk each reads a quarter, each at its own energy."""
    costly = {"read_energy": 1000.0, "write_energy": 1000.0}
    allocation = _solve(tmp_path, SPLIT_N, tiles=[N_TILE], hardware=_ring(hbm_1=costly, hbm_2=costly, hbm_3=costly))
    energy = AllocationIR.from_internal(allocation).performance.energy
    reads = {row.resource: row.pj_per_iteration for row in energy.components if row.resource.startswith("hbm")}
    assert reads["hbm_1.rw_port_1"] == pytest.approx(reads["hbm_0.rw_port_1"] * 1000.0 / 15.625, rel=0.01)
    assert reads["hbm_1.rw_port_1"] == pytest.approx(reads["hbm_3.rw_port_1"])


def test_a_weight_copied_in_each_hbm_is_read_where_its_reader_is(tmp_path):
    allocation = _solve(tmp_path, {"memories": HBMS})
    tr, route = _route(allocation, "w")
    assert route.pairs == ((0, 0), (1, 1), (2, 2), (3, 3))
    assert _ring_share(route) == 0
    assert _hbm_shares(allocation, tr, route) == dict.fromkeys(HBMS, 0.25)


def test_an_output_split_lands_a_block_in_each_slice_s_hbm(tmp_path):
    allocation = _solve(tmp_path, SPLIT_N)
    _, route = _route(allocation, "out")
    assert [memory.name for memory in route.targets] == HBMS
    assert route.pairs == ((0, 0), (1, 1), (2, 2), (3, 3))
    assert _ring_share(route) == 0
    ir = AllocationIR.from_internal(allocation)
    assert ir.edge_layouts["out"].memories == HBMS and ir.edge_layouts["out"].axis == 1


def test_an_output_without_a_layout_lands_in_the_hbm_nearest_each_producing_core(tmp_path):
    _, route = _route(_solve(tmp_path, SPLIT_N, out={}), "out")
    assert [memory.name for memory in route.targets] == HBMS
    assert _ring_share(route) == 0


def test_an_input_without_a_layout_is_refused_by_name(tmp_path):
    with pytest.raises(ValueError, match=r"input w has no layout over the off-chip memories \['hbm_0', 'hbm_1'"):
        _solve(tmp_path, {})


def test_a_split_that_does_not_divide_its_axis_is_refused(tmp_path):
    with pytest.raises(ValueError, match="axis 0 of extent 1 does not split evenly"):
        _solve(tmp_path, SPLIT_N, out={"memories": HBMS, "split": 0})


def test_an_output_copied_in_several_memories_is_refused(tmp_path):
    with pytest.raises(ValueError, match="output out cannot be a copy"):
        _solve(tmp_path, SPLIT_N, out={"memories": HBMS})


@pytest.mark.parametrize(
    "args",
    [{"memories": "hbm_0"}, {"split": 1}, {"memories": [], "split": 0}, {"memories": HBMS, "split": -1}],
)
def test_a_malformed_layout_is_refused(args):
    validator = MappingValidator({"layers": [], "runtime_args": {"w": args}})
    assert not validator.validate()


def test_other_runtime_args_are_left_to_their_readers():
    validator = MappingValidator({"layers": [], "runtime_args": {"w": {"layout": "(d0, d1) -> (d1, d0)", "depth": 2}}})
    assert validator.validate()


def test_a_layout_names_off_chip_memories(tmp_path):
    with pytest.raises(ValueError, match=r"w lives in \['sram_0'\], but the off-chip memories are"):
        _solve(tmp_path, {"memories": ["sram_0"]})


def _generated(tmp_path, w):
    """The matmul mapped by the generic generator from a mapping file that only lays out its edges, and the split it
    chose across the cores."""
    _matmul(tmp_path)
    runtime_args = {"a": {"memories": HBMS}, "w": w, "out": SPLIT_N}
    (tmp_path / "layouts.yaml").write_text(yaml.safe_dump({"runtime_args": runtime_args}))
    estimate = evaluate_mapping(
        RING,
        str(tmp_path / "matmul.onnx"),
        str(tmp_path),
        str(tmp_path / "layouts.yaml"),
        SolveOptions(artifacts=False),
    )
    (layer,) = open_yaml(tmp_path / "group_0" / "mapping.yaml")["layers"]
    return estimate.context.get("allocation"), layer["inter_core_tiling"]


def test_the_generic_generator_splits_the_cores_as_a_layout_splits_the_weight(tmp_path):
    allocation, split = _generated(tmp_path, SPLIT_N)
    assert split == [[{"dim": "D1", "split": 4}]]
    tr, route = _route(allocation, "w")
    assert _ring_share(route) == 0
    assert _hbm_shares(allocation, tr, route) == dict.fromkeys(HBMS, 0.25)


def test_the_generic_generator_prefers_the_axis_a_layout_splits(tmp_path):
    """Split along k, each core reduces the block of its own slice's HBM, the iterations walking n rather than k."""
    allocation, split = _generated(tmp_path, {"memories": HBMS, "split": 0})
    assert split == [[{"dim": "D2", "split": 4}]]
    tr, route = _route(allocation, "w")
    assert _ring_share(route) == 0
    assert _hbm_shares(allocation, tr, route) == dict.fromkeys(HBMS, 0.25)


def _mlp(tmp_path, runtime_args, hardware=RING):
    """Two matmuls in two fused groups, the hidden tensor between them crossing the cut."""
    shapes = {"a": [1, 256], "w1": [256, 512], "w2": [512, 512]}
    inputs = [helper.make_tensor_value_info(name, TensorProto.BFLOAT16, shape) for name, shape in shapes.items()]
    result = helper.make_tensor_value_info("out", TensorProto.BFLOAT16, [1, 512])
    nodes = [
        helper.make_node("MatMul", ["a", "w1"], ["hidden"], name="Up"),
        helper.make_node("MatMul", ["hidden", "w2"], ["out"], name="Down"),
    ]
    graph = helper.make_graph(nodes, "g", inputs, [result])
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 19)]), tmp_path / "mlp.onnx")
    layers = [
        {"name": name, "core_allocation": [[0, 1, 2, 3]], "inter_core_tiling": [[{"dim": "D1", "split": 4}]]}
        for name in ("Up", "Down")
    ]
    groups = [{"name": name, "layers": [name], "intra_core_tiling": []} for name in ("Up", "Down")]
    (tmp_path / "mapping.yaml").write_text(
        yaml.safe_dump({"layers": layers, "fused_groups": groups, "runtime_args": runtime_args})
    )
    mapping = str(tmp_path / "mapping.yaml")
    estimate = evaluate_mapping(
        hardware, str(tmp_path / "mlp.onnx"), str(tmp_path), mapping, SolveOptions(artifacts=False)
    )
    return estimate.context.get("sub_mappings")


LAID_OUT = {name: SPLIT_N for name in ("w1", "w2", "out")} | {"a": {"memories": HBMS}}


def test_a_layout_declared_for_one_end_of_a_fusion_cut_holds_for_the_other(tmp_path):
    layouts = [mapping.edge_layouts for mapping in _mlp(tmp_path, LAID_OUT | {"Up_cut_out": SPLIT_N})]
    assert layouts[0]["Up_cut_out"] == layouts[1]["Up_cut_in"]
    assert [memory.name for memory in layouts[1]["Up_cut_in"].memories] == HBMS


def test_a_tensor_crossing_a_fusion_cut_is_held_once(tmp_path):
    with pytest.raises(
        InfeasibleAllocationError, match=r"^hbm_0 holds a \+ w1/4 \+ hidden/4 \+ w2/4 \+ out/4 = 193 KiB > 100 KiB$"
    ):
        _mlp(tmp_path, LAID_OUT | {"Up_cut_out": SPLIT_N}, hardware=_ring_with_hbms_of(100 * 8 * 2**10))


def test_the_two_ends_of_a_fusion_cut_must_agree(tmp_path):
    runtime_args = LAID_OUT | {"Up_cut_in": SPLIT_N, "Up_cut_out": {"memories": ["hbm_0"]}}
    with pytest.raises(
        ValueError, match="Up_cut_out and Up_cut_in are the two ends of one fusion cut but declare memories"
    ):
        _mlp(tmp_path, runtime_args)


def test_the_ends_of_a_fusion_cut_merge_what_they_declare(tmp_path):
    transposed = {"layout": "(d0, d1) -> (d1, d0)"}
    mappings = _mlp(tmp_path, LAID_OUT | {"Up_cut_out": transposed, "Up_cut_in": SPLIT_N})
    assert mappings[0].edge_layouts["Up_cut_out"] == mappings[1].edge_layouts["Up_cut_in"]
    assert str(mappings[1].runtime_args["Up_cut_in"]) == str(mappings[0].runtime_args["Up_cut_out"])


def test_a_codegen_layout_declared_for_one_end_of_a_fusion_cut_holds_for_the_other(tmp_path):
    """Its producer writes the buffer its consumer reads, so both ends take the one declared layout."""
    transposed = {"Up_cut_out": {"layout": "(d0, d1) -> (d1, d0)"}}
    mappings = _mlp(tmp_path, transposed, hardware="stream/inputs/examples/hardware/tpu_like_quad_core.yaml")
    assert str(mappings[1].runtime_args["Up_cut_in"]) == str(mappings[0].runtime_args["Up_cut_out"])
    assert str(mappings[1].runtime_args["Up_cut_in"]) == "(d0, d1) -> (d1, d0)"


def test_an_input_and_an_output_split_along_different_axes_walk_every_pairing_of_their_blocks():
    """Two iterations walk the input's axis and three the output's: six iterations, where one axis pairs them."""
    held, target = ["in 0", "in 1"], ["out 0", "out 1", "out 2"]
    assert walked_iterations(held, target, one_axis=False) == [(h, t) for h in held for t in target]
    assert walked_iterations(held, target[:2], one_axis=True) == [("in 0", "out 0"), ("in 1", "out 1")]
    assert walked_iterations(None, target, one_axis=False) == [(None, t) for t in target]


def test_where_an_output_lands_is_no_placement_of_its_edge():
    """An output lands by its layout or beside its producing cores (landing_of), which its edge alone cannot say."""
    accelerator = AcceleratorFactory(_ring_data()).create()
    with pytest.raises(ValueError, match="holds no operand"):
        placements_of(accelerator, Mapping(), OutEdge(name="out", inputs=()), OUTPUT)


def test_the_reports_carry_each_off_chip_memory_s_capacity_and_access_width(tmp_path):
    estimate = _estimate(tmp_path, SPLIT_N)
    (accesses,) = estimate.context.get("group_memory_accesses").values()
    widths = {(row["memory"], row["read_bandwidth"], row["write_bandwidth"]) for row in accesses["offchip"]}
    assert widths == {(hbm, 8, 8) for hbm in HBMS}
    memories = estimate.context.get("accelerator").get_ir()["memories"]
    assert {hbm: memories[hbm]["capacity_bits"] for hbm in HBMS} == dict.fromkeys(HBMS, 10**10)
