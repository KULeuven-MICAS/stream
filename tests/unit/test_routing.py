"""Routing a transfer: each target from the nearest source holding its data, over one Steiner tree per source."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import onnx
import pytest
import yaml
from onnx import TensorProto, helper

from stream.api import SolveOptions, evaluate_mapping
from stream.cost_model import communication_manager
from stream.cost_model.communication_manager import DemandItem, MulticastPathPlan, box_bits
from stream.cost_model.communication_manager import _dijkstra as dijkstra
from stream.hardware.architecture.accelerator import Accelerator
from stream.hardware.architecture.memory import Memory
from stream.hardware.ports import OUTPUT, input_role
from stream.opt.allocation.constraint_optimization.space import communicating_pairs
from stream.stages.parsing.accelerator_parser import parse_accelerator
from stream.workload.blocks import build_block

OFFCHIP = {
    "offchip": True,
    "views": [],
    "capacity_bits": 10_000_000_000,
    "ports": {
        "rw_port_1": {
            "bandwidth": 64,
            "serves": {"any": ["read", "write"]},
            "read_energy": 15.625,
            "write_energy": 15.625,
        }
    },
}
"""An off-chip memory: one 64-bit port serving every operand both ways."""


def _hardware(path: Path, name: str, cores: dict[int, str], memories: dict[str, Any], connections: str) -> Path:
    """Write a hardware description to ``path`` and return the path.

    ``memories`` maps a name to :data:`OFFCHIP` or to the views of an on-chip memory with one read and one write port.
    ``connections`` is a YAML list of links and buses, each naming the ``memories`` it joins.
    """

    def declared(views: Any) -> dict[str, Any]:
        if views is OFFCHIP:
            return OFFCHIP
        cores = [int(view.split(".")[0]) for view in views]
        ports = {port: {"from": port, "cores": cores} for port in ("r_port_1", "w_port_1")}
        return {"offchip": False, "views": views, "ports": ports}

    data = {
        "name": name,
        "cores": cores,
        "unit_energy_cost": 0,
        "memories": {memory: declared(views) for memory, views in memories.items()},
        "connections": [
            {key: value for key, value in connection.items() if key != "memories"}
            | {"ends": {name: list(declared(memories[name])["ports"]) for name in connection["memories"]}}
            for connection in yaml.safe_load(connections)
        ],
    }
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


TWO_CLUSTERS_MEMORIES = {
    **{f"m{i}": [f"{i}.sram_128K_A"] for i in range(6)},
    "m6": OFFCHIP,
    **{f"w{i}": [f"{i}.sram_128K_W"] for i in range(6)},
}
TWO_CLUSTERS_CONNECTIONS = """
- {type: bus, memories: [m0, m1, m2], bandwidth: 1000}
- {type: bus, memories: [m3, m4, m5], bandwidth: 1000}
- {type: bus, memories: [m2, m5, m6], bandwidth: 100}
- {type: link, memories: [m2, m5], bandwidth: 300}
- {type: bus, memories: [w0, w1, w2, w3, w4, w5, m6], bandwidth: 100}
"""
"""Two clusters of three cores on a bus each. Hubs m2 and m5 share a link and a bus with off-chip m6. Each core also has
a weights memory w<core> that only m6 reaches."""

TWO_CORES_MEMORIES = {
    "m0": ["0.sram_128K_A"],
    "m1": ["1.sram_128K_A"],
    "m2": OFFCHIP,
    "w0": ["0.sram_128K_W"],
    "w1": ["1.sram_128K_W"],
}
"""Two cores, each with an activation memory m<core> and a weights memory w<core>, and off-chip m2."""

HALF_A, HALF_B = ((0, 7), (0, 15)), ((8, 15), (0, 15))
WHOLE = ((0, 15), (0, 15))


@pytest.fixture(scope="module")
def accelerator(tmp_path_factory: pytest.TempPathFactory) -> Accelerator:
    path = tmp_path_factory.mktemp("hw") / "two_clusters.yaml"
    cores = dict.fromkeys(range(6), "testing_core1.yaml")
    return parse_accelerator(_hardware(path, "two-clusters", cores, TWO_CLUSTERS_MEMORIES, TWO_CLUSTERS_CONNECTIONS))


def _memory(accelerator: Accelerator, end: int | str, role: str) -> Memory:
    """The memory named ``end``, or the one core ``end`` holds the operand of ``role`` in."""
    return accelerator.memories[end] if isinstance(end, str) else accelerator.memory_of(accelerator.get_core(end), role)


def route(accelerator: Accelerator, sources, targets, demand, shares_memory=lambda one, other: False):
    """Route from the output memories of ``sources`` to the first-input memories of ``targets`` (core ids or memory
    names)."""
    plan, _ = accelerator.communication_manager.route(
        tuple(_memory(accelerator, end, OUTPUT) for end in sources),
        tuple(_memory(accelerator, end, input_role(1)) for end in targets),
        tuple(demand),
        16,
        shares_memory,
    )
    return plan


def shares(plan) -> dict[str, float]:
    return {str(link): round(share, 3) for link, share in plan.link_shares}


def test_overlapping_boxes_count_their_common_part_once():
    assert box_bits([((0, 3), (0, 3)), ((2, 5), (0, 3)), ((0, 3), (0, 3))], 1) == 24


def test_each_target_is_served_by_the_copy_in_its_own_cluster(accelerator):
    """Hubs 2 and 5 both hold the tensor; each cluster reads its own hub's copy, so nothing crosses the hub bus and
    each cluster bus carries the whole tensor, the two at once."""
    demand = [DemandItem(t, (0, 1), WHOLE) for t in range(4)]
    plan = route(accelerator, (2, 5), (0, 1, 3, 4), demand)
    assert plan.pairs == ((0, 0), (0, 1), (1, 2), (1, 3))
    assert shares(plan) == {"Bus(m0,m1,m2, bw=1000)": 1.0, "Bus(m3,m4,m5, bw=1000)": 1.0}


def test_a_link_is_charged_only_the_part_that_is_not_already_in_its_target(tmp_path):
    """Two cores sharing a memory read twelve rows, eight held there and four by a core across a link: the link
    carries those four, a third of what the transfer delivers."""
    path = _hardware(
        tmp_path / "local.yaml",
        "local",
        {0: "shared_testing_core1.yaml", 1: "shared_testing_core2.yaml", 2: "testing_core1.yaml"},
        {"l1": ["0.sram_64KB", "1.sram_64KB"], "m2": ["2.sram_128K_A"], "w2": ["2.sram_128K_W"], "m3": OFFCHIP},
        "[{type: link, memories: [l1, m2], bandwidth: 64}, {type: link, memories: [l1, m3], bandwidth: 64},"
        " {type: link, memories: [w2, m3], bandwidth: 64}]",
    )
    quarter = ((0, 3), (0, 15))
    demand = [DemandItem(0, (0,), HALF_A), DemandItem(1, (0,), ((8, 11), (0, 15))), DemandItem(1, (1,), quarter)]
    plan = route(parse_accelerator(path), (0, 2), (0, 1), demand)
    assert shares(plan) == {"CL(m2, l1, bw=64)": round(64 / 192, 3)}


def test_a_broadcast_crosses_each_link_once(accelerator):
    """The same data to both cores of a cluster is sent over its bus once."""
    plan = route(accelerator, (2,), (0, 1), [DemandItem(0, (0,), WHOLE), DemandItem(1, (0,), WHOLE)])
    assert shares(plan) == {"Bus(m0,m1,m2, bw=1000)": 1.0}


def test_a_scatter_charges_each_link_only_the_part_crossing_it(accelerator):
    """Off-chip memory sends one half to each cluster: both halves leave over the hub bus, one per cluster bus."""
    plan = route(accelerator, ("m6",), (0, 3), [DemandItem(0, (0,), HALF_A), DemandItem(1, (0,), HALF_B)])
    assert shares(plan) == {"Bus(m2,m5,m6, bw=100)": 1.0, "Bus(m0,m1,m2, bw=1000)": 0.5, "Bus(m3,m4,m5, bw=1000)": 0.5}


def test_a_hop_between_clusters_takes_the_faster_link(accelerator):
    """From one hub to the other, the 300-wide link beats the 100-wide hub bus."""
    plan = route(accelerator, (2,), (5,), [DemandItem(0, (0,), WHOLE)])
    assert shares(plan) == {"CL(m2, m5, bw=300)": 1.0}


def test_a_target_sharing_memory_with_its_source_moves_nothing(accelerator):
    """The route still reaches the target, so the transfer is not mistaken for one within a single core."""
    plan = route(accelerator, (2,), (0,), [DemandItem(0, (0,), WHOLE)], shares_memory=lambda one, other: True)
    assert plan.pairs == ((0, 0),)
    assert plan.links_used and all(share == 0 for _, share in plan.link_shares)


def test_off_chip_memory_never_relays_data_between_cores(tmp_path):
    """Two cores with wide links to DRAM and a narrow one between them: the data takes the narrow link, since a
    route cannot pass through DRAM."""
    path = _hardware(
        tmp_path / "relay.yaml",
        "relay",
        dict.fromkeys(range(2), "testing_core1.yaml"),
        TWO_CORES_MEMORIES,
        "[{type: link, memories: [w0, m2], bandwidth: 1000}, {type: link, memories: [w1, m2], bandwidth: 1000},"
        " {type: link, memories: [m0, m2], bandwidth: 1000}, {type: link, memories: [m1, m2], bandwidth: 1000},"
        " {type: link, memories: [m0, m1], bandwidth: 10}]",
    )
    plan = route(parse_accelerator(path), (0,), (1,), [DemandItem(0, (0,), WHOLE)])
    assert shares(plan) == {"CL(m0, m1, bw=10)": 1.0}


def _offchip_fed(tmp_path: Path, cores: int, links: list[tuple[str, str, int]]) -> Accelerator:
    """``cores`` cores and off-chip memory ``dram``, ``links`` joining the cores' activation memories ``m<core>`` and
    ``dram``; each core's weights memory ``w<core>`` has a link to ``dram``."""
    memories = {f"m{i}": [f"{i}.sram_128K_A"] for i in range(cores)} | {
        f"w{i}": [f"{i}.sram_128K_W"] for i in range(cores)
    }
    links = links + [(f"w{i}", "dram", 64) for i in range(cores)]
    path = _hardware(
        tmp_path / "hw.yaml",
        "hw",
        dict.fromkeys(range(cores), "testing_core1.yaml"),
        memories | {"dram": OFFCHIP},
        yaml.safe_dump([{"type": "link", "memories": [a, b], "bandwidth": w} for a, b, w in links]),
    )
    return parse_accelerator(path)


FORWARDING = [("dram", "m0", 1000), ("dram", "m1", 1000), ("m0", "m1", 1300)]
"""A link from off-chip memory to each of two cores, which share a faster link."""


def ends(plan: MulticastPathPlan) -> dict[tuple[str, str], float]:
    return {(link.sender.name, link.receiver.name): round(share, 3) for link, share in plan.link_shares}


def reads(plan: MulticastPathPlan, accelerator: Accelerator) -> dict[str, float]:
    """The share of the delivered data each memory reads to send it on."""
    shares = {m.name: plan.share_at(m, True, plan.links_at(m, True)) for m in accelerator.memories.values()}
    return {name: share for name, share in shares.items() if share}


def test_a_target_with_its_own_data_is_not_fed_through_another(tmp_path):
    """Each core's own half comes over its own off-chip link, however fast the link between the cores."""
    plan = route(
        _offchip_fed(tmp_path, 2, FORWARDING),
        ("dram",),
        (0, 1),
        [DemandItem(0, (0,), HALF_A), DemandItem(1, (0,), HALF_B)],
    )
    assert ends(plan) == {("dram", "m0"): 0.5, ("dram", "m1"): 0.5}


def test_a_broadcast_from_off_chip_memory_may_also_enter_striped(tmp_path):
    """The tree brings a broadcast over one off-chip link; the striped route half over each, each core forwarding it."""
    accelerator = _offchip_fed(tmp_path, 2, FORWARDING)
    demand = [DemandItem(0, (0,), WHOLE), DemandItem(1, (0,), WHOLE)]
    tree, striped = _plans(accelerator, ("dram",), (0, 1), demand, 1)
    assert ends(tree) == {("dram", "m0"): 1.0, ("m0", "m1"): 1.0}
    assert ends(striped) == {("dram", "m0"): 0.5, ("m0", "m1"): 0.5, ("dram", "m1"): 0.5, ("m1", "m0"): 0.5}
    assert reads(striped, accelerator) == {"dram": 1.0, "m0": 0.5, "m1": 0.5}
    assert len(_plans(accelerator, (0,), (1,), demand[:1], 1)) == 1


def test_a_striped_route_carries_only_the_data_every_entry_core_reads(tmp_path):
    """Neither core needs the other's rows, so no route parks them in its memory to forward them."""
    uneven = [DemandItem(0, (0,), ((0, 11), (0, 15))), DemandItem(1, (0,), ((12, 15), (0, 15)))]
    (plan,) = _plans(_offchip_fed(tmp_path, 2, FORWARDING), ("dram",), (0, 1), uneven, 1)
    assert ends(plan) == {("dram", "m0"): 0.75, ("dram", "m1"): 0.25}


def test_split_data_does_not_funnel_through_a_hub_it_already_crosses(tmp_path):
    """A hub carrying one half would be busy twice as long carrying both, so each half enters over its own hub."""
    links = [("dram", "m0", 10), ("dram", "m1", 10)] + [(a, b, 1000) for a in ("m0", "m1") for b in ("m2", "m3")]
    plan = route(
        _offchip_fed(tmp_path, 4, links), ("dram",), (2, 3), [DemandItem(0, (0,), HALF_A), DemandItem(1, (0,), HALF_B)]
    )
    assert sorted(share for (sender, _), share in ends(plan).items() if sender == "dram") == [0.5, 0.5]


def test_a_relay_reads_only_the_overlap_of_what_it_holds_and_passes_on(tmp_path):
    """Down a chain of overlapping windows each core reads again only the rows it holds that the next ones read."""
    accelerator = _offchip_fed(tmp_path, 3, [("dram", "m0", 100), ("m0", "m1", 1000), ("m1", "m2", 1000)])
    windows = [((0, 9), (0, 15)), ((5, 14), (0, 15)), ((8, 17), (0, 15))]
    plan = route(accelerator, ("dram",), (0, 1, 2), [DemandItem(i, (0,), w) for i, w in enumerate(windows)])
    assert {name: round(share * 18, 3) for name, share in reads(plan, accelerator).items()} == {
        "dram": 18.0,
        "m0": 5.0,
        "m1": 7.0,
    }


@pytest.mark.parametrize("halo", [0, 2])
def test_a_scatter_round_a_ring_searches_about_once_per_target(tmp_path, monkeypatch, halo):
    """128 cores on a ring fed through one, each reading its own rows and its neighbours' edges: targets of equal
    size share a search, redone only when their path took more data."""
    accelerator = _offchip_fed(
        tmp_path, 128, [("dram", "m0", 1000)] + [(f"m{i}", f"m{(i + 1) % 128}", 1300) for i in range(128)]
    )
    searches = []

    def counted(*args):
        searches.append(args)
        return dijkstra(*args)

    monkeypatch.setattr(communication_manager, "_dijkstra", counted)
    rows = [(max(0, 8 * i - halo), min(1023, 8 * i + 7 + halo)) for i in range(128)]
    route(accelerator, ("dram",), tuple(range(128)), [DemandItem(i, (0,), (rows[i], (0, 63))) for i in range(128)])
    assert len(searches) <= 4 * 128


def _solve_pair(tmp_path: Path, between: int, link_energy: float = 0):
    """A matmul split over the output columns of two cores, each with a narrow link from off-chip memory."""
    sram = {f"m{i}": [f"{i}.sram_2MB"] for i in range(2)}
    links = [("m0", "dram", 16), ("m1", "dram", 16), ("m0", "m1", between)]
    path = _hardware(
        tmp_path / "hw.yaml",
        "pair",
        dict.fromkeys(range(2), "tpu_like.yaml"),
        sram | {"dram": OFFCHIP},
        yaml.safe_dump([{"type": "link", "memories": [a, b], "bandwidth": w} for a, b, w in links]),
    )
    data = yaml.safe_load(path.read_text()) | {"unit_energy_cost": link_energy}
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    (tmp_path / "mapping.yaml").write_text(
        "layers:\n- {name: projection, core_allocation: [[0, 1]], inter_core_tiling: [[{dim: D1, split: 2}]]}\n"
        "fused_groups:\n- {name: g, layers: [projection]}\n"
    )

    def value(name: str, shape: list[int]):
        return helper.make_tensor_value_info(name, TensorProto.BFLOAT16, shape)

    node = helper.make_node("MatMul", ["x", "w"], ["y"], name="projection")
    graph = helper.make_graph(
        [node], "projection", [value("x", [64, 256]), value("w", [256, 256])], [value("y", [64, 256])]
    )
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)]), tmp_path / "mm.onnx")
    options = SolveOptions(artifacts=False)
    return evaluate_mapping(
        str(path), str(tmp_path / "mm.onnx"), str(tmp_path), str(tmp_path / "mapping.yaml"), options
    )


@pytest.mark.parametrize(("between", "striped"), [(1024, True), (4, False)])
def test_the_allocator_takes_the_striped_route_only_where_forwarding_on_chip_pays(tmp_path, between, striped):
    """Both cores read all of x: with a fast link between them half enters over each, with a slow one the tree."""
    routes = _solve_pair(tmp_path, between).context.get("allocation").solution.transfer_routes
    (load,) = [plan for tr, plan in routes.items() if tr.name == "Transfer(x)"]
    from_dram = sorted(share for (sender, _), share in ends(load).items() if sender == "dram")
    assert from_dram == ([0.5, 0.5] if striped else [1.0, 1.0])


def test_each_link_is_billed_the_bits_it_carries(tmp_path):
    """Every link's energy is the bits it carries at its energy per bit, the halves of a split and of a stripe too."""
    allocations = _solve_pair(tmp_path, 1024, link_energy=160).context.get("group_allocations")
    for allocation in allocations.values():
        performance = allocation["performance"]
        billed = {
            c["resource"]: c["pj_per_iteration"] for c in performance["energy"]["components"] if c["kind"] == "link"
        }
        rows = [row for row in performance["memory_ports"] if row["kind"] == "link"]
        assert rows and billed
        for row in rows:
            bandwidth = 16 if "dram" in row["resource"] else 1024
            assert billed[row["resource"]] == pytest.approx(row["bits_per_iteration"] * 160 / bandwidth)


def test_cores_in_one_memory_need_no_route(tmp_path):
    """Two cores sharing their memory, reachable from each other only through DRAM: the data is already there."""
    path = _hardware(
        tmp_path / "shared.yaml",
        "shared",
        {0: "shared_testing_core1.yaml", 1: "shared_testing_core2.yaml"},
        {"l1": ["0.sram_64KB", "1.sram_64KB"], "m2": OFFCHIP},
        "[{type: link, memories: [l1, m2], bandwidth: 64}, {type: link, memories: [l1, m2], bandwidth: 64}]",
    )
    plan = route(parse_accelerator(path), (0,), (1,), [DemandItem(0, (0,), WHOLE)])
    assert plan.pairs == ((0, 0),)
    assert not plan.links_used


def test_a_bus_is_at_a_memory_only_where_a_hop_of_the_route_ends_there(accelerator):
    """A route from m2 over the bus to m6 and over the link to m5 reaches m5 only through the link."""
    m2, m5, m6 = (accelerator.memories[f"m{i}"] for i in (2, 5, 6))
    links = {str(d["cl"]): d["cl"] for _, _, d in accelerator.network.edges(data=True)}
    bus, link = links["Bus(m2,m5,m6, bw=100)"], links["CL(m2, m5, bw=300)"]
    plan = MulticastPathPlan((m2,), (m5, m6), 2, (bus, link), ((0, 0), (0, 1)), hops=((m2, m6, bus), (m2, m5, link)))
    assert plan.links_at(m5, leaving=False) == (link,) and plan.links_at(m6, leaving=False) == (bus,)
    assert plan.links_at(m2, leaving=True) == (bus, link) and plan.links_at(m2, leaving=False) == ()


def test_buses_of_the_same_width_are_distinct_links(accelerator):
    """Each cluster bus is its own link, so traffic on one does not count against the other."""
    buses = {data["cl"] for _, _, data in accelerator.network.edges(data=True) if data["cl"].members}
    assert sorted(str(bus) for bus in buses) == [
        "Bus(m0,m1,m2, bw=1000)",
        "Bus(m2,m5,m6, bw=100)",
        "Bus(m3,m4,m5, bw=1000)",
        "Bus(w0,w1,w2,w3,w4,w5,m6, bw=100)",
    ]


def test_an_aie_route_pairs_its_cores_as_the_code_generator_does(tmp_path):
    """On AIE the code generator matches cores by spatial index, so every route pairs exactly those cores."""
    estimate = evaluate_mapping(
        "stream/inputs/aie/hardware/whole_array_strix.yaml",
        "stream/inputs/aie/workload/gemm_256_8192_2048.onnx",
        str(tmp_path),
        options=SolveOptions(nb_cols_to_use=8, artifacts=False),
    )
    routes = estimate.context.get("allocation").solution.transfer_routes
    assert routes
    for plan in routes.values():
        paired = {(plan.sources[i], plan.targets[j]) for i, j in plan.pairs}
        assert paired == set(communicating_pairs(plan.sources, plan.targets))


def test_partial_sums_are_added_where_their_paths_meet(accelerator):
    """Partial sums on 3 and 4 meet at hub 5, which adds them, so the link from 5 to 2 carries the sum once; a bus
    adds nothing, so both partials cross the bus they share."""
    plan = route(accelerator, (3, 4), (2,), [DemandItem(0, (0, 1), WHOLE, reduce=True)])
    assert plan.pairs == ((0, 0), (1, 0))
    assert shares(plan) == {
        "Bus(m3,m4,m5, bw=1000)": 2.0,
        "CL(m5, m2, bw=300)": 1.0,
    }


def test_partial_sums_bound_for_dram_are_reduce_scattered(tmp_path):
    """Four memories on a ring hold partial sums of one tensor for DRAM, which cannot add: each completes a quarter
    and writes it, so each ring link carries the quarters passing it rather than every partial."""
    links = "".join(
        f"  - {{type: link, memories: [m{a}, m{b}], bandwidth: {w}}}\n"
        for a, b, w in ((0, 1, 100), (1, 2, 100), (2, 3, 100), (3, 0, 100), (0, 4, 1000), (1, 4, 1000))
        + ((2, 4, 1000), (3, 4, 1000))
    )
    memories = {f"m{i}": [f"{i}.sram_128K_A"] for i in range(4)} | {f"w{i}": [f"{i}.sram_128K_W"] for i in range(4)}
    path = _hardware(
        tmp_path / "ring.yaml",
        "ring",
        dict.fromkeys(range(4), "testing_core1.yaml"),
        memories | {"m4": OFFCHIP},
        links + "  - {type: bus, memories: [w0, w1, w2, w3, m4], bandwidth: 100}\n",
    )
    plan = route(parse_accelerator(path), (0, 1, 2, 3), ("m4",), [DemandItem(0, (0, 1, 2, 3), WHOLE, reduce=True)])
    ring = {link: share for link, share in shares(plan).items() if ", m4," not in link}
    to_dram = {link: share for link, share in shares(plan).items() if ", m4," in link}
    assert sorted(to_dram.values()) == [0.25] * 4
    assert sum(ring.values()) == pytest.approx(4 * 0.75)
    assert max(ring.values()) <= 0.5


def _plans(accelerator: Accelerator, sources, targets, demand, alternatives: int):
    return accelerator.communication_manager.get_possible_transfer_plan(
        tuple(_memory(accelerator, end, OUTPUT) for end in sources),
        tuple(_memory(accelerator, end, input_role(1)) for end in targets),
        tuple(demand),
        16,
        lambda one, other: False,
        alternatives,
    )


def _parallel(tmp_path: Path) -> Accelerator:
    """Two cores whose activation memories are joined by two equal links, beside a slow bus."""
    path = _hardware(
        tmp_path / "parallel.yaml",
        "parallel",
        dict.fromkeys(range(2), "testing_core1.yaml"),
        TWO_CORES_MEMORIES,
        "[{type: link, memories: [m0, m1], bandwidth: 64}, {type: link, memories: [m0, m1], bandwidth: 64},"
        " {type: bus, memories: [m0, m1, w0, w1, m2], bandwidth: 8}]",
    )
    return parse_accelerator(path)


def test_a_transfer_is_offered_each_of_two_equally_fast_connections(tmp_path):
    """With two equal links between the same memories, the alternatives use one link each."""
    plans = _plans(_parallel(tmp_path), (0,), (1,), [DemandItem(0, (0,), WHOLE)], 3)
    assert sorted(str(link) for plan in plans for link in plan.links_used) == [
        "CL#1(m0, m1, bw=64)",
        "CL(m0, m1, bw=64)",
    ]
    assert all(share == 1.0 for plan in plans for _, share in plan.link_shares)


def test_routes_over_the_same_links_that_share_them_differently_are_each_offered(tmp_path):
    """Two parts from one memory take one link each; the next alternative swaps them, which loads the links
    differently, so it is offered too."""
    demand = [DemandItem(0, (0,), ((0, 11), (0, 15))), DemandItem(0, (1,), ((12, 15), (0, 15)))]
    plans = _plans(_parallel(tmp_path), (0, 0), (1,), demand, 3)
    assert len({frozenset(plan.links_used) for plan in plans}) == 1
    assert sorted(tuple(sorted(shares(plan).values())) for plan in plans) == [(0.25, 0.75)] * 2
    assert shares(plans[0]) != shares(plans[1])


def test_a_transfer_with_one_way_there_is_offered_one_route(accelerator):
    """Cores reachable only over their cluster's bus get one route."""
    assert len(_plans(accelerator, (0,), (1,), [DemandItem(0, (0,), WHOLE)], 3)) == 1


def test_the_first_alternative_is_the_fastest_route(accelerator):
    demand = [DemandItem(0, (0,), HALF_A), DemandItem(1, (0,), HALF_B)]
    (first, *_) = _plans(accelerator, ("m6",), (0, 3), demand, 3)
    assert first == route(accelerator, ("m6",), (0, 3), demand)


@pytest.mark.slow
def test_more_routes_never_make_a_solve_slower(tmp_path):
    """The routes offered with alternatives include the one offered without, so the solve can only do better."""
    latencies = [  # simba's mesh offers several equally fast routes between most chiplets
        evaluate_mapping(
            "stream/inputs/examples/hardware/simba.yaml",
            build_block("attention", heads=2, seq=32, d_head=32),
            str(tmp_path / str(n)),
            options=SolveOptions(artifacts=False, route_alternatives=n),
        ).cycles
        for n in (1, 3)
    ]
    assert latencies[1] <= latencies[0]


def _solved(hardware: Path, tmp_path: Path, name_of: dict[str, str]) -> tuple[float, float, list]:
    """The latency, energy and routes of the two-convolution workload on ``hardware``, its memories named back."""
    context = evaluate_mapping(
        str(hardware),
        "stream/inputs/testing/workload/2conv_1_8_32_32_16_32_3.onnx",
        str(tmp_path / hardware.stem),
        options=SolveOptions(artifacts=False),
    ).context
    energy = sum(group["performance"]["energy"]["total_pj"] for group in context.get("group_allocations").values())

    def names(memories) -> tuple[str, ...]:
        return tuple(name_of[memory.name] for memory in memories)

    def route(plan: MulticastPathPlan) -> tuple:
        links = sorted((names(link.memories), link.parallel) for link in plan.links_used)
        return names(plan.sources), names(plan.targets), links

    routes = sorted((tr.name, *route(plan)) for tr, plan in context.get("allocation").solution.transfer_routes.items())
    return context.get("total_latency"), energy, routes


def test_renaming_the_memories_changes_neither_the_solve_nor_its_routes(tmp_path):
    """Equal routes are told apart by the cores viewing their memories, as before memories had names, not by name."""
    source = Path("stream/inputs/examples/hardware/meta_prototype_dual_core_simd_offchip.yaml")
    shutil.copytree(source.parent / "cores", tmp_path / "cores")
    data = yaml.safe_load(source.read_text())
    names = sorted(data["memories"])
    renamed = {name: f"m{len(names) - i}" for i, name in enumerate(names)}
    data["memories"] = {renamed[name]: memory for name, memory in data["memories"].items()}
    for connection in data["connections"]:
        connection["ends"] = {renamed[name]: ports for name, ports in connection["ends"].items()}
    shutil.copy(source, tmp_path / "original.yaml")
    (tmp_path / "renamed.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
    original = _solved(tmp_path / "original.yaml", tmp_path, {name: name for name in names})
    assert _solved(tmp_path / "renamed.yaml", tmp_path, {new: old for old, new in renamed.items()}) == original
