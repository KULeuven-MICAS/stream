"""Routing a transfer: each target from the nearest source holding its data, over one Steiner tree per source."""

from __future__ import annotations

from typing import Any

import pytest
import yaml

from stream.api import SolveOptions, evaluate_mapping
from stream.cost_model.communication_manager import DemandItem, box_bits
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


def _hardware(path, name: str, cores: dict[int, str], memories: dict[str, Any], connections: str) -> str:
    """Write a description of ``cores`` (testing core files) at ``path``: ``memories`` maps a name to :data:`OFFCHIP`
    or to the views of an on-chip memory, whose cores all reach its one read and one write port, and ``connections``
    is the YAML list of its links and buses, each by the ``memories`` it joins, through every port of each."""

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
    return str(path)


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
"""Two clusters of three cores on a bus each, whose hubs 2 and 5 share a bus with off-chip memory 6 and a link; every
core has a memory of its own, m<core>, and a weights memory w<core> that only off-chip memory reaches."""

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
    """Route between the memories ``sources`` write their output in and ``targets`` read their first input in, each a
    core or a memory's name."""
    return accelerator.communication_manager.route(
        tuple(_memory(accelerator, end, OUTPUT) for end in sources),
        tuple(_memory(accelerator, end, input_role(1)) for end in targets),
        tuple(demand),
        16,
        shares_memory,
    )


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
        {
            "m0": ["0.sram_128K_A"],
            "m1": ["1.sram_128K_A"],
            "m2": OFFCHIP,
            "w0": ["0.sram_128K_W"],
            "w1": ["1.sram_128K_W"],
        },
        "[{type: link, memories: [w0, m2], bandwidth: 1000}, {type: link, memories: [w1, m2], bandwidth: 1000},"
        " {type: link, memories: [m0, m2], bandwidth: 1000}, {type: link, memories: [m1, m2], bandwidth: 1000},"
        " {type: link, memories: [m0, m1], bandwidth: 10}]",
    )
    plan = route(parse_accelerator(path), (0,), (1,), [DemandItem(0, (0,), WHOLE)])
    assert shares(plan) == {"CL(m0, m1, bw=10)": 1.0}


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
    """Hub m2 reaching off-chip m6 over the bus it shares with hub m5, and m5 over their link: m5's side passes the
    link only, although m5 is on the bus too, so its port is charged the link's bits and not the bus's."""
    from stream.cost_model.communication_manager import MulticastPathPlan

    m2, m5, m6 = (accelerator.memories[f"m{i}"] for i in (2, 5, 6))
    links = {str(d["cl"]): d["cl"] for _, _, d in accelerator.network.edges(data=True)}
    bus, link = links["Bus(m2,m5,m6, bw=100)"], links["CL(m2, m5, bw=300)"]
    plan = MulticastPathPlan((m2,), (m5, m6), 2, (bus, link), ((0, 0), (0, 1)), hops=((m2, m6, bus), (m2, m5, link)))
    assert plan.links_at(m5, leaving=False) == (link,) and plan.links_at(m6, leaving=False) == (bus,)
    assert plan.links_at(m2, leaving=True) == (bus, link) and plan.links_at(m2, leaving=False) == ()
    without_hops = MulticastPathPlan((m2,), (m5, m6), 2, (bus, link), ((0, 0), (0, 1)))
    assert set(without_hops.links_at(m5, leaving=False)) == {bus, link}


def test_buses_of_the_same_width_are_distinct_links(accelerator):
    """Each cluster bus is its own link, so traffic on one does not count against the other."""
    buses = {link for _, _, data in accelerator.network.edges(data=True) if (link := data["cl"]).members}
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


def test_a_transfer_is_offered_each_of_two_equally_fast_connections(tmp_path):
    """Two links between the same memories: the first route takes one, the next, routed as if the first carried the
    transfer, the other, so that the solve can put two transfers of one slot on one each."""
    path = _hardware(
        tmp_path / "parallel.yaml",
        "parallel",
        dict.fromkeys(range(2), "testing_core1.yaml"),
        {
            "m0": ["0.sram_128K_A"],
            "m1": ["1.sram_128K_A"],
            "m2": OFFCHIP,
            "w0": ["0.sram_128K_W"],
            "w1": ["1.sram_128K_W"],
        },
        "[{type: link, memories: [m0, m1], bandwidth: 64}, {type: link, memories: [m0, m1], bandwidth: 64},"
        " {type: bus, memories: [m0, m1, w0, w1, m2], bandwidth: 8}]",
    )
    plans = _plans(parse_accelerator(path), (0,), (1,), [DemandItem(0, (0,), WHOLE)], 3)
    assert sorted(str(link) for plan in plans for link in plan.links_used) == [
        "CL#1(m0, m1, bw=64)",
        "CL(m0, m1, bw=64)",
    ]
    assert all(share == 1.0 for plan in plans for _, share in plan.link_shares)


def test_a_transfer_with_one_way_there_is_offered_one_route(accelerator):
    """Each cluster's bus is the only way between its cores: the alternatives all take it, so only one is kept."""
    assert len(_plans(accelerator, (0,), (1,), [DemandItem(0, (0,), WHOLE)], 3)) == 1


def test_the_first_alternative_is_the_fastest_route(accelerator):
    demand = [DemandItem(0, (0,), HALF_A), DemandItem(1, (0,), HALF_B)]
    (first, *_) = _plans(accelerator, ("m6",), (0, 3), demand, 3)
    assert first == route(accelerator, ("m6",), (0, 3), demand)
    assert shares(first) == shares(route(accelerator, ("m6",), (0, 3), demand))


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
