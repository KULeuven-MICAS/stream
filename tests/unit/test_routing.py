"""Routing a transfer: each target from the nearest source holding its data, over one Steiner tree per source."""

from __future__ import annotations

import pytest

from stream.api import SolveOptions, evaluate_mapping
from stream.cost_model.communication_manager import DemandItem, box_bits
from stream.hardware.architecture.accelerator import Accelerator
from stream.opt.allocation.constraint_optimization.space import communicating_pairs
from stream.stages.parsing.accelerator_parser import parse_accelerator

TWO_CLUSTERS = """
name: two-clusters
cores:
  0: testing_core1.yaml
  1: testing_core1.yaml
  2: testing_core1.yaml
  3: testing_core1.yaml
  4: testing_core1.yaml
  5: testing_core1.yaml
  6: offchip.yaml
offchip_core_id: 6
unit_energy_cost: 0
core_connectivity:
  - type: bus
    cores: [0, 1, 2]
    bandwidth: 1000
  - type: bus
    cores: [3, 4, 5]
    bandwidth: 1000
  - type: bus
    cores: [2, 5, 6]
    bandwidth: 100
  - type: link
    cores: [2, 5]
    bandwidth: 300
"""
"""Two clusters of three cores on a bus each, whose hubs 2 and 5 share a bus with off-chip memory 6 and a link."""

HALF_A, HALF_B = ((0, 7), (0, 15)), ((8, 15), (0, 15))
WHOLE = ((0, 15), (0, 15))


@pytest.fixture(scope="module")
def accelerator(tmp_path_factory: pytest.TempPathFactory) -> Accelerator:
    path = tmp_path_factory.mktemp("hw") / "two_clusters.yaml"
    path.write_text(TWO_CLUSTERS)
    return parse_accelerator(str(path))


def route(accelerator: Accelerator, sources, targets, demand, shares_memory=lambda one, other: False):
    core = accelerator.get_core
    return accelerator.communication_manager.route(
        tuple(core(i) for i in sources), tuple(core(i) for i in targets), tuple(demand), 16, shares_memory
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
    assert shares(plan) == {"Bus(0,1,2, bw=1000)": 1.0, "Bus(3,4,5, bw=1000)": 1.0}


def test_a_link_is_charged_only_the_part_that_is_not_already_in_its_target(tmp_path):
    """Two cores sharing a memory read twelve rows, eight held there and four by a core across a link: the link
    carries those four, a third of what the transfer delivers."""
    path = tmp_path / "local.yaml"
    path.write_text(
        "name: local\ncores:\n  0: shared_testing_core1.yaml\n  1: shared_testing_core2.yaml\n"
        "  2: testing_core1.yaml\n  3: offchip.yaml\noffchip_core_id: 3\nunit_energy_cost: 0\ncore_connectivity:\n"
        "  - {type: link, cores: [0, 2], bandwidth: 64}\n  - {type: link, cores: [0, 3], bandwidth: 64}\n"
        "core_memory_sharing:\n  - 0, 1\n"
    )
    quarter = ((0, 3), (0, 15))
    demand = [DemandItem(0, (0,), HALF_A), DemandItem(1, (0,), ((8, 11), (0, 15))), DemandItem(1, (1,), quarter)]
    plan = route(parse_accelerator(str(path)), (0, 2), (0, 1), demand)
    assert shares(plan) == {"CL(Core(2, zigzag.compute), Core(0, zigzag.compute), bw=64)": round(64 / 192, 3)}


def test_a_broadcast_crosses_each_link_once(accelerator):
    """The same data to both cores of a cluster is sent over its bus once."""
    plan = route(accelerator, (2,), (0, 1), [DemandItem(0, (0,), WHOLE), DemandItem(1, (0,), WHOLE)])
    assert shares(plan) == {"Bus(0,1,2, bw=1000)": 1.0}


def test_a_scatter_charges_each_link_only_the_part_crossing_it(accelerator):
    """Off-chip memory sends one half to each cluster: both halves leave over the hub bus, one per cluster bus."""
    plan = route(accelerator, (6,), (0, 3), [DemandItem(0, (0,), HALF_A), DemandItem(1, (0,), HALF_B)])
    assert shares(plan) == {"Bus(2,5,6, bw=100)": 1.0, "Bus(0,1,2, bw=1000)": 0.5, "Bus(3,4,5, bw=1000)": 0.5}


def test_a_hop_between_clusters_takes_the_faster_link(accelerator):
    """From one hub to the other, the 300-wide link beats the 100-wide hub bus."""
    plan = route(accelerator, (2,), (5,), [DemandItem(0, (0,), WHOLE)])
    assert shares(plan) == {"CL(Core(2, zigzag.compute), Core(5, zigzag.compute), bw=300)": 1.0}


def test_a_target_sharing_memory_with_its_source_moves_nothing(accelerator):
    """The route still reaches the target, so the transfer is not mistaken for one within a single core."""
    plan = route(accelerator, (2,), (0,), [DemandItem(0, (0,), WHOLE)], shares_memory=lambda one, other: True)
    assert plan.pairs == ((0, 0),)
    assert plan.links_used and all(share == 0 for _, share in plan.link_shares)


def test_off_chip_memory_never_relays_data_between_cores(tmp_path):
    """Two cores with wide links to DRAM and a narrow one between them: the data takes the narrow link, since a
    route cannot pass through DRAM."""
    path = tmp_path / "relay.yaml"
    path.write_text(
        "name: relay\ncores:\n  0: testing_core1.yaml\n  1: testing_core1.yaml\n  2: offchip.yaml\n"
        "offchip_core_id: 2\nunit_energy_cost: 0\ncore_connectivity:\n"
        "  - {type: link, cores: [0, 2], bandwidth: 1000}\n  - {type: link, cores: [1, 2], bandwidth: 1000}\n"
        "  - {type: link, cores: [0, 1], bandwidth: 10}\n"
    )
    plan = route(parse_accelerator(str(path)), (0,), (1,), [DemandItem(0, (0,), WHOLE)])
    assert shares(plan) == {"CL(Core(0, zigzag.compute), Core(1, zigzag.compute), bw=10)": 1.0}


def test_cores_in_one_memory_need_no_route(tmp_path):
    """Two cores sharing their memory, reachable from each other only through DRAM: the data is already there."""
    path = tmp_path / "shared.yaml"
    path.write_text(
        "name: shared\ncores:\n  0: shared_testing_core1.yaml\n  1: shared_testing_core2.yaml\n  2: offchip.yaml\n"
        "offchip_core_id: 2\nunit_energy_cost: 0\ncore_connectivity:\n"
        "  - {type: link, cores: [0, 2], bandwidth: 64}\n  - {type: link, cores: [1, 2], bandwidth: 64}\n"
        "core_memory_sharing:\n  - 0, 1\n"
    )
    plan = route(parse_accelerator(str(path)), (0,), (1,), [DemandItem(0, (0,), WHOLE)])
    assert plan.pairs == ((0, 0),)
    assert not plan.links_used


def test_buses_of_the_same_width_are_distinct_links(accelerator):
    """Each cluster bus is its own link, so traffic on one does not count against the other."""
    buses = {link for _, _, data in accelerator.cores.edges(data=True) if (link := data["cl"]).members}
    assert sorted(str(bus) for bus in buses) == ["Bus(0,1,2, bw=1000)", "Bus(2,5,6, bw=100)", "Bus(3,4,5, bw=1000)"]


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
        "Bus(3,4,5, bw=1000)": 2.0,
        "CL(Core(5, zigzag.compute), Core(2, zigzag.compute), bw=300)": 1.0,
    }


def test_partial_sums_bound_for_dram_are_reduce_scattered(tmp_path):
    """Four memories on a ring hold partial sums of one tensor for DRAM, which cannot add: each completes a quarter
    and writes it, so each ring link carries the quarters passing it rather than every partial."""
    path = tmp_path / "ring.yaml"
    links = "".join(
        f"  - {{type: link, cores: [{a}, {b}], bandwidth: {w}}}\n"
        for a, b, w in ((0, 1, 100), (1, 2, 100), (2, 3, 100), (3, 0, 100), (0, 4, 1000), (1, 4, 1000))
        + ((2, 4, 1000), (3, 4, 1000))
    )
    path.write_text(
        "name: ring\ncores:\n"
        + "".join(f"  {i}: testing_core1.yaml\n" for i in range(4))
        + "  4: offchip.yaml\noffchip_core_id: 4\nunit_energy_cost: 0\ncore_connectivity:\n"
        + links
    )
    plan = route(parse_accelerator(str(path)), (0, 1, 2, 3), (4,), [DemandItem(0, (0, 1, 2, 3), WHOLE, reduce=True)])
    ring = {link: share for link, share in shares(plan).items() if ", Core(4," not in link}
    to_dram = {link: share for link, share in shares(plan).items() if ", Core(4," in link}
    assert sorted(to_dram.values()) == [0.25] * 4
    assert sum(ring.values()) == pytest.approx(4 * 0.75)
    assert max(ring.values()) <= 0.5
