from typing import Any, Literal

from zigzag.hardware.architecture.core import Core

from stream.hardware.architecture.accelerator import Accelerator
from stream.hardware.architecture.noc.communication_link import CommunicationLink


def intersections(a: list[Any], b: list[Any]):
    """Get the intersections of two lists of ranges.
    https://stackoverflow.com/questions/40367461/intersection-of-two-lists-of-ranges-in-python

    Args:
        a (list): The first list.
        b (list): The second list.

    Returns:
        list: The intersections between the two lists.
    """
    ranges: list[Any] = []
    i = j = 0
    while i < len(a) and j < len(b):
        a_left, a_right = a[i]
        b_left, b_right = b[j]

        if a_right < b_right:
            i += 1
        else:
            j += 1

        if a_right >= b_left and b_right >= a_left:
            end_pts = sorted([a_left, a_right, b_left, b_right])
            middle = (end_pts[1], end_pts[2])
            ranges.append(middle)

    ri = 0
    while ri < len(ranges) - 1:
        if ranges[ri][1] == ranges[ri + 1][0]:
            ranges[ri : ri + 2] = [(ranges[ri][0], ranges[ri + 1][1])]

        ri += 1

    return ranges


def get_core_capacities(accelerator: Accelerator, mem_op: str, core_ids: list[int]):
    core_capacities = {}
    for core_id in core_ids:
        core_name = f"Core {core_id}"
        core = accelerator.get_core(core_id)
        top_instance = accelerator.get_top_instance_of_core(core, mem_op)
        core_capacities[core_name] = top_instance.size
    return core_capacities


def have_shared_memory(a: Core, b: Core):
    """Returns True if core a and core b have a shared top level memory"""
    top_level_memory_instances_a = set(
        [level.memory_instance for level, out_degree in a.memory_hierarchy.out_degree() if out_degree == 0]
    )
    top_level_memory_instances_b = set(
        [level.memory_instance for level, out_degree in b.memory_hierarchy.out_degree() if out_degree == 0]
    )
    for memory_instance_a in top_level_memory_instances_a:
        if memory_instance_a in top_level_memory_instances_b:
            return True
    return False


def get_bidirectional_edges(
    core_a: Core,
    core_b: Core,
    bandwidth: float,
    unit_energy_cost: float,
    link_type: Literal["bus"] | Literal["link"],
) -> list[tuple[Core, Core, dict[str, CommunicationLink]]]:
    """Create a list with two edges: from A to B and B to A."""
    bus = CommunicationLink("Any", "Any", bandwidth, unit_energy_cost)
    link_a_to_b = CommunicationLink(core_a, core_b, bandwidth, unit_energy_cost)
    link_b_to_a = CommunicationLink(core_b, core_a, bandwidth, unit_energy_cost)

    if have_shared_memory(core_a, core_b):
        # No edge if the cores have a shared memory
        return []

    return [
        #  A -> B
        (
            core_a,
            core_b,
            {"cl": bus if link_type == "bus" else link_a_to_b},
        ),
        # B -> A
        (
            core_b,
            core_a,
            {"cl": bus if link_type == "bus" else link_b_to_a},
        ),
    ]
