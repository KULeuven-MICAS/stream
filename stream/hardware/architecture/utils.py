from typing import TYPE_CHECKING, Any

from zigzag.hardware.architecture.core import Core

if TYPE_CHECKING:
    from stream.hardware.architecture.accelerator import Accelerator


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


def get_core_capacities(accelerator: "Accelerator", mem_op: str, core_ids: list[int]):
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
