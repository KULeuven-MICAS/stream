from typing import TYPE_CHECKING

from zigzag.datatypes import MemoryOperand

if TYPE_CHECKING:
    from stream.hardware.architecture.accelerator import Accelerator


from typing import TYPE_CHECKING, Any

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


def get_core_capacities(accelerator: "Accelerator", mem_op: MemoryOperand, core_ids: list[int]):
    core_capacities: dict[str, float] = {}
    for core_id in core_ids:
        core_name = f"Core {core_id}"
        core = accelerator.get_core(core_id)
        top_instance = accelerator.get_top_instance_of_core(core, mem_op)
        core_capacities[core_name] = top_instance.size
    return core_capacities
