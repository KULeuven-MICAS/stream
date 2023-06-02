import numpy as np
import networkx as nx
from networkx import DiGraph

from stream.classes.hardware.architecture.communication_link import CommunicationLink
from zigzag.classes.hardware.architecture.core import Core


def have_shared_memory(a, b):
    """Returns True if core a and core b have a shared top level memory

    Args:
        a (Core): First core
        b (Core): Second core
    """
    top_level_memory_instances_a = set(
        [
            level.memory_instance
            for level, out_degree in a.memory_hierarchy.out_degree()
            if out_degree == 0
        ]
    )
    top_level_memory_instances_b = set(
        [
            level.memory_instance
            for level, out_degree in b.memory_hierarchy.out_degree()
            if out_degree == 0
        ]
    )
    for memory_instance_a in top_level_memory_instances_a:
        if memory_instance_a in top_level_memory_instances_b:
            return True
    return False


def shared_memories_with_offchip(
    cores,
    offchip_core,
    unit_energy_cost,
):
    """Return a graph of the cores where each core is connected to the offchip core.
    No edges are inserted between cores themselves because they are assumed to be connected through shared memories in their core definition.

    Args:
        cores (list): Core objects.
        offchip_core (Core): The offchip core that is added.
        unit_energy_cost (float): The unit energy cost of having a communication-link active. This does not include the involved memory read/writes

    """
    edges = []

    # No edges between the cores because they are connected through shared memories

    # If there is an offchip core, add a single link for writing to and a single link for reading from the offchip
    if offchip_core:
        offchip_read_bandwidth = offchip_core.mem_r_bw_dict["O"][0]
        offchip_write_bandwidth = offchip_core.mem_w_bw_dict["O"][0]
        # if the offchip core has only one port
        if len(offchip_core.mem_hierarchy_dict["O"][0].port_list) == 1:
            to_offchip_link = CommunicationLink(
                offchip_core,
                "Any",
                offchip_write_bandwidth,
                unit_energy_cost,
                bidirectional=True,
            )
            from_offchip_link = to_offchip_link
        # if the offchip core has more than one port
        else:
            to_offchip_link = CommunicationLink(
                "Any", offchip_core, offchip_write_bandwidth, unit_energy_cost
            )
            from_offchip_link = CommunicationLink(
                offchip_core, "Any", offchip_read_bandwidth, unit_energy_cost
            )
        if not isinstance(offchip_core, Core):
            raise ValueError("The given offchip_core is not a Core object.")
        for core in cores:
            edges.append((core, offchip_core, {"cl": to_offchip_link}))
            edges.append((offchip_core, core, {"cl": from_offchip_link}))

    # Build the graph using the constructed list of edges
    H = DiGraph(edges)

    return H
