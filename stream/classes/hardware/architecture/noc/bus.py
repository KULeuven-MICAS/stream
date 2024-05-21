import numpy as np
import networkx as nx
from networkx import DiGraph

from stream.classes.hardware.architecture.noc.communication_link import CommunicationLink
from zigzag.hardware.architecture.Core import Core
from zigzag.datatypes import Constants

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


def get_bus(
    cores,
    bandwidth,
    unit_energy_cost,
    pooling_core=None,
    simd_core=None,
    offchip_core=None,
):
    """Return a graph of the cores where each core is connected to a single bus.

    Args:
        cores (list): list of core objects
        bandwidth (int): bandwidth of the communication bus
        unit_energy_cost (float): The unit energy cost of having a communication-link active. This does not include the involved memory read/writes.
        pooling_core (Core, optional): If provided, the pooling core that is added.
        simd_core (Core, optional): If provided, the simd core that is added.
        offchip_core (Core, optional): If provided, the offchip core that is added.
        offchip_bandwidth (int, optional): If offchip_core is provided, this is the
    """
    bus = CommunicationLink("Any", "Any", bandwidth, unit_energy_cost)

    edges = []
    pairs = [(a, b) for idx, a in enumerate(cores) for b in cores[idx + 1:]]
    for pair in pairs:
        (sender, receiver) = pair
        if not have_shared_memory(sender, receiver):
            edges.append((sender, receiver, {"cl": bus}))
            edges.append((receiver, sender, {"cl": bus}))

    # If there is a pooling core, also add two edges from each core to the pooling core: one in each direction
    if pooling_core:
        if not isinstance(pooling_core, Core):
            raise ValueError("The given pooling_core is not a Core object.")
        for core in cores:
            if not have_shared_memory(core, pooling_core):
                edges.append(
                    (
                        core,
                        pooling_core,
                        {
                            "cl": CommunicationLink(
                                core, pooling_core, bandwidth, unit_energy_cost
                            )
                        },
                    )
                )
                edges.append(
                    (
                        pooling_core,
                        core,
                        {
                            "cl": CommunicationLink(
                                pooling_core, core, bandwidth, unit_energy_cost
                            )
                        },
                    )
                )

    # If there is a simd core, also add two edges from each core to the pooling core: one in each direction
    # For now, assume the simd operations come for free, so bandwidth is infinite and unit energy cost is 0
    simd_bandwidth = float("inf")
    simd_unit_energy_cost = 0
    if simd_core:
        if not isinstance(simd_core, Core):
            raise ValueError("The given simd_core is not a Core object.")
        for core in cores:
            if not have_shared_memory(core, simd_core):
                edges.append(
                    (
                        core,
                        simd_core,
                        {
                            "cl": bus
                        },
                    )
                )
                edges.append(
                    (
                        simd_core,
                        core,
                        {
                            "cl": bus
                        },
                    )
                )
        # If there is a pooling core, also add two edges from/to the pooling core
        if pooling_core:
            if not have_shared_memory(pooling_core, simd_core):
                edges.append(
                    (
                        pooling_core,
                        simd_core,
                        {
                            "cl": bus
                        },
                    )
                )
                edges.append(
                    (
                        simd_core,
                        pooling_core,
                        {
                            "cl": bus
                        },
                    )
                )

    # If there is an offchip core, add a single link for writing to and a single link for reading from the offchip
    if offchip_core:
        output_operand = Constants.OUTPUT_MEM_OP
        offchip_read_bandwidth = offchip_core.mem_r_bw_dict[output_operand][0]
        offchip_write_bandwidth = offchip_core.mem_w_bw_dict[output_operand][0]
        # if the offchip core has only one port
        if len(offchip_core.mem_hierarchy_dict[output_operand][0].port_list) == 1:
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
        if pooling_core:
            edges.append((pooling_core, offchip_core, {"cl": to_offchip_link}))
            edges.append((offchip_core, pooling_core, {"cl": from_offchip_link}))
        if simd_core:
            edges.append((simd_core, offchip_core, {"cl": to_offchip_link}))
            edges.append((offchip_core, simd_core, {"cl": from_offchip_link}))

    # Build the graph using the constructed list of edges
    H = DiGraph(edges)

    return H
