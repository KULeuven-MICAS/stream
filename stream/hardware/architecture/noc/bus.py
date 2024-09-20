from zigzag.datatypes import Constants
from zigzag.hardware.architecture.core import Core

from stream.hardware.architecture.accelerator import CoreGraph
from stream.hardware.architecture.noc.communication_link import CommunicationLink
from stream.hardware.architecture.utils import get_bidirectional_edges


def get_bus(
    cores: list[Core],
    bandwidth: int,
    unit_energy_cost: float,
    pooling_core: Core | None = None,
    simd_core: Core | None = None,
    offchip_core: Core | None = None,
):
    """Return a graph of the cores where each core is connected to a single bus.

    Args:
        cores: list of core objects
        bandwidth: bandwidth of the communication bus
        unit_energy_cost: The unit energy cost of having a communication-link active. This does not include the
        involved memory read/writes.
        pooling_core: If provided, the pooling core that is added.
        simd_core: If provided, the simd core that is added.
        offchip_core: If provided, the offchip core that is added.
        offchip_bandwidth: If offchip_core is provided, this is the
    """

    def get_edges_bus(core_a: Core, core_b: Core):
        return get_bidirectional_edges(core_a, core_b, bandwidth, unit_energy_cost, link_type="bus")

    def get_edges_link(core_a: Core, core_b: Core):
        return get_bidirectional_edges(core_a, core_b, bandwidth, unit_energy_cost, link_type="link")

    edges: list[tuple[Core, Core, dict[str, CommunicationLink]]] = []
    pairs = [(a, b) for idx, a in enumerate(cores) for b in cores[idx + 1 :]]

    for core_a, core_b in pairs:
        edges += get_edges_bus(core_a, core_b)

    # If there is a pooling core, also add two edges from each core to the pooling core: one in each direction
    if pooling_core:
        for core in cores:
            edges += get_edges_link(core, pooling_core)

    # If there is a simd core, also add two edges from each core to the pooling core: one in each direction
    if simd_core:
        for core in cores:
            edges += get_edges_bus(core, simd_core)

    # If there is a pooling core, also add two edges from/to the pooling core
    if pooling_core and simd_core:
        edges += get_edges_bus(pooling_core, simd_core)

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
            to_offchip_link = CommunicationLink("Any", offchip_core, offchip_write_bandwidth, unit_energy_cost)
            from_offchip_link = CommunicationLink(offchip_core, "Any", offchip_read_bandwidth, unit_energy_cost)
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
    return CoreGraph(edges)
