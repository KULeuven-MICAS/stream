import numpy as np
from zigzag.datatypes import Constants
from zigzag.hardware.architecture.core import Core

from stream.hardware.architecture.accelerator import CoreGraph
from stream.hardware.architecture.noc.communication_link import CommunicationLink
from stream.hardware.architecture.utils import get_bidirectional_edges


def get_2d_mesh(
    cores: list[Core],
    nb_rows: int,
    nb_cols: int,
    bandwidth: int,
    unit_energy_cost: float,
    pooling_core: Core | None = None,
    simd_core: Core | None = None,
    offchip_core: Core | None = None,
):
    """Return a 2D mesh graph of the cores where each core is connected to its N, E, S, W neighbor.
    We build the mesh by iterating through the row and then moving to the next column.
    Each connection between two cores includes two links, one in each direction, each with specified bandwidth.
    Thus there are a total of ((nb_cols-1)*2*nb_rows + (nb_rows-1)*2*nb_cols) links in the noc.
    If a pooling_core is provided, it is added with two directional links with each core, one in each direction.
    Thus, 2*nb_rows*nb_cols more links are added.
    If an offchip_core is provided, it is added with two directional links with each core, one in each direction.
    Thus, 2*nb_rows*nb_cols (+2 if a pooling core is present)

    Args:
        cores: list of core objects
        nb_rows: the number of rows in the 2D mesh
        nb_cols: the number of columns in the 2D mesh
        bandwidth: bandwidth of each created directional link in bits per clock cycle
        unit_energy_cost: The unit energy cost of having a communication-link active. This does not include the
        involved memory read/writes.
        pooling_core: If provided, the pooling core that is added.
        simd_core: If provided, the simd core that is added.
        offchip_core: If provided, the offchip core that is added.
        offchip_bandwidth: If offchip_core is provided, this is the
    """

    def get_edges(core_a: Core, core_b: Core):
        return get_bidirectional_edges(core_a, core_b, bandwidth, unit_energy_cost, link_type="link")

    def get_edges_simd(core_a: Core, core_b: Core):
        simd_bandwidth = float("inf")
        simd_unit_energy_cost = 0
        return get_bidirectional_edges(core_a, core_b, simd_bandwidth, simd_unit_energy_cost, link_type="link")

    cores_array = np.asarray(cores).reshape((nb_rows, nb_cols), order="F")
    edges: list[tuple[Core, Core, dict[str, CommunicationLink]]] = []

    # Horizontal edges
    for row in cores_array:
        pairs = zip(row, row[1:])
        for west_core, east_core in pairs:
            edges += get_edges(west_core, east_core)

    # Vertical edges
    for col in cores_array.T:
        pairs = zip(col, col[1:])
        for north_core, south_core in pairs:
            edges += get_edges(north_core, south_core)

    # If there is a pooling core, also add two edges from each core to the pooling core: one in each direction
    if pooling_core:
        for core in cores:
            edges += get_edges(core, pooling_core)

    # If there is a simd core, also add two edges from each core to the pooling core: one in each direction
    # For now, assume the simd operations come for free, so bandwidth is infinite and unit energy cost is 0

    if simd_core:
        for core in cores:
            edges += get_edges_simd(core, simd_core)

    # If there is a pooling core, also add two edges from/to the pooling core
    if simd_core and pooling_core:
        edges += get_edges_simd(pooling_core, simd_core)

    # If there is an offchip core, add a single link for writing to and a single link for reading from the offchip
    if offchip_core:
        offchip_read_bandwidth = offchip_core.mem_r_bw_dict[Constants.OUTPUT_MEM_OP][0]
        offchip_write_bandwidth = offchip_core.mem_w_bw_dict[Constants.OUTPUT_MEM_OP][0]
        # if the offchip core has only one port
        if len(offchip_core.mem_hierarchy_dict[Constants.OUTPUT_MEM_OP][0].port_list) == 1:
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
