import numpy as np
import networkx as nx
from networkx import DiGraph

from stream.classes.hardware.architecture.communication_link import CommunicationLink
from zigzag.classes.hardware.architecture.core import Core

def get_2d_mesh(cores, nb_rows, nb_cols, bandwidth, unit_energy_cost, pooling_core=None, simd_core=None, offchip_core=None):
    """Return a 2D mesh graph of the cores where each core is connected to its N, E, S, W neighbour.
    We build the mesh by iterating through the row and then moving to the next column.
    Each connection between two cores includes two links, one in each direction, each with specified bandwidth.
    Thus there are a total of ((nb_cols-1)*2*nb_rows + (nb_rows-1)*2*nb_cols) links in the noc.
    If a pooling_core is provided, it is added with two directional links with each core, one in each direction.
    Thus, 2*nb_rows*nb_cols more links are added.
    If an offchip_core is provided, it is added with two directional links with each core, one in each direction.
    Thus, 2*nb_rows*nb_cols (+2 if a pooling core is present)

    Args:
        cores (list): list of core objects
        nb_rows (int): the number of rows in the 2D mesh
        nb_cols (int): the number of columns in the 2D mesh
        bandwidth (int): bandwidth of each created directional link in bits per clock cycle
        unit_energy_cost (float): The unit energy cost of having a communication-link active. This does not include the involved memory read/writes.
        pooling_core (Core, optional): If provided, the pooling core that is added.
        simd_core (Core, optional): If provided, the simd core that is added.
        offchip_core (Core, optional): If provided, the offchip core that is added.
        offchip_bandwidth (int, optional): If offchip_core is provided, this is the 
    """

    cores_array = np.asarray(cores).reshape((nb_rows, nb_cols), order='F')
    
    edges = []
    # Horizontal edges
    for row in cores_array:
        # From left to right
        pairs = zip(row, row[1:])
        for pair in pairs:
            (sender, receiver) = pair
            edges.append((sender, receiver, {'cl': CommunicationLink(sender, receiver, bandwidth, unit_energy_cost)}))
        # From right to left
        pairs = zip(reversed(row), reversed(row[:-1]))
        for pair in pairs:
            (sender, receiver) = pair
            edges.append((sender, receiver, {'cl': CommunicationLink(sender, receiver, bandwidth, unit_energy_cost)}))
    # Vertical edges
    for col in cores_array.T:
        # From top to bottom (bottom is highest idx)
        pairs = zip(col, col[1:])
        for pair in pairs:
            (sender, receiver) = pair
            edges.append((sender, receiver, {'cl': CommunicationLink(sender, receiver, bandwidth, unit_energy_cost)}))
        # From bottom to top
        pairs = zip(reversed(col), reversed(col[:-1]))
        for pair in pairs:
            (sender, receiver) = pair
            edges.append((sender, receiver, {'cl': CommunicationLink(sender, receiver, bandwidth, unit_energy_cost)}))

    # If there is a pooling core, also add two edges from each core to the pooling core: one in each direction
    if pooling_core:
        if not isinstance(pooling_core, Core):
            raise ValueError("The given pooling_core is not a Core object.")
        for core in cores:
            edges.append((core, pooling_core, {'cl': CommunicationLink(core, pooling_core, bandwidth, unit_energy_cost)}))
            edges.append((pooling_core, core, {'cl': CommunicationLink(pooling_core, core, bandwidth, unit_energy_cost)}))

    # If there is a simd core, also add two edges from each core to the pooling core: one in each direction
    # For now, assume the simd operations come for free, so bandwidth is infinite and unit energy cost is 0
    simd_bandwidth = float('inf')
    simd_unit_energy_cost = 0
    if simd_core:
        if not isinstance(simd_core, Core):
            raise ValueError("The given simd_core is not a Core object.")
        for core in cores:
            edges.append((core, simd_core, {'cl': CommunicationLink(core, simd_core, simd_bandwidth, simd_unit_energy_cost)}))
            edges.append((simd_core, core, {'cl': CommunicationLink(simd_core, core, simd_bandwidth, simd_unit_energy_cost)}))
        # If there is a pooling core, also add two edges from/to the pooling core
        if pooling_core:
            edges.append((pooling_core, simd_core, {'cl': CommunicationLink(pooling_core, simd_core, simd_bandwidth, simd_unit_energy_cost)}))
            edges.append((simd_core, pooling_core, {'cl': CommunicationLink(simd_core, pooling_core, simd_bandwidth, simd_unit_energy_cost)}))

    # If there is an offchip core, add a single link for writing to and a single link for reading from the offchip
    if offchip_core:
        offchip_read_bandwidth = offchip_core.mem_r_bw_dict['O'][0]
        offchip_write_bandwidth = offchip_core.mem_w_bw_dict['O'][0]
        # if the offchip core has only one port
        if len(offchip_core.mem_hierarchy_dict['O'][0].port_list) == 1:
            to_offchip_link = CommunicationLink(offchip_core, "Any", offchip_write_bandwidth, unit_energy_cost, bidirectional=True)
            from_offchip_link = to_offchip_link
        # if the offchip core has more than one port
        else:
            to_offchip_link = CommunicationLink("Any", offchip_core, offchip_write_bandwidth, unit_energy_cost)
            from_offchip_link = CommunicationLink(offchip_core, "Any", offchip_read_bandwidth, unit_energy_cost)
        if not isinstance(offchip_core, Core):
            raise ValueError("The given offchip_core is not a Core object.")
        for core in cores:
            edges.append((core, offchip_core, {'cl': to_offchip_link}))
            edges.append((offchip_core, core, {'cl': from_offchip_link}))
        if pooling_core:
            edges.append((pooling_core, offchip_core, {'cl': to_offchip_link}))
            edges.append((offchip_core, pooling_core, {'cl': from_offchip_link}))
        if simd_core:
            edges.append((simd_core, offchip_core, {'cl': to_offchip_link}))
            edges.append((offchip_core, simd_core, {'cl': from_offchip_link}))

    # Build the graph using the constructed list of edges
    H = DiGraph(edges)
    
    return H
