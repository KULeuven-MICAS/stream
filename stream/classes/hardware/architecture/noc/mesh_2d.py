import numpy as np
import networkx as nx
from networkx import DiGraph , MultiDiGraph

# from stream.classes.hardware.architecture.communication_link import CommunicationLink
# from zigzag.classes.hardware.architecture.core import Core

from stream.classes.hardware.architecture.noc.communication_link import CommunicationLink
from zigzag.datatypes import Constants
from stream.classes.hardware.architecture.stream_core import Core

# From the AIE-MLs perspective, the throughput of each of the loads and store is 256 bits per clock cycle.
core_to_core_bw = 256  # bandwidth of every link connecting two neighboring cores
core_to_mem_tile_bw = 32 * 6

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


def get_2d_mesh(
    cores: list[Core],
    nb_rows: int,
    nb_cols: int,
    bandwidth: int,  
    unit_energy_cost: float,
    pooling_core: Core | None = None,
    simd_core: Core | None = None,
    offchip_core: Core | None = None,
    parallel_links_flag: bool | None = False, # if this is True, the exploration will consider multiple parallel links
    use_shared_mem_flag: bool | None = True, # if this is True, the exploration will consider multiple parallel links
):
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
    ########### Beginning of the logic for adding the links representing the shared memory
    # At the moment there is a shared memory link in 4 directions

    offchip_read_channels_num = 2
    offchip_write_channels_num = 2
    memTile_read_channels_num = 2
    memTile_write_channels_num = 2

    cores_array = np.asarray(cores).reshape((nb_rows, nb_cols), order="C")
    edges = []
    # Horizontal edges
    for row in cores_array:
        # From left to right
        pairs = zip(row, row[1:])
        for pair in pairs:
            (sender, receiver) = pair
            if(sender.core_type == 1 or receiver.core_type == 1):  # skip memTile cores
                continue
            if use_shared_mem_flag:
                if not have_shared_memory(sender, receiver):
                    edges.append(
                        (
                            sender,
                            receiver,
                            {
                                "cl": CommunicationLink(
                                    sender, receiver, core_to_core_bw, unit_energy_cost
                                )
                            },
                        )
                    )

        # From right to left
        pairs = zip(reversed(row), reversed(row[:-1]))
        for pair in pairs:
            (sender, receiver) = pair
            if(sender.core_type == 1 or receiver.core_type == 1):  # skip memTile cores
                continue
            if use_shared_mem_flag:
                if not have_shared_memory(sender, receiver):
                    edges.append(
                        (
                            sender,
                            receiver,
                            {
                                "cl": CommunicationLink(
                                    sender, receiver, core_to_core_bw, unit_energy_cost
                                )
                            },
                        )
                    )
           
    # Vertical edges
    for col in cores_array.T:
        # From top to bottom (bottom is highest idx)
        pairs = zip(col, col[1:])
        for pair in pairs:
            (sender, receiver) = pair
            if(sender.core_type == 1 or receiver.core_type == 1):  # skip memTile cores
                continue
           
            if use_shared_mem_flag:
                if not have_shared_memory(sender, receiver):
                    edges.append(
                        (
                            sender,
                            receiver,
                            {
                                "cl": CommunicationLink(
                                    sender, receiver, core_to_core_bw, unit_energy_cost
                                )
                            },
                        )
                    )
            
        # From bottom to top
        pairs = zip(reversed(col), reversed(col[:-1]))
        for pair in pairs:
            (sender, receiver) = pair
            if(sender.core_type == 1 or receiver.core_type == 1):  # skip memTile cores
                continue
            if use_shared_mem_flag:
                if not have_shared_memory(sender, receiver):
                    edges.append(
                        (
                            sender,
                            receiver,
                            {
                                "cl": CommunicationLink(
                                    sender, receiver, core_to_core_bw, unit_energy_cost
                                )
                            },
                        )
                    )
    ########### End of the logic for adding the links representing the shared memory
                
    ########### Beginning of the logic for building the AXI4-Stream network
    
    # counters to keep track of the number of channels already represented with a CommunicationLink
    offchip_read_count  = 0
    offchip_write_count = 0
    memTile_write_count = 0
    memTile_read_count = 0

    # calculate the maximum number of generic CommunicationLinks needed to 
    max_channels_num = max(offchip_read_channels_num, offchip_write_channels_num, memTile_read_channels_num, memTile_write_channels_num)  # this should be the largest of the above channels

    for i in range(max_channels_num):  # a parametrizable number of channels
        generic_test_link = CommunicationLink(
            "Any" + str(i), "Any" + str(i), bandwidth, unit_energy_cost
        )            
        # the purpose of these flags is to ensure that we increment the corresponding channels counter only once after adding any number of edges to/from the corresponding core
        offchip_read_flag = False
        offchip_write_flag = False
        memTile_read_flag = False
        memTile_write_flag = False

        # what are all possible connections for connecting any to any?
        # (1) add a read and write edge from/to every core (including all memTiles) and the offchip core
        for core in cores:
            if core.core_type == 1:
                if offchip_read_count < offchip_read_channels_num and memTile_write_count < memTile_write_channels_num:
                    edges.append((core, offchip_core, {"cl":  generic_test_link}))  # in each iteration of the outer loop, it acts as a 1 read channel of the offchip core and 1 write channel of every memTile
                    offchip_read_flag = True
                    memTile_write_flag = True
            else:
                if offchip_read_count < offchip_read_channels_num:
                    edges.append((core, offchip_core, {"cl":  generic_test_link}))  # in each iteration of the outer loop, it acts as 1 read channel of the offchip core and 1 write channel of every compute core
                    if pooling_core:
                        edges.append((pooling_core, offchip_core, {"cl":  generic_test_link}))
                        edges.append((offchip_core, pooling_core, {"cl":  generic_test_link}))
                    offchip_read_flag = True
            
        for core in cores:    
            if core.core_type == 1:
                if offchip_write_count < offchip_write_channels_num and memTile_read_count < memTile_read_channels_num:
                    edges.append((offchip_core, core, {"cl":  generic_test_link}))  # in each iteration of the outer loop, it acts as 1 write channel of the offchip core and 1 read channel of every memTile
                    offchip_write_flag = True
                    memTile_read_flag = True
            else:
                if offchip_write_count < offchip_write_channels_num:
                    edges.append((offchip_core, core, {"cl":  generic_test_link}))  # in each iteration of the outer loop, it acts as 1 write channel of the offchip core and 1 read channel of every compute core
                    if pooling_core:
                        edges.append((pooling_core, offchip_core, {"cl":  generic_test_link}))
                        edges.append((offchip_core, pooling_core, {"cl":  generic_test_link}))
                    offchip_write_flag = True

        # (2) add a read and write edge between every core and every other core (including all memTiles)
        for core_1 in cores:
            if core_1.core_type == 1:
                for core_2 in cores:
                    if core_1 == core_2:  # no edges between a core and itself
                        continue
                    if core_2.core_type == 1:
                        if memTile_read_count < memTile_read_channels_num and memTile_write_count < memTile_write_channels_num:
                            edges.append((core_1, core_2, {"cl": generic_test_link})) # in each iteration of the outer loop, it acts as 1 read of every memTile and 1 write channel of every other memTile
                            memTile_read_flag = True
                            memTile_write_flag = True
                    else:
                        if memTile_write_count < memTile_write_channels_num:
                            edges.append((core_1, core_2, {"cl": generic_test_link})) # in each iteration of the outer loop, it acts as 1 write channel of each memTile and 1 read channel of each compute core
                            memTile_write_flag = True
            else:
                for core_2 in cores:
                    if core_1 == core_2: # no edges between a core and itself
                        continue
                    if core_2.core_type == 1:
                        if memTile_read_count < memTile_read_channels_num:
                            edges.append((core_1, core_2, {"cl": generic_test_link})) # in each iteration of the outer loop, it acts as 1 read channel of each memTile and 1 write channel of each compute core
                            memTile_read_flag = True
                    else:
                        edges.append((core_1, core_2, {"cl": generic_test_link})) # in each iteration of the outer loop, it acts as 1 read channel of every compute core and 1 write channel of every other compute core
        # increment the counters corresponding to the true flags
        if offchip_read_flag:
            offchip_read_count += 1
        if offchip_write_flag:
            offchip_write_count += 1 
        if memTile_read_flag:
            memTile_read_count += 1
        if memTile_write_flag:
            memTile_write_count += 1
    ########### End of the logic for building the AXI4-Stream network
            
    # Build the graph using the constructed list of edges
    single_digraph = DiGraph(edges)
    multi_digraph = MultiDiGraph(edges)

    if(parallel_links_flag == True):
        H = multi_digraph
    else:
        H = single_digraph

    print("===== Printing the chosen H at the end of get_2d_mesh =====")
    print(H)
    print("===================================================")

    return H