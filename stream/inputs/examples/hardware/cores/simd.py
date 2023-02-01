from stream.classes.hardware.architecture.simd_array import SimdArray
from zigzag.classes.hardware.architecture.memory_hierarchy import MemoryHierarchy
from zigzag.classes.hardware.architecture.memory_instance import MemoryInstance
from zigzag.classes.hardware.architecture.core import Core
from stream.classes.hardware.architecture.simd_unit import SimdUnit

def get_memory_hierarchy(multiplier_array):
    """Memory hierarchy variables"""
    ''' size=#bit, bw=(read bw, write bw), cost=(read word energy, write work energy) '''
    inf = MemoryInstance(name="rf_16B", size=1_000_000_000, r_bw=64 * 8, w_bw=64 * 8, r_cost=0.01, w_cost=0.01, area=0, r_port=0, w_port=0, rw_port=2, latency=0)  # rd E per bit 0.0625

    memory_hierarchy_graph = MemoryHierarchy(operational_array=multiplier_array)

    '''
    fh: from high = wr_in_by_high 
    fl: from low = wr_in_by_low 
    th: to high = rd_out_to_high
    tl: to low = rd_out_to_low
    '''
    # memory_hierarchy_graph.add_memory(memory_instance=rf1, operands=('I1',),
    #                                   port_alloc=({'fh': 'w_port_1', 'tl': 'r_port_1', 'fl': None, 'th': None},),
    #                                   served_dimensions=set())
    memory_hierarchy_graph.add_memory(memory_instance=inf, operands=('I1', 'I2', 'O'),
                                      port_alloc=(
                                        {'fh': 'rw_port_1', 'tl': 'rw_port_2', 'fl': None, 'th': None},
                                        {'fh': 'rw_port_1', 'tl': 'rw_port_2', 'fl': None, 'th': None},
                                        {'fh': 'rw_port_1', 'tl': 'rw_port_2', 'fl': 'rw_port_1', 'th': 'rw_port_2'}
                                        ),
                                      served_dimensions='all')

    # from visualization.graph.memory_hierarchy import visualize_memory_hierarchy_graph
    # visualize_memory_hierarchy_graph(memory_hierarchy_graph)
    return memory_hierarchy_graph


def get_operational_array():
    pooling_unit_input_precision = [8, 8]
    pooling_energy = 0.1
    pooling_area = 0.01
    dimensions = {'D1': 64}
    pooling_unit = SimdUnit(pooling_unit_input_precision, pooling_energy, pooling_area)
    pooling_array = SimdArray(pooling_unit, dimensions)
    return pooling_array



def get_dataflows():
    return [{'D1': ('K', 64)}]


def get_core(id):
    operational_array = get_operational_array()
    memory_hierarchy = get_memory_hierarchy(operational_array)
    dataflows = get_dataflows()
    core = Core(id, operational_array, memory_hierarchy, dataflows)
    return core
