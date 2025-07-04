from stream.opt.partitioning.TemporalLoop import TemporalLoop
from stream.workload.mapping import TILING_T


def convert_outer_cn_loops(outer_cn_loops: TILING_T):
    """Converts a list of string-defined outer-cn loops to outer-cn TemporalLoop objects.

    Args:
        outer_cn_loops (list): The list of string-defined outer-cn loops
        node: The original node.

    NOTE `HintLoopGenerationStage` already clears out the invalid unrollings
    """
    assert all(isinstance(factor, int) for _, factor in outer_cn_loops)
    return [TemporalLoop(layer_dim, loop_size) for layer_dim, loop_size in outer_cn_loops if loop_size > 1]
