from stream.classes.opt.splitting.TemporalLoop import TemporalLoop
from stream.classes.workload.computation_node import ComputationNode
from zigzag.datatypes import LayerDim

import logging


logger = logging.getLogger(__name__)


def get_rest_loops(total_loop_dim: dict[str, int], to_be_excluded_loops: list[TemporalLoop]) -> list[TemporalLoop]:
    """
    This function return a list of the rest temporal loops after remove the to_be_excluded_loops from the total_loop_dim.
    """
    rest_loops = []
    to_be_excluded_loops = {TM_loop.dimension: TM_loop.size for TM_loop in to_be_excluded_loops}
    for loop_name, loop_value_total in total_loop_dim.items():
        if loop_name in to_be_excluded_loops:
            loop_value_to_be_gone = to_be_excluded_loops[loop_name]
            loop_value_left = loop_value_total // loop_value_to_be_gone
            if loop_value_left > 1:
                rest_loops.append(TemporalLoop(loop_name, loop_value_left))
        else:
            if loop_value_total > 1:
                rest_loops.append(TemporalLoop(loop_name, loop_value_total))
    return rest_loops


def find_the_closest_divisible_factor_within_a_range(total, factor, a_range):
    """
    This function find the closest divisible factor within a range.
    E.g., if the total loop size 26, the factor is 10, and the range is 2,
    the function will try all the values between 10/2 and 10*2, and return 13 as result.
    """
    lower_bound = max(2, factor // a_range)
    upper_bound = min(total, factor * a_range)
    new_factor_candidates = [(i, abs(factor - i)) for i in range(lower_bound, upper_bound + 1) if total % i == 0]

    new_factor = min(new_factor_candidates, key=lambda tup: tup[1])[0]
    return new_factor


def convert_inner_cn_loops(inner_cn_loops: list, layer: ComputationNode):
    """Converts a list of string-defined inner-cn loops to outer-cn TemporalLoop objects.
    "all" as a inner-cn dimension size is converted to the layer's dimension size.

    Args:
        inner_cn_loops (list): A list of string-defined inner-cn loops.
        layer (ComputationNode): The original layer.
    """
    inner_loops = []
    for loop_name, loop_size in inner_cn_loops:
        if loop_name in layer.layer_dims:
            if loop_size == "all" or layer.layer_dim_sizes[loop_name] < loop_size:
                inner_loops.append(TemporalLoop(loop_name, layer.layer_dim_sizes[loop_name]))
            elif layer.layer_dim_sizes[loop_name] % loop_size == 0:
                inner_loops.append(TemporalLoop(loop_name, loop_size))
            else:
                try:
                    # find the closest factor within 50x.
                    new_loop_size = find_the_closest_divisible_factor_within_a_range(
                        layer.layer_dim_sizes[loop_name], loop_size, 50
                    )
                    inner_loops.append(TemporalLoop(loop_name, new_loop_size))
                    logger.info(
                        f"For layer {int(layer.id)}, the inner CN dimension {loop_name} size is adjusted from {loop_size} to {new_loop_size}."
                    )
                except:
                    raise ValueError(f"({loop_name}, {loop_size}) is not a valid inner CN loop.")
    outer_loops = get_rest_loops(layer.layer_dim_sizes, inner_loops)
    return outer_loops


def convert_outer_cn_loops(outer_cn_loops: list[tuple[str, int | str]], layer: ComputationNode):
    """Converts a list of string-defined outer-cn loops to outer-cn TemporalLoop objects.
    "all" in ("K", "all") is converted to the size of that dimension for the layer.

    Args:
        outer_cn_loops (list): The list of string-defined outer-cn loops
        layer (ComputationNode): The original layer.
    """
    outer_loops: list[TemporalLoop] = []
    for layer_dim_str, loop_size in outer_cn_loops:
        assert isinstance(layer_dim_str, str)
        layer_dim = LayerDim(layer_dim_str)

        if layer_dim in layer.layer_dim_sizes.layer_dims:
            if isinstance(loop_size, str):
                if loop_size == "all":
                    outer_loops.append(TemporalLoop(layer_dim, layer.layer_dim_sizes[layer_dim]))
                else:
                    raise ValueError("Loop hint is string but not 'all'")
            else:
                if layer.layer_dim_sizes[layer_dim] < loop_size:
                    outer_loops.append(TemporalLoop(layer_dim, layer.layer_dim_sizes[layer_dim]))
                elif layer.layer_dim_sizes[layer_dim] % loop_size == 0:
                    outer_loops.append(TemporalLoop(layer_dim, loop_size))
                else:
                    # Increase the loop size of the layer until it is divisible by the outer loop
                    new_layer_dim_size = layer.layer_dim_sizes[layer_dim] + 1
                    while new_layer_dim_size % loop_size != 0:
                        new_layer_dim_size += 1
                    # Set the new loop size of the layer
                    logger.warn(
                        f"Layer {layer}: {layer_dim} {layer.layer_dim_sizes[layer_dim]} -> {new_layer_dim_size}"
                    )
                    layer.layer_dim_sizes[layer_dim] = new_layer_dim_size
                    outer_loops.append(TemporalLoop(layer_dim, loop_size))
    return outer_loops


def convert_outer_cn_loops_with_k(outer_cn_loops: list, layer: ComputationNode, split_factor: int):
    """Converts a list of string-defined outer-cn loops to outer-cn TemporalLoop objects.
    Adds output channel (K) outer-cn loops to the already provided outer-cn loops depending on the split factor.

    Args:
        outer_cn_loops (list): The list of string-defined outer-cn loops.
        layer (ComputationNode): The original layer.
        split_factor (int): The number of output channel splits that will be added.
    """
    if not isinstance(split_factor, int):
        raise ValueError("The number of K splits should be an integer.")
    if split_factor > 1:
        outer_cn_loops += [(LayerDim("K"), split_factor)]
    outer_loops = convert_outer_cn_loops(outer_cn_loops, layer)
    return outer_loops
