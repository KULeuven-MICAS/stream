import logging
from typing import TYPE_CHECKING, Any, TypeAlias

from numpy.typing import NDArray
from zigzag.cost_model.cost_model import CostModelEvaluation
from zigzag.datatypes import MemoryOperand

from stream.cost_model.core_cost import CoreCostEntry

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from stream.hardware.architecture.accelerator import Accelerator

ARRAY_T: TypeAlias = NDArray[Any]


def get_too_large_operands(
    cme: CoreCostEntry | CostModelEvaluation, accelerator: "Accelerator", core_id: int
) -> list[MemoryOperand]:
    """Create a list of memory operands for which an extra memory level (i.e. offchip) was added.

    Args:
        cme (CostModelEvaluation): The CostModelEvaluation containing information wrt the memory utilization.
        accelerator (Accelerator): The accelerator object containing the different cores.
        core_id: The id of the core of which we wish to get the too large operands.
    """
    too_large_operands: list[MemoryOperand] = []
    core = accelerator.get_core(core_id)
    core_nb_memory_levels = core.memory_hierarchy.nb_levels
    mapping = getattr(cme, "mapping", None)
    if not mapping or not hasattr(mapping, "data_elem_per_level"):
        return too_large_operands
    for layer_operand, lvl in mapping.data_elem_per_level.items():
        memory_operand = cme.layer.memory_operand_links.layer_to_mem_op(layer_operand)
        if len(lvl) > core_nb_memory_levels[memory_operand] + 1:  # +1 because of spatial level
            too_large_operands.append(memory_operand)
    return too_large_operands


def contains_wildcard(tiling) -> bool:
    """Returns wether the given tiling contains a wildcard number `*`. The wildcard must later be replaced by the
    constraint optimization into the optimal number of tiles"""
    return any(tiling == "*" for _, tiling in tiling)


def return_tiling_type(tiling):
    if contains_wildcard(tiling):
        raise ValueError(
            "Tiling contains wildcard. Use `replace_wildcard_tiling` to replace the wildcard with a number of tiles."
        )
    return tiling  # type: ignore


def get_inter_core_tiling_size(node) -> int:
    inter_core_tiling = node.inter_core_tiling
    if inter_core_tiling and not contains_wildcard(inter_core_tiling):
        total_tiling_size = 1
        for tiling_dim, tiling_size in inter_core_tiling:
            if tiling_size == "all":
                # If the inter_core_tiling is 'all', we assume it means all cores in the layer
                # and return the size of the layer dimension.
                assert node.layer_dim_sizes, "Layer dimension sizes must be defined for 'all' inter_core_tiling."
                assert isinstance(node.layer_dim_sizes, dict), "Layer dimension sizes must be a dictionary."
                tiling_size_updated = node.layer_dim_sizes[tiling_dim]
            else:
                tiling_size_updated = tiling_size
            assert isinstance(tiling_size_updated, int), f"Tiling size must be an integer, got {tiling_size}."
            total_tiling_size *= tiling_size_updated
        return total_tiling_size
    return 1
