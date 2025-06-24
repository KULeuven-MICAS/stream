import logging
from typing import TypeAlias

from zigzag.datatypes import LayerDim
from zigzag.workload.layer_attributes import LayerDimSizes

from stream.utils import contains_wildcard
from stream.workload.computation.computation_node import LOOP_RANGES_T
from stream.workload.mapping import TILING_T

logger = logging.getLogger(__name__)

GroupAllocation: TypeAlias = dict[tuple[tuple[int, int], ...], int]


class GroupIdManager:
    """
    Manages group IDs for tiles based on inter-core and intra-core tiling and layer dimension sizes.
    Used to determine which tiles belong to the same group/core for allocation and cost model evaluation.
    """

    def __init__(
        self,
        layer_dim_sizes: LayerDimSizes,
        intra_core_tiling: TILING_T,
        inter_core_tiling: TILING_T,
    ):
        self.__id_count = 0
        self.groups: GroupAllocation = {}
        self.layer_dim_sizes = layer_dim_sizes
        self.intra_core_tiling: list[tuple[LayerDim, int]] = intra_core_tiling
        self.inter_core_tiling = inter_core_tiling
        self.inter_core_tiled_dims = [layer_dim for layer_dim, _ in inter_core_tiling]

    def __get_range_identifier(self, tile_loop_ranges: LOOP_RANGES_T):
        """
        Returns a tuple identifier for the tile's inter-core loop ranges.
        Raises ValueError if required dimensions are missing.
        """
        if not all(layer_dim in tile_loop_ranges for layer_dim, _ in self.inter_core_tiling):
            raise ValueError(
                f"Given inter core tiling {self.inter_core_tiling} contains layer dims that are not "
                f"in the provided tile_loop_ranges."
            )

        return tuple(
            self.__get_range_identifier_single_dim(layer_dim, tile_loop_ranges[layer_dim])
            for layer_dim, _ in self.inter_core_tiling
        )

    def get_group_id(self, tile_loop_ranges: LOOP_RANGES_T) -> int:
        """
        Returns the group ID for a tile, based on its loop ranges and the current tiling.
        If there is no constant operand, returns 0 (all nodes share the same group).
        """
        if contains_wildcard(self.inter_core_tiling):
            # In this case, the tiles should not be split between cores yet
            return 0

        # Differentiate based on node's inter core tiling
        range_identifier = self.__get_range_identifier(tile_loop_ranges)

        # This tile belongs together with previously seen tiles
        if range_identifier in self.groups:
            return self.groups[range_identifier]

        # New group
        new_group_id = self.__get_and_raise_id()
        self.groups[range_identifier] = new_group_id
        return new_group_id

    def __get_and_raise_id(self):
        curr_id = self.__id_count
        self.__id_count += 1
        return curr_id

    def __get_range_identifier_single_dim(self, inter_core_layer_dim: LayerDim, current_range: tuple[int, int]):
        """Given the layer dim for which the tile should be split inter core, and the range of the tile for this
        layer dim, return a range identifier that determines the group ID for this tile range. The group ID for the
        full tile is an aggregate of the tile identifiers of all tile ranges of layer dims that are split inter-core.

        The total layer size of the given layer dim is split up into N equal parts, where N is the number of intra core
        splits for the given layer dim (that will also be split inter core). The range identifier corresponds to the
        given range modulo the size of the N equal parts.
        """
        nb_intra_core_splits = next(
            (split for layer_dim, split in self.intra_core_tiling if layer_dim == inter_core_layer_dim), 1
        )
        range_size_per_intra_split = self.layer_dim_sizes[inter_core_layer_dim] // nb_intra_core_splits
        range_adjusted_to_intra_split = tuple(i % range_size_per_intra_split for i in current_range)
        return range_adjusted_to_intra_split
