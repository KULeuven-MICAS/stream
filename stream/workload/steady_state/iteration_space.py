# --------------------------------------------------------------------------- #
#  steady_state_iteration_space.py                                            #
# --------------------------------------------------------------------------- #
from __future__ import annotations

import warnings
from collections.abc import Iterable, Sequence
from enum import Flag, auto
from math import prod

from zigzag.datatypes import LayerDim, LayerOperand
from zigzag.workload.layer_node import LoopRelevancyInfo

from stream.workload.computation.computation_node import ComputationNode
from stream.workload.mapping import TILING_T


class ComputeTileReuse(Flag):
    NOT_SET = auto()
    REUSE = auto()
    NO_REUSE = auto()


class MemTileReuse(Flag):
    NOT_SET = auto()
    REUSE = auto()
    NO_REUSE = auto()


# --------------------------------------------------------------------------- #
# 1.  IterationVariable – one intra-core tiling loop                          #
# --------------------------------------------------------------------------- #
class IterationVariable:
    """
    Represents a single intra-core tiling loop (innermost → outermost order).

    Parameters
    ----------
    dimension : LayerDim
        Which logical loop dimension (e.g. `LayerDim.H`).
    size      : int
        Loop trip-count **within a single steady-state slice**.
    relevant  : bool
        *True* ⇢ this loop varies between consecutive steady-state iterations
        (R/PR in ZigZag terminology); once a relevant loop is encountered,
        every **outer** loop is marked relevant as well.
    """

    def __init__(self, dimension: LayerDim, size: int, relevant: bool, spatial: bool = False) -> None:
        self.dimension: LayerDim = dimension
        self.size: int = int(size)
        self.relevant: bool = bool(relevant)
        self._compute_tile_reuse: ComputeTileReuse = ComputeTileReuse.NOT_SET
        self._mem_tile_reuse: MemTileReuse = MemTileReuse.NOT_SET
        self.spatial: bool = bool(spatial)

    # ---------- nice aliases ------------------------------------------------
    def __iter__(self):
        yield from (self.dimension, self.size, self.relevant, self.compute_tile_reuse, self.spatial)

    def __repr__(self):
        prefix = "S" if self.spatial else "T"
        tag = "R" if self.relevant else "IR"
        return f"{prefix}({self.dimension.name},{self.size},{tag})"

    def __eq__(self, other):
        if not isinstance(other, IterationVariable):
            return NotImplemented
        return (
            self.dimension == other.dimension
            and self.size == other.size
            and self.relevant == other.relevant
            and self.compute_tile_reuse == other.compute_tile_reuse
            and self.mem_tile_reuse == other.mem_tile_reuse
            and self.spatial == other.spatial
        )

    def __hash__(self):
        return hash((self.dimension, self.size, self.relevant, self.compute_tile_reuse, self.spatial))

    # Getter and setter for compute and mem tile reuse attribute
    @property
    def compute_tile_reuse(self) -> ComputeTileReuse:
        return self._compute_tile_reuse

    @compute_tile_reuse.setter
    def compute_tile_reuse(self, value: ComputeTileReuse) -> None:
        if not isinstance(value, ComputeTileReuse):
            raise ValueError(f"Expected ComputeTileReuse, got {type(value)}")
        self._compute_tile_reuse = value

    @property
    def mem_tile_reuse(self) -> MemTileReuse:
        return self._mem_tile_reuse

    @mem_tile_reuse.setter
    def mem_tile_reuse(self, value: MemTileReuse) -> None:
        if not isinstance(value, MemTileReuse):
            raise ValueError(f"Expected MemTileReuse, got {type(value)}")
        self._mem_tile_reuse = value


# --------------------------------------------------------------------------- #
# 2.  SteadyStateIterationSpace                                               #
# --------------------------------------------------------------------------- #
class SteadyStateIterationSpace:
    """
    Ordered container for all intra-core loops that together define *one*
    steady-state slice.

    The list is stored **innermost → outermost** (same order as the original
    `intra_core_tiling` from ZigZag).
    """

    # ............................................... basic constructor ....
    def __init__(self, variables: list[IterationVariable]) -> None:
        self.variables: list[IterationVariable] = list(variables)

    def __eq__(self, other):
        if not isinstance(other, SteadyStateIterationSpace):
            return NotImplemented
        return self.variables == other.variables

    def __hash__(self):
        return hash(tuple(self.variables))

    # ..................................................................... #
    # ── CLASS FACTORY  (replaces the old helper in SteadyStateScheduler) ── #
    # ..................................................................... #
    @classmethod
    def from_loop_info(
        cls,
        *,
        loop_relevancy: LoopRelevancyInfo,
        intra_core_tiling: Iterable[tuple[LayerDim, int]],
        operand: LayerOperand,
        inter_core_tiling: TILING_T | None = None,
    ) -> SteadyStateIterationSpace:
        """
        Build the SSIS for **one operand** of a computation node.

        *The logic is identical to the former* `extract_steady_state_iteration_space`
        *in* `SteadyStateScheduler`, but self-contained and reusable.*

        Parameters
        ----------
        loop_relevancy : zigzag.workload.layer_node.LoopRelevancyInfo
            Contains *R* / *PR* loop information for the operand.
        intra_core_tiling : list[tuple[LayerDim, int]]
            Ordered (innermost→outermost) tiling definition: `(dimension, size)`.
        operand : LayerOperand
            Which operand we are analysing (needed for relevancy lookup).
        inter_core_tiling : Iterable[tuple[LayerDim, int]], optional
            Ordered (innermost→outermost) tiling definition for inter-core tiling.
            Defaults to empty iterable. Is used for transfer nodes as innermost iteration variable.
        """
        if inter_core_tiling is None:
            inter_core_tiling = []
        # collect all R  +  PR descendants
        relevant_dims = set(loop_relevancy.get_r_or_pr_layer_dims(operand))
        # add PR loops
        for dim in list(relevant_dims):
            relevant_dims.update(loop_relevancy.pr_dims[operand].get(dim, []))

        # Spatial inter_core_tiling loop variables
        variables: list[IterationVariable] = []
        for dim, size in inter_core_tiling:
            is_rel = dim in relevant_dims
            variables.append(IterationVariable(dim, size, is_rel, spatial=True))

        # Temporal intra_core_tiling loop variables
        seen_ir = False
        for dim, size in intra_core_tiling:
            is_rel = dim in relevant_dims
            iter_var = IterationVariable(dim, size, is_rel)
            if not is_rel:
                seen_ir = True
            elif seen_ir and is_rel:
                # once we have seen an IR loop, subsequent R loops can't have mem reuse
                iter_var.mem_tile_reuse = MemTileReuse.NO_REUSE
            variables.append(iter_var)

        return cls(variables)

    @classmethod
    def from_computation_node(cls, *, node: ComputationNode, multiplicity: int = 1) -> SteadyStateIterationSpace:
        # Temporal intra_core_tiling loop variables only
        variables: list[IterationVariable] = []
        intra_core_tiling = node.intra_core_tiling
        adjusted = False
        for dim, size in intra_core_tiling:
            if not adjusted and size >= multiplicity:
                adj_size = size // multiplicity  # Adjust for multiplicity on first large enough loop
                adjusted = True
            else:
                adj_size = size
            is_rel = True  # All dimensions are relevant for the computation node
            iter_var = IterationVariable(dim, adj_size, is_rel)
            variables.append(iter_var)
        return cls(variables)

    # ..................................................................... #
    # ── Convenience helpers                                                 #
    # ..................................................................... #
    @property
    def slices_per_full(self) -> int:
        """
        *How many* steady-state slices constitute one **full tensor**.

        Equals the product of loop sizes that are *relevant* (vary between
        iterations).  Falls back to **1** if nothing varies.
        """
        sizes = [v.size for v in self.variables if v.relevant]
        return prod(sizes) if sizes else 1

    @property
    def slice_volume(self) -> int:
        """Total elements/bits in *one* slice (product of *all* loop sizes)."""
        return prod(v.size for v in self.variables) if self.variables else 1

    def shape_mem(self, spatial_relevant: Sequence[LayerDim]) -> tuple[int, ...]:
        """
        Returns the shape of the relevant iteration space kept local in a memtile.
        """
        spatial_shape = prod(iv.size for iv in self.variables if iv.spatial and iv.dimension in spatial_relevant)
        spatial_shape = (spatial_shape,) if spatial_shape > 1 else ()
        temporal_shape = self.nb_local_tensors_mem()
        temporal_shape = (temporal_shape,) if temporal_shape > 1 else ()
        if len(spatial_shape) > 0 and len(temporal_shape) > 0:
            warnings.warn(
                "Both spatial and temporal shapes have more than one dimension. "
                "This mixes temporal reuse / distribute, which seems to be unsupported. "
                "Skipping the temporal shape for now, will break if reuse factor > 1.",
                stacklevel=1,
            )
            return spatial_shape
        return temporal_shape + spatial_shape

    def reuse_factor_compute(self) -> int:
        """
        Returns the number of time a tensor is reused in a compute tile.
        """
        return prod(
            iv.size
            for iv in self.get_temporal_variables()
            if not iv.relevant and ComputeTileReuse.REUSE in iv.compute_tile_reuse
        )

    def nb_local_tensors_compute(self) -> int:
        """
        Returns the number of tensors that are kept local in a compute tile.
        """
        return prod(
            iv.size
            for iv in self.get_temporal_variables()
            if iv.relevant and ComputeTileReuse.REUSE in iv.compute_tile_reuse
        )

    def nb_local_tensors_mem(self) -> int:
        """
        Returns the number of tensors that are kept local in a mem tile.
        """
        return prod(
            iv.size for iv in self.get_temporal_variables() if iv.relevant and MemTileReuse.REUSE in iv.mem_tile_reuse
        )

    def reuse_factor_mem(self) -> int:
        """
        Returns the number of time a tensor is reused in a mem tile.
        """
        return (
            prod(
                iv.size
                for iv in self.get_temporal_variables()
                if not iv.relevant and MemTileReuse.REUSE in iv.mem_tile_reuse
            )
            // self.reuse_factor_compute()
        )

    def get_temporal_variables(self) -> list[IterationVariable]:
        """
        Returns the list of temporal iteration variables (i.e. those that are not spatial).
        """
        return [iv for iv in self.variables if not iv.spatial]

    def get_temporal_dimensions(self) -> list[LayerDim]:
        """
        Returns the list of temporal loop dimensions (i.e. those that are not spatial).
        """
        return [iv.dimension for iv in self.get_temporal_variables()]

    def get_temporal_sizes(self) -> list[int]:
        """
        Returns the list of sizes of temporal iteration variables.
        """
        return [iv.size for iv in self.get_temporal_variables()]

    def get_temporal_compute_tile_reuses(self) -> list[ComputeTileReuse]:
        """
        Returns the list of reuses of temporal iteration variables.
        """
        return [iv.compute_tile_reuse for iv in self.get_temporal_variables()]

    def get_temporal_mem_tile_reuses(self) -> list[MemTileReuse]:
        """
        Returns the list of reuses of temporal iteration variables.
        """
        return [iv.mem_tile_reuse for iv in self.get_temporal_variables()]

    @classmethod
    def merge_iteration_spaces(cls, ssis_list: list[SteadyStateIterationSpace]) -> SteadyStateIterationSpace:
        """
        Merges multiple SteadyStateIterationSpace into one, combining the relevancy
        If one of the iteration variables is relevant, the merged one is relevant.
        TODO: maybe not necessary anymore this method
        """
        iter_vars = []
        for ssis in ssis_list:
            print(ssis)
        for iter_var in zip(*ssis_list, strict=True):
            iv = iter_var[0]
            for other_iv in iter_var[1:]:
                if other_iv.relevant:
                    iv.relevant = True
            iter_vars.append(iv)
        result = SteadyStateIterationSpace(iter_vars)
        return result

    # ..................................................................... #
    # ── Iteration / pretty printing                                         #
    # ..................................................................... #
    def __iter__(self):
        return iter(self.variables)

    def __len__(self):
        return len(self.variables)

    def __repr__(self):
        inside = ", ".join(repr(v) for v in self.variables)
        return f"SSIS([{inside}])"
