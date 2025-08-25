# --------------------------------------------------------------------------- #
#  steady_state_iteration_space.py                                            #
# --------------------------------------------------------------------------- #
from __future__ import annotations

from collections.abc import Iterable, Sequence
from enum import Flag, auto
from math import prod

from zigzag.datatypes import LayerDim, LayerOperand
from zigzag.workload.layer_node import LoopRelevancyInfo

from stream.workload.mapping import TILING_T


class IterationVariableReuse(Flag):
    NOT_SET = auto()  # default value is not set
    MEM_TILE_REUSE = auto()
    MEM_TILE_NO_REUSE = auto()
    COMPUTE_TILE_REUSE = auto()
    COMPUTE_TILE_NO_REUSE = auto()


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
        self._reuse: IterationVariableReuse = IterationVariableReuse.NOT_SET
        self.spatial: bool = bool(spatial)

    # ---------- nice aliases ------------------------------------------------
    def __iter__(self):
        yield from (self.dimension, self.size, self.relevant, self.reuse, self.spatial)

    def __repr__(self):
        tag = "R" if self.relevant else "IR"
        return f"IterVar({self.dimension.name},{self.size},{tag},{self.reuse},spatial={self.spatial})"

    def __eq__(self, other):
        if not isinstance(other, IterationVariable):
            return NotImplemented
        return (
            self.dimension == other.dimension
            and self.size == other.size
            and self.relevant == other.relevant
            and self.reuse == other.reuse
            and self.spatial == other.spatial
        )

    def __hash__(self):
        return hash((self.dimension, self.size, self.relevant, self.reuse, self.spatial))

    # Getter and setter for reuse attribute
    @property
    def reuse(self) -> IterationVariableReuse:
        return self._reuse

    @reuse.setter
    def reuse(self, value: IterationVariableReuse) -> None:
        if not isinstance(value, IterationVariableReuse):
            raise ValueError(f"Expected IterationVariableReuse, got {type(value)}")
        self._reuse = value


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
        for dim, size in intra_core_tiling:
            is_rel = dim in relevant_dims
            variables.append(IterationVariable(dim, size, is_rel))

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
        return temporal_shape + spatial_shape

    def reuse_factor_compute(self) -> int:
        """
        Returns the number of time a tensor is reused in a compute tile.
        """
        return prod(
            iv.size
            for iv in self.get_temporal_variables()
            if not iv.relevant and IterationVariableReuse.COMPUTE_TILE_REUSE in iv.reuse
        )

    def nb_local_tensors_compute(self) -> int:
        """
        Returns the number of tensors that are kept local in a compute tile.
        """
        return prod(
            iv.size
            for iv in self.get_temporal_variables()
            if iv.relevant and IterationVariableReuse.COMPUTE_TILE_REUSE in iv.reuse
        )

    def nb_local_tensors_mem(self) -> int:
        """
        Returns the number of tensors that are kept local in a mem tile.
        """
        return prod(
            iv.size
            for iv in self.get_temporal_variables()
            if iv.relevant and IterationVariableReuse.MEM_TILE_REUSE in iv.reuse
        )

    def reuse_factor_mem(self) -> int:
        """
        Returns the number of time a tensor is reused in a mem tile.
        """
        return (
            prod(
                iv.size
                for iv in self.get_temporal_variables()
                if not iv.relevant and IterationVariableReuse.MEM_TILE_REUSE in iv.reuse
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

    def get_temporal_resues(self) -> list[IterationVariableReuse]:
        """
        Returns the list of reuses of temporal iteration variables.
        """
        return [iv.reuse for iv in self.get_temporal_variables()]

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
