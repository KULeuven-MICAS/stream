# --------------------------------------------------------------------------- #
#  steady_state_iteration_space.py                                            #
# --------------------------------------------------------------------------- #
from __future__ import annotations

from collections.abc import Iterable
from enum import Enum, auto
from math import prod

from zigzag.datatypes import LayerOperand
from zigzag.workload.layer_node import LoopRelevancyInfo

from stream.datatypes import LayerDim


class Reuse(Enum):
    NOT_SET = auto()
    REUSE = auto()
    NO_REUSE = auto()
    NOT_APPLICABLE = auto()

    def __str__(self):
        if self == Reuse.NOT_SET:
            return "Not Set"
        elif self == Reuse.REUSE:
            return "Reuse"
        elif self == Reuse.NO_REUSE:
            return "No Reuse"
        else:
            return "Not Applicable"

    def __repr__(self):
        return str(self)


class LoopEffect(Enum):
    """
    Captures how a loop dimension affects a given node/operand's tensor semantics.

    - ABSENT:    This dimension is not part of the operand's logical index space.
                 The loop may exist only to align with a global iteration space.
    - INVARIANT: This dimension exists for the operand, but does not change which
                 element/slice is addressed across iterations of this loop
                 (loop-invariant for this operand).
    - VARYING:   This dimension exists for the operand and changes the addressed
                 element/slice across iterations of this loop.
    """

    ABSENT = auto()
    INVARIANT = auto()
    VARYING = auto()

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return str(self)


class IterationVariableType(Enum):
    KERNEL = auto()
    SPATIAL = auto()
    TEMPORAL = auto()
    SPATIOTEMPORAL = auto()  # for substitution temporal loops of other operators spatial loops


# --------------------------------------------------------------------------- #
#  1. IterationVariable – one tiling loop                                     #
# --------------------------------------------------------------------------- #
class IterationVariable:
    """
    Represents a single steady state tiling loop.

    Parameters
    ----------
    dimension : LayerDim
        Which logical loop dimension (e.g. `LayerDim.H`).
    size : int
        Loop trip-count **within a single steady-state slice**.
    effect : LoopEffect
        How this loop dimension affects the node/operand tensor being analyzed.

        - LoopEffect.VARYING   ⇢ this loop changes the addressed element/slice.
        - LoopEffect.INVARIANT ⇢ loop exists but does not change the addressed element/slice.
        - LoopEffect.ABSENT    ⇢ this dimension is not part of the operand's logical index space
                                 (kept only for global iteration-space alignment).
    type : IterationVariableType
        Whether this is a spatial, temporal, kernel, or spatiotemporal loop.

    Notes
    -----
    Historically, ZigZag-style "relevant" vs "irrelevant" (R/PR vs IR) was sufficient
    to express whether an operand varies between steady-state iterations.
    In a global iteration-space setting, we also need to represent loop dimensions
    that do not apply to an operand at all (ABSENT), while still keeping the loop
    around for uniformity across nodes.
    """

    def __init__(
        self,
        dimension: LayerDim,
        size: int,
        effect: LoopEffect,
        type: IterationVariableType = IterationVariableType.TEMPORAL,
    ) -> None:
        self.dimension: LayerDim = dimension
        self.size: int = int(size)
        self.effect: LoopEffect = effect
        if effect == LoopEffect.ABSENT:
            self._reuse = Reuse.NOT_APPLICABLE
        else:
            self._reuse: Reuse = Reuse.NOT_SET
        self.type: IterationVariableType = type

    # ---------- derived convenience ----------------------------------------
    @property
    def relevant(self) -> bool:
        """
        Backward-compatible alias.

        True iff this loop affects the addressed element/slice for this operand.
        """
        return self.effect == LoopEffect.VARYING

    @property
    def applicable(self) -> bool:
        """True iff this loop dimension is part of the operand's logical index space."""
        return self.effect != LoopEffect.ABSENT

    # ---------- nice aliases ------------------------------------------------
    def __iter__(self):
        yield from (
            self.dimension,
            self.size,
            self.effect,
            self.reuse,
            self.type,
        )

    def __repr__(self) -> str:
        if self.type == IterationVariableType.SPATIAL:
            prefix = "S"
        elif self.type == IterationVariableType.KERNEL:
            prefix = "K"
        elif self.type == IterationVariableType.SPATIOTEMPORAL:
            prefix = "ST"
        else:
            prefix = "T"

        if self.effect == LoopEffect.VARYING:
            tag = "V"
        elif self.effect == LoopEffect.INVARIANT:
            tag = "I"
        else:
            tag = "A"  # absent

        return f"{prefix}({self.dimension},{self.size},{tag})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, IterationVariable):
            return NotImplemented
        return (
            self.dimension == other.dimension
            and self.size == other.size
            and self.effect == other.effect
            and self.reuse == other.reuse
            and self.type == other.type
        )

    def __hash__(self) -> int:
        return hash((self.dimension, self.size, self.effect, self.reuse, self.type))

    # ---------- reuse ------------------------------------------
    @property
    def reuse(self) -> Reuse:
        return self._reuse

    @reuse.setter
    def reuse(self, value: Reuse) -> None:
        if not isinstance(value, Reuse):
            raise ValueError(f"Expected Reuse, got {type(value)}")
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
    def __init__(self, variables: tuple[IterationVariable]) -> None:
        self.variables: tuple[IterationVariable] = tuple(variables)

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
        # inter_core_tiling: TILING_T | None = None,
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
        # if inter_core_tiling is None:
        #     inter_core_tiling = []
        # collect all R  +  PR descendants
        relevant_dims = set(loop_relevancy.get_r_or_pr_layer_dims(operand))
        # add PR loops
        for dim in list(relevant_dims):
            relevant_dims.update(loop_relevancy.pr_dims[operand].get(dim, []))

        # Spatial inter_core_tiling loop variables
        variables: list[IterationVariable] = []
        # for dim, size in inter_core_tiling:
        #     is_rel = dim in relevant_dims
        #     variables.append(IterationVariable(dim, size, is_rel, spatial=True))

        # Temporal intra_core_tiling loop variables
        seen_ir = False
        for dim, size in intra_core_tiling:
            is_rel = dim in relevant_dims
            iter_var = IterationVariable(dim, size, is_rel)
            if not is_rel:
                seen_ir = True
            elif seen_ir and is_rel:
                # once we have seen an IR loop, subsequent R loops can't have mem reuse
                iter_var.reuse = Reuse.NO_REUSE
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

    def shape_mem(self, num_memtiles: int = 1) -> tuple[int, ...]:
        """
        Returns the shape of the relevant iteration space kept local in a memtile.
        Kernel dimensions may be misordered.
        """
        # start with kernel shape
        # (ordering may be a bit off here)
        shape = tuple(var.size for var in self.get_kernel_variables() if var.relevant)

        # add relevant spatial shape
        spatial_shape = tuple(
            var.size for var in self.get_spatial_variables() if var.relevant and var.size != num_memtiles
        )
        shape = spatial_shape + shape

        # add dims reused in memtile
        local_tensors = self.nb_local_tensors_mem()
        if local_tensors > 1:
            shape = (local_tensors,) + shape

        return shape

    def reuse_factor(self) -> int:
        """
        Returns the number of time a tensor is reused.
        """
        return prod(iv.size for iv in self.get_temporal_variables() if not iv.relevant and iv.reuse == Reuse.REUSE)

    def nb_local_tensors(self) -> int:
        """
        Returns the number of tensors tiles that are kept local.
        """
        last_reuse = None
        for var in reversed(self.get_temporal_variables()):
            if var.reuse == Reuse.REUSE:
                last_reuse = var
                break
        if last_reuse:
            last_reuse_index = self.get_temporal_variables().index(last_reuse)
            reuse_iters = self.get_temporal_variables()[: last_reuse_index + 1]
        else:
            reuse_iters = []

        return prod(iv.size for iv in reuse_iters if iv.relevant)

    def reuse_summary(self) -> dict:
        """
        JSON-serializable summary of the compute- and mem-tile reuse decisions.

        Should be called after the MILP solver has set all ``compute_tile_reuse``
        and ``mem_tile_reuse`` flags via ``update_transfer_reuse_levels()``.

        Returns
        -------
        dict with keys:
          ``compute_reuse_factor``
              Product of non-relevant loop sizes marked REUSE at the compute tile.
              1 means no reuse (identity of multiplication).
          ``mem_reuse_factor``
              Analogous factor for the memory tile (divided by compute factor).
              1 means no additional mem-tile reuse beyond the compute level.
              NOT present in the dict when no memory tile is involved
              (all mem_tile_reuse flags are NOT_SET or NOT_APPLICABLE).
          ``loops``
              Per-dimension breakdown of applicable temporal loops, innermost first:
              ``dim``, ``size``, ``relevant``,
              ``compute_tile_reuse`` (str), ``mem_tile_reuse`` (str or None).
        """
        applicable = self.get_applicable_temporal_variables()
        loops = [
            {
                "dim": str(iv.dimension),
                "size": iv.size,
                "relevant": iv.relevant,
                "reuse": str(iv.reuse),
            }
            for iv in applicable
        ]
        result: dict = {
            "reuse_factor": self.reuse_factor(),
            "loops": loops,
        }
        return result

    def get_kernel_variables(self) -> list[IterationVariable]:
        """
        Returns the list of kernel iteration variables.
        """
        return [iv for iv in self.variables if iv.type == IterationVariableType.KERNEL]

    def get_spatial_variables(self) -> list[IterationVariable]:
        """
        Returns the list of temporal iteration variables (i.e. those that are not spatial).
        """
        return [iv for iv in self.variables if iv.type == IterationVariableType.SPATIAL]

    def get_spatio_temporal_variables(self) -> list[IterationVariable]:
        return [iv for iv in self.variables if iv.type == IterationVariableType.SPATIOTEMPORAL]

    def get_temporal_variables(self) -> list[IterationVariable]:
        """
        Returns the list of temporal iteration variables (i.e. those that are not spatial).
        """
        return [iv for iv in self.variables if iv.type == IterationVariableType.TEMPORAL]

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

    def get_temporal_reuses(self) -> list[Reuse]:
        """
        Returns the list of reuses of temporal iteration variables.
        """
        return [iv.reuse for iv in self.get_temporal_variables()]

    def get_applicable_temporal_variables(self) -> list[IterationVariable]:
        """
        Returns the list of temporal iteration variables that are applicable to the operand.
        """
        return [iv for iv in self.get_temporal_variables() if iv.applicable]

    def get_applicable_temporal_dimensions(self) -> list[LayerDim]:
        """
        Returns the list of temporal loop dimensions that are applicable to the operand.
        """
        return [iv.dimension for iv in self.get_temporal_variables() if iv.applicable]

    def get_applicable_temporal_sizes(self) -> list[int]:
        """
        Returns the list of sizes of temporal iteration variables that are applicable to the operand.
        """
        return [iv.size for iv in self.get_temporal_variables() if iv.applicable]

    def get_applicable_temporal_relevancies(self) -> list[bool]:
        """
        Returns the list of relevancies of temporal iteration variables that are applicable to the operand.
        """
        return [iv.relevant for iv in self.get_temporal_variables() if iv.applicable]

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
