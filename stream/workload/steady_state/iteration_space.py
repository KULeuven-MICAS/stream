# --------------------------------------------------------------------------- #
#  steady_state_iteration_space.py                                            #
# --------------------------------------------------------------------------- #
from __future__ import annotations

from enum import Enum, auto
from math import prod

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
    halo : int
        Elements along its axis beyond the step that a tile of a sliding window also holds (0 for disjoint tiles).

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
        halo: int = 0,
    ) -> None:
        self.dimension: LayerDim = dimension
        self.size: int = int(size)
        self.effect: LoopEffect = effect
        if effect == LoopEffect.ABSENT:
            self._reuse = Reuse.NOT_APPLICABLE
        else:
            self._reuse: Reuse = Reuse.NOT_SET
        self.type: IterationVariableType = type
        self.halo: int = halo

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

        halo = f",H{self.halo}" if self.halo else ""
        return f"{prefix}({self.dimension},{self.size},{tag}{halo})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, IterationVariable):
            return NotImplemented
        return (
            self.dimension == other.dimension
            and self.size == other.size
            and self.effect == other.effect
            and self.reuse == other.reuse
            and self.type == other.type
            and self.halo == other.halo
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
    def __init__(self, variables: tuple[IterationVariable, ...]) -> None:
        self.variables: tuple[IterationVariable, ...] = tuple(variables)

    def __eq__(self, other):
        if not isinstance(other, SteadyStateIterationSpace):
            return NotImplemented
        return self.variables == other.variables

    def __hash__(self):
        return hash(tuple(self.variables))

    # ..................................................................... #
    # ── CLASS FACTORY ── #
    # ..................................................................... #

    # ..................................................................... #
    # ── Convenience helpers                                                 #
    # ..................................................................... #

    def reuse_factor(self) -> int:
        """
        Returns the number of time a tensor is reused.
        """
        return prod(iv.size for iv in self.get_temporal_variables() if not iv.relevant and iv.reuse == Reuse.REUSE)

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
