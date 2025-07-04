# --------------------------------------------------------------------------- #
#  steady_state_iteration_space.py                                            #
# --------------------------------------------------------------------------- #
from __future__ import annotations

from collections.abc import Iterable
from math import prod

from zigzag.datatypes import LayerDim, LayerOperand
from zigzag.workload.layer_node import LoopRelevancyInfo


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

    def __init__(self, dimension: LayerDim, size: int, relevant: bool) -> None:
        self.dimension: LayerDim = dimension
        self.size: int = int(size)
        self.relevant: bool = bool(relevant)

    # ---------- nice aliases ------------------------------------------------
    def __iter__(self):
        yield from (self.dimension, self.size, self.relevant)

    def __repr__(self):
        tag = "R" if self.relevant else "C"
        return f"IterVar({self.dimension.name},{self.size},{tag})"


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
        """
        # collect all R  +  PR descendants
        relevant_dims = set(loop_relevancy.get_r_or_pr_layer_dims(operand))
        # add PR grandchildren
        for dim in list(relevant_dims):
            relevant_dims.update(loop_relevancy.pr_dims[operand].get(dim, []))

        variables: list[IterationVariable] = []
        seen_first_relevant = False
        for dim, size in intra_core_tiling:
            is_rel = dim in relevant_dims or seen_first_relevant
            if is_rel:
                seen_first_relevant = True
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
