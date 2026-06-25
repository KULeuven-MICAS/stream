"""Per-accelerator *setup* (configuration) cost models.

Some accelerators pay overhead that the dataflow cost model (compute + operand transfer)
does not capture: before a kernel runs, the host core programs the accelerator's
configuration registers (for SNAX gemmx: the accfg CSRs that set up the streamers'
address-generation loops and the functional units), and the *first* use of the
accelerator additionally pays a one-time cold-start cost (runtime init, the first DMA).
Measurements on the gemmx cluster show these are **two distinct components** with
different multi-layer behaviour:

- a **one-time activation** cost, paid once per accelerator per inference (it does *not*
  repeat per layer), and
- a **per-layer config** cost, paid once per computation node (per layer launch),
  independent of the inner temporal tiling.

So a 2-layer MLP costs ``activation + 2*per_layer``, not ``2*(activation + per_layer)``
— the activation amortises across layers. Both are added in
:class:`~stream.cost_model.steady_state_scheduler.SteadyStateScheduler` *outside* the
steady-state iteration multiplier.

A core declares its model with an optional ``setup_cost`` block in its hardware YAML::

    setup_cost:
      kind: accfg
      cycles_per_csr: 12.0          # host cycles to issue+apply one config-register write
      activation_cycles: 288        # one-time cold start (runtime + first DMA), per accelerator
      base_csrs: 6                  # per-layer accel control + primary-unit (GEMM) config
      csrs_per_operand_streamer: 7  # per-layer address-gen CSRs per active data streamer
      per_operation:                # type-aware extra per-layer CSRs, keyed by operation kind
        default: 0
        requantized_matmul: 0       # gemmx SIMD requant overlaps the GEMM pipeline
      restream_cycles_per_tile: 90  # fixed cost per extra streamed-axis block (operand reload)
      restream_tile_lanes: 8        # array lanes per output dim (ceil(extent/lanes) blocks)
      restream_per_row_block: 23    # added per block of the OTHER output dim (2-D grid); 0 -> 1-D

There is also an optional **re-stream** term, charged per layer like the config: with a
stationary operand (SNAX gemmx is B-stationary) each extra block of the *streamed* output
axis re-streams the other operand, a cost zigzag's idealized (L1-resident) transfer model
overlaps with compute. The cost of one such re-stream is not constant — it scales with the
*other* (non-streamed) output dim, because the moving operand is re-streamed across the
whole output-tile grid. So the term is two-dimensional::

    restream = (n_blocks - 1) * (restream_cycles_per_tile + restream_per_row_block * m_blocks)

with ``n_blocks = ceil(last_output_dim / restream_tile_lanes)`` (the streamed axis) and
``m_blocks = ceil(other_output_dim / restream_tile_lanes)``. It is **zero** for a single
streamed-axis block, so single-output-column workloads are unchanged. Setting
``restream_per_row_block = 0`` recovers the original **1-D** term
``(n_blocks - 1) * restream_cycles_per_tile`` (the fall-back); ``restream_cycles_per_tile =
0`` **and** ``restream_per_row_block = 0`` (the defaults) disable re-stream entirely.

Cores **without** a ``setup_cost`` block get :class:`ZeroSetupCostModel` — i.e. the
framework's standard behaviour of zero setup overhead is preserved for every existing
accelerator/core definition. The model is intentionally *close-but-not-cycle-accurate*:
the per-layer term estimates configuration-register writes from the execution (how many
operand streamers it engages, which functional units its operation type needs) times a
per-write cost; the activation + re-stream terms are measured constants.
"""

from __future__ import annotations

import abc
from typing import Any


class SetupCostModel(abc.ABC):
    """Estimates an accelerator's setup overhead: a one-time activation + per-layer config."""

    @abc.abstractmethod
    def cycles(self, node: Any) -> int:
        """Per-layer config cycles charged for running ``node`` (a ComputationNode) on this core."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def activation_cycles(self) -> int:
        """One-time cold-start cycles charged on this core's first use in an inference."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def is_zero(self) -> bool:
        """True if this model never adds overhead (lets callers skip work)."""
        raise NotImplementedError


class ZeroSetupCostModel(SetupCostModel):
    """The framework default: no setup overhead. Used for any core without a spec."""

    def cycles(self, node: Any) -> int:  # noqa: ARG002
        return 0

    @property
    def activation_cycles(self) -> int:
        return 0

    @property
    def is_zero(self) -> bool:
        return True

    def __repr__(self) -> str:
        return "ZeroSetupCostModel()"


def _count_operand_streamers(node: Any) -> int:
    """Number of distinct operand tensors a node streams (inputs + outputs).

    Each operand is moved by its own streamer/data-mover, so this is the count of
    address-generators the accelerator must configure for this execution. Falls back to
    a sensible default when a node does not expose operand lists.
    """
    inputs = getattr(node, "inputs", None)
    outputs = getattr(node, "outputs", None)
    n = 0
    if inputs is not None:
        n += len(inputs)
    if outputs is not None:
        n += len(outputs)
    return n if n > 0 else 3  # GEMM-like default (two inputs + one output)


def _operation_kind(node: Any) -> str:
    """Normalised operation type of a node (e.g. 'gemm', 'requantized_matmul')."""
    t = getattr(node, "type", None)
    return str(t).strip().lower() if t is not None else "default"


def _axis_blocks(node: Any, axes: "tuple[int, ...]", lanes: int) -> int:
    """Product of ``ceil(extent / lanes)`` over the given output-tensor ``axes``.

    ``axes`` index ``node.outputs[0].shape`` (negative allowed). The result is how many
    spatial *blocks* those axes tile into -- i.e. how many times the accelerator walks the
    corresponding temporal loop(s). Returns 1 if the shape/axes cannot be read, so an
    unreadable node contributes no re-stream.

    This is the dataflow-agnostic primitive behind the re-stream term: a caller names the
    axes of the output-tile grid that a given operand must be re-streamed over, and gets the
    block count for that operand's re-stream loop -- no GEMM-specific axis assumption baked in.
    """
    if lanes <= 0 or not axes:
        return 1
    try:
        shape = node.outputs[0].shape
    except Exception:
        return 1
    prod = 1
    for a in axes:
        try:
            extent = int(shape[a])
        except Exception:
            return 1
        prod *= max(1, -(-extent // lanes))  # ceil
    return prod


class AccfgSetupCostModel(SetupCostModel):
    """Config-register-count setup model (SNAX-accfg style).

    ``setup_cycles = round(cycles_per_csr * n_csr)`` where::

        n_csr = base_csrs
              + csrs_per_operand_streamer * n_active_streamers
              + per_operation[op_kind]

    The streamer count is read from the node (so a node with more operands costs more to
    configure), and the per-operation term makes the estimate vary with the *kind* of
    execution the accelerator runs (e.g. a plain matmul vs. a requantising matmul that
    also programs the SIMD/rescale unit).
    """

    def __init__(
        self,
        cycles_per_csr: float,
        activation_cycles: int = 0,
        base_csrs: int = 0,
        csrs_per_operand_streamer: int = 0,
        per_operation: dict[str, int] | None = None,
        restream_cycles_per_tile: int = 0,
        restream_tile_lanes: int = 0,
        restream_per_row_block: int = 0,
        restream_axes: "tuple[int, ...]" = (-1,),
        refill_axes: "tuple[int, ...]" = (-2,),
    ) -> None:
        self.cycles_per_csr = float(cycles_per_csr)
        self._activation_cycles = int(activation_cycles)
        self.base_csrs = int(base_csrs)
        self.csrs_per_operand_streamer = int(csrs_per_operand_streamer)
        self.per_operation = {str(k).strip().lower(): int(v) for k, v in (per_operation or {}).items()}
        self._default_op_csrs = self.per_operation.get("default", 0)
        # Moving-operand re-stream / pipeline-refill term. A fixed-dataflow accelerator pays a
        # fixed, *reduction-independent* refill bubble at every iteration of the temporal
        # loop(s) that lie OUTSIDE the re-streamed (moving) operand's reuse footprint: at each
        # such boundary the streamer restarts that operand's access pattern from the top and the
        # spatial array's input FIFO underruns / the systolic pipeline re-fills. The idealized
        # roofline (double-buffered, "enough ports", pipeline filled once) hides all reloads, so
        # it charges zero for these -- they are the residual this term restores. It is an INPUT-
        # feed cost, *not* an output drain (doubling the output buffer removes none of it) and
        # *not* a bandwidth stall (it is K-independent: the re-streamed data volume is unchanged).
        #
        # ``restream_axes`` / ``refill_axes`` name which output-tile-grid axes (indices into the
        # output shape) play each role, so the term is not hard-coded to a GEMM's N/M layout:
        #   n_refills  = prod(ceil(dim/lanes) for axis in restream_axes) - 1   # outer re-stream loops
        #   inner_fill = prod(ceil(dim/lanes) for axis in refill_axes)         # extent re-traversed per refill
        #   restream   = n_refills * (restream_cycles_per_tile + restream_per_row_block*inner_fill)
        # Defaults (-1,)/(-2,) reproduce the SNAX-gemmx (B-stationary) case exactly: A is
        # re-streamed once per output-N block, each refill re-fills M row-blocks. A different
        # accelerator (weight-stationary, output-stationary, conv) names different axes; 0 cost
        # constants (the default) disable the term entirely.
        self.restream_cycles_per_tile = int(restream_cycles_per_tile)
        self.restream_tile_lanes = int(restream_tile_lanes)
        # restream_per_row_block == 0 -> the original 1-D term (no per-refill inner-fill scaling).
        self.restream_per_row_block = int(restream_per_row_block)
        self.restream_axes = tuple(int(a) for a in restream_axes)
        self.refill_axes = tuple(int(a) for a in refill_axes)

    def _operation_csrs(self, op_kind: str) -> int:
        return self.per_operation.get(op_kind, self._default_op_csrs)

    def cycles(self, node: Any) -> int:
        n_streamers = _count_operand_streamers(node)
        op_kind = _operation_kind(node)
        n_csr = self.base_csrs + self.csrs_per_operand_streamer * n_streamers + self._operation_csrs(op_kind)
        config = int(round(self.cycles_per_csr * max(n_csr, 0)))
        restream = 0
        if self.restream_cycles_per_tile or self.restream_per_row_block:
            # # times the moving operand is re-streamed = iterations of the outer (restream)
            # loop(s) beyond the first; each pays a fixed refill + a part scaling with the
            # inner extent (refill_axes) re-traversed per refill. Axes are configurable so this
            # is dataflow-agnostic; defaults reproduce the gemmx B-stationary (N-outer) case.
            n_refills = _axis_blocks(node, self.restream_axes, self.restream_tile_lanes) - 1
            inner_fill = _axis_blocks(node, self.refill_axes, self.restream_tile_lanes)
            per_refill = self.restream_cycles_per_tile + self.restream_per_row_block * inner_fill
            restream = max(n_refills, 0) * per_refill  # 0 for a single re-stream block
        return config + restream

    @property
    def activation_cycles(self) -> int:
        return self._activation_cycles

    @property
    def is_zero(self) -> bool:
        return False

    def __repr__(self) -> str:
        return (
            f"AccfgSetupCostModel(cycles_per_csr={self.cycles_per_csr}, "
            f"activation_cycles={self._activation_cycles}, base_csrs={self.base_csrs}, "
            f"csrs_per_operand_streamer={self.csrs_per_operand_streamer}, per_operation={self.per_operation})"
        )


_MODEL_BUILDERS = {
    "accfg": lambda spec: AccfgSetupCostModel(
        cycles_per_csr=spec["cycles_per_csr"],
        activation_cycles=spec.get("activation_cycles", 0),
        base_csrs=spec.get("base_csrs", 0),
        csrs_per_operand_streamer=spec.get("csrs_per_operand_streamer", 0),
        per_operation=spec.get("per_operation"),
        restream_cycles_per_tile=spec.get("restream_cycles_per_tile", 0),
        restream_tile_lanes=spec.get("restream_tile_lanes", 0),
        restream_per_row_block=spec.get("restream_per_row_block", 0),
        restream_axes=tuple(spec.get("restream_axes", (-1,))),
        refill_axes=tuple(spec.get("refill_axes", (-2,))),
    ),
}


def build_setup_cost_model(spec: dict[str, Any] | None) -> SetupCostModel:
    """Build a :class:`SetupCostModel` from a core's optional ``setup_cost`` YAML block.

    ``None`` (no block) -> :class:`ZeroSetupCostModel`, so cores that do not opt in keep
    the framework's standard zero-overhead behaviour. An unknown ``kind`` is a hard error
    (a typo should not silently disable the overhead).
    """
    if not spec:
        return ZeroSetupCostModel()
    kind = str(spec.get("kind", "accfg")).strip().lower()
    try:
        builder = _MODEL_BUILDERS[kind]
    except KeyError as exc:
        raise ValueError(
            f"unknown setup_cost kind {kind!r}; known kinds: {sorted(_MODEL_BUILDERS)}"
        ) from exc
    return builder(spec)
