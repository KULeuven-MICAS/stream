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
      restream_cycles_per_tile: 220 # per extra output-tile block on the streamed axis
      restream_tile_lanes: 8        # array lanes on that axis (ceil(extent/lanes) blocks)

There is also an optional **re-stream** term, charged per layer like the config: with a
stationary operand (SNAX gemmx is B-stationary) each extra block of the streamed output
axis re-streams the other operand, a cost zigzag's idealized (L1-resident) transfer model
overlaps with compute. It is ``(ceil(last_output_dim / restream_tile_lanes) - 1) *
restream_cycles_per_tile`` — **zero** for a single block, so single-output-block workloads
are unchanged. ``restream_cycles_per_tile = 0`` (default) disables it.

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


def _restream_tile_count(node: Any, lanes: int) -> int:
    """How many tiles the re-stream axis (the node's last output dim) splits into.

    For a stationary-operand dataflow (e.g. SNAX gemmx is B-stationary), each block of
    the streamed output axis re-streams the other operand. The number of blocks is
    ``ceil(extent / lanes)`` of the last output dimension. Returns 1 (no extra re-stream)
    when the extent cannot be read.
    """
    if lanes <= 0:
        return 1
    try:
        extent = int(node.outputs[0].shape[-1])
    except Exception:
        return 1
    return max(1, -(-extent // lanes))  # ceil


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
    ) -> None:
        self.cycles_per_csr = float(cycles_per_csr)
        self._activation_cycles = int(activation_cycles)
        self.base_csrs = int(base_csrs)
        self.csrs_per_operand_streamer = int(csrs_per_operand_streamer)
        self.per_operation = {str(k).strip().lower(): int(v) for k, v in (per_operation or {}).items()}
        self._default_op_csrs = self.per_operation.get("default", 0)
        # Per-output-tile re-stream: with a stationary operand, each extra block of the
        # streamed output axis re-streams the other operand -- a cost zigzag's idealized
        # (resident-operand) transfer model overlaps with compute. 0 (default) -> no term.
        self.restream_cycles_per_tile = int(restream_cycles_per_tile)
        self.restream_tile_lanes = int(restream_tile_lanes)

    def _operation_csrs(self, op_kind: str) -> int:
        return self.per_operation.get(op_kind, self._default_op_csrs)

    def cycles(self, node: Any) -> int:
        n_streamers = _count_operand_streamers(node)
        op_kind = _operation_kind(node)
        n_csr = self.base_csrs + self.csrs_per_operand_streamer * n_streamers + self._operation_csrs(op_kind)
        config = int(round(self.cycles_per_csr * max(n_csr, 0)))
        restream = 0
        if self.restream_cycles_per_tile:
            tiles = _restream_tile_count(node, self.restream_tile_lanes)
            restream = (tiles - 1) * self.restream_cycles_per_tile  # 0 for a single block
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
