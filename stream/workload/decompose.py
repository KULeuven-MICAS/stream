"""Registry mapping an op type to a node->Workload decomposer of affine sub-operators; overlays add entries via
``register_decomposition``."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from stream.workload.node import ComputationNode
    from stream.workload.workload import Workload

Decomposer = Callable[["ComputationNode"], "Workload"]

_REGISTRY: dict[str, Decomposer] = {}
_LOAD_STATE: dict[str, bool] = {"builtins": False}


def register_decomposition(op_type: str, decomposer: Decomposer) -> None:
    """Register (or override) the affine sub-operator decomposition for op ``op_type``."""
    _REGISTRY[op_type] = decomposer


def _ensure_builtins() -> None:
    """Register the in-tree decomposers lazily here (not in the decomposer modules) to avoid an import cycle."""
    if _LOAD_STATE["builtins"]:
        return
    _LOAD_STATE["builtins"] = True
    from stream.workload.normalization import NORMALIZATION_OPS, decompose_normalization  # noqa: PLC0415
    from stream.workload.rewrites.flash_attention import decompose_attention_block  # noqa: PLC0415

    for op_type in NORMALIZATION_OPS:
        register_decomposition(op_type, decompose_normalization)
    register_decomposition("AttentionBlock", decompose_attention_block)
    _load_plugins()


def _load_plugins() -> None:
    """Register out-of-tree decomposers from the ``stream.decompositions`` entry-point group (name = op type, object =
    decomposer)."""
    import logging  # noqa: PLC0415
    from importlib.metadata import entry_points  # noqa: PLC0415

    try:
        eps = entry_points(group="stream.decompositions")
    except Exception as exc:  # pragma: no cover - importlib.metadata edge cases
        logging.getLogger(__name__).debug("decomposition entry-point discovery failed: %s", exc)
        return
    for ep in eps:
        try:
            register_decomposition(ep.name, ep.load())
        except Exception as exc:  # pragma: no cover - a broken plugin must not break the registry
            logging.getLogger(__name__).warning("skipping decomposition plugin %r: %s", ep.name, exc)


def has_decomposition(node: ComputationNode) -> bool:
    """Whether ``node``'s op type has a registered affine sub-operator decomposition."""
    _ensure_builtins()
    return getattr(node, "type", None) in _REGISTRY


def decompose(node: ComputationNode) -> Workload | None:
    """The affine sub-operator subgraph of ``node``, or ``None`` if its op type has no decomposition."""
    _ensure_builtins()
    decomposer = _REGISTRY.get(getattr(node, "type", None))
    return decomposer(node) if decomposer is not None else None
