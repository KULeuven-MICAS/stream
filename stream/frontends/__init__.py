"""Pluggable workload ingestion: each format is a :class:`WorkloadFrontend` in a registry.

Engines depend on this protocol and the registry, never on a concrete frontend module.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from importlib.metadata import entry_points
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from stream.workload.workload import Workload

logger = logging.getLogger(__name__)

__all__ = [
    "WorkloadFrontend",
    "FrontendConfig",
    "register_frontend",
    "available_frontends",
    "frontend_for",
    "load_workload",
]

ENTRY_POINT_GROUP = "stream.frontends"


@dataclass(frozen=True)
class FrontendConfig:
    """Ingestion options common to every frontend; frontend-specific knobs go in ``options``."""

    options: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class WorkloadFrontend(Protocol):
    """Turns a model source into the internal affine :class:`~stream.workload.workload.Workload`."""

    name: str

    def can_load(self, source: Any) -> bool:
        """Whether this frontend recognises ``source`` (a path, a model proto, an exported program …)."""
        ...

    def load(self, source: Any, config: FrontendConfig | None = None) -> Workload:
        """Parse ``source`` into a workload graph."""
        ...


_REGISTRY: dict[str, WorkloadFrontend] = {}
_LOAD_STATE: dict[str, bool] = {"plugins": False}


def register_frontend(frontend: WorkloadFrontend) -> WorkloadFrontend:
    """Register ``frontend`` under its ``name`` (idempotent; last registration of a name wins)."""
    _REGISTRY[frontend.name] = frontend
    return frontend


def _load_plugins() -> None:
    """Discover out-of-tree frontends declared under the ``stream.frontends`` entry-point group."""
    if _LOAD_STATE["plugins"]:
        return
    _LOAD_STATE["plugins"] = True
    try:
        eps = entry_points(group=ENTRY_POINT_GROUP)
    except Exception as exc:  # pragma: no cover - importlib.metadata edge cases
        logger.debug("frontend entry-point discovery failed: %s", exc)
        return
    for ep in eps:
        try:
            obj = ep.load()
            # Instantiate a class, use an instance as-is. Key on type(): a class also passes
            # isinstance() against a runtime_checkable Protocol, so the Protocol check is wrong here.
            register_frontend(obj() if isinstance(obj, type) else obj)
        except Exception as exc:  # pragma: no cover - a broken plugin must not break ingestion
            logger.warning("skipping frontend plugin %r: %s", ep.name, exc)


def available_frontends() -> tuple[WorkloadFrontend, ...]:
    """Every registered frontend (built-in + discovered plugins)."""
    _ensure_builtins()
    _load_plugins()
    return tuple(_REGISTRY.values())


def frontend_for(source: Any, config: FrontendConfig | None = None) -> WorkloadFrontend:
    """The first registered frontend that accepts ``source``. Raises if none does."""
    for frontend in available_frontends():
        if frontend.can_load(source):
            return frontend
    names = ", ".join(sorted(_REGISTRY)) or "<none>"
    raise ValueError(f"no registered frontend can load {source!r} (available: {names})")


def load_workload(source: Any, config: FrontendConfig | None = None) -> Workload:
    """Ingest ``source`` into a :class:`~stream.workload.workload.Workload` via the matching frontend."""
    return frontend_for(source, config).load(source, config)


def _ensure_builtins() -> None:
    """Register the in-tree frontends, imported lazily so the registry never imports a concrete frontend."""
    if "onnx" not in _REGISTRY:
        from stream.frontends.onnx import OnnxFrontend  # noqa: PLC0415

        register_frontend(OnnxFrontend())
    if "torch_export" not in _REGISTRY:
        from stream.frontends.torch_export import TorchExportFrontend  # noqa: PLC0415

        register_frontend(TorchExportFrontend())
