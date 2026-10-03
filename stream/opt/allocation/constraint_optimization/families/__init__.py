"""Constraint families: optional groups of constraints and quantities an allocator builds on request."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, runtime_checkable

from stream.plugins import load_group

if TYPE_CHECKING:
    from stream.opt.allocation.constraint_optimization.quantities import QuantityRegistry
    from stream.opt.allocation.constraint_optimization.transfer_and_tensor_allocation import (
        TransferAndTensorAllocator,
    )

FAMILY_GROUP = "stream.constraint_families"
SLOT_PRESSURE = "slot_pressure"
FamilySpec = str | Mapping[str, Any]


class ConstraintFamily(Protocol):
    """``declare`` runs before the overlap is defined and ``constrain`` after it, both with the registry."""

    name: ClassVar[str]

    def declare(self, alloc: TransferAndTensorAllocator, q: QuantityRegistry) -> None: ...

    def constrain(self, alloc: TransferAndTensorAllocator, q: QuantityRegistry) -> None: ...


@runtime_checkable
class ReportingFamily(Protocol):
    """A family that adds sections to the solved schedule's performance report."""

    def report(self, alloc: TransferAndTensorAllocator, q: QuantityRegistry) -> dict[str, Any]: ...


def available_families() -> dict[str, Callable[..., ConstraintFamily]]:
    """Family factories by name from the ``stream.constraint_families`` entry points."""
    factories: dict[str, Callable[..., ConstraintFamily]] = {}
    for plugin in load_group(FAMILY_GROUP):
        if plugin.name in factories and factories[plugin.name] is not plugin.obj:
            raise ValueError(f"Constraint family {plugin.name!r} is registered twice")
        factories[plugin.name] = plugin.obj
    return factories


def parse_spec(spec: FamilySpec) -> tuple[str, dict[str, Any]]:
    """``"memory_ports"`` or ``{"memory_ports": {"burst": True}}`` to a name and its options."""
    if isinstance(spec, str):
        return spec, {}
    if len(spec) != 1:
        raise ValueError(f"A family entry names exactly one family, got {sorted(spec)}")
    ((name, options),) = spec.items()
    return name, dict(options or {})


def load_families(
    specs: Sequence[FamilySpec], factories: Mapping[str, Callable[..., ConstraintFamily]] | None = None
) -> tuple[ConstraintFamily, ...]:
    """Instantiate the families ``specs`` name, in order; an unknown name raises with the known ones."""
    if not specs:
        return ()
    if isinstance(specs, str):
        raise TypeError(f"Constraint families are a list of names, got the string {specs!r}")
    known = available_families() if factories is None else factories
    families: list[ConstraintFamily] = []
    for name, options in map(parse_spec, specs):
        if name not in known:
            raise KeyError(f"Unknown constraint family {name!r}; available: {', '.join(sorted(known)) or 'none'}")
        if any(f.name == name for f in families):
            raise ValueError(f"Constraint family {name!r} is selected twice")
        families.append(known[name](**options))
    return tuple(families)
