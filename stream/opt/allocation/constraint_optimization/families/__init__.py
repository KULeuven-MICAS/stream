"""Constraint families: the groups of constraints and quantities an allocation model is built from."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from importlib import import_module
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, runtime_checkable

from stream.plugins import load_group, overlay_allowlist

if TYPE_CHECKING:
    from stream.opt.allocation.constraint_optimization.formulation import FormulationContext
    from stream.opt.solver import ObjectiveLevel

FAMILY_GROUP = "stream.constraint_families"
SLOT_PRESSURE = "slot_pressure"
TOTAL_LATENCY = "total_latency"
FamilySpec = str | Mapping[str, Any]
Build = Callable[["FormulationContext"], None]

LATENCY, OFFCHIP_TRAFFIC, DMA_PEAKS, BUFFERING, ROUTE_HOPS = 5, 4, 3, 2, 1
"""The priorities of Stream's objective levels: the latency decides first, then the off-chip traffic, the DMA
channel peaks, the buffering depth and the route length each break the ties of the level above."""

OFFCHIP_TIMED = "offchip_timed"
"""Registered by a family whose latency already pays for the time the off-chip links take."""

DEFAULT_FAMILIES: tuple[str, ...] = (
    "placement",
    "path_choice",
    "reuse_rates",
    "link_contention",
    "memory_capacity",
    "object_fifo_depth",
    "buffer_descriptors",
    "slot_latency",
    "reuse_levels",
    "output_reuse",
    "reuse_compatibility",
    "spatial_reuse",
    "overlap",
    "memory_ports",
    "dma_channels",
    "offchip_traffic",
)
"""Stream's own families, in every default set and built in this order where their requirements allow."""

BUILTIN_FAMILIES: dict[str, str] = {
    "placement": "routing:Placement",
    "path_choice": "routing:PathChoice",
    "reuse_rates": "reuse:ReuseRates",
    "link_contention": "routing:LinkContention",
    "memory_capacity": "memory:MemoryCapacity",
    "object_fifo_depth": "memory:ObjectFifoDepth",
    "buffer_descriptors": "memory:BufferDescriptors",
    "slot_latency": "latency:SlotLatency",
    "reuse_levels": "reuse:ReuseLevels",
    "output_reuse": "reuse:OutputReuse",
    "reuse_compatibility": "reuse:ReuseCompatibility",
    "spatial_reuse": "reuse:SpatialReuse",
    "overlap": "overlap:Overlap",
    "dma_channels": "dma:DmaChannels",
    "offchip_traffic": "offchip:OffchipTraffic",
    "aie2_object_fifo_depth": "aie2:AIE2ObjectFifoDepth",
    "aie2_buffer_descriptors": "aie2:AIE2BufferDescriptors",
    "aie2_memory_reuse": "aie2:AIE2MemoryReuse",
    "aie2_dma_channels": "aie2:AIE2DmaChannels",
    "memory_ports": "memory_ports:MemoryPorts",
}
"""Stream's own families by name, as ``module:factory`` within this package."""


class ConstraintFamily(Protocol):
    """Constraints and the quantities they define. ``build`` runs after every family that provides a name in
    ``requires``, and defines what ``provides`` names; a family is shared by the solves of a run, so it keeps
    no state of its own. The core decision variables exist before any family runs."""

    name: ClassVar[str]
    requires: ClassVar[tuple[str, ...]]
    provides: ClassVar[tuple[str, ...]]

    def build(self, ctx: FormulationContext) -> None: ...


@runtime_checkable
class DeclaringFamily(Protocol):
    """A family that defines quantities others read before its own ``build`` can run: ``declare`` runs once
    what ``declare_requires`` names is provided, and provides ``declares``."""

    declare_requires: ClassVar[tuple[str, ...]]
    declares: ClassVar[tuple[str, ...]]

    def declare(self, ctx: FormulationContext) -> None: ...


@runtime_checkable
class ObjectiveFamily(Protocol):
    """A family that contributes to the lexicographic objective once every family has built: the levels of one
    name, which share a priority, are summed into one."""

    def objective(self, ctx: FormulationContext) -> list[ObjectiveLevel]: ...


@runtime_checkable
class ReportingFamily(Protocol):
    """A family that adds sections to the solved allocation's performance report."""

    def report(self, ctx: FormulationContext) -> dict[str, Any]: ...


@dataclass(frozen=True)
class FamilySelection:
    """The families of a solve with the options each was built with, and ``steps``, the ``(family name,
    build)`` calls that build the model, in order."""

    families: tuple[ConstraintFamily, ...]
    options: tuple[dict[str, Any], ...]
    steps: tuple[tuple[str, Build], ...]

    def specs(self) -> tuple[tuple[str, dict[str, Any]], ...]:
        """Each family's name and options, in the order they were selected."""
        return tuple((family.name, options) for family, options in zip(self.families, self.options, strict=True))


def available_families() -> dict[str, Callable[..., ConstraintFamily]]:
    """Family factories by name: Stream's own and those of the ``stream.constraint_families`` entry points, discovered
    once per overlay allowlist, as every solve of a sweep resolves its families."""
    return dict(_discovered(overlay_allowlist()))


@cache
def _discovered(allow: frozenset[str] | None) -> dict[str, Callable[..., ConstraintFamily]]:
    factories: dict[str, Callable[..., ConstraintFamily]] = {}
    for name, path in BUILTIN_FAMILIES.items():
        module, factory = path.split(":")
        factories[name] = getattr(import_module(f"{__name__}.{module}"), factory)
    for plugin in load_group(FAMILY_GROUP, allow):
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


def load_families(specs: Sequence[FamilySpec]) -> FamilySelection:
    """Instantiate the families ``specs`` name and order their steps; an unknown name raises with the known ones, a
    requirement no selected family provides with the family that needs it, and a selection that defines no total
    latency, the first objective level, with the family that does."""
    if isinstance(specs, str):
        raise TypeError(f"Constraint families are a list of names, got the string {specs!r}")
    known = available_families() if specs else {}
    families: list[ConstraintFamily] = []
    options: list[dict[str, Any]] = []
    for name, family_options in map(parse_spec, specs):
        if name not in known:
            raise KeyError(f"Unknown constraint family {name!r}; available: {', '.join(sorted(known)) or 'none'}")
        if any(f.name == name for f in families):
            raise ValueError(f"Constraint family {name!r} is selected twice")
        try:
            families.append(known[name](**family_options))
        except TypeError as exc:
            raise TypeError(f"Constraint family {name!r}: {exc}") from None
        options.append(family_options)
    steps = _build_order(families)
    if not any(TOTAL_LATENCY in _provided(family) for family in families):
        raise ValueError(f"The constraint families {[f.name for f in families]} define no {TOTAL_LATENCY}; add overlap")
    return FamilySelection(tuple(families), tuple(options), steps)


def drop_families[Spec: FamilySpec](specs: Iterable[Spec], names: Iterable[str]) -> tuple[Spec, ...]:
    """``specs`` less the families ``names`` and every family that then requires what none of the rest provide."""
    kept = list(specs)
    dropped = set(names)
    if not dropped:
        return tuple(kept)
    known = available_families()
    while True:
        kept = [spec for spec in kept if parse_spec(spec)[0] not in dropped]
        factories = [known.get(parse_spec(spec)[0]) for spec in kept]
        provided = {name for factory in factories for name in _provided(factory)}
        unmet = {
            parse_spec(spec)[0]
            for spec, factory in zip(kept, factories, strict=True)
            if any(name not in provided for name in _required(factory))
        }
        if not unmet:
            return tuple(kept)
        dropped |= unmet


def _provided(family: Any) -> tuple[str, ...]:
    return (*getattr(family, "provides", ()), *getattr(family, "declares", ()))


def _required(family: Any) -> tuple[str, ...]:
    return (*getattr(family, "requires", ()), *getattr(family, "declare_requires", ()))


def _build_order(families: Sequence[ConstraintFamily]) -> tuple[tuple[str, Build], ...]:
    """A step runs once every step providing what it requires has: Stream's own families otherwise keep the
    order of :data:`DEFAULT_FAMILIES`, and any other family runs as early as its requirements allow."""
    steps: list[tuple[tuple[int, int], str, Build, tuple[str, ...], tuple[str, ...]]] = []
    for index, family in enumerate(families):
        rank = (1, DEFAULT_FAMILIES.index(family.name)) if family.name in DEFAULT_FAMILIES else (0, index)
        if isinstance(family, DeclaringFamily):
            steps.append((rank, family.name, family.declare, family.declare_requires, family.declares))
            steps.append((rank, family.name, family.build, (*family.requires, *family.declares), family.provides))
        else:
            steps.append((rank, family.name, family.build, family.requires, family.provides))
    providers: dict[str, list[int]] = {}
    for i, (*_, provides) in enumerate(steps):
        for name in provides:
            providers.setdefault(name, []).append(i)
    for _, family, _, requires, _ in steps:
        for name in requires:
            if name not in providers:
                raise ValueError(f"Constraint family {family!r} requires {name!r}, which no selected family provides")
    pending = sorted(range(len(steps)), key=lambda i: steps[i][0])
    done: set[int] = set()
    order: list[tuple[str, Build]] = []
    while pending:
        ready = next(
            (i for i in pending if all(p in done or p == i for r in steps[i][3] for p in providers[r])),
            None,
        )
        if ready is None:
            cycle = sorted({steps[i][1] for i in pending})
            raise ValueError(f"Constraint families {cycle} require each other's quantities")
        pending.remove(ready)
        done.add(ready)
        order.append((steps[ready][1], steps[ready][2]))
    return tuple(order)
