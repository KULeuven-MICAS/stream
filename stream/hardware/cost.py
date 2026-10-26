"""Coarse area and access-energy cost of a hardware bundle, and the budget guard around it."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from stream.hardware.bundle import HardwareBundle
from stream.plugins import load_group

__all__ = [
    "BudgetVerdict",
    "ComputeCost",
    "CoreCost",
    "HardwareBudget",
    "HardwareBudgetExceededError",
    "HardwareCostReport",
    "MemoryCost",
    "assert_within_budget",
    "check_budget",
    "evaluate_bundle_cost",
]

HARDWARE_COST_MODEL_GROUP = "stream.hardware_cost_model"


@dataclass(frozen=True)
class MemoryCost:
    """One memory, priced once: a physical memory of the accelerator, seen by ``core_ids``, or a level private to one
    core."""

    name: str
    core_ids: tuple[int, ...]
    style: str
    instances: int
    bits_per_instance: int
    total_bits: int
    access_width_bits: int
    nb_ports: int
    area_mm2: float
    read_energy_pj: float
    write_energy_pj: float
    authored_read_energy_pj: float | None


@dataclass(frozen=True)
class ComputeCost:
    core_id: int
    core_name: str
    modelled: bool
    nb_units: int
    input_format: str
    accumulator_format: str
    precision_declared: bool
    gate_equivalents: float
    area_mm2: float
    energy_per_op_pj: float
    authored_energy_per_op_pj: float | None


@dataclass(frozen=True)
class CoreCost:
    core_id: int
    name: str
    core_type: str
    memories: list[MemoryCost]  # its private levels; the memories it shares with others are the report's
    compute: ComputeCost | None
    area_mm2: float


@dataclass(frozen=True)
class HardwareCostReport:
    bundle_name: str
    technology_node: str
    technology_declared: bool
    cores: list[CoreCost]
    memories: list[MemoryCost]  # the accelerator's memories, each seen by its cores' top-level instances
    total_area_mm2: float
    memory_area_mm2: float
    compute_area_mm2: float
    on_die_memory_bits: int
    off_die_memory_bits: int
    peak_access_energy_pj_per_cycle: float
    off_die_access_energy_pj_per_cycle: float
    warnings: list[str] = field(default_factory=list)
    authored_disagreements: list[str] = field(default_factory=list)
    accuracy_claim: str | None = None

    @property
    def compute_modelled(self) -> bool | None:
        computes = [c.compute for c in self.cores if c.compute is not None]
        if not computes:
            return None
        return all(c.modelled for c in computes)

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "compute_modelled": self.compute_modelled}


@dataclass(frozen=True)
class _Density:
    read_pj_per_bit: float
    write_pj_per_bit: float
    sram_um2_per_bit: float
    mac_um2: float
    mac_pj: float


# Coarse 16 nm public defaults; detailed per-node models plug in via HARDWARE_COST_MODEL_GROUP.
_DENSITIES: dict[str, _Density] = {
    "n16": _Density(0.16, 0.18, 0.10, 200.0, 0.5),
}
DEFAULT_TECH_NODE = "n16"
DRAM_PJ_PER_BIT = 4.0


def evaluate_bundle_cost(bundle: HardwareBundle, technology_node: str | None = None) -> HardwareCostReport:
    plugins = load_group(HARDWARE_COST_MODEL_GROUP)
    if plugins:
        return plugins[-1].obj(bundle, technology_node=technology_node)
    return _evaluate_builtin(bundle, technology_node)


def _evaluate_builtin(bundle: HardwareBundle, technology_node: str | None) -> HardwareCostReport:
    warnings: list[str] = []
    declared = bool(technology_node or bundle.technology_node)
    node = (technology_node or bundle.technology_node or DEFAULT_TECH_NODE).lower()
    if not declared:
        warnings.append(f"bundle declares no `technology_node`; priced at {DEFAULT_TECH_NODE}")
    if node not in _DENSITIES:
        warnings.append(f"node '{node}' unknown to the built-in model; priced at {DEFAULT_TECH_NODE}")
        node = DEFAULT_TECH_NODE
    density = _DENSITIES[node]

    data = bundle.validated_data()
    cores_data = data["cores"]
    viewed = {view for memory in data["memories"].values() for view in memory["views"]}

    core_costs: list[CoreCost] = []
    for core_id, core in sorted(cores_data.items()):
        private = [name for name in _levels(core) if (core_id, name) not in viewed]
        memories = [_memory_cost(name, [(core_id, core, name)], True, density) for name in private]
        core_name = str(core.get("name", f"core_{core_id}"))
        compute = _compute_cost(core_id, core_name, core, density)
        core_costs.append(
            CoreCost(
                core_id=core_id,
                name=core_name,
                core_type=str(core.get("type", "")),
                memories=memories,
                compute=compute,
                area_mm2=sum(m.area_mm2 for m in memories) + (compute.area_mm2 if compute else 0.0),
            )
        )
    shared = []
    for name, memory in data["memories"].items():
        if memory["views"]:
            levels = [(cid, cores_data[cid], level) for cid, level in memory["views"]]
            shared.append(_memory_cost(name, levels, not memory["offchip"], density))
        else:
            shared.append(_own_memory_cost(name, memory, density))

    memories_all = [*shared, *(m for c in core_costs for m in c.memories)]
    on_die_memories = [m for m in memories_all if m.style == "sram"]
    off_die_memories = [m for m in memories_all if m.style == "off_die"]
    computes = [c.compute for c in core_costs if c.compute and c.compute.modelled]
    peak_energy_pj = sum(m.instances * m.nb_ports * m.read_energy_pj for m in on_die_memories) + sum(
        c.nb_units * c.energy_per_op_pj for c in computes
    )
    warnings = list(dict.fromkeys(warnings))

    memory_area = sum(m.area_mm2 for m in memories_all)
    compute_area = sum(c.compute.area_mm2 for c in core_costs if c.compute)
    return HardwareCostReport(
        bundle_name=bundle.name,
        technology_node=node,
        technology_declared=declared,
        cores=core_costs,
        memories=shared,
        total_area_mm2=memory_area + compute_area,
        memory_area_mm2=memory_area,
        compute_area_mm2=compute_area,
        on_die_memory_bits=sum(m.total_bits for m in on_die_memories),
        off_die_memory_bits=sum(m.total_bits for m in off_die_memories),
        peak_access_energy_pj_per_cycle=peak_energy_pj,
        off_die_access_energy_pj_per_cycle=sum(m.instances * m.nb_ports * m.read_energy_pj for m in off_die_memories),
        warnings=warnings,
    )


def _levels(core: dict[str, Any]) -> list[str]:
    """The memory levels a core file declares: a ZigZag core's by name, an AIE tile's one `memory`."""
    return list(core["memories"]) if "memories" in core else ["memory"]


def _level(core: dict[str, Any], name: str) -> tuple[int, int, dict[str, int]]:
    """A level's instances (one per array element it does not serve), bits per instance and port widths by name."""
    if "memories" not in core:
        width = int(core["memory"]["bandwidth_max"])
        return 1, int(core["memory"]["capacity"]), {"read": width, "write": width}
    level = core["memories"][name]
    served = set(level["served_dimensions"])
    instances = 1
    for dim, size in _oa_sizes(core).items():
        if dim not in served:
            instances *= max(int(size), 1)
    return instances, int(level["size"]), {port["name"]: int(port["bandwidth_max"]) for port in level["ports"]}


def _memory_cost(
    name: str, views: list[tuple[int, dict[str, Any], str]], on_die: bool, density: _Density
) -> MemoryCost:
    """One memory priced once from its views, which agree on its size: its ports are those any view declares, a port
    two views name alike counted once."""
    levels = [_level(core, level) for _, core, level in views]
    if len({(instances, bits) for instances, bits, _ in levels}) != 1:
        raise ValueError(f"the views of memory {name} disagree on its size: {[lv[:2] for lv in levels]}")
    instances, bits, _ = levels[0]
    ports = {port: width for *_, widths in levels for port, width in widths.items()}
    core_ids = tuple(core_id for core_id, _, _ in views)
    return _priced(name, core_ids, instances, bits, max(ports.values(), default=0), max(len(ports), 1), on_die, density)


def _own_memory_cost(name: str, memory: dict[str, Any], density: _Density) -> MemoryCost:
    """A memory no core views, priced from what it declares: its capacity and its own ports."""
    widths = [int(port["bandwidth"]) for port in memory["ports"].values()]
    return _priced(
        name,
        (),
        1,
        memory["capacity_bits"],
        max(widths, default=0),
        max(len(widths), 1),
        not memory["offchip"],
        density,
    )


def _priced(  # noqa: PLR0913
    name: str,
    core_ids: tuple[int, ...],
    instances: int,
    bits: int,
    width: int,
    nb_ports: int,
    on_die: bool,
    density: _Density,
) -> MemoryCost:
    if on_die:
        read_energy, write_energy = width * density.read_pj_per_bit, width * density.write_pj_per_bit
        area_mm2 = bits * instances * density.sram_um2_per_bit / 1e6
    else:
        read_energy = write_energy = width * DRAM_PJ_PER_BIT
        area_mm2 = 0.0
    return MemoryCost(
        name=name,
        core_ids=core_ids,
        style="sram" if on_die else "off_die",
        instances=instances,
        bits_per_instance=bits,
        total_bits=bits * instances,
        access_width_bits=width,
        nb_ports=nb_ports,
        area_mm2=area_mm2,
        read_energy_pj=read_energy,
        write_energy_pj=write_energy,
        authored_read_energy_pj=None,
    )


def _compute_cost(core_id: int, core_name: str, core: dict[str, Any], density: _Density) -> ComputeCost | None:
    array = core.get("operational_array")
    if not isinstance(array, dict):
        return ComputeCost(
            core_id=core_id,
            core_name=core_name,
            modelled=False,
            nb_units=0,
            input_format="",
            accumulator_format="",
            precision_declared=False,
            gate_equivalents=0.0,
            area_mm2=0.0,
            energy_per_op_pj=0.0,
            authored_energy_per_op_pj=None,
        )

    nb_units = 1
    for size in array.get("sizes") or []:
        nb_units *= max(int(size), 0)
    if nb_units <= 0:
        return None

    precision = core.get("operand_precision") or {}
    return ComputeCost(
        core_id=core_id,
        core_name=core_name,
        modelled=True,
        nb_units=nb_units,
        input_format=str(precision.get("input", "")),
        accumulator_format=str(precision.get("accumulator", "")),
        precision_declared=bool(precision),
        gate_equivalents=0.0,
        area_mm2=nb_units * density.mac_um2 / 1e6,
        energy_per_op_pj=density.mac_pj,
        authored_energy_per_op_pj=None,
    )


def _oa_sizes(core: dict[str, Any]) -> dict[str, int]:
    array = core.get("operational_array") or {}
    dims = array.get("dimensions") or []
    sizes = array.get("sizes") or []
    return {str(dim): int(size) for dim, size in zip(dims, sizes, strict=False)}


class HardwareBudgetExceededError(ValueError):
    """A hardware variant exceeds its silicon or power budget."""


@dataclass(frozen=True)
class HardwareBudget:
    max_area_mm2: float | None = None
    max_energy_pj_per_cycle: float | None = None
    label: str = "baseline"

    @classmethod
    def from_bundle(
        cls,
        bundle: HardwareBundle,
        headroom: float = 0.0,
        technology_node: str | None = None,
    ) -> HardwareBudget:
        report = evaluate_bundle_cost(bundle, technology_node=technology_node)
        scale = 1.0 + headroom
        return cls(
            max_area_mm2=report.total_area_mm2 * scale,
            max_energy_pj_per_cycle=report.peak_access_energy_pj_per_cycle * scale,
            label=f"{bundle.name} x{scale:g}",
        )


@dataclass(frozen=True)
class BudgetVerdict:
    ok: bool
    violations: list[str]
    report: HardwareCostReport

    def __str__(self) -> str:
        return "within budget" if self.ok else "; ".join(self.violations)


def check_budget(
    bundle: HardwareBundle,
    budget: HardwareBudget,
    technology_node: str | None = None,
) -> BudgetVerdict:
    report = evaluate_bundle_cost(bundle, technology_node=technology_node)
    violations: list[str] = []
    if budget.max_area_mm2 is not None and report.total_area_mm2 > budget.max_area_mm2:
        violations.append(
            f"area {report.total_area_mm2:.3f} mm2 exceeds budget {budget.max_area_mm2:.3f} mm2 "
            f"({report.total_area_mm2 / budget.max_area_mm2:.2f}x)"
        )
    if (
        budget.max_energy_pj_per_cycle is not None
        and report.peak_access_energy_pj_per_cycle > budget.max_energy_pj_per_cycle
    ):
        violations.append(
            f"peak access energy {report.peak_access_energy_pj_per_cycle:.3f} pJ/cycle exceeds budget "
            f"{budget.max_energy_pj_per_cycle:.3f} pJ/cycle"
        )
    return BudgetVerdict(ok=not violations, violations=violations, report=report)


def assert_within_budget(
    bundle: HardwareBundle,
    budget: HardwareBudget,
    technology_node: str | None = None,
) -> HardwareCostReport:
    verdict = check_budget(bundle, budget, technology_node=technology_node)
    if not verdict.ok:
        raise HardwareBudgetExceededError(f"Hardware bundle '{bundle.name}' vs budget '{budget.label}': {verdict}")
    return verdict.report
