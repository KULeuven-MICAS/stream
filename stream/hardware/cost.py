"""Coarse area and access-energy cost of a hardware bundle, and the budget guard around it."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from stream.hardware.bundle import HardwareBundle
from stream.parser.core_validator import core_kind_from_type
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
    core_id: int
    core_name: str
    memory_name: str
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
    counted: bool
    alias_group: str | None


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
    on_die: bool
    memories: list[MemoryCost]
    compute: ComputeCost | None
    area_mm2: float


@dataclass(frozen=True)
class HardwareCostReport:
    bundle_name: str
    technology_node: str
    technology_declared: bool
    cores: list[CoreCost]
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
OFF_DIE_KINDS = frozenset({"offchip", "shim"})


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

    cores_data = bundle.validated_data()["cores"]
    alias_of, alias_owner = _alias_maps(bundle)

    core_costs: list[CoreCost] = []
    for core_id, core in sorted(cores_data.items()):
        core_name = str(core.get("name", f"core_{core_id}"))
        on_die = (core_kind_from_type(core.get("type")) or "compute") not in OFF_DIE_KINDS
        memories = _memory_costs(core_id, core_name, core, density, on_die, alias_of, alias_owner)
        compute = _compute_cost(core_id, core_name, core, density) if on_die else None
        core_costs.append(
            CoreCost(
                core_id=core_id,
                name=core_name,
                core_type=str(core.get("type", "")),
                on_die=on_die,
                memories=memories,
                compute=compute,
                area_mm2=sum(m.area_mm2 for m in memories) + (compute.area_mm2 if compute else 0.0),
            )
        )

    memories_all = [(c, m) for c in core_costs for m in c.memories]
    computes = [c.compute for c in core_costs if c.compute and c.compute.modelled]
    peak_energy_pj = sum(m.instances * m.nb_ports * m.read_energy_pj for _, m in memories_all if m.counted) + sum(
        c.nb_units * c.energy_per_op_pj for c in computes
    )
    off_die = [(c, m) for c, m in memories_all if not c.on_die]
    warnings = list(dict.fromkeys(warnings))

    memory_area = sum(m.area_mm2 for _, m in memories_all)
    compute_area = sum(c.compute.area_mm2 for c in core_costs if c.compute)
    return HardwareCostReport(
        bundle_name=bundle.name,
        technology_node=node,
        technology_declared=declared,
        cores=core_costs,
        total_area_mm2=memory_area + compute_area,
        memory_area_mm2=memory_area,
        compute_area_mm2=compute_area,
        on_die_memory_bits=sum(m.total_bits for _, m in memories_all if m.counted),
        off_die_memory_bits=sum(m.total_bits for _, m in off_die),
        peak_access_energy_pj_per_cycle=peak_energy_pj,
        off_die_access_energy_pj_per_cycle=sum(m.instances * m.nb_ports * m.read_energy_pj for _, m in off_die),
        warnings=warnings,
    )


def _alias_maps(bundle: HardwareBundle) -> tuple[dict[tuple[int, str], str], dict[str, tuple[int, str]]]:
    alias_of: dict[tuple[int, str], str] = {}
    alias_owner: dict[str, tuple[int, str]] = {}
    for group in bundle.memory_aliases:
        refs = [(int(r.split(".")[0]), r.split(".", 1)[1]) for r in group]
        label = "+".join(f"{cid}.{name}" for cid, name in refs)
        alias_owner[label] = refs[0]
        for ref in refs:
            alias_of[ref] = label
    return alias_of, alias_owner


def _memory_costs(
    core_id: int,
    core_name: str,
    core: dict[str, Any],
    density: _Density,
    on_die: bool,
    alias_of: dict[tuple[int, str], str],
    alias_owner: dict[str, tuple[int, str]],
) -> list[MemoryCost]:
    is_zigzag = "memories" in core
    declared: dict[str, dict[str, Any]] = dict(core["memories"]) if is_zigzag else {"memory": core.get("memory", {})}
    oa_sizes = _oa_sizes(core) if is_zigzag else {}

    costs: list[MemoryCost] = []
    for mem_name, mem in declared.items():
        if is_zigzag:
            ports = mem.get("ports") or []
            nb_ports = max(len(ports), 1)
            width = max((int(p.get("bandwidth_max", 0)) for p in ports), default=0)
            bits = int(mem.get("size", 0))
            served = set(mem.get("served_dimensions") or [])
            instances = 1
            for dim, size in oa_sizes.items():
                if dim not in served:
                    instances *= max(int(size), 1)
        else:
            instances, nb_ports = 1, 2
            width = int(mem.get("bandwidth_max", 0))
            bits = int(mem.get("capacity", 0))

        group_label = alias_of.get((core_id, mem_name))
        counted = on_die and (group_label is None or alias_owner[group_label] == (core_id, mem_name))

        if on_die:
            style = "sram"
            read_energy = width * density.read_pj_per_bit
            write_energy = width * density.write_pj_per_bit
            area_mm2 = bits * instances * density.sram_um2_per_bit / 1e6 if counted else 0.0
        else:
            style = "off_die"
            read_energy = width * DRAM_PJ_PER_BIT
            write_energy = width * DRAM_PJ_PER_BIT
            area_mm2 = 0.0

        costs.append(
            MemoryCost(
                core_id=core_id,
                core_name=core_name,
                memory_name=mem_name,
                style=style,
                instances=instances,
                bits_per_instance=bits,
                total_bits=bits * instances,
                access_width_bits=width,
                nb_ports=nb_ports,
                area_mm2=area_mm2,
                read_energy_pj=read_energy,
                write_energy_pj=write_energy,
                authored_read_energy_pj=None,
                counted=counted,
                alias_group=group_label,
            )
        )
    return costs


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
