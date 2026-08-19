"""Analytical area and access-energy model for a hardware bundle.

TPU7x YAMLs price growth at zero (``area: 0``, ``unit_energy_cost: 0``), so an agent minimising
latency wins by enlarging every memory and port. This module prices that unpriced axis, computing
everything from the quantities a mutation edits -- capacity, access width, port count, instance
count, array dimensions, operand precision -- never from a stored constant.

Area is in bitcell-equivalents (one 6T HD SRAM bitcell at the node), converted to mm2. Three cost
classes: SRAM macro (bits/eta * portfactor + IO_SLICE * width * ports), flop/latch array (fewer
than REGISTER_MIN_ROWS rows), and MAC array (NAND2 gate-equivalents from dimensions x precision).
Access energy per bit is a bank term (Horowitz ISSCC 2014, scaled by bitcell shrink and V^2) plus
a wire term (C.V^2 over half a macro side): linear in width, sub-linear in capacity.

Authored r_cost/w_cost/unit_energy values are left untouched and keep driving the engine; this
module reports its own figure alongside plus their ratio, and the budget guard uses the computed
one. Every constant below states where it comes from. Order-of-magnitude model: see ACCURACY_CLAIM.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

from stream.hardware.bundle import HardwareBundle
from stream.parser.core_validator import core_kind_from_type

__all__ = [
    "ACCURACY_CLAIM",
    "DEFAULT_TECH_NODE",
    "TECH_NODES",
    "BudgetVerdict",
    "ComputeCost",
    "CoreCost",
    "HardwareBudget",
    "HardwareBudgetExceededError",
    "HardwareCostReport",
    "MemoryCost",
    "TechNode",
    "assert_within_budget",
    "check_budget",
    "evaluate_bundle_cost",
]

# ── Technology constants ────────────────────────────────────────────────────────────────────────
# `sram_bitcell_um2`: published high-density 6T bitcell area for the node.
# `nand2_um2`: 2-input NAND in a 6-track library = 6 · (metal pitch) tall by 2 · CPP wide.
# `vdd_v`: nominal core supply.


@dataclass(frozen=True)
class TechNode:
    name: str
    sram_bitcell_um2: float
    nand2_um2: float
    vdd_v: float


TECH_NODES: dict[str, TechNode] = {
    # N7: 57 nm CPP, 40 nm minimum metal pitch, 0.027 µm² HD bitcell.
    "n7": TechNode("n7", sram_bitcell_um2=0.027, nand2_um2=6 * 0.040 * 2 * 0.057, vdd_v=0.75),
    # N5: 51 nm CPP, 30 nm minimum metal pitch, 0.021 µm² HD bitcell.
    "n5": TechNode("n5", sram_bitcell_um2=0.021, nand2_um2=6 * 0.030 * 2 * 0.051, vdd_v=0.75),
    # N4 is an N5 derivative: same bitcell, same pitches.
    "n4": TechNode("n4", sram_bitcell_um2=0.021, nand2_um2=6 * 0.030 * 2 * 0.051, vdd_v=0.75),
    # N3(E): 45 nm CPP, 23 nm minimum metal pitch, 0.0199 µm² HD bitcell — SRAM barely shrank
    # from N5, which is why memory dominates the area of a modern accelerator.
    "n3": TechNode("n3", sram_bitcell_um2=0.0199, nand2_um2=6 * 0.023 * 2 * 0.045, vdd_v=0.70),
}

DEFAULT_TECH_NODE = "n5"
"""Used when a bundle does not declare `technology_node`. Recorded as a warning, never silently."""

# ── Model constants ─────────────────────────────────────────────────────────────────────────────

SRAM_ARRAY_EFFICIENCY = 0.70
"""Bitcell area / macro area for a dense SRAM macro. Published dense macros land in 0.60–0.75."""

PORT_AREA_COEFF = 0.15
"""Cell growth per extra port, per axis. Set so a 2-port cell is 1.32x a 6T cell (8T ~= 1.3x)."""

IO_SLICE_BITCELLS = 100.0
"""Bitcell areas per accessed bit per port for the column IO stack (SA + write driver + mux + latch)."""

FLOP_BITCELL_RATIO = 5.0
"""Flip-flop bit / 6T bitcell. A DFF is ~6 tracks by ~11 CPP: ~0.10 um^2 at N5 vs 0.021 um^2."""

REGISTER_MIN_ROWS = 8
"""Below this many rows (ceil(bits/width)) a memory has no array structure and is flop-based."""

FULL_ADDER_NAND2 = 5.0
"""Gate-equivalents of a 1-bit full adder."""

FLOAT_DATAPATH_OVERHEAD = 1.3
"""Exponent path, normalisation and rounding, relative to the bare significand datapath."""

LOGIC_UTILIZATION = 0.6
"""Placed-and-routed standard-cell utilisation for a datapath block."""

WIRE_CAP_PF_PER_MM = 0.2
"""Global-metal capacitance per mm per bit. Roughly node-invariant; wires are why big memories cost."""

WRITE_READ_ENERGY_RATIO = 1.1
"""Write energy / read energy. Matches the authored w_cost:r_cost ratios (220:200, 130:120)."""

# Horowitz, ISSCC 2014 plenary: 32-bit read from an 8 KB SRAM = 5 pJ at 45 nm.
SRAM_BANK_ENERGY_PJ_PER_BIT_45NM = 5.0 / 32
# Intel 45 nm 6T HD SRAM bitcell (IEDM 2007).
SRAM_BITCELL_UM2_45NM = 0.346
# 45 nm 6-track NAND2, from a ~160 nm CPP / ~140 nm metal pitch library.
NAND2_UM2_45NM = 6 * 0.140 * 2 * 0.160
VDD_45NM_V = 1.1
# Calibrated on Horowitz's 45 nm fp32 multiply (3.7 pJ) + fp32 add (0.9 pJ) against this model's
# 3900 gate-equivalents. Reproduces his fp16-multiply and int8-multiply entries within ~2x.
GATE_ENERGY_FJ_45NM = 1.2

DRAM_ENERGY_PJ_PER_BIT = 4.0
"""HBM2E/HBM3-class energy per transferred bit, PHY included. Off-die: costs energy, not die area."""

AUTHORED_DISAGREEMENT_FACTOR = 2.0
"""Ratio beyond which an authored r_cost/unit_energy is reported as disagreeing with the model."""

ACCURACY_CLAIM = (
    "order-of-magnitude model with the right derivatives, not a signoff estimate: treat absolute "
    "figures as +/-2x, and relative comparisons between two bundles of the same shape as much tighter"
)
"""This module's own published tolerance, carried on every report so consumers quote it."""

OFF_DIE_KINDS = frozenset({"offchip", "shim"})
"""Core kinds that front external memory: they carry no on-die array of their declared capacity."""

# ── Operand precision ───────────────────────────────────────────────────────────────────────────
# Multiplier cost follows the *significand* width for floats and the full width for integers.

_FLOAT_FORMATS: dict[str, int] = {
    # name -> significand bits, implicit leading 1 included
    "fp8": 4,  # E4M3
    "fp8_e5m2": 3,
    "bf16": 8,
    "fp16": 11,
    "tf32": 11,
    "fp32": 24,
    "fp64": 53,
}
_INT_FORMATS: dict[str, int] = {"int4": 4, "int8": 8, "int16": 16, "int32": 32, "int64": 64}

DEFAULT_INPUT_FORMAT = "bf16"
DEFAULT_ACCUMULATOR_FORMAT = "fp32"


def _datapath_width(fmt: str | int) -> tuple[int, float]:
    """(multiplier/adder operand width, float overhead) for a declared operand format."""
    if isinstance(fmt, int):
        return fmt, 1.0
    key = str(fmt).strip().lower()
    if key in _FLOAT_FORMATS:
        return _FLOAT_FORMATS[key], FLOAT_DATAPATH_OVERHEAD
    if key in _INT_FORMATS:
        return _INT_FORMATS[key], 1.0
    raise ValueError(
        f"Unknown operand format '{fmt}'. Known: {sorted(_FLOAT_FORMATS)} and {sorted(_INT_FORMATS)}, "
        "or an integer bit width."
    )


# ── Report ──────────────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class MemoryCost:
    core_id: int
    core_name: str
    memory_name: str
    style: str  # "sram" | "register" | "off_die"
    instances: int
    bits_per_instance: int
    total_bits: int
    access_width_bits: int
    nb_ports: int
    area_mm2: float
    read_energy_pj: float
    """Modelled energy of one full-width read of one instance."""
    write_energy_pj: float
    authored_read_energy_pj: float | None
    """`r_cost` as authored, when non-zero. ZigZag charges it per max-bandwidth-wide access."""
    counted: bool
    """False when this is an aliased view of a memory already priced under another core."""
    alias_group: str | None


@dataclass(frozen=True)
class ComputeCost:
    core_id: int
    core_name: str
    modelled: bool
    """False when the core schema carries no array description (aie2 tiles) — unknown, not zero."""
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
    """Per-bundle silicon cost. `total_area_mm2` and `peak_access_energy_pj_per_cycle` are the two
    budgetable figures; everything else is the breakdown that explains them."""

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
    """Ceiling if every on-die port and MAC ran full-width for one cycle; monotone in what a mutation edits."""
    off_die_access_energy_pj_per_cycle: float
    warnings: list[str] = field(default_factory=list)
    authored_disagreements: list[str] = field(default_factory=list)
    """One line per memory/array whose authored per-access energy differs from the model by more than
    :data:`AUTHORED_DISAGREEMENT_FACTOR`; a countable list, not only a warnings sentence."""
    accuracy_claim: str = ACCURACY_CLAIM
    """What this module says its own numbers are worth. See :data:`ACCURACY_CLAIM`."""

    @property
    def compute_modelled(self) -> bool | None:
        """True/False if every compute core was/wasn't modelled; None when no core carries compute."""
        computes = [c.compute for c in self.cores if c.compute is not None]
        if not computes:
            return None
        return all(c.modelled for c in computes)

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "compute_modelled": self.compute_modelled}


# ── Cost evaluation ─────────────────────────────────────────────────────────────────────────────


def _tech_energy_scale(tech: TechNode, cell_um2_45nm: float) -> float:
    """Dynamic energy scaling 45 nm -> target node: capacitance shrinks with the linear dimension,
    energy with V^2."""
    linear_shrink = math.sqrt(tech.sram_bitcell_um2 / cell_um2_45nm) if cell_um2_45nm else 1.0
    return linear_shrink * (tech.vdd_v / VDD_45NM_V) ** 2


def _memory_area_and_energy(
    bits_per_instance: int,
    width_bits: int,
    nb_ports: int,
    instances: int,
    tech: TechNode,
) -> tuple[str, float, float]:
    """(style, total area in mm², read energy in pJ for one full-width access of one instance)."""
    width_bits = max(width_bits, 1)
    rows = max(1, math.ceil(bits_per_instance / width_bits))
    port_factor = (1.0 + PORT_AREA_COEFF * (max(nb_ports, 1) - 1)) ** 2

    if rows < REGISTER_MIN_ROWS:
        style = "register"
        bitcells_per_instance = bits_per_instance * FLOP_BITCELL_RATIO * port_factor
    else:
        style = "sram"
        bitcells_per_instance = (
            bits_per_instance / SRAM_ARRAY_EFFICIENCY * port_factor + IO_SLICE_BITCELLS * width_bits * nb_ports
        )

    area_um2_per_instance = bitcells_per_instance * tech.sram_bitcell_um2
    area_mm2 = area_um2_per_instance * instances / 1e6

    # Wire energy scales with how far a bit travels inside its own macro: half a side of its square.
    side_mm = math.sqrt(area_um2_per_instance / 1e6)
    bank_energy = SRAM_BANK_ENERGY_PJ_PER_BIT_45NM * _tech_energy_scale(tech, SRAM_BITCELL_UM2_45NM)
    wire_energy = WIRE_CAP_PF_PER_MM * tech.vdd_v**2  # C·V² on a full-swing global wire
    energy_per_bit = bank_energy + wire_energy * 0.5 * side_mm
    return style, area_mm2, energy_per_bit * width_bits


def _zigzag_memory_geometry(mem: dict[str, Any], oa_sizes: dict[str, int]) -> tuple[int, int, int]:
    """(instances, access width in bits, port count) for one ZigZag memory declaration."""
    ports = mem.get("ports") or []
    nb_ports = max(len(ports), 1)
    width = max((int(p.get("bandwidth_max", 0)) for p in ports), default=0)
    served = set(mem.get("served_dimensions") or [])
    # A memory not served along a dimension is replicated across it — `served_dimensions: []` on the
    # weight registers means one register per PE.
    instances = 1
    for dim, size in oa_sizes.items():
        if dim not in served:
            instances *= max(int(size), 1)
    return instances, width, nb_ports


def _compute_cost(
    core_id: int,
    core_name: str,
    core: dict[str, Any],
    tech: TechNode,
    warnings: list[str],
) -> ComputeCost | None:
    array = core.get("operational_array")
    if not isinstance(array, dict):
        # aie2 tiles describe no array: report "not modelled", not 0 (which would claim free compute).
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
        return None  # memory-only core: the array exists in the schema but has no units

    precision = core.get("operand_precision") or {}
    declared = bool(precision)
    if not declared:
        warnings.append(
            f"core '{core_name}' declares no `operand_precision`; "
            f"priced as {DEFAULT_INPUT_FORMAT} x {DEFAULT_ACCUMULATOR_FORMAT}"
        )
    input_format = precision.get("input", DEFAULT_INPUT_FORMAT)
    accumulator_format = precision.get("accumulator", DEFAULT_ACCUMULATOR_FORMAT)

    m_in, in_overhead = _datapath_width(input_format)
    m_acc, acc_overhead = _datapath_width(accumulator_format)
    gates = FULL_ADDER_NAND2 * m_in**2 * in_overhead + FULL_ADDER_NAND2 * m_acc * acc_overhead

    area_mm2 = nb_units * gates * tech.nand2_um2 / LOGIC_UTILIZATION / 1e6
    energy_pj = gates * GATE_ENERGY_FJ_45NM / 1000 * _tech_energy_scale(tech, NAND2_UM2_45NM)
    authored = array.get("unit_energy")
    return ComputeCost(
        core_id=core_id,
        core_name=core_name,
        modelled=True,
        nb_units=nb_units,
        input_format=str(input_format),
        accumulator_format=str(accumulator_format),
        precision_declared=declared,
        gate_equivalents=gates,
        area_mm2=area_mm2,
        energy_per_op_pj=energy_pj,
        authored_energy_per_op_pj=float(authored) if authored else None,
    )


def evaluate_bundle_cost(bundle: HardwareBundle, technology_node: str | None = None) -> HardwareCostReport:
    """Price a bundle: area in mm², peak access energy in pJ/cycle, plus the full breakdown.

    Everything is derived from the bundle's own declarations, so mutating a capacity, a port width,
    a port count or an array dimension moves the result.
    """
    warnings: list[str] = []
    node_name = (technology_node or bundle.technology_node or DEFAULT_TECH_NODE).lower()
    declared = bool(technology_node or bundle.technology_node)
    if not declared:
        warnings.append(f"bundle declares no `technology_node`; priced at {DEFAULT_TECH_NODE}")
    if node_name not in TECH_NODES:
        raise ValueError(f"Unknown technology node '{node_name}'. Known: {sorted(TECH_NODES)}.")
    tech = TECH_NODES[node_name]

    cores_data = bundle.validated_data()["cores"]

    # An aliased memory is one physical macro seen from several cores: the first ref owns it and is
    # billed, the rest are views.
    alias_of: dict[tuple[int, str], str] = {}
    alias_owner: dict[str, tuple[int, str]] = {}
    for group in bundle.memory_aliases:
        refs = [(int(r.split(".")[0]), r.split(".", 1)[1]) for r in group]
        label = "+".join(f"{cid}.{name}" for cid, name in refs)
        alias_owner[label] = refs[0]
        for ref in refs:
            alias_of[ref] = label

    core_costs: list[CoreCost] = []
    for core_id, core in sorted(cores_data.items()):
        core_name = str(core.get("name", f"core_{core_id}"))
        on_die = (core_kind_from_type(core.get("type")) or "compute") not in OFF_DIE_KINDS
        memories = _memory_costs(core_id, core_name, core, tech, on_die, alias_of, alias_owner, warnings)
        compute = _compute_cost(core_id, core_name, core, tech, warnings) if on_die else None
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
    # "Peak" = every counted port and every MAC running at full width for one cycle.
    peak_energy_pj = sum(m.instances * m.nb_ports * m.read_energy_pj for _, m in memories_all if m.counted) + sum(
        c.nb_units * c.energy_per_op_pj for c in computes
    )
    off_die = [(c, m) for c, m in memories_all if not c.on_die]

    disagreements = _authored_disagreements(core_costs)
    if disagreements:
        warnings.append(
            "authored access energies disagree with the model by more than "
            f"{AUTHORED_DISAGREEMENT_FACTOR:g}x (the model is what the budget uses): " + "; ".join(disagreements)
        )
    warnings = list(dict.fromkeys(warnings))  # one line per distinct issue, not per repeated core

    memory_area = sum(m.area_mm2 for _, m in memories_all)
    compute_area = sum(c.compute.area_mm2 for c in core_costs if c.compute)
    return HardwareCostReport(
        bundle_name=bundle.name,
        technology_node=tech.name,
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
        authored_disagreements=disagreements,
    )


def _memory_costs(  # noqa: PLR0913 -- one call site; splitting it would only move the arguments
    core_id: int,
    core_name: str,
    core: dict[str, Any],
    tech: TechNode,
    on_die: bool,
    alias_of: dict[tuple[int, str], str],
    alias_owner: dict[str, tuple[int, str]],
    warnings: list[str],
) -> list[MemoryCost]:
    """Price every memory one core declares."""
    # ZigZag cores declare a named hierarchy; aie2 cores a single unnamed tile memory.
    is_zigzag = "memories" in core
    declared: dict[str, dict[str, Any]] = dict(core["memories"]) if is_zigzag else {"memory": core.get("memory", {})}
    oa_sizes = _oa_sizes(core) if is_zigzag else {}
    if not is_zigzag:
        warnings.append(f"core '{core_name}' declares no memory ports; priced as 1R1W")

    costs: list[MemoryCost] = []
    for mem_name, mem in declared.items():
        if is_zigzag:
            instances, width, nb_ports = _zigzag_memory_geometry(mem, oa_sizes)
            bits = int(mem.get("size", 0))
            authored_r = float(mem.get("r_cost") or 0.0) or None
        else:
            instances, nb_ports = 1, 2  # aie2: one tile memory, read + write
            width = int(mem.get("bandwidth_max", 0))
            bits = int(mem.get("capacity", 0))
            authored_r = None

        alias_key = (core_id, mem_name)
        group_label = alias_of.get(alias_key)
        counted = on_die and (group_label is None or alias_owner[group_label] == alias_key)

        if on_die:
            style, area_mm2, read_energy = _memory_area_and_energy(bits, width, nb_ports, instances, tech)
        else:
            style, area_mm2 = "off_die", 0.0
            read_energy = DRAM_ENERGY_PJ_PER_BIT * width

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
                area_mm2=area_mm2 if counted else 0.0,
                read_energy_pj=read_energy,
                write_energy_pj=read_energy * WRITE_READ_ENERGY_RATIO,
                authored_read_energy_pj=authored_r,
                counted=counted,
                alias_group=group_label,
            )
        )
    return costs


def _disagrees(modelled: float, authored: float | None) -> bool:
    if not authored or modelled <= 0:
        return False
    ratio = modelled / authored
    return ratio > AUTHORED_DISAGREEMENT_FACTOR or ratio < 1 / AUTHORED_DISAGREEMENT_FACTOR


def _authored_disagreements(core_costs: list[CoreCost]) -> list[str]:
    """Where the model and the authored per-access energies part company, so it can be argued with."""
    lines: list[str] = []
    for core in core_costs:
        for mem in core.memories:
            if _disagrees(mem.read_energy_pj, mem.authored_read_energy_pj):
                lines.append(
                    f"{core.name}.{mem.memory_name}: authored {mem.authored_read_energy_pj:.3g} pJ, "
                    f"modelled {mem.read_energy_pj:.3g} pJ"
                )
        compute = core.compute
        if compute and compute.modelled and _disagrees(compute.energy_per_op_pj, compute.authored_energy_per_op_pj):
            lines.append(
                f"{core.name}.operational_array: authored {compute.authored_energy_per_op_pj:.3g} pJ/op, "
                f"modelled {compute.energy_per_op_pj:.3g} pJ/op"
            )
    return sorted(set(lines))


def _oa_sizes(core: dict[str, Any]) -> dict[str, int]:
    array = core.get("operational_array") or {}
    dims = array.get("dimensions") or []
    sizes = array.get("sizes") or []
    return {str(dim): int(size) for dim, size in zip(dims, sizes, strict=False)}


# ── Budget guard ────────────────────────────────────────────────────────────────────────────────


class HardwareBudgetExceededError(ValueError):
    """A hardware variant costs more silicon (or power) than its budget allows."""


@dataclass(frozen=True)
class HardwareBudget:
    """A hard ceiling on what a hardware variant may cost.

    The default is the baseline bundle's own cost, so "better" means better at equal-or-less
    silicon. Without that, minimising latency degenerates into growing everything.
    """

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
        """Budget equal to this bundle's cost, optionally with fractional `headroom` on top."""
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
    """Price `bundle` and compare it against `budget`. Pure arithmetic on the YAML — no solve."""
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
    """Raise :class:`HardwareBudgetExceededError` when the bundle busts its budget."""
    verdict = check_budget(bundle, budget, technology_node=technology_node)
    if not verdict.ok:
        raise HardwareBudgetExceededError(f"Hardware bundle '{bundle.name}' vs budget '{budget.label}': {verdict}")
    return verdict.report
