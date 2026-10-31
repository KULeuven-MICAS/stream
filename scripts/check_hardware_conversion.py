"""Check that a hardware file converted from the core format describes the accelerator the original did.

Each file is summarised by the Stream that reads it: per memory its capacity, ports, measured bandwidth, DMA and
banks; the connections with the ports they pass; the cores; and the latency, energy and memory occupancy of one
workload's solve. The two summaries are then compared. A memory is known by its views (``<core id>.<instance>``
joined by ``+``), the off-chip memory as ``offchip``, and ports and connections by what they are, so names may change.

    python scripts/check_hardware_conversion.py check OLD.yaml NEW.yaml --old-python "/opt/stream-old/bin/python"
    python scripts/check_hardware_conversion.py summarize HARDWARE.yaml SUMMARY.json
    python scripts/check_hardware_conversion.py compare OLD.json NEW.json

``summarize`` runs under either Stream; ``check`` runs it for OLD.yaml under ``--old-python``. The exit status is 1 if
the summaries differ. See docs/source/hardware.md, "Converting a hardware file from the core format".
"""

from __future__ import annotations

import argparse
import contextlib
import json
import shlex
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

from zigzag.datatypes import MemoryOperand

from stream.api import SolveOptions, evaluate_mapping
from stream.cost_model.bandwidth import BandwidthModel
from stream.stages.parsing.accelerator_parser import parse_accelerator

OFFCHIP = "offchip"
EVERY_OPERAND = {"input1", "input2", "output"}
DEFAULT_WORKLOAD = Path(__file__).resolve().parents[1] / "stream/inputs/testing/workload/2conv_1_8_32_32_16_32_3.onnx"


def _number(value: float) -> str:
    return f"{value:.6g}"


def _model(model: BandwidthModel) -> str:
    text = f"ceiling={_number(model.ceiling)} contiguous={_number(model.contiguous)}"
    if model.burst:
        text += f" burst={model.burst}"
    for side, table in sorted(model.strided.items()):
        if table:
            text += f" {side}=" + ",".join(f"{span}:{_number(rate)}" for span, rate in sorted(table.items()))
    return text


def _services(serves: frozenset[tuple[str, str]], own: bool) -> str:
    """What a port serves. A port the memory declares itself (``own``) serves reads and writes, so an off-chip core's
    datapath directions count as those, and serving every operand is serving ``any``."""
    if not own:
        return ",".join(sorted(f"{role}:{direction}" for role, direction in serves))
    roles: dict[str, set[str]] = defaultdict(set)
    for role, direction in serves:
        roles[direction.removesuffix("_by_datapath")].add(role)
    every = {direction for direction, served in roles.items() if "any" in served or served >= EVERY_OPERAND}
    pairs = [f"any:{direction}" for direction in every]
    pairs += [f"{role}:{d}" for d, served in roles.items() if d not in every for role in served]
    return ",".join(sorted(pairs))


def _port(port: Any, cores: tuple[int, ...], channels: int | None) -> str:
    text = (
        f"cores={','.join(map(str, sorted(cores))) or '-'} serves={_services(port.serves, not cores)} "
        f"{_model(port.bandwidth)} energy={_number(port.read_energy_per_bit)}/{_number(port.write_energy_per_bit)}"
    )
    return text + (f" channels={channels}" if channels else "")


def _memory(
    offchip: bool, capacity_bits: int, ports: list[str], measured: Any, dma: dict | None, banks: Any
) -> dict[str, Any]:
    return {
        "offchip": offchip,
        "capacity_bits": capacity_bits,
        "ports": sorted(ports),
        "measured": _model(measured) if measured else None,
        "dma": dma,
        "banks": asdict(banks) if banks else None,
    }


def _connection(kind: str, bandwidth: float, unit_energy_cost: float, ends: dict[str, set[str]]) -> dict[str, Any]:
    return {
        "type": kind,
        "bandwidth": _number(bandwidth),
        "unit_energy_cost": _number(unit_energy_cost),
        "ends": {memory: sorted(ports) for memory, ports in sorted(ends.items())},
    }


def _core(core: Any) -> dict[str, Any]:
    coordinates = None if core.col_id is None else [core.col_id, core.row_id]
    return {"type": core.core_type, "coordinates": coordinates, "operator_types": core.operator_types}


def _memory_key(memory: Any) -> str:
    if memory.offchip:
        return OFFCHIP
    return "+".join(sorted(f"{core.id}.{instance}" for core, instance in memory.views)) or memory.name


def _summary(accelerator: Any) -> tuple[dict[str, Any], Any]:
    """The facts of an accelerator in the memories format, and the memory key of an occupancy row."""
    registry = {port.key: port for port in accelerator.ports}
    memories = {}
    for memory in accelerator.memories.values():
        channels = sorted(port.channels for port in memory.ports.values() if port.channels)
        dma = {"channels": channels, **(asdict(memory.dma) if memory.dma else {})} if channels else None
        ports = [
            _port(port, port.core_ids, port.channels) for key, port in registry.items() if key.memory == memory.name
        ]
        memories[_memory_key(memory)] = _memory(
            memory.offchip, memory.capacity_bits, ports, accelerator.bandwidth.get(memory), dma, memory.banks
        )
    connections = {}
    for _, _, data in accelerator.network.edges(data=True):
        link = data["cl"]
        key = id(link) if link.members else (frozenset(memory.name for memory in link.memories), link.parallel)
        ends = {
            _memory_key(memory): {
                _port(registry[port], registry[port].core_ids, registry[port].channels)
                for port in (memory.ports[name].key for name in names)
                if port in registry
            }
            for memory, names in link.ports.items()
        }
        connections[key] = _connection("bus" if link.members else "link", link.bandwidth, link.unit_energy_cost, ends)
    facts = {
        "cores": {str(core.id): _core(core) for core in accelerator.core_list},
        "memories": memories,
        "connections": list(connections.values()),
    }
    return facts, lambda row: _memory_key(accelerator.memories[row["memory"]])


def _top_instances(core: Any) -> dict[str, int]:
    """A core-format core's top-level memory instances and their sizes in bits."""
    if core.namespace == "aie2":
        return {"memory": core.memory_capacity_bits}
    hierarchy = core.memory_hierarchy
    levels = (hierarchy.get_memory_levels(operand)[-1] for operand in hierarchy.get_operands())
    return {level.name: level.memory_instance.size for level in levels}


def _allocated_instance(core: Any) -> str:
    """The instance a core-format core's memory was sized by and its links reached: the top level of ``I1``."""
    if core.namespace == "aie2":
        return "memory"
    hierarchy = core.memory_hierarchy
    if MemoryOperand("I1") in hierarchy.get_operands():
        return hierarchy.get_memory_levels(MemoryOperand("I1"))[-1].name
    return min(_top_instances(core))


def _routes_elsewhere(accelerator: Any, core: Any) -> bool:
    """Whether a core-format core's traffic entered and left through the first core of its `core_memory_sharing`
    group, as it did where that one had links, leaving the core's own links unused."""
    owner = accelerator.memory_of(core)
    return owner is not core and accelerator.cores.degree(owner) > 0


def _core_format_views(accelerator: Any, tops: dict[int, dict[str, int]]) -> dict[tuple[int, str], str]:
    """The memory key of each top-level instance of a core-format accelerator: one memory per `core_memory_sharing`
    group (by instance name, else the first core's ``I1`` instance) and per `memory_aliases` group."""
    parent: dict[tuple[int, str], tuple[int, str]] = {}

    def root(view: tuple[int, str]) -> tuple[int, str]:
        while parent.get(view, view) != view:
            view = parent[view]
        return view

    for core_id, instances in tops.items():
        owner = accelerator.shared_mem_group_ids.get(core_id, core_id)
        for instance in instances:
            shared = instance if instance in tops[owner] else _allocated_instance(accelerator.get_core(owner))
            parent[root((core_id, instance))] = root((owner, shared))
    for alias, owner in accelerator.memory_aliases.items():
        parent[root(alias)] = root(owner)
    groups: dict[tuple[int, str], list[tuple[int, str]]] = defaultdict(list)
    for core_id, instances in tops.items():
        for instance in instances:
            groups[root((core_id, instance))].append((core_id, instance))
    return {
        view: "+".join(sorted(f"{core_id}.{instance}" for core_id, instance in members))
        for members in groups.values()
        for view in members
    }


def _core_format_summary(accelerator: Any) -> tuple[dict[str, Any], Any]:
    """:func:`_summary` of an accelerator parsed by the core-format Stream, whose memories are its cores' top-level
    instances, grouped by :func:`_core_format_views`, and its off-chip core's memory."""
    offchip_id = accelerator.offchip_core_id
    cores = {core.id: core for core in accelerator.core_list}
    offchip = cores[offchip_id]
    tops = {core_id: _top_instances(core) for core_id, core in cores.items() if core_id != offchip_id}
    key_of = _core_format_views(accelerator, tops)

    def memory_key(core_id: int, instance: str | None = None) -> str:
        if core_id == offchip_id:
            return OFFCHIP
        return key_of[(core_id, instance or _allocated_instance(cores[core_id]))]

    def memories_of(core_id: int) -> set[str]:
        return {OFFCHIP} if core_id == offchip_id else {key_of[(core_id, instance)] for instance in tops[core_id]}

    def port_text(port: Any) -> str:
        if port.key.share_group != offchip_id:
            return _port(port, port.core_ids, None)
        return _port(port, (), getattr(offchip, f"dma_{port.key.port}", None))

    ports_on: dict[str, list[str]] = defaultdict(list)
    port_memory = {}
    for port in accelerator.ports:
        share_group, instance, _ = port.key
        port_memory[port.key] = memory_key(share_group, "memory" if instance == "dma" else instance)
        ports_on[port_memory[port.key]].append(port_text(port))
    measured = {memory_key(core_id): model for core_id, model in accelerator.bandwidth.items()}
    banks = {memory_key(core_id): cores[core_id].banks for core_id in tops if cores[core_id].banks}
    memories = {
        key: _memory(False, tops[core_id][instance], ports_on[key], measured.get(key), None, banks.get(key))
        for (core_id, instance), key in key_of.items()
    }
    offchip_dma = None
    if offchip.namespace == "aie2" and offchip.dma_channel_bits:
        offchip_dma = {"channels": sorted([offchip.dma_mm2s, offchip.dma_s2mm])}
        if offchip.dma_buffer_descriptors:
            offchip_dma |= {"buffer_descriptors": offchip.dma_buffer_descriptors, "iterations": offchip.dma_iterations}
    memories[OFFCHIP] = _memory(
        True, offchip.get_memory_capacity(), ports_on[OFFCHIP], measured.get(OFFCHIP), offchip_dma, None
    )
    connections = {}
    for a, b, data in accelerator.cores.edges(data=True):
        link = data["cl"]
        if not link.members and (_routes_elsewhere(accelerator, a) or _routes_elsewhere(accelerator, b)):
            continue
        ends: dict[str, set[str]] = defaultdict(set)
        for core in link.members or (a, b):
            for key in memories_of(core.id):
                ends[key] |= {
                    port_text(port) for port in accelerator.ports.ports_of(core) if port_memory[port.key] == key
                }
        key = id(link) if link.members else frozenset((a.id, b.id))
        connections[key] = _connection("bus" if link.members else "link", link.bandwidth, link.unit_energy_cost, ends)
    facts = {
        "cores": {str(core_id): _core(core) for core_id, core in cores.items() if core_id != offchip_id},
        "memories": memories,
        "connections": list(connections.values()),
    }
    return facts, lambda row: memory_key(row["core_id"])


def _solve(hardware: str, workload: str, mapping: str | None, cols: int | None, memory_key: Any) -> dict[str, Any]:
    """The latency, energy and, per memory, the peak resident bits and the capacity over the groups of a solve."""
    options = SolveOptions(artifacts=False) if cols is None else SolveOptions(artifacts=False, nb_cols_to_use=cols)
    with tempfile.TemporaryDirectory() as output, contextlib.redirect_stdout(sys.stderr):
        context = evaluate_mapping(hardware, workload, output, mapping=mapping, options=options).context
    energy, occupancy = 0.0, {}
    for allocation in context.get("group_allocations").values():
        performance = allocation["performance"]
        energy += performance["energy"]["total_pj"]
        held: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for row in performance["memory_occupancy"] or ():
            held[memory_key(row)][0] += row["resident_bits"]
            held[memory_key(row)][1] += row["capacity_bits"]
        for key, (resident, capacity) in held.items():
            peak = occupancy.get(key, (0, 0))
            occupancy[key] = (max(peak[0], resident), max(peak[1], capacity))
    return {"latency": context.get("total_latency"), "energy_pj": energy, "occupancy": occupancy}


def summarize(hardware: str, workload: str | None, mapping: str | None, cols: int | None) -> dict[str, Any]:
    accelerator = parse_accelerator(hardware)
    facts, memory_key = (_summary if hasattr(accelerator, "memories") else _core_format_summary)(accelerator)
    if workload is not None:
        facts["solve"] = _solve(hardware, workload, mapping, cols, memory_key)
    return facts


def _relative(old: float, new: float) -> float:
    return abs(new - old) / abs(old) if old else abs(new)


def _same_occupancy(old: list[tuple[int, int]], new: list[tuple[int, int]], rtol: float) -> bool:
    """Whether the memories hold the same bits up to a permutation of equal memories, as a tie between them may break
    either way."""
    return len(old) == len(new) and all(
        a[1] == b[1] and _relative(a[0], b[0]) <= rtol for a, b in zip(old, new, strict=True)
    )


def _changes(label: str, old: list[str], new: list[str]) -> list[str]:
    lost, added = Counter(old) - Counter(new), Counter(new) - Counter(old)
    return [f"{label} lost: {item}" for item in lost.elements()] + [
        f"{label} added: {item}" for item in added.elements()
    ]


def _head(connection: dict[str, Any]) -> str:
    return (
        f"{connection['type']} of {connection['bandwidth']} bits/cycle, energy {connection['unit_energy_cost']}, "
        f"joining {' & '.join(connection['ends'])}"
    )


def _entry_differences(old: dict[str, Any], new: dict[str, Any], section: str, noun: str) -> list[str]:
    differences = []
    for key in sorted(old[section].keys() | new[section].keys()):
        a, b = old[section].get(key), new[section].get(key)
        if a is None or b is None:
            differences.append(f"{noun} {key}: only in the {'new' if a is None else 'old'} file")
            continue
        for field in (field for field in a if a[field] != b[field]):
            if field == "ports":
                differences += _changes(f"{noun} {key}: port", a[field], b[field])
            else:
                differences.append(f"{noun} {key}: {field} {a[field]} -> {b[field]}")
    return differences


def _connection_differences(old: list[dict[str, Any]], new: list[dict[str, Any]]) -> list[str]:
    """Connections lost and added, and for one of the same kind, rate and memories, the ports that changed."""
    by_head: list[dict[str, list[dict[str, Any]]]] = [defaultdict(list), defaultdict(list)]
    for side, connections in zip(by_head, (old, new), strict=True):
        for connection in connections:
            side[_head(connection)].append(connection["ends"])
    differences = []
    for head in sorted(by_head[0].keys() | by_head[1].keys()):
        a, b = by_head[0][head], by_head[1][head]
        for ends_a, ends_b in zip([e for e in a if e not in b], [e for e in b if e not in a], strict=False):
            for memory in ends_a:
                differences += _changes(f"{head}: port at {memory}", ends_a[memory], ends_b.get(memory, []))
        if len(a) != len(b):
            word = "lost" if len(a) > len(b) else "added"
            differences += [f"connection {word}: {head}"] * abs(len(a) - len(b))
    return differences


def _solve_differences(old: dict[str, Any], new: dict[str, Any], rtol: float) -> list[str]:
    differences = [
        f"solve: {metric} {old[metric]} -> {new[metric]}"
        for metric in ("latency", "energy_pj")
        if _relative(old[metric], new[metric]) > rtol
    ]
    held = old["occupancy"], new["occupancy"]
    if _same_occupancy(*(sorted(map(tuple, h.values()), key=lambda v: (v[1], v[0])) for h in held), rtol):
        return differences
    for key in sorted(held[0].keys() | held[1].keys()):
        a, b = held[0].get(key, (0, 0)), held[1].get(key, (0, 0))
        if a[1] != b[1] or _relative(a[0], b[0]) > rtol:
            differences.append(f"solve: {key} holds {a[0]} of {a[1]} bits -> {b[0]} of {b[1]}")
    return differences


def compare(old: dict[str, Any], new: dict[str, Any], rtol: float) -> list[str]:
    """Every difference between two summaries, one line each."""
    differences = _entry_differences(old, new, "cores", "core") + _entry_differences(old, new, "memories", "memory")
    differences += _connection_differences(old["connections"], new["connections"])
    if "solve" in old and "solve" in new:
        differences += _solve_differences(old["solve"], new["solve"], rtol)
    return differences


def _report(old: dict[str, Any], new: dict[str, Any], rtol: float) -> int:
    differences = compare(old, new, rtol)
    print("\n".join(differences))
    counts = ", ".join(f"{len(new[section])} {section}" for section in ("cores", "memories", "connections"))
    verdict = f"{len(differences)} difference(s)" if differences else "same accelerator"
    if "solve" in old and "solve" in new:
        verdict += "; " + ", ".join(
            f"{metric} {old['solve'][metric]:.6g} -> {new['solve'][metric]:.6g}" for metric in ("latency", "energy_pj")
        )
    print(f"{counts}: {verdict}")
    return 1 if differences else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    solve = argparse.ArgumentParser(add_help=False)
    solve.add_argument("--workload", default=str(DEFAULT_WORKLOAD), help="an ONNX workload to solve on both files")
    solve.add_argument("--mapping", help="a mapping YAML; without it, the mapping generator proposes one")
    solve.add_argument("--cols", type=int, help="SolveOptions.nb_cols_to_use")
    solve.add_argument("--no-solve", action="store_true", help="compare the structure only")
    summary = commands.add_parser("summarize", parents=[solve], help="write the summary of one hardware file")
    summary.add_argument("hardware")
    summary.add_argument("output")
    both = commands.add_parser("compare", help="compare two summaries")
    both.add_argument("old")
    both.add_argument("new")
    check = commands.add_parser("check", parents=[solve], help="summarise and compare an old and a new file")
    check.add_argument("old")
    check.add_argument("new")
    check.add_argument("--old-python", required=True, help="the command running Python with the core-format Stream")
    for command in (both, check):
        command.add_argument("--rtol", type=float, default=0.01, help="relative tolerance of the solved figures")
    args = parser.parse_args()

    if args.command == "compare":
        return _report(json.loads(Path(args.old).read_text()), json.loads(Path(args.new).read_text()), args.rtol)
    workload = None if args.no_solve else str(Path(args.workload).resolve())
    if args.command == "summarize":
        Path(args.output).write_text(json.dumps(summarize(args.hardware, workload, args.mapping, args.cols), indent=1))
        return 0
    flags = ["--no-solve"] if workload is None else ["--workload", workload]
    if args.mapping:
        flags += ["--mapping", str(Path(args.mapping).resolve())]
    if args.cols is not None:
        flags += ["--cols", str(args.cols)]
    with tempfile.TemporaryDirectory() as scratch:
        old_summary = Path(scratch) / "old.json"
        command = [*shlex.split(args.old_python), __file__, "summarize", args.old, str(old_summary), *flags]
        subprocess.run(command, check=True, stdout=sys.stderr)
        old = json.loads(old_summary.read_text())
    return _report(old, summarize(args.new, workload, args.mapping, args.cols), args.rtol)


if __name__ == "__main__":
    sys.exit(main())
