"""A picture of an accelerator as Stream understands it: its cores, the memories their top-level instances are views
of, the memories a namespace lets a core reach without a DMA, each memory's ports, and the links and buses between the
memories.

``python -m stream.visualization.hardware_graph <hardware.yaml> <out.svg>`` draws a hardware description; a solve
draws its accelerator as ``hardware_graph.svg`` in its output folder."""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

import networkx as nx

from stream.hardware.ports import (
    OUTPUT,
    READ,
    READ_BY_DATAPATH,
    WRITE,
    WRITE_BY_DATAPATH,
    declared_ports,
    input_role,
)
from stream.opt.allocation.constraint_optimization.hardware import NamespaceConfig, namespaces_for
from stream.workload._svg import write_svg

if TYPE_CHECKING:
    from stream.hardware.architecture.accelerator import Accelerator
    from stream.hardware.architecture.core import Core
    from stream.hardware.architecture.memory import Memory
    from stream.hardware.architecture.noc.communication_link import CommunicationLink
    from stream.hardware.ports import Port
    from stream.opt.allocation.constraint_optimization.hardware import HardwareNamespace

PALETTE = {
    "compute": ("#dbeafe", "#2a78d6"),
    "memory core": ("#ece0ff", "#7d5bd1"),
    "memory": ("#d6f5e3", "#1baf7a"),
    "off-chip memory": ("#fff1d6", "#c98a12"),
    "link": ("#f1f3f5", "#8a929b"),
    "bus": ("#ffe6d1", "#eb6834"),
}
ROLES = (input_role(1), input_role(2), OUTPUT)
KIBI = 1024
DIRECTIONS = {READ: "read", WRITE: "write", READ_BY_DATAPATH: "datapath read", WRITE_BY_DATAPATH: "datapath write"}


def draw_hardware(accelerator: Accelerator, path: str) -> None:
    """Draw ``accelerator`` as an SVG at ``path``: a column of cores, each with where it holds each operand and its
    DMA, a column of memories, each with its views, capacity, banks and ports, and a column of the connections between
    memories. A core points at each memory it is a view of, and with a dashed arrow at each one it reaches without a
    DMA as its namespace shares a neighbour's memory; a memory and each connection it is an end of point at each other,
    both carrying data either way."""
    boxes: dict[str, tuple[list[str], str]] = {}
    edges: list[tuple[str, str]] = []
    styles: dict[tuple[str, str], str] = {}
    namespaces = namespaces_for(accelerator, NamespaceConfig(accelerator))
    memory_box = {}
    for i, memory in enumerate(sorted(accelerator.memories.values(), key=_memory_order)):
        memory_box[memory] = name = f"m{i:04d}"
        boxes[name] = (_memory_lines(accelerator, memory), "off-chip memory" if memory.offchip else "memory")
    for core in accelerator.core_list:
        name = f"c{core.id:04d}"
        boxes[name] = (_core_lines(accelerator, core), _core_kind(core))
        edges += [(name, memory_box[memory]) for memory in accelerator.memories_of(core)]
        for memory in _shared(accelerator, namespaces, core):
            edges.append((name, memory_box[memory]))
            styles[edges[-1]] = "dashed"
    for i, (link, ends) in enumerate(_connections(accelerator)):
        name = f"x{i:04d}"
        boxes[name] = (_connection_lines(link, ends), "bus" if link.members else "link")
        edges += [(memory_box[end], name) for end in ends]
        styles.update({(memory_box[end], name): "both" for end in ends})
    graph = nx.DiGraph()
    graph.add_nodes_from(boxes)
    graph.add_edges_from(edges)
    write_svg(path, boxes, edges, graph, PALETTE, styles)


def _shared(accelerator: Accelerator, namespaces: list[HardwareNamespace], core: Core) -> list[Memory]:
    """The memories, not its own, that ``core`` reaches without a DMA: a neighbour's, as its namespace shares it."""
    own = set(accelerator.memories_of(core))
    reached = {
        memory
        for other in accelerator.core_list
        if other is not core and any(ns.shares_memory(core, other) for ns in namespaces)
        for memory in accelerator.memories_of(other)
    }
    return sorted(reached - own, key=lambda memory: memory.name)


def _core_kind(core: Core) -> str:
    return "compute" if core.kind == "compute" else "memory core"


def _core_lines(accelerator: Accelerator, core: Core) -> list[str]:
    """Its id, name and type, where it sits, what it runs, and per top-level instance the operands it holds there
    and the memory that instance is a view of."""
    lines = [f"Core {core.id} · {core.name}", core.core_type]
    if core.col_id is not None:
        lines.append(f"at column {core.col_id}, row {core.row_id}")
    if operators := getattr(core, "operator_types", None):
        lines.append("runs " + _listed(operators, 5))
    held: dict[str, list[str]] = {}
    for role in ROLES:
        held.setdefault(core.top_instance(role), []).append(_role(role))
    for instance in sorted(core.top_instances()):
        memory = accelerator.memory_viewed(core, instance)
        lines.append(f"{instance} → {memory.name}: holds {', '.join(held.get(instance, ['nothing it computes']))}")
    for instance, banks in sorted(core.memory_banks.items()):
        lines.append(f"{instance} banks: {banks.count} × {banks.width} B, stripe {banks.stripe}")
    if dma := _dma(core):
        lines.append(dma)
    if core.max_object_fifo_depth:
        lines.append(f"object fifos up to {core.max_object_fifo_depth} deep")
    return lines


def _dma(core: Core) -> str | None:
    """Its DMA channels each way, their width and its buffer descriptors, where its backend has a DMA."""
    if core.namespace != "aie2" or not core.dma_channel_bits:
        return None
    descriptors = f", {core.dma_buffer_descriptors} BDs" if core.dma_buffer_descriptors else ""
    return f"DMA {core.dma_mm2s} mm2s + {core.dma_s2mm} s2mm channels × {core.dma_channel_bits} b{descriptors}"


def _memory_lines(accelerator: Accelerator, memory: Memory) -> list[str]:
    """Its name and kind, capacity, views, banks, and each port: its width, what it serves and which cores reach
    it."""
    lines = [
        f"Memory {memory.name}" + (" · off-chip" if memory.offchip else ""),
        f"kind {memory.kind}, {_bits(memory.capacity_bits)}",
    ]
    lines.append("views " + (", ".join(f"{core.id}.{instance}" for core, instance in memory.views) or "none"))
    if memory.banks is not None:
        lines.append(f"banks {memory.banks.count} × {memory.banks.width} B, stripe {memory.banks.stripe}")
    for port in sorted(declared_ports(memory), key=lambda p: p.key.port):
        lines.append(_port_line(port))
    return lines


def _port_line(port: Port) -> str:
    serves: dict[str, list[str]] = {}
    for operand, direction in sorted(port.serves):
        serves.setdefault(DIRECTIONS[direction], []).append(_role(operand))
    served = "; ".join(f"{direction} {', '.join(operands)}" for direction, operands in sorted(serves.items()))
    cores = f"cores {', '.join(str(core_id) for core_id in port.core_ids)}" if port.core_ids else "its own"
    return f"port {port.key.port}: {port.bandwidth.ceiling:g} b/cycle, {served}; {cores}"


def _connections(accelerator: Accelerator) -> list[tuple[CommunicationLink, tuple[Memory, ...]]]:
    """Each declared connection once, a link by one of its two directions, with the memories it joins."""
    found: dict[object, tuple[CommunicationLink, tuple[Memory, ...]]] = {}
    for _, _, data in accelerator.network.edges(data=True):
        link = data["cl"]
        ends = tuple(sorted(link.memories, key=lambda memory: memory.name))
        key = id(link) if link.members else (ends, link.parallel)
        found.setdefault(key, (link, ends))
    return sorted(found.values(), key=lambda item: ([m.name for m in item[1]], item[0].parallel))


def _connection_lines(link: CommunicationLink, ends: tuple[Memory, ...]) -> list[str]:
    """Its kind, the memories it joins, its bandwidth and energy, and the ports of each memory it passes."""
    nth = f" #{link.parallel}" if link.parallel else ""
    head = f"Bus{nth}" if link.members else f"Link{nth} {ends[0].name} ↔ {ends[1].name}"
    return [
        head,
        f"{link.bandwidth:g} b/cycle each way" if not link.members else f"{link.bandwidth:g} b/cycle shared",
        f"{link.unit_energy_cost:g} pJ/word",
        *(f"{end.name} through {', '.join(link.ports[end]) or 'no port'}" for end in ends),
    ]


def _memory_order(memory: Memory) -> tuple:
    """Off-chip memory last, the others by the lowest core seeing them."""
    return (memory.offchip, min((core.id for core in memory.cores), default=-1), memory.name)


def _role(role: str) -> str:
    """``output`` as ``O``, ``input<k>`` as ``I<k>``, as a core file names them; ``any`` as itself."""
    if role == OUTPUT:
        return "O"
    return f"I{role.removeprefix('input')}" if role.startswith("input") else role


def _listed(items: list[str], most: int) -> str:
    return ", ".join(items[:most]) + (f" and {len(items) - most} more" if len(items) > most else "")


def _bits(bits: int) -> str:
    size, unit = bits / 8, "B"
    for bigger in ("KiB", "MiB", "GiB", "TiB"):
        if size < KIBI:
            break
        size, unit = size / KIBI, bigger
    return f"{size:.3g} {unit}"


def main() -> None:
    from stream.stages.parsing.accelerator_parser import parse_accelerator  # noqa: PLC0415

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hardware", help="the hardware description, a .yaml")
    parser.add_argument("out", help="the SVG to write")
    args = parser.parse_args()
    draw_hardware(parse_accelerator(args.hardware), args.out)


if __name__ == "__main__":
    main()
