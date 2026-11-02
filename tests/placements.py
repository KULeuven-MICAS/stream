"""Hardware with an off-chip memory per slice, and solves of workloads whose inputs and outputs are placed over them."""

from collections import defaultdict
from pathlib import Path

import yaml
from zigzag.utils import open_yaml

from stream.api import SolveOptions, evaluate_mapping
from stream.mapping.mapping import offchip_holdings, with_cut_landings
from stream.parser.accelerator_factory import AcceleratorFactory
from stream.parser.accelerator_validator import AcceleratorValidator

RING = "stream/inputs/testing/hardware/tpu_like_ring_hbm.yaml"
IRONWOOD = "stream/inputs/testing/hardware/tpu_v7_ironwood_hbm_per_chiplet.yaml"
TPU_V7 = "stream/inputs/examples/hardware/tpu_v7_ironwood.yaml"
HBMS = ["hbm_0", "hbm_1", "hbm_2", "hbm_3"]
COPIED = {"placement": {"memories": HBMS}}
POOLED = {"placement": {"memories": ["hbm_0"]}}


def split_on(axis: int, memories=HBMS) -> dict:
    return {"placement": {"memories": memories, "split": axis}}


def hardware(path=RING, *, capacity_bits=None, hbm_bandwidth=None, sram_bits=None, dma=False, ports=None, **fields):
    """The hardware at ``path`` with each HBM of ``capacity_bits``, its port and link at ``hbm_bandwidth``, its port a
    DMA's with ``dma`` and changed as ``ports`` names it, each core's SRAM of ``sram_bits``, and ``fields`` replaced."""
    validator = AcceleratorValidator(open_yaml(path), path)
    assert validator.validate()
    data = validator.normalized_data | fields
    for name, memory in data["memories"].items():
        if name in HBMS:
            port = memory["ports"]["rw_port_1"]
            memory["capacity_bits"] = capacity_bits or memory["capacity_bits"]
            port |= {"bandwidth": hbm_bandwidth or port["bandwidth"]} | (ports or {}).get(name, {})
            if dma:
                port["channels"] = 2
                memory["dma"] = {"buffer_descriptors": 16, "iterations": 64}
    for connection in data["connections"]:
        if hbm_bandwidth and set(connection["ends"]) & set(HBMS):
            connection["bandwidth"] = hbm_bandwidth
    for core in data["cores"].values() if sram_bits else ():
        core["memories"]["sram_2MB"]["size"] = sram_bits
    return AcceleratorFactory(data).create()


def tpu_like_slices(n: int) -> dict:
    """The cores and SRAMs of ``n`` slices laid out as the first of ``RING``."""
    data = open_yaml(RING)
    sram = data["memories"]["sram_0"]
    ports = sram["ports"].items()
    srams = {
        f"sram_{s}": sram | {"views": [f"{s}.sram_2MB"], "ports": {name: port | {"cores": [s]} for name, port in ports}}
        for s in range(n)
    }
    return {"cores": dict.fromkeys(range(n), data["cores"][0]), "memories": srams}


def ring_of(n: int):
    """``RING`` with ``n`` slices, each with its own HBM stack."""
    data = open_yaml(RING)
    first = ({"sram_0", "sram_1"}, {"sram_0", "hbm_0"})
    ring_link, hbm_link = (next(c for c in data["connections"] if set(c["ends"]) == ends) for ends in first)
    rw, hbm_port = ring_link["ends"]["sram_0"], hbm_link["ends"]["hbm_0"]
    slices = tpu_like_slices(n)
    data |= {
        "name": f"ring_of_{n}",
        "cores": slices["cores"],
        "memories": slices["memories"] | {f"hbm_{s}": data["memories"]["hbm_0"] for s in range(n)},
        "connections": [ring_link | {"ends": {f"sram_{s}": rw, f"sram_{(s + 1) % n}": rw}} for s in range(n)]
        + [hbm_link | {"ends": {f"sram_{s}": rw, f"hbm_{s}": hbm_port}} for s in range(n)],
    }
    validator = AcceleratorValidator(data, RING)
    assert validator.validate()
    return AcceleratorFactory(validator.normalized_data).create()


def solve(hardware, workload, tmp_path, mapping=None, **stage_options):
    """``workload`` on ``hardware`` mapped by ``mapping``, or by the generic generator if None, with the stage options
    given a value."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = None
    if mapping is not None:
        path = str(tmp_path / "mapping.yaml")
        Path(path).write_text(yaml.safe_dump(mapping))
    options = SolveOptions(artifacts=False, stage_options={k: v for k, v in stage_options.items() if v is not None})
    return evaluate_mapping(hardware, workload, str(tmp_path), path, options)


def route_of(allocation, edge: str):
    """The transfer moving edge ``edge``'s tensor, and the route the solve chose for it."""
    workload = allocation.problem.workload
    (tr,) = (
        tr
        for tr in workload.get_transfer_nodes()
        if any(n.name == edge for n in (*workload.predecessors(tr), *workload.successors(tr)))
    )
    return tr, allocation.mapping.get(tr).resource_allocation[0]


def offchip_reads(estimate) -> dict[tuple[str, str], int]:
    """What each off-chip memory reads of each tensor over all groups, by memory and tensor name."""
    reads: dict[tuple[str, str], int] = defaultdict(int)
    for group in estimate.context.get("group_memory_accesses").values():
        for row in group["offchip"]:
            reads[row["memory"], row["tensor"]] += row["read"]
    return {key: read for key, read in reads.items() if read}


def held_once(estimate, held=offchip_holdings):
    """``held`` of the step's groups, each tensor crossing a fusion cut counted once where it lands."""
    context = estimate.context
    accelerator, workloads = context.get("accelerator"), context.get("sub_workloads")
    return held(accelerator, workloads, with_cut_landings(accelerator, workloads, context.get("sub_mappings")))
