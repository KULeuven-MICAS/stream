"""Fuzz the memory model: solve small random workloads on random accelerators and check the tensor placements.

Each case is a seed. ``python -m tests.fuzz.memory_fuzz --seeds 0:200 --out out/fuzz`` runs a range in parallel and
writes one JSON per seed plus a summary grouped by cause. Each case is also solved with its lists reordered, which must
give the same latency, and with its memories renamed, which may change the latency through route tie-breaks
(status ``route_sensitive``)."""

from __future__ import annotations

import argparse
import copy
import json
import logging
import multiprocessing
import random
import re
import tempfile
import traceback
from collections import Counter
from pathlib import Path
from typing import Any
from unittest.mock import patch

import yaml

from stream.api import SolveOptions, evaluate_mapping
from stream.hardware.ports import OUTPUT
from stream.ir.infeasibility import InfeasibleAllocationError
from stream.mapping.mapping import role_of
from stream.opt.allocation.constraint_optimization.allocation_model import AllocationModel
from stream.parser.accelerator_factory import AcceleratorFactory
from stream.parser.accelerator_validator import AcceleratorValidator
from stream.workload.blocks import build_block
from stream.workload.node import ComputationNode, InEdge

CORES = Path("stream/inputs/examples/hardware/cores")
TESTING_CORES = Path("stream/inputs/testing/hardware/cores")
MAC = [CORES / "tpu_like.yaml", CORES / "eyeriss_like.yaml", CORES / "simba_chiplet.yaml", CORES / "fusemax_array.yaml"]
DUAL_TOP = [TESTING_CORES / "testing_core1.yaml", CORES / "meta_prototype.yaml"]
VECTOR = [CORES / "simd.yaml", CORES / "fusemax_vec.yaml"]
# a TensorCore: two arrays and a memory core that all view one VMEM
TENSORCORE = [CORES / "tpu_v7_mxu.yaml", CORES / "tpu_v7_vpu.yaml", CORES / "tpu_v7_vmem.yaml"]
SCALES = (1 / 1024, 1 / 128, 1 / 32, 1 / 8, 1 / 2, 1, 2)
BLOCKS = {
    "swiglu": lambda r: {"tokens": r.choice([8, 16, 32]), "d_model": r.choice([32, 64]), "d_ff": r.choice([64, 128])},
    "rmsnorm": lambda r: {"tokens": r.choice([8, 16]), "d_model": r.choice([32, 64])},
    "attention": lambda r: {"heads": r.choice([1, 2]), "seq": r.choice([16, 32]), "d_head": r.choice([16, 32])},
    "gqa": lambda r: {"groups": r.choice([1, 2]), "reps": 2, "seq": r.choice([16, 32]), "d_head": 16},
    "linear_attention": lambda r: {"seq": r.choice([16, 32]), "d_k": 16, "d_v": 16},
    "moe": lambda r: {"tokens": 16, "d_model": 32, "d_ff": 64, "experts": 2, "capacity": 8},
}


def _tops(core: dict[str, Any]) -> dict[str, list[str]]:
    """The top-level instances of a core file, each with the operands it holds."""
    tops: dict[str, list[str]] = {}
    for name, level in core["memories"].items():
        for operand in level["operands"]:
            tops = {n: [o for o in ops if o != operand] for n, ops in tops.items()}
            tops.setdefault(name, []).append(operand)
    return {name: ops for name, ops in tops.items() if ops}


def _onchip(cores: list[dict[str, Any]], views: list[tuple[int, str]]) -> dict[str, Any]:
    """An on-chip memory with ``views``, each port of the first view reached by all view cores."""
    core_id, instance = views[0]
    names = [port["name"] for port in cores[core_id]["memories"][instance]["ports"]]
    ids = [c for c, _ in views]
    return {
        "offchip": False,
        "views": [f"{c}.{i}" for c, i in views],
        "ports": {name: {"from": name, "cores": ids} for name in names},
    }


def _dram(r: random.Random) -> dict[str, Any]:
    """The off-chip memory, with either one shared port or one port per operand."""
    energy = {"read_energy": 15.625, "write_energy": 15.625}
    if r.random() < 0.3:
        ports = {
            f"rw_port_{k}": {"bandwidth": r.choice([64, 256]), "serves": {role: ["read", "write"]}} | energy
            for k, role in enumerate(("input1", "input2", "output"), start=1)
        }
    else:
        ports = {"rw_port_1": {"bandwidth": r.choice([64, 256]), "serves": {"any": ["read", "write"]}} | energy}
    return {"offchip": True, "views": [], "capacity_bits": 10_000_000_000, "ports": ports}


def generate(seed: int) -> dict[str, Any]:  # noqa: PLR0912, PLR0915 -- one draw per choice
    """A random case: the accelerator description (core files inlined) and the workload to solve on it."""
    r = random.Random(seed)
    files = [r.choice(MAC) for _ in range(r.randint(1, 3))] + [r.choice(VECTOR) for _ in range(r.randint(1, 2))]
    files += [r.choice(DUAL_TOP) for _ in range(r.choice([0, 0, 1]))]
    r.shuffle(files)
    tensorcore = r.random() < 0.2
    if tensorcore:
        files = TENSORCORE + files
    cores = [yaml.safe_load(path.read_text()) for path in files]
    # give each top-level instance its own memory, then merge some single-top-level cores into shared memories
    memories: dict[str, list[tuple[int, str]]] = {}
    for core_id, core in enumerate(cores):
        for instance in _tops(core):
            level = core["memories"][instance]
            level["size"] = max(8192, int(level["size"] * r.choice(SCALES)) // 8 * 8)
            memories[f"m{len(memories)}"] = [(core_id, instance)]
    if tensorcore:
        # merge the TensorCore's three VMEM views, copying the memory core's level onto the others
        vmem = [f"m{i}" for i in range(3)]
        lead = cores[2]["memories"]["vmem"]
        for name in vmem[:2]:
            ((core_id, instance),) = memories.pop(name)
            for key in ("size", "ports", "r_cost", "w_cost"):
                cores[core_id]["memories"][instance][key] = copy.deepcopy(lead[key])
            memories["m2"].append((core_id, instance))
    single = [n for n, views in memories.items() if len(views) == 1 and len(_tops(cores[views[0][0]])) == 1]
    r.shuffle(single)
    while len(single) >= 2 and r.random() < 0.6:
        group = [single.pop() for _ in range(min(len(single), r.randint(2, 3)))]
        lead_id, lead_instance = memories[group[0]][0]
        lead = cores[lead_id]["memories"][lead_instance]
        for name in group[1:]:
            ((core_id, instance),) = memories.pop(name)
            level = cores[core_id]["memories"][instance]
            # views of one memory must agree on its size and ports
            for key in ("size", "ports", "r_cost", "w_cost", "banks"):
                if key in lead:
                    level[key] = copy.deepcopy(lead[key])
                else:
                    level.pop(key, None)
            memories[group[0]].append((core_id, instance))
    onchip = list(memories)
    r.shuffle(onchip)
    connections: list[dict[str, Any]] = []
    for i in range(1, len(onchip)):
        if r.random() < 0.3 and i >= 2:
            members = [onchip[i], *r.sample(onchip[:i], r.randint(1, min(3, i)))]
            connections.append({"type": "bus", "memories": members, "bandwidth": r.choice([32, 64, 128, 256])})
        else:
            other = r.choice(onchip[:i])
            connections.append({"type": "link", "memories": [onchip[i], other], "bandwidth": r.choice([16, 64, 256])})
    for _ in range(r.randint(0, 2)):
        if len(onchip) >= 2:
            a, b = r.sample(onchip, 2)
            connections.append({"type": "link", "memories": [a, b], "bandwidth": r.choice([16, 64])})
    if r.random() < 0.5:
        reach = r.sample(onchip, r.randint(1, len(onchip)))
        connections.append({"type": "bus", "memories": [*reach, "dram"], "bandwidth": r.choice([64, 128, 512])})
    else:
        for name in r.sample(onchip, r.randint(1, len(onchip))):
            connections.append({"type": "link", "memories": [name, "dram"], "bandwidth": r.choice([32, 64, 128])})
    declared = {name: _onchip(cores, views) for name, views in memories.items()} | {"dram": _dram(r)}
    bandwidth = {}
    if r.random() < 0.3:
        measured = r.choice([*onchip, "dram"])
        bandwidth[measured] = {"ceiling": r.choice([64.0, 256.0]), "contiguous": r.choice([32.0, 128.0])}
    elif r.random() < 0.3:
        # measure one port of one memory
        name = r.choice([*onchip, "dram"])
        port = r.choice(sorted(declared[name]["ports"]))
        bandwidth[f"{name}.{port}"] = {"ceiling": 64.0, "contiguous": r.choice([16.0, 64.0])}

    data = {
        "name": f"fuzz_{seed}",
        "cores": dict(enumerate(cores)),
        "unit_energy_cost": 0,
        "memories": declared,
        # each connection goes through every port of each memory it joins
        "connections": [
            {key: value for key, value in c.items() if key != "memories"}
            | {"ends": {name: list(declared[name]["ports"]) for name in c["memories"]}}
            for c in connections
        ],
        **({"bandwidth": bandwidth} if bandwidth else {}),
    }
    block = r.choice(sorted(BLOCKS))
    return {"seed": seed, "hardware": data, "block": block, "params": BLOCKS[block](r)}


def reordered(data: dict[str, Any], seed: int) -> dict[str, Any]:
    """The same accelerator with its memories, views, ports and bus members shuffled."""
    r = random.Random(seed + 1_000_003)
    out = copy.deepcopy(data)
    items = list(out["memories"].items())
    r.shuffle(items)
    out["memories"] = {}
    for name, memory in items:
        memory["views"] = r.sample(memory["views"], len(memory["views"]))
        ports = list(memory["ports"].items())
        r.shuffle(ports)
        memory["ports"] = {port: _shuffled(r, declared) for port, declared in ports}
        out["memories"][name] = memory
    for connection in out["connections"]:
        if connection["type"] == "bus":
            ends = list(connection["ends"].items())
            r.shuffle(ends)
            connection["ends"] = dict(ends)
    return out


def _shuffled(r: random.Random, port: dict[str, Any]) -> dict[str, Any]:
    return port | {"cores": r.sample(port["cores"], len(port["cores"]))} if "cores" in port else port


def renamed(data: dict[str, Any], seed: int) -> dict[str, Any]:
    """The same accelerator with its memories renamed and its connections shuffled."""
    r = random.Random(seed + 2_000_003)
    names = list(data["memories"])
    new = dict(zip(names, (f"v{i}_{n}" for i, n in enumerate(r.sample(names, len(names)))), strict=True))
    out = copy.deepcopy(data)
    out["memories"] = {new[name]: views for name, views in data["memories"].items()}
    out["connections"] = r.sample(out["connections"], len(out["connections"]))
    for connection in out["connections"]:
        connection["ends"] = {new[m]: ports for m, ports in connection["ends"].items()}
    out["bandwidth"] = {}
    for key, value in data.get("bandwidth", {}).items():
        memory, dot, port = key.partition(".")
        out["bandwidth"][new[memory] + dot + port] = value
    return out


def build(data: dict[str, Any]):
    validator = AcceleratorValidator(copy.deepcopy(data), "fuzz.yaml")
    if not validator.validate():
        raise ValueError(f"invalid description: {validator.errors}")
    return AcceleratorFactory(validator.normalized_data).create()


def solve(data: dict[str, Any], block: str, params: dict[str, Any]) -> tuple[Any, list[Any]]:
    """The solved context, and every allocation model the solve built, each with its space and solution."""
    solved: list[tuple[Any, Any]] = []
    original = AllocationModel.solve

    def record(self, *args, **kwargs):
        solution = original(self, *args, **kwargs)
        solved.append((self, solution))
        return solution

    with tempfile.TemporaryDirectory() as out, patch.object(AllocationModel, "solve", record):
        estimate = evaluate_mapping(
            build(data), build_block(block, **params), out, options=SolveOptions(artifacts=False)
        )
    return estimate.context, solved


def check(context: Any, solved: list[tuple[Any, Any]]) -> list[str]:
    """The placement and route invariants this solve breaks."""
    broken: list[str] = []
    for group, allocation in context.get("group_allocations").items():
        for row in allocation["performance"]["memory_occupancy"]:
            if row["resident_bits"] > row["capacity_bits"]:
                broken.append(f"{group}: {row['memory']} holds {row['resident_bits']} > {row['capacity_bits']} bits")
    for model, solution in solved:
        broken += _placements(context.get("accelerator"), model.space, solution.tensor_placements)
        for tr, route in solution.transfer_routes.items():
            broken += _route(model.space, solution.tensor_placements, tr, route)
    return broken


def _placements(accelerator: Any, space: Any, placed: dict[Any, Any]) -> list[str]:
    """Check each tensor is placed in one of its choices and in the memories its producers and readers use."""
    broken = []
    for t, placement in placed.items():
        if placement not in space.tensor_choices[t]:
            broken.append(f"{t.name} placed in {placement}, not one of its choices")
        if any(m is not accelerator.memories.get(m.name) for m in placement):
            broken.append(f"{t.name} placed in a memory the accelerator lacks: {placement}")
    for node in space.workload.nodes():
        if isinstance(node, InEdge):
            broken += [
                f"edge tensor {t.name} placed in {placed[t]}, not off chip"
                for t in node.outputs
                if t in placed and set(placed[t]) != {accelerator.offchip_memory}
            ]
        if not isinstance(node, ComputationNode):
            continue
        cores = space.cores(node)
        for t in node.inputs:
            reads = {accelerator.memory_of(c, role_of(node, t)) for c in cores}
            if t in placed and not set(placed[t]) <= reads:
                names = sorted(m.name for m in reads)
                broken.append(f"{node.name} reads {t.name} in {placed[t]}, its cores read it in {names}")
        for t in node.outputs:
            if t in placed and tuple(placed[t]) != tuple(accelerator.memory_of(c, OUTPUT) for c in cores):
                broken.append(f"{node.name} writes {t.name} to {placed[t]}, not its cores' output memories")
    return broken


def _route(space: Any, placed: dict[Any, Any], tr: Any, route: Any) -> list[str]:
    """Check a transfer takes an offered route from its input's memory to its copies' memories."""
    broken = []
    if route not in space.path_choices[tr]:
        broken.append(f"{tr.name} takes a route it was not offered")
    source = placed.get(tr.inputs[0])
    if source is not None and not set(route.sources) <= set(source):
        broken.append(f"{tr.name} reads from {route.sources}, its input sits in {source}")
    for copy_ in tr.outputs:
        if copy_ in placed and not set(placed[copy_]) <= set(route.targets):
            broken.append(f"{copy_.name} sits in {placed[copy_]}, {tr.name} delivers to {route.targets}")
    return broken + _route_reaches(space, tr, route)


def _route_reaches(space: Any, tr: Any, route: Any) -> list[str]:
    """Check each target of the route is reachable from its paired source."""
    edges: dict[Any, set[Any]] = {}
    for link in route.links_used:
        ends = (
            [(link.sender, link.receiver)]
            if not link.members
            else [(a, b) for a in link.members for b in link.members if a is not b]
        )
        for a, b in ends:
            edges.setdefault(a, set()).add(b)
    broken = []
    for source, target in space.pairs(tr, route):
        if source is target or space.hardware.shares_memory(source, target):
            continue
        seen, frontier = {source}, [source]
        while frontier:
            for nxt in edges.get(frontier.pop(), ()):
                if nxt not in seen:
                    seen.add(nxt)
                    frontier.append(nxt)
        if target not in seen:
            broken.append(f"{tr.name}: no link of its route reaches {target.name} from {source.name}")
    return broken


def run(seed: int) -> dict[str, Any]:
    logging.disable(logging.WARNING)

    case = generate(seed)
    result: dict[str, Any] = {"seed": seed, "block": case["block"], "params": case["params"]}
    try:
        context, solved = solve(case["hardware"], case["block"], case["params"])
    except InfeasibleAllocationError as exc:
        return result | {"status": "infeasible", "reason": str(exc).splitlines()[0][:300]}
    except Exception as exc:  # noqa: BLE001 -- a fuzz case reports whatever breaks
        return result | {"status": "error", "error": _signature(exc), "traceback": traceback.format_exc()}
    result |= {"status": "ok", "latency": context.get("total_latency"), "broken": check(context, solved)}
    for label, transform in (("reordered", reordered), ("renamed", renamed)):
        try:
            other, _ = solve(transform(case["hardware"], seed), case["block"], case["params"])
            result[f"{label}_latency"] = other.get("total_latency")
        except Exception as exc:  # noqa: BLE001
            result[f"{label}_error"] = _signature(exc)
            result[f"{label}_traceback"] = traceback.format_exc()
    if result["broken"] or "reordered_error" in result or result.get("reordered_latency") != result["latency"]:
        result["status"] = "broken"
    elif "renamed_error" in result or result.get("renamed_latency") != result["latency"]:
        result["status"] = "route_sensitive"
    if result["status"] != "ok":
        result["hardware"] = case["hardware"]
    return result


def _signature(exc: BaseException) -> str:
    """The exception type and message, with numbers and memory names stripped so one cause groups its cases."""
    message = re.sub(r"\b\d+(\.\d+)?\b", "N", str(exc).splitlines()[0] if str(exc) else "")
    return f"{type(exc).__name__}: {re.sub(r'[mv]\d+(_m\d+|_dram)?', 'M', message)[:200]}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="0:50", help="start:stop")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    start, stop = map(int, args.seeds.split(":"))
    args.out.mkdir(parents=True, exist_ok=True)
    statuses: Counter[str] = Counter()
    causes: Counter[str] = Counter()
    with multiprocessing.get_context("spawn").Pool(args.jobs, maxtasksperchild=4) as pool:
        for result in pool.imap_unordered(run, range(start, stop)):
            (args.out / f"{result['seed']}.json").write_text(json.dumps(result, indent=1, default=str))
            statuses[result["status"]] += 1
            errors = (result.get("error"), result.get("reordered_error"), result.get("renamed_error"))
            for cause in (*errors, *result.get("broken", [])[:1]):
                if cause:
                    causes[cause] += 1
            print(result["seed"], result["status"], result.get("error") or result.get("reason") or "", flush=True)
    summary = {"statuses": dict(statuses), "causes": causes.most_common()}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
