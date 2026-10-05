"""Two fused 3x3 convolutions, the mappings of the windowed scenarios, a brute-force oracle of what each tile reads
straight from the access maps, and the same quantities read off a solved allocation."""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from itertools import accumulate
from math import prod
from pathlib import Path
from typing import Any, cast

import numpy as np
import onnx
from onnx import TensorProto, helper, shape_inference
from xdsl.ir.affine import AffineDimExpr

from stream.api import SolveOptions, evaluate_mapping
from stream.frontends import load_workload
from stream.opt.allocation.constraint_optimization.space import DecisionSpace
from stream.workload.affine_transform import AffineTransform
from stream.workload.node import ComputationNode, Tensor, TransferNode
from stream.workload.workload import Workload

MAPPINGS = Path(__file__).parent / "fixtures" / "conv_windows"
HARDWARE = "stream/inputs/examples/hardware/{}.yaml"


@dataclass(frozen=True)
class Scenario:
    """conv2's ``stride``, its output ``rows`` per iteration of the fused OY loop (None: one iteration), the cores
    each conv splits its output columns over, whether a 3x3 max pool takes conv2's place, the padding of both, and
    the mapping fixture when it is not the scenario's own."""

    stride: int
    rows: int | None
    conv1_cores: tuple[int, ...]
    conv2_cores: tuple[int, ...]
    pool: bool = False
    pads: int = 1
    mapping: str = ""

    @property
    def second(self) -> str:
        return "Pool" if self.pool else "Conv2"


SCENARIOS = {
    "s1": Scenario(1, None, (0,), (0,)),
    "s2": Scenario(1, 8, (0,), (0,)),
    "s3": Scenario(1, None, (0, 1, 2, 3), (0, 1, 2, 3)),
    "s4": Scenario(1, 8, (0, 1), (2, 3)),
    "s5": Scenario(2, 4, (0,), (0,)),
    "s3_shared": Scenario(1, None, (0, 1), (0, 1)),
    "s4_shared": Scenario(1, 8, (0,), (1,)),
    "pool": Scenario(1, 8, (0, 1), (2, 3), pool=True),
    "valid": Scenario(1, None, (0,), (0,), pads=0, mapping="s1"),
    "s5_named": Scenario(2, 4, (0,), (0,)),
    "s6": Scenario(2, None, (0, 1, 2, 3), (0, 1, 2, 3), mapping="s3"),
}


def conv(name: str, x: str, w: str, y: str, k: int = 3, stride: int = 1, pads: int = 1) -> onnx.NodeProto:
    return helper.make_node("Conv", [x, w], [y], name=name, kernel_shape=[k, k], pads=[pads] * 4, strides=[stride] * 2)


def onnx_graph(nodes: list[onnx.NodeProto], shapes: dict[str, list[int]], outputs: list[str]) -> str:
    """The bf16 ONNX path of ``nodes`` over graph inputs of ``shapes``, shapes inferred."""
    values = [helper.make_tensor_value_info(name, TensorProto.BFLOAT16, shape) for name, shape in shapes.items()]
    results = [helper.make_tensor_value_info(name, TensorProto.BFLOAT16, None) for name in outputs]
    graph = helper.make_graph(nodes, "windows", values, results)
    model = shape_inference.infer_shapes(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]))
    path = f"{tempfile.mkdtemp()}/windows.onnx"
    onnx.save(model, path)
    return path


@cache
def conv_chain(stride: int = 1, pool: bool = False, pads: int = 1) -> str:
    """input (1,8,32,32) -> Conv1 (16, 3x3) -> Conv2 (32, 3x3, ``stride``), or with ``pool`` a 3x3 max pool of the
    same window, both padded by ``pads``, as an ONNX path."""
    window = {"kernel_shape": [3, 3], "pads": [pads] * 4, "strides": [stride] * 2}
    second = (
        helper.make_node("MaxPool", ["conv1_out"], ["out"], name="Pool", **window)
        if pool
        else conv("Conv2", "conv1_out", "w2", "out", stride=stride, pads=pads)
    )
    shapes = {"input": [1, 8, 32, 32], "w1": [16, 8, 3, 3]} | ({} if pool else {"w2": [32, 16, 3, 3]})
    return onnx_graph([conv("Conv1", "input", "w1", "conv1_out", pads=pads), second], shapes, ["out"])


def _sizes(node: ComputationNode) -> list[int]:
    """Each dim's extent where a tensor axis is that dim alone, else 3: every window of these chains is 3x3."""
    sizes = [3] * node.num_dims
    for tensor in node.tensors:
        for expr, size in zip(node.get_mapping(tensor).results, tensor.shape, strict=True):
            if isinstance(expr, AffineDimExpr):
                sizes[expr.position] = size
    return sizes


def reads(node: ComputationNode, tensor: Tensor, outputs: set[tuple[int, ...]]) -> set[tuple[int, ...]]:
    """Every in-bounds element of ``tensor`` that computing the ``outputs`` elements of ``node`` reads."""
    if not outputs:
        return set()
    transform = AffineTransform.from_affine_map(node.get_mapping(tensor))
    out_dims = [expr.position for expr in node.get_mapping(node.outputs[0]).results]
    used = [d for d in range(node.num_dims) if transform.A[:, d].any()]
    known, free = [d for d in used if d in out_dims], [d for d in used if d not in out_dims]
    rows = np.unique(np.array(sorted(outputs))[:, [out_dims.index(d) for d in known]], axis=0)
    ranges = [np.arange(_sizes(node)[d]) for d in free]
    grid = np.stack(np.meshgrid(*ranges, indexing="ij"), -1).reshape(-1, len(free)) if free else np.zeros((1, 0), int)
    points = np.hstack([np.repeat(rows, len(grid), 0), np.tile(grid, (len(rows), 1))])
    index = points @ transform.A[:, known + free].T + transform.b
    inside = ((index >= 0) & (index < np.array(tensor.shape))).all(1)
    return set(map(tuple, index[inside].tolist()))


def box(shape: tuple[int, ...], rows: range, cols: range) -> set[tuple[int, ...]]:
    return {(b, k, y, x) for b in range(shape[0]) for k in range(shape[1]) for y in rows for x in cols}


def block(size: int, parts: int, j: int) -> range:
    return range(j * size // parts, (j + 1) * size // parts)


def extent_of(elements: set[tuple[int, ...]]) -> tuple[int, ...]:
    index = np.array(sorted(elements))
    return tuple(int(n) for n in index.max(0) - index.min(0) + 1)


def _new(per_iteration: list[set]) -> list[set]:
    """What each iteration needs that no earlier one did: the halo of a sliding window stays resident."""
    seen = [set(), *accumulate(per_iteration, set.union)]
    return [now - before for now, before in zip(per_iteration, seen, strict=False)]


@cache
def oracle(name: str | Scenario) -> dict[str, Any]:
    """The scenario's tile quantities by enumeration: conv2's tiles fix what conv1 must have computed by each
    iteration, conv1 computes what is new, and each core reads what its tile's elements read."""
    scenario = SCENARIOS[name] if isinstance(name, str) else name
    conv1, conv2 = load_workload(conv_chain(scenario.stride, scenario.pool, scenario.pads)).get_computation_nodes()
    mid, inp = conv2.inputs[0], conv1.inputs[0]
    shape1, shape2 = conv1.outputs[0].shape, conv2.outputs[0].shape
    rows = scenario.rows or shape2[2]
    c1, c2 = scenario.conv1_cores, scenario.conv2_cores
    iterations = shape2[2] // rows
    work2 = [
        [box(shape2, range(i * rows, (i + 1) * rows), block(shape2[3], len(c2), j)) for j in range(len(c2))]
        for i in range(iterations)
    ]
    read2 = [[reads(conv2, mid, tile) for tile in tiles] for tiles in work2]
    new1 = _new([set().union(*tiles) for tiles in read2])
    owned = [box(shape1, range(shape1[2]), block(shape1[3], len(c1), j)) for j in range(len(c1))]
    work1 = [[new & own for own in owned] for new in new1]
    read1 = [[reads(conv1, inp, tile) for tile in tiles] for tiles in work1]
    moved2 = list(zip(*[_new(list(core)) for core in zip(*read2, strict=True)], strict=True))
    moved1 = list(zip(*[_new(list(core)) for core in zip(*read1, strict=True)], strict=True))
    i = min(1, iterations - 1)
    return {
        "iterations": iterations,
        "conv1_tile": tuple(extent_of(tile) for tile in work1[i]),
        "out_tile": tuple(extent_of(tile) for tile in work2[i]),
        "input_staged": tuple(extent_of(tile) for tile in read1[i]),
        "input_moved": tuple(extent_of(tile) for tile in moved1[i]),
        "window": tuple(extent_of(tile) for tile in read2[i]),
        "moved": tuple(extent_of(tile) for tile in moved2[i]),
        "sources": {
            dst: {src: len(moved & own) for src, own in zip(c1, owned, strict=True) if moved & own}
            for dst, moved in zip(c2, moved2[i], strict=True)
        },
        "conv1_rows": tuple(len({e[2] for e in new}) for new in new1),
        "input_moved_rows": tuple(len({e[2] for e in moved[0]}) for moved in moved1),
    }


@cache
def solve(name: str, hardware: str) -> dict[str, Any]:
    """The quantities of :func:`oracle` that the allocation the scenario's mapping solves to reports, and the solver's
    own quantities of conv2's input transfer."""
    scenario = SCENARIOS[name]
    with tempfile.TemporaryDirectory() as out:
        mapping_path = str(MAPPINGS / f"{scenario.mapping or name}.yaml")
        workload_path = conv_chain(scenario.stride, scenario.pool, scenario.pads)
        estimate = evaluate_mapping(
            HARDWARE.format(hardware), workload_path, out, mapping_path, SolveOptions(artifacts=False)
        )
    allocation = estimate.context.get("allocation")
    workload: Workload = allocation.problem.workload
    mapping, ssis, space = allocation.problem.mapping, allocation.problem.ssis, DecisionSpace(allocation.problem)
    conv1, conv2 = (cast(ComputationNode, workload.get_node_by_name(n)) for n in ("Conv1", scenario.second))
    into1, into2 = (
        next(t for t in workload.predecessors(n) if isinstance(t, TransferNode) and n.inputs[0] in t.outputs)
        for n in (conv1, conv2)
    )

    def per_core(node: ComputationNode, tile: Callable[[int], Tensor]) -> tuple[tuple[int, ...], ...]:
        return tuple(tuple(tile(core).shape) for core in range(len(mapping.get(node).resource_allocation[0])))

    mid, inp = conv2.inputs[0], conv1.inputs[0]
    moved = workload.get_tensor_of_transfer_to_single_core(mid, into2, mapping, ssis=ssis[mid])
    route = allocation.solution.transfer_routes[into2]
    handed = {(route.sources[i], route.targets[j]): n for (i, j), n in space.overlaps(into2).items()}
    handed = handed or {pair: prod(moved.shape) for pair in space.pairs(into2, route)}
    oy = workload.get_dims(conv2)[2]
    work = workload.get_sliding_work(oy, allocation.problem.fusion_splits.get(oy, 1))
    routes = allocation.solution.transfer_routes
    return {
        "iterations": allocation.problem.iterations,
        "conv1_tile": per_core(conv1, lambda c: workload.get_tensor_single_core(conv1.outputs[0], conv1, mapping, c)),
        "out_tile": per_core(conv2, lambda c: workload.get_tensor_single_core(conv2.outputs[0], conv2, mapping, c)),
        "input_staged": per_core(conv1, lambda c: workload.get_tensor_single_core(inp, into1, mapping, c)),
        "input_moved": per_core(
            conv1, lambda c: workload.get_tensor_of_transfer_to_single_core(inp, into1, mapping, c, ssis[inp])
        ),
        "window": per_core(conv2, lambda c: workload.get_tensor_single_core(mid, into2, mapping, c)),
        "moved": per_core(
            conv2, lambda c: workload.get_tensor_of_transfer_to_single_core(mid, into2, mapping, c, ssis[mid])
        ),
        "sources": {dst.id: {src.id: n for (src, d), n in handed.items() if d == dst} for _, dst in handed},
        "conv1_rows": work.get(conv1, (conv1.outputs[0].shape[2],)),
        "input_moved_rows": work.get(into1, (inp.shape[2],)),
        "in_place": allocation.solution.route_cycles[into2] == 0,
        "halos": {loop.type.name: loop.halo for loop in ssis[mid] if loop.halo},
        "linked_bits": space.moved_bits(into2, route),
        "linked_pairs": [pair for pair in space.pairs(into2, route) if not space.hardware.shares_memory(*pair)],
        "received_bits": [
            (
                sum(space.moved_bits(tr, routes[tr], target=t) for t in routes[tr].targets),
                space.moved_bits(tr, routes[tr]),
            )
            for tr in (into1, into2)
        ],
        "input_copied": space.copied_bits(inp),
        "waited_on": space.is_waited_on(conv1),
        "first_tiles": {node.name: extra for node, extra in space.warmup.items()},
        "fill": allocation.solution.latency.fill,
    }
