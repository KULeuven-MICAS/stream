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
from stream.opt.allocation.constraint_optimization.space import communicating_pairs
from stream.workload.affine_transform import AffineTransform
from stream.workload.node import ComputationNode, Tensor, TransferNode
from stream.workload.workload import Workload

MAPPINGS = Path(__file__).parent / "fixtures" / "conv_windows"
HARDWARE = "stream/inputs/examples/hardware/{}.yaml"


@dataclass(frozen=True)
class Scenario:
    """conv2's ``stride``, its output ``rows`` per iteration of the fused OY loop (None: one iteration), and the cores
    each conv splits its output columns over."""

    stride: int
    rows: int | None
    conv1_cores: tuple[int, ...]
    conv2_cores: tuple[int, ...]


SCENARIOS = {
    "s1": Scenario(1, None, (0,), (0,)),
    "s2": Scenario(1, 8, (0,), (0,)),
    "s3": Scenario(1, None, (0, 1, 2, 3), (0, 1, 2, 3)),
    "s4": Scenario(1, 8, (0, 1), (2, 3)),
    "s5": Scenario(2, 4, (0,), (0,)),
    "s3_shared": Scenario(1, None, (0, 1), (0, 1)),
    "s4_shared": Scenario(1, 8, (0,), (1,)),
}


@cache
def conv_chain(stride: int = 1) -> str:
    """input (1,8,32,32) -> Conv1 (16, 3x3, pad 1) -> Conv2 (32, 3x3, pad 1, ``stride``), bf16, as an ONNX path."""
    side = 32 // stride
    values = [
        helper.make_tensor_value_info(name, TensorProto.BFLOAT16, shape)
        for name, shape in (("input", [1, 8, 32, 32]), ("w1", [16, 8, 3, 3]), ("w2", [32, 16, 3, 3]))
    ]
    out = helper.make_tensor_value_info("out", TensorProto.BFLOAT16, [1, 32, side, side])
    attrs = {"kernel_shape": [3, 3], "pads": [1, 1, 1, 1]}
    nodes = [
        helper.make_node("Conv", ["input", "w1"], ["conv1_out"], name="Conv1", **attrs),
        helper.make_node("Conv", ["conv1_out", "w2"], ["out"], name="Conv2", strides=[stride, stride], **attrs),
    ]
    graph = helper.make_graph(nodes, "conv_chain", values, [out])
    model = shape_inference.infer_shapes(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]))
    path = f"{tempfile.mkdtemp()}/conv_chain_{stride}.onnx"
    onnx.save(model, path)
    return path


def _sizes(node: ComputationNode) -> list[int]:
    sizes = [0] * node.num_dims
    for tensor in node.tensors:
        for expr, size in zip(node.get_mapping(tensor).results, tensor.shape, strict=True):
            if isinstance(expr, AffineDimExpr):
                sizes[expr.position] = size
    return sizes


def reads(node: ComputationNode, tensor: Tensor, outputs: set[tuple[int, ...]]) -> set[tuple[int, ...]]:
    """Every in-bounds element of ``tensor`` that computing the ``outputs`` elements of ``node`` reads."""
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


def _box(shape: tuple[int, ...], rows: range, cols: range) -> set[tuple[int, ...]]:
    return {(b, k, y, x) for b in range(shape[0]) for k in range(shape[1]) for y in rows for x in cols}


def _block(size: int, parts: int, j: int) -> range:
    return range(j * size // parts, (j + 1) * size // parts)


def _shape(elements: set[tuple[int, ...]]) -> tuple[int, ...]:
    index = np.array(sorted(elements))
    return tuple(int(n) for n in index.max(0) - index.min(0) + 1)


def _new(per_iteration: list[set]) -> list[set]:
    """What each iteration needs that no earlier one did: the halo of a sliding window stays resident."""
    seen = [set(), *accumulate(per_iteration, set.union)]
    return [now - before for now, before in zip(per_iteration, seen, strict=False)]


@cache
def oracle(name: str) -> dict[str, Any]:
    """The scenario's tile quantities by enumeration: conv2's tiles fix what conv1 must have computed by each
    iteration, conv1 computes what is new, and each core reads what its tile's elements read."""
    scenario = SCENARIOS[name]
    conv1, conv2 = load_workload(conv_chain(scenario.stride)).get_computation_nodes()
    mid, inp = conv2.inputs[0], conv1.inputs[0]
    shape1, shape2 = conv1.outputs[0].shape, conv2.outputs[0].shape
    rows = scenario.rows or shape2[2]
    c1, c2 = scenario.conv1_cores, scenario.conv2_cores
    iterations = shape2[2] // rows
    work2 = [
        [_box(shape2, range(i * rows, (i + 1) * rows), _block(shape2[3], len(c2), j)) for j in range(len(c2))]
        for i in range(iterations)
    ]
    read2 = [[reads(conv2, mid, tile) for tile in tiles] for tiles in work2]
    new1 = _new([set().union(*tiles) for tiles in read2])
    owned = [_box(shape1, range(shape1[2]), _block(shape1[3], len(c1), j)) for j in range(len(c1))]
    work1 = [[new & own for own in owned] for new in new1]
    read1 = [[reads(conv1, inp, tile) for tile in tiles] for tiles in work1]
    moved2 = list(zip(*[_new(list(core)) for core in zip(*read2, strict=True)], strict=True))
    moved1 = list(zip(*[_new(list(core)) for core in zip(*read1, strict=True)], strict=True))
    i = min(1, iterations - 1)
    return {
        "iterations": iterations,
        "conv1_tile": tuple(_shape(tile) for tile in work1[i]),
        "out_tile": tuple(_shape(tile) for tile in work2[i]),
        "input_staged": tuple(_shape(tile) for tile in read1[i]),
        "input_moved": tuple(_shape(tile) for tile in moved1[i]),
        "window": tuple(_shape(tile) for tile in read2[i]),
        "moved": tuple(_shape(tile) for tile in moved2[i]),
        "sources": {
            dst: {src: len(moved & own) for src, own in zip(c1, owned, strict=True) if moved & own}
            for dst, moved in zip(c2, moved2[i], strict=True)
        },
        "conv1_rows": tuple(len({e[2] for e in new}) for new in new1),
        "input_moved_rows": tuple(len({e[2] for e in moved[0]}) for moved in moved1),
    }


@cache
def solve(name: str, hardware: str) -> dict[str, Any]:
    """The quantities of :func:`oracle` that the allocation the scenario's mapping solves to reports."""
    with tempfile.TemporaryDirectory() as out:
        mapping_path = str(MAPPINGS / f"{name}.yaml")
        workload_path = conv_chain(SCENARIOS[name].stride)
        estimate = evaluate_mapping(
            HARDWARE.format(hardware), workload_path, out, mapping_path, SolveOptions(artifacts=False)
        )
    allocation = estimate.context.get("allocation")
    workload: Workload = allocation.problem.workload
    mapping = allocation.problem.mapping
    conv1, conv2 = (cast(ComputationNode, workload.get_node_by_name(n)) for n in ("Conv1", "Conv2"))
    into1, into2 = (
        next(t for t in workload.predecessors(n) if isinstance(t, TransferNode) and n.inputs[0] in t.outputs)
        for n in (conv1, conv2)
    )

    def per_core(node: ComputationNode, tile: Callable[[int], Tensor]) -> tuple[tuple[int, ...], ...]:
        return tuple(tuple(tile(core).shape) for core in range(len(mapping.get(node).resource_allocation[0])))

    ssis = allocation.problem.ssis
    mid, inp = conv2.inputs[0], conv1.inputs[0]
    moved = workload.get_tensor_of_transfer_to_single_core(mid, into2, mapping, ssis=ssis[mid])
    route = allocation.solution.transfer_routes[into2]
    overlaps = workload.get_transfer_overlaps(into2, mapping, ssis[mid])
    pairs = communicating_pairs(route.sources, route.targets, overlaps)
    share = {pair: prod(moved.shape) // sum(d == pair[1] for _, d in pairs) for pair in pairs}
    share |= {(route.sources[i], route.targets[j]): n for (i, j), n in overlaps.items()}
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
        "sources": {dst.id: {src.id: n for (src, d), n in share.items() if d == dst} for _, dst in pairs},
        "in_place": allocation.solution.route_cycles[into2] == 0,
        "halos": {loop.type.name: loop.halo for loop in ssis[mid] if loop.halo},
    }
