"""What one firing of a transfer moves: per copy and per source-target pair, less what the target memory holds."""

import tempfile
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from conv_windows import HARDWARE, conv, onnx_graph
from onnx import helper
from placements import offchip_reads

from stream.api import SolveOptions, default_families, evaluate_mapping
from stream.hardware.ports import READ
from stream.opt.allocation.constraint_optimization.families import available_families, traffic
from stream.opt.allocation.constraint_optimization.families.latency import _shared_firing_rate, link_busy
from stream.opt.allocation.constraint_optimization.families.overlap import _transfer_run_ahead
from stream.opt.allocation.constraint_optimization.formulation import (
    DecisionVariables,
    FormulationContext,
    ResourceLedger,
)
from stream.opt.allocation.constraint_optimization.quantities import QuantityRegistry
from stream.opt.allocation.constraint_optimization.space import DecisionSpace
from stream.opt.solver import SolverBackend, SolverParams, SolverVarType, create_solver
from stream.workload.demand import DemandItem, moved_elements
from stream.workload.node import TransferNode

TPU = HARDWARE.format("tpu_like_quad_core")
SIMBA = HARDWARE.format("simba")


def _estimate(workload: str, mapping: str) -> Any:
    with tempfile.TemporaryDirectory() as out:
        path = Path(out) / "mapping.yaml"
        path.write_text(mapping)
        return evaluate_mapping(TPU, workload, out, str(path), SolveOptions(artifacts=False))


def _solve(workload: str, mapping: str) -> Any:
    return _estimate(workload, mapping).context.get("allocation")


def _model(solved_model: Any, workload: str, mapping: str, tmp_path: Path, hardware: str = TPU, family: Any = None):
    """The solved model, with ``family`` built after Stream's own."""
    path = tmp_path / "mapping.yaml"
    path.write_text(mapping)
    if family is None:
        return solved_model(hardware, workload, str(tmp_path), str(path), SolveOptions(artifacts=False))
    options = SolveOptions(artifacts=False, families=[*default_families(hardware), family.name])
    available = {**available_families(), family.name: family}
    return solved_model(hardware, workload, str(tmp_path), str(path), options, families_available=available)


def _local_copy_bits(model: Any) -> list[float]:
    """Per local copy on a chosen route, the bits per iteration charged to its ports."""
    value = model.model.value
    return [
        round(value(stream.bits_per_cycle * stream.active_cycles))
        for stream in traffic.local_copies(model.context)
        if any(side.port for side in stream.sides) and value(stream.active_cycles) > 0
    ]


def _chosen(model: Any, tr: TransferNode) -> Any:
    """The route the solve chose for ``tr``."""
    return next(choice for (t, choice), y in model.context.vars.y.items() if t is tr and model.model.value(y) > 0.5)


def _transfer(workload: Any, name: str) -> TransferNode:
    return next(n for n in workload.get_transfer_nodes() if n.name == name)


def _reaching(space: DecisionSpace, tr: TransferNode, route: Any, copy: Any) -> float:
    """Bits one firing of ``tr`` moves on ``route`` that reach ``copy``."""
    index = tr.outputs.index(copy)
    return (
        moved_elements(item for item in route.delivered_items if index in item.copies)
        * tr.inputs[0].operand_type.bitwidth
    )


def _demand(space: DecisionSpace, tr: TransferNode, route: Any, local: bool) -> tuple[DemandItem, ...]:
    """What ``tr`` reads on one core with local copies allowed."""
    tilings = space.mapping.get(tr).inter_core_tiling
    return space.workload.get_transfer_demand(
        tr,
        space.mapping,
        tilings[0] if tilings else (),
        1,
        1,
        False,
        route.targets,
        lambda node, tensor: route.targets,
        lambda items: (0.0,),
        ssis=space.ssis.get,
        fusion_splits=space.problem.fusion_splits,
        allow_local_copy=True,
        shares_memory=lambda source, target: local,
    )


def _contributions(allocation: Any, tr: TransferNode, field: str = "latency_contribution_cycles") -> list[float]:
    """The cycles per iteration the solve charges ``tr`` in its slot."""
    return [
        entry[field]
        for slot in allocation.solution.slot_latency_breakdown["slots"]
        for entry in slot["transfer_contributors"]
        if entry["transfer"] == tr.name
    ]


def _head(order: str = "QKV", relu: bool = False) -> str:
    """A bf16 attention head: I[64,64] times wQ, wK, wV [64,32] in ``order``; scores Q K^T; softmax; times V. With
    ``relu``, a Relu produces I."""
    nodes = [helper.make_node("Relu", ["X"], ["I"], name="Pre")] if relu else []
    nodes += [helper.make_node("MatMul", ["I", f"w{p}"], [p], name=f"Proj{p}") for p in order]
    nodes += [
        helper.make_node("Transpose", ["K"], ["KT"], name="Transpose", perm=[1, 0]),
        helper.make_node("MatMul", ["Q", "KT"], ["scores"], name="QK"),
        helper.make_node("Softmax", ["scores"], ["attn"], name="Softmax", axis=-1),
        helper.make_node("MatMul", ["attn", "V"], ["Y"], name="PV"),
    ]
    shapes = {"X" if relu else "I": [64, 64], "wQ": [64, 32], "wK": [64, 32], "wV": [64, 32]}
    return onnx_graph(nodes, shapes, ["Y"])


HEAD_MAPPING = """layers:
- {name: ProjQ, core_allocation: [[0]]}
- {name: ProjK, core_allocation: [[0]]}
- {name: ProjV, core_allocation: [[0]]}
- {name: Transpose, core_allocation: [[0]]}
- {name: QK, core_allocation: [[0]]}
- {name: Softmax, core_allocation: [[0]]}
- {name: PV, core_allocation: [[0]]}
fused_groups:
- name: G
  layers: [ProjQ, ProjK, ProjV, QK, Softmax, PV]
  intra_core_tiling:
  - {dim: QK.D0, tile: 16}
"""
HEAD_TWO_CORES = (
    HEAD_MAPPING.replace("ProjK, core_allocation: [[0]]", "ProjK, core_allocation: [[1]]")
    .replace("ProjV, core_allocation: [[0]]", "ProjV, core_allocation: [[1]]")
    .replace("Transpose, core_allocation: [[0]]", "Transpose, core_allocation: [[1]]")
)
RELU_HEAD = HEAD_MAPPING.replace(
    "layers:\n- {name: ProjQ", "layers:\n- {name: Pre, core_allocation: [[0]]}\n- {name: ProjQ"
).replace("layers: [ProjQ", "layers: [Pre, ProjQ")


def test_a_copy_kept_across_a_loop_its_reader_skips_is_the_source_of_a_local_copy():
    """Query tiles of 16 rows on core 0: I has the query tile ProjQ reads and all of I that ProjK and ProjV read. The
    whole of I is held across the query loop (reuse 4) and is the source of the query tiles' local copy: only it moves,
    64*64*16 = 65536 bits (512 cycles on the 128-bit bus) every 4 iterations, 128 cycles an iteration."""
    allocation = _solve(_head(), HEAD_MAPPING)
    space, tr = DecisionSpace(allocation.problem), _transfer(allocation.problem.workload, "Transfer(I)")
    route = allocation.solution.transfer_routes[tr]
    smaller, larger = sorted(tr.outputs, key=space.copied_bits)
    assert (_reaching(space, tr, route, smaller), _reaching(space, tr, route, larger)) == (0, 65536)
    assert allocation.solution.reuse_levels[larger] == 0
    assert _contributions(allocation, tr) == [128]


def _small_chain() -> str:
    """input (1,4,8,16) -> Conv1 (8, 3x3, pad 1) -> Conv2 (8, 3x3, pad 1), bf16."""
    nodes = [conv("Conv1", "input", "w1", "conv1_out"), conv("Conv2", "conv1_out", "w2", "out")]
    return onnx_graph(nodes, {"input": [1, 4, 8, 16], "w1": [8, 4, 3, 3], "w2": [8, 8, 3, 3]}, ["out"])


SPLIT_COLUMNS = """layers:
- name: Conv1
  core_allocation: [[0, 1, 2, 3]]
  inter_core_tiling: [[{dim: D1, split: 4}]]
- name: Conv2
  core_allocation: [[0, 1, 2, 3]]
  inter_core_tiling: [[{dim: D1, split: 4}]]
fused_groups:
- {name: G, layers: [Conv1, Conv2]}
"""


def test_a_link_takes_as_long_as_the_halo_it_carries():
    """Both convs split their 16 columns over four cores: each core sends each neighbour one column of conv1_out,
    8*8*16 = 1024 bits, and keeps its own 4. The 32-bit link carries that column alone: 1024 / 32 = 32 cycles."""
    allocation = _solve(_small_chain(), SPLIT_COLUMNS)
    space, tr = DecisionSpace(allocation.problem), _transfer(allocation.problem.workload, "Transfer(conv1_out)")
    route = allocation.solution.transfer_routes[tr]
    assert space.moved_bits(tr, route) == 6 * 1024
    assert space.transfer_latency_for_path(tr, route) == 32


def _residual() -> str:
    """input (1,4,16,16) -> Conv1 (4, 3x3, pad 1) -> Add(conv1_out, input), bf16."""
    add = helper.make_node("Add", ["conv1_out", "input"], ["out"], name="Add")
    return onnx_graph(
        [conv("Conv1", "input", "w1", "conv1_out"), add], {"input": [1, 4, 16, 16], "w1": [4, 4, 3, 3]}, ["out"]
    )


def _two_readers() -> str:
    """input (1,4,16,16) -> Conv1 (8, 3x3, pad 1) and Conv2 (8, 1x1) -> Add, bf16."""
    nodes = [
        conv("Conv1", "input", "w1", "a"),
        helper.make_node("Conv", ["input", "w2"], ["b"], name="Conv2", kernel_shape=[1, 1]),
        helper.make_node("Add", ["a", "b"], ["out"], name="Add"),
    ]
    return onnx_graph(nodes, {"input": [1, 4, 16, 16], "w1": [8, 4, 3, 3], "w2": [8, 4, 1, 1]}, ["out"])


CHANNELS_INSIDE_ROWS = """layers:
- {name: Conv1, core_allocation: [[0]]}
- {name: Conv2, core_allocation: [[0]]}
- {name: Add, core_allocation: [[0]]}
fused_groups:
- name: G
  layers: [Conv1, Conv2, Add]
  intra_core_tiling:
  - {dim: Conv1.D6, tile: 4}
  - {dim: Conv1.D2, tile: 4}
"""


def test_a_local_copy_held_across_a_loop_is_charged_once_per_firing(solved_model, tmp_path):
    """Rows tiled by 4 outside output channels tiled by 4 (two steps): Conv2's rows 4s..4s+3 are copied from Conv1's
    4s-1..4s+4. Both copies are held across the channel loop (reuse 2): the local copy reads and writes 4*4*16*16 = 4096
    bits a firing, 2048 an iteration."""
    model = _model(solved_model, _two_readers(), CHANNELS_INSIDE_ROWS, tmp_path)
    tr = _transfer(model.space.workload, "Transfer(input)")
    assert [model.get_tensor_reuse_levels()[copy] for copy in tr.outputs] == [0, 0]
    assert sum(_local_copy_bits(model)) == 2048


def test_each_copy_moves_and_is_timed_at_its_own_reuse():
    """Query tiles of 16 rows on core 0, ProjK and ProjV on core 1 reading all of I, held across the query loop (reuse
    4). The query rows both copies read move every iteration, 16*64*16 = 16384 bits, 128 cycles on the 128-bit bus;
    the other 48 rows, 49152 bits, 384 cycles, every 4th: 16384 + 49152/4 = 28672 bits, 128 + 96 = 224 cycles."""
    allocation = _solve(_head(), HEAD_TWO_CORES)
    node = next(n for n in allocation.get_ir()["steady_state"]["tiled_graph"]["nodes"] if n["name"] == "Transfer(I)")
    assert (node["copy_moved_bits"], node["copy_reuse_factors"]) == ([16384, 49152], [1, 4])
    assert (node["moved_bits"], node["reuse_factor"]) == (28672, 1)
    assert _contributions(allocation, _transfer(allocation.problem.workload, "Transfer(I)")) == [224]


def test_a_shared_box_fires_at_its_shared_firing_rate_whichever_copy_comes_first():
    """As above, but I's first copy is the one held across the query loop (reuse 4): the query rows still move every
    iteration, 224 cycles an iteration, and the breakdown reports the reuse of the most often firing copy, 1."""
    allocation = _solve(_head("KVQ"), HEAD_TWO_CORES)
    tr = _transfer(allocation.problem.workload, "Transfer(I)")
    node = next(n for n in allocation.get_ir()["steady_state"]["tiled_graph"]["nodes"] if n["name"] == tr.name)
    assert node["copy_reuse_factors"] == [4, 1]
    assert _contributions(allocation, tr) == [224]
    assert _contributions(allocation, tr, "reuse_factor") == [1]


def test_a_local_copy_is_charged_at_its_own_reuse_not_the_first_copys(solved_model, tmp_path):
    """I's first copy is the whole of I (reuse 4), its second the query tile, copied from it: the local copy reads and
    writes the query tile, 16*64*16 = 16384 bits, at the query copy's reuse of 1."""
    model = _model(solved_model, _head("KVQ"), HEAD_MAPPING, tmp_path)
    demand = _chosen(model, _transfer(model.space.workload, "Transfer(I)")).demand
    assert [(item.copies, item.copied_from) for item in demand if item.copied_from is not None] == [((1,), 0)]
    assert _local_copy_bits(model) == [16384]


class _WideUnheld:
    """Hold no copy of I that ProjK reads across any loop: it is reloaded every iteration."""

    name = "wide_unheld"
    requires: tuple[str, ...] = ()
    provides: tuple[str, ...] = ()

    def build(self, ctx: Any) -> None:
        for tr in ctx.space.transfer_nodes:
            for copy in tr.outputs:
                if any(n.name == "ProjK" for n in ctx.space.workload.successors(tr) if copy in n.inputs):
                    ctx.model.add_constr(ctx.vars.z_stop[(copy, -1)] == 1, name=f"unheld_{copy.name}")


def test_a_copy_not_kept_is_no_source_and_the_smaller_copy_is_moved(solved_model, tmp_path):
    """With the whole of I reloaded every iteration the query tile cannot be copied from it across the query loop: the
    route without the local copy is chosen. The whole of I, 65536 bits, is sent once for both copies, the query copy's
    16384 among them."""
    model = _model(solved_model, _head(), HEAD_MAPPING, tmp_path, family=_WideUnheld)
    tr = _transfer(model.space.workload, "Transfer(I)")
    route = _chosen(model, tr)
    assert [item for item in route.demand if item.copied_from is not None] == []
    proj_q = next(n for n in model.space.workload.successors(tr) if n.name == "ProjQ")
    query = next(copy for copy in tr.outputs if copy in proj_q.inputs)
    assert _reaching(model.space, tr, route, query) == 16384
    assert model.space.moved_bits(tr, route) == 65536


ROWS_OVER_TWO_CORES = """layers:
- name: Conv1
  core_allocation: [[0, 1]]
  inter_core_tiling: [[{dim: D2, split: 2}]]
- name: Add
  core_allocation: [[0, 1]]
  inter_core_tiling: [[{dim: D2, split: 2}]]
fused_groups:
- name: G
  layers: [Conv1, Add]
  intra_core_tiling:
  - {dim: Conv1.D2, tile: 4}
"""


def test_a_fused_step_of_a_split_dim_starts_past_every_core():
    """16 rows in tiles of 4, split over two cores: step s of core c is tile c + 2s. The Add's window on core 1 at
    step 1 is tile 3, rows 12..15; Conv1's there rows 11..15, its halo row above clipped at the end."""
    allocation = _solve(_residual(), ROWS_OVER_TWO_CORES)
    workload, tr = allocation.problem.workload, _transfer(allocation.problem.workload, "Transfer(input)")
    (dim,) = allocation.problem.fusion_splits
    rows = {}
    for copy in tr.outputs:
        reader = next(n for n in workload.successors(tr) if copy in n.inputs)
        split = workload._tiling(reader, allocation.problem.mapping)
        rows[reader.name] = workload._read_box((copy, reader, split, 1, 2), {dim}, {dim: 1})[2]
    assert rows == {"Add": (12, 15), "Conv1": (11, 15)}


def test_a_link_is_busy_and_charged_for_the_copies_crossing_it(solved_model, tmp_path):
    """HEAD_TWO_CORES on simba: chiplet 1 passes the query rows both copies read, 16384 bits a firing, on to chiplet
    0 over a 256-bit link, every iteration: the link is busy 16384 / 256 = 64 cycles and carries 16384 bits an
    iteration; the other 48 rows never cross it."""
    model = _model(solved_model, _head(), HEAD_TWO_CORES, tmp_path, hardware=SIMBA)
    value, tr = model.model.value, _transfer(model.space.workload, "Transfer(I)")
    name = "CL(simba_chiplet_1, simba_chiplet_0"
    busy = [
        sum(value(term) for term in terms)
        for (link, slot), terms in link_busy(model.context).items()
        if str(link).startswith(name) and slot == model.space.slot_of[tr]
    ]
    assert busy == [64]
    carried = sum(
        value(stream.bits_per_cycle * stream.active_cycles) * share
        for stream in traffic.dma_streams(model.context)
        for link, share in stream.link_shares
        if str(link).startswith(name) and stream.choice is _chosen(model, tr)
    )
    assert carried == 16384


def test_off_chip_memory_is_read_at_each_copy_sets_own_rate():
    """HEAD_TWO_CORES, four iterations: the query rows both copies read, 16384 bits, are read off chip every
    iteration, the other 49152 bits once."""
    estimate = _estimate(_head(), HEAD_TWO_CORES)
    accelerator = estimate.context.get("accelerator")
    rate = accelerator.offchip_rate(accelerator.offchip_memory, READ)
    reads = offchip_reads(estimate)[(accelerator.offchip_memory.name, "I")]
    assert reads == 4 * ceil(16384 / rate) + ceil(49152 / rate)


def test_a_copy_its_source_holds_in_its_memory_is_neither_moved_nor_copied(solved_model, tmp_path):
    """The Relu producing I and every reader of I on core 0: the Relu's output already holds the query tile in that
    memory, so nothing is copied or moved. Were the source elsewhere, the query tile would be a local copy from the
    whole of I."""
    model = _model(solved_model, _head(relu=True), RELU_HEAD, tmp_path)
    space, tr = model.space, _transfer(model.space.workload, "Transfer(I)")
    route = _chosen(model, tr)
    local_copies = [
        [item.copies for item in _demand(space, tr, route, local) if item.copied_from is not None]
        for local in (True, False)
    ]
    assert local_copies == [[], [(0,)]]
    assert space.moved_bits(tr, route) == 0
    assert _local_copy_bits(model) == []


def _basic_residual() -> str:
    """input (1,8,16,16) -> Conv1 (8, 3x3, pad 1) -> Relu1 -> Conv2 (8, 3x3, pad 1) -> Add1 with input, bf16."""
    nodes = [
        conv("Conv1", "input", "w1", "c1"),
        helper.make_node("Relu", ["c1"], ["r1"], name="Relu1"),
        conv("Conv2", "r1", "w2", "c2"),
        helper.make_node("Add", ["c2", "input"], ["out"], name="Add1"),
    ]
    return onnx_graph(nodes, {"input": [1, 8, 16, 16], "w1": [8, 8, 3, 3], "w2": [8, 8, 3, 3]}, ["out"])


BASIC_RESIDUAL = """layers:
- name: Conv1
  core_allocation: [[0, 1, 2, 3]]
  inter_core_tiling: [[{dim: D2, split: 4}]]
- name: Relu1
  core_allocation: [[0, 1, 2, 3]]
  inter_core_tiling: [[{dim: D2, split: 4}]]
- name: Conv2
  core_allocation: [[0, 1, 2, 3]]
  inter_core_tiling: [[{dim: D2, split: 4}]]
- name: Add1
  core_allocation: [[5]]
fused_groups:
- name: G
  layers: [Conv1, Relu1, Conv2, Add1]
  intra_core_tiling:
  - {dim: Conv1.D1, tile: 4}
"""


def test_copies_sliding_together_share_what_each_does_not_hold_at_the_same_firing(solved_model, tmp_path):
    """Columns in tiles of 4, rows split over cores 0-3, the Add on the SIMD core. Conv1's copies gain input columns
    0-5, 6-9, 10-13, 14-15 and the Add reads 4s..4s+3: off-chip memory sends the union, 0-5, 4-9, 8-13, 12-15, 22
    columns of 8*16 elements, 45056 bits over the run, 88 cycles a firing on the 128-bit bus. That is 6, 6, 6 and 4
    columns a firing, 704 elements on average: the first three run at most 192/704 of a firing ahead, 24 cycles."""
    model = _model(solved_model, _basic_residual(), BASIC_RESIDUAL, tmp_path)
    space, tr = model.space, _transfer(model.space.workload, "Transfer(input)")
    route = _chosen(model, tr)
    assert space.moved_bits(tr, route) * space.iterations == 45056
    assert space.transfer_latency_for_path(tr, route) == 88
    assert model.model.value(_transfer_run_ahead(model.context)[tr]) == pytest.approx(24)


@dataclass(frozen=True)
class _Named:
    name: str


def test_a_shared_box_fires_exactly_at_its_shared_firing_rate():
    """Two copies sharing a box, one firing every 4th iteration, the other every 2nd: the box fires every 2nd, a rate
    of exactly 1/2, however far a solve would push it up."""
    model = create_solver(SolverBackend.ORTOOLS_GSCIP, "shared_firing_rate")
    model.set_param(SolverParams.VERBOSITY, 0)
    held, half = _Named("held"), _Named("half")
    levels = {(held, -1): 1, (held, 0): 4, (half, -1): 1, (half, 0): 2}
    z_stop = {key: model.add_var(vtype=SolverVarType.BINARY, name=f"z_{key[0].name}_{key[1]}") for key in levels}
    for (_, stop), z in z_stop.items():
        model.add_constr(z == (stop == 0))
    space = SimpleNamespace(reuse_levels=levels, stops=lambda copy: (-1, 0))
    variables = DecisionVariables({}, {}, z_stop, {}, {})
    ctx = FormulationContext(space, variables, model, QuantityRegistry(), ResourceLedger())  # type: ignore[arg-type]
    rate = _shared_firing_rate(ctx, [held, half])  # type: ignore[arg-type]
    model.set_objective(rate, sense="maximize")
    model.optimize()
    assert model.value(rate) == 0.5


def _run_ahead_beside_a_halo() -> str:
    """input (1,2,4,16) read by Conv1 (1x1) feeding Conv2 (5x5, pad 2) and by ConvB (3x3, pad 1), added, bf16."""
    nodes = [
        conv("Conv1", "input", "w1", "c1", k=1, pads=0),
        conv("Conv2", "c1", "w2", "c2", k=5, pads=2),
        conv("ConvB", "input", "wb", "cb"),
        helper.make_node("Add", ["c2", "cb"], ["out"], name="Add1"),
    ]
    shapes = {"input": [1, 2, 4, 16], "w1": [2, 2, 1, 1], "w2": [2, 2, 5, 5], "wb": [2, 2, 3, 3]}
    return onnx_graph(nodes, shapes, ["out"])


def _strided_skip() -> str:
    """input (1,2,4,32) -> Conv1 (3x3, stride 2, pad 1) -> Conv2 (3x3, pad 1), and Conv3 (1x1, stride 2) on input,
    added, bf16."""
    nodes = [
        conv("Conv1", "input", "w1", "c1", stride=2),
        conv("Conv2", "c1", "w2", "c2"),
        conv("Conv3", "input", "w3", "c3", k=1, stride=2, pads=0),
        helper.make_node("Add", ["c2", "c3"], ["out"], name="Add1"),
    ]
    shapes = {"input": [1, 2, 4, 32], "w1": [2, 2, 3, 3], "w2": [2, 2, 3, 3], "w3": [2, 2, 1, 1]}
    return onnx_graph(nodes, shapes, ["out"])


RUN_AHEAD = """layers:
- {name: Conv1, core_allocation: [[0]]}
- {name: Conv2, core_allocation: [[0]]}
- {name: ConvB, core_allocation: [[0]]}
- {name: Add1, core_allocation: [[0]]}
fused_groups:
- name: G
  layers: [Conv1, Conv2, ConvB, Add1]
  intra_core_tiling:
  - {dim: Conv1.D1, tile: 4}
"""


def test_a_copy_running_ahead_is_a_local_copy_only_at_the_firings_the_larger_copy_holds_it(solved_model, tmp_path):
    """Four firings of 4 columns. Conv2's 5x5 window makes Conv1 run 2 columns ahead: it reads columns 0-5, 6-9,
    10-13, 14-15. ConvB reads 0-4, 3-8, 7-12, 11-15. Only by the last firing has ConvB's window held Conv1's, so only
    there is Conv1's copy a local copy, with its columns 14-15."""
    model = _model(solved_model, _run_ahead_beside_a_halo(), RUN_AHEAD, tmp_path)
    tr = _transfer(model.space.workload, "Transfer(input)")
    demand = _demand(model.space, tr, _chosen(model, tr), local=False)
    assert [(item.box[0], item.box[-1]) for item in demand if item.copied_from is not None] == [((3, 3), (14, 15))]


def test_a_copy_lagging_behind_is_copied_from_what_the_larger_copy_held_at_an_earlier_firing(solved_model, tmp_path):
    """Four firings of 4 output columns, stride 2. Conv1 runs 1 column ahead and reads input columns 0-9, 9-17,
    17-25, 25-31; Conv3 reads 8s..8s+6. Column 8s lies behind Conv1's window from firing 1 on, but Conv1's copy held
    it a firing before: Conv3's copy is a local copy at every firing."""
    model = _model(solved_model, _strided_skip(), RUN_AHEAD.replace("ConvB", "Conv3"), tmp_path)
    space, tr = model.space, _transfer(model.space.workload, "Transfer(input)")
    conv3 = next(n for n in space.workload.successors(tr) if n.name == "Conv3")
    skip = next(c for c, copy in enumerate(tr.outputs) if copy in conv3.inputs)
    skipped = [item for item in _demand(space, tr, _chosen(model, tr), local=False) if skip in item.copies]
    assert {s for item in skipped for s in range(item.box[0][0], item.box[0][1] + 1)} == {0, 1, 2, 3}
    assert all(item.copied_from is not None for item in skipped)


def test_a_copy_set_counts_on_a_link_the_same_box_from_each_source_it_crosses_from():
    """Two sources each send the same 4 elements of 8 bits to one copy set over one link: the link carries both,
    64 bits."""
    link = _Named("link")
    sent = [DemandItem(0, (s,), ((0, 3),), copies=(0, 1)) for s in (0, 1)]
    path = SimpleNamespace(
        hops=(("m0", "t", link), ("m1", "t", link)), hop_demand=((sent[0],), (sent[1],)), demand=sent
    )
    tr = SimpleNamespace(inputs=[SimpleNamespace(operand_type=SimpleNamespace(bitwidth=8))])
    space = object.__new__(DecisionSpace)
    assert space.copy_set_link_bits(tr, path, (0, 1)) == {link: 64}  # type: ignore[arg-type]
