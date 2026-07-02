"""M05 Level-2: WL repeated-block detection and the StructuralDedupStage annotation."""

from __future__ import annotations

from xdsl.dialects.builtin import bf16, i8
from xdsl.ir.affine import AffineMap

from stream.stages.context import StageContext
from stream.stages.generation.structural_dedup import StructuralDedupStage
from stream.stages.stage import LeafStage, MainStage
from stream.workload.node import ComputationNode, InEdge, OutEdge
from stream.workload.structure import find_repeated_blocks, refine_colours
from stream.workload.tensor import Tensor
from stream.workload.workload import Workload

_GEMM = (
    AffineMap.from_callable(lambda m, k, n: (m, k)),
    AffineMap.from_callable(lambda m, k, n: (k, n)),
    AffineMap.from_callable(lambda m, k, n: (m, n)),
)
_RELU = (AffineMap.from_callable(lambda m, n: (m, n)), AffineMap.from_callable(lambda m, n: (m, n)))


def _nblock_workload(n_blocks: int, hidden: int = 8, perturb_block: int | None = None) -> Workload:
    """A chain of ``n_blocks`` identical [Gemm -> Relu] blocks. ``perturb_block`` changes one block's
    Gemm weight precision (a change deep inside that block)."""
    nodes: list = []
    activation = Tensor.create("x0", bf16, (hidden, hidden))
    nodes.append(InEdge(name="x0", outputs=(activation,)))
    for i in range(n_blocks):
        weight_dtype = i8 if perturb_block == i else bf16
        weight = Tensor.create(f"W{i}", weight_dtype, (hidden, hidden))
        gemm_out = Tensor.create(f"g{i}", bf16, (hidden, hidden))
        relu_out = Tensor.create(f"r{i}", bf16, (hidden, hidden))
        nodes.append(InEdge(name=f"W{i}", outputs=(weight,)))
        nodes.append(
            ComputationNode(
                type="Gemm", name=f"gemm{i}", inputs=(activation, weight), outputs=(gemm_out,), operand_mapping=_GEMM
            )
        )
        nodes.append(
            ComputationNode(
                type="Relu", name=f"relu{i}", inputs=(gemm_out,), outputs=(relu_out,), operand_mapping=_RELU
            )
        )
        activation = relu_out
    nodes.append(OutEdge(name="out", inputs=(activation,)))
    return Workload(nodes)


def test_identical_blocks_form_full_multiplicity_classes():
    n = 4
    classes = find_repeated_blocks(_nblock_workload(n))
    assert len(classes) == 2  # one Gemm class, one Relu class
    assert all(c.multiplicity == n for c in classes)


def test_near_miss_splits_only_the_changed_class():
    n = 4
    classes = find_repeated_blocks(_nblock_workload(n, perturb_block=2))
    multiplicities = sorted(c.multiplicity for c in classes)
    assert multiplicities == [n - 1, n]  # Gemms drop to n-1 (perturbed one splits off); Relus stay n
    perturbed = {node.name for c in classes for node in c.nodes}
    assert "gemm2" not in perturbed  # the changed node is no longer in a repeated class


def test_wl_refinement_distinguishes_chain_positions():
    """With a refinement radius, position leaks in, so a linear chain has no full-multiplicity class."""
    workload = _nblock_workload(4)
    refined = find_repeated_blocks(workload, rounds=2)
    assert all(c.multiplicity < 4 for c in refined)
    colours = refine_colours(workload, rounds=2)
    gemms = [n for n in workload.get_computation_nodes() if n.type == "Gemm"]
    assert len({colours[g] for g in gemms}) > 1  # WL separates the chain positions


def test_structural_dedup_stage_annotates_block_classes():
    workload = _nblock_workload(3)
    ctx = StageContext.from_kwargs(workload=workload)
    result = MainStage([StructuralDedupStage, LeafStage], ctx).run()[0]
    block_classes = result.get("block_classes")
    gemm_ids = {block_classes[f"gemm{i}"] for i in range(3)}
    relu_ids = {block_classes[f"relu{i}"] for i in range(3)}
    assert len(gemm_ids) == 1  # all three Gemms share a class
    assert len(relu_ids) == 1
    assert gemm_ids != relu_ids
