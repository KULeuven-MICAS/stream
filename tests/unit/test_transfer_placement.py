"""A transfer's copy lives in the memories its reader reads it from, not in every memory the transfer reaches."""

import onnx
import pytest
import yaml
from onnx import TensorProto, helper
from zigzag.utils import open_yaml

from stream.api import evaluate_mapping
from stream.opt.allocation.constraint_optimization.space import DecisionSpace
from stream.parser.accelerator_factory import AcceleratorFactory
from stream.parser.accelerator_validator import AcceleratorValidator
from stream.workload.blocks import build_block
from stream.workload.node import ComputationNode, InEdge, TransferType

TPU_V7 = "stream/inputs/examples/hardware/tpu_v7_ironwood.yaml"


def test_each_copy_of_a_multicast_is_placed_with_its_reader(tmp_path):
    """The softmax's max runs on the tensor cores and its exp on the vector core; each copy of the scores stays with
    the node that reads it."""
    workload = build_block("attention", batch=1, heads=2, seq=16, d_head=16)
    allocation = evaluate_mapping(
        "stream/inputs/examples/hardware/tpu_like_quad_core.yaml", workload, str(tmp_path)
    ).context.get("allocation")
    space = DecisionSpace(allocation.problem)
    spread = 0
    for tr in space.transfer_nodes:
        targets = {memory for choice in space.tensor_choices[tr.outputs[0]] for memory in choice}
        for reader in space.workload.successors(tr):
            if not isinstance(reader, ComputationNode):
                continue
            copy = next(t for t in tr.outputs if t in reader.inputs)
            readers = {memory for choice in space.placements(reader, copy) for memory in choice}
            assert space.candidate_memories(copy) <= readers, (tr.name, copy.name)
            spread += len(tr.outputs) > 1 and targets != readers
    assert spread


def test_a_copy_for_a_reader_on_one_core_holds_the_whole_tensor(tmp_path):
    """The softmax's exp runs unsplit on the vector core, so the copy of the scores it reads is the whole tile, though
    the transfer also feeds the max split over the tensor cores."""
    workload = build_block("attention", batch=1, heads=2, seq=16, d_head=16)
    allocation = evaluate_mapping(
        "stream/inputs/examples/hardware/tpu_like_quad_core.yaml", workload, str(tmp_path)
    ).context.get("allocation")
    problem = allocation.problem
    whole = 0
    for tr in problem.workload.get_transfer_nodes():
        for reader in problem.workload.successors(tr):
            copy = next(t for t in tr.outputs if t in reader.inputs)
            split = isinstance(reader, ComputationNode) and problem.mapping.get(reader).inter_core_tiling[0]
            if isinstance(reader, ComputationNode) and not split and len(tr.outputs) > 1:
                assert problem.workload.get_tensor_single_core(copy, tr, problem.mapping).shape == copy.shape
                moved = problem.workload.get_tensor_of_transfer_to_single_core(
                    copy, tr, problem.mapping, ssis=problem.ssis[copy]
                )
                assert moved.shape == copy.shape
                whole += 1
    assert whole


def test_a_transfer_sits_on_the_cores_computing_at_its_ends(tmp_path):
    """A transfer's compute cores are exactly the cores of the computation nodes at its ends."""
    workload = build_block("attention", batch=1, heads=2, seq=16, d_head=16)
    allocation = evaluate_mapping(
        "stream/inputs/examples/hardware/tpu_like_quad_core.yaml", workload, str(tmp_path)
    ).context.get("allocation")
    space = DecisionSpace(allocation.problem)
    for tr in space.transfer_nodes:
        ends = (*space.workload.predecessors(tr), *space.workload.successors(tr))
        computed = {core for n in ends if isinstance(n, ComputationNode) for core in space.cores(n)}
        assert computed and space.compute_cores(tr) == computed
        assert all(core.kind == "compute" for core in computed)


def _validated(path):
    validator = AcceleratorValidator(open_yaml(path), path)
    assert validator.validate()
    return validator.normalized_data


def _folded_ironwood():
    """Ironwood's four TensorCores folded into two columns, as a ring is laid out."""
    data = _validated(TPU_V7)
    for core in range(8, 16):
        col, row = data["core_coordinates"][core]
        data["core_coordinates"][core] = [col - 2, row + 4]
    return AcceleratorFactory(data).create()


def _matmul_problem(tmp_path, m, k, n, layer=None, accelerator=None):
    inputs = [
        helper.make_tensor_value_info("a", TensorProto.BFLOAT16, [m, k]),
        helper.make_tensor_value_info("w", TensorProto.BFLOAT16, [k, n]),
    ]
    out = helper.make_tensor_value_info("out", TensorProto.BFLOAT16, [m, n])
    graph = helper.make_graph([helper.make_node("MatMul", ["a", "w"], ["out"], name="MatMul")], "g", inputs, [out])
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 19)]), tmp_path / "matmul.onnx")
    mapping = None
    if layer is not None:
        groups = [{"name": "g", "layers": ["MatMul"], "intra_core_tiling": []}]
        (tmp_path / "mapping.yaml").write_text(yaml.safe_dump({"layers": [layer], "fused_groups": groups}))
        mapping = str(tmp_path / "mapping.yaml")
    hardware = accelerator or _folded_ironwood()
    estimate = evaluate_mapping(hardware, str(tmp_path / "matmul.onnx"), str(tmp_path), mapping)
    return estimate.context.get("allocation").problem


def test_cores_working_in_memory_tiles_of_their_own_are_served_from_them(tmp_path):
    """The weights of a matmul split over the eight MXUs are staged a slice in each TensorCore's VMEM."""
    problem = _matmul_problem(tmp_path, 16, 1024, 4096)
    (staged,) = (
        problem.mapping.get(tr).memory_allocation
        for tr in problem.workload.get_transfer_nodes()
        if any(isinstance(src, InEdge) and src.name == "w" for src in problem.workload.predecessors(tr))
    )
    assert [[memory.name for memory in choice] for choice in staged] == [["vmem_0", "vmem_1", "vmem_2", "vmem_3"]]


@pytest.mark.parametrize(
    ("cores", "split", "gathered"),
    [
        ([0, 1, 4, 5, 8, 9, 12, 13], [("D0", 2), ("D1", 4)], [("D0", 2), ("D1", 2)]),
        ([1, 0, 5, 4, 9, 8, 13, 12], [("D0", 2), ("D1", 4)], [("D0", 2), ("D1", 2)]),
        ([0, 1, 4, 5, 8, 9, 12, 13], [("D1", 4), ("D0", 2)], [("D1", 4)]),
    ],
)
def test_a_split_over_two_dims_is_gathered_in_the_memory_tiles_of_its_own_cores(tmp_path, cores, split, gathered):
    """The two MXUs of a TensorCore take consecutive tiles of the split; their VMEM gathers exactly those two."""
    layer = {
        "name": "MatMul",
        "core_allocation": [cores],
        "inter_core_tiling": [[{"dim": dim, "split": factor} for dim, factor in split]],
    }
    problem = _matmul_problem(tmp_path, 64, 256, 512, layer, AcceleratorFactory(_validated(TPU_V7)).create())
    (matmul,) = problem.workload.get_computation_nodes()
    position = {
        problem.workload.leading_dim(dim)[0]: f"D{i}" for i, dim in enumerate(problem.workload.get_dims(matmul))
    }
    (gather,) = (tr for tr in problem.workload.get_transfer_nodes() if tr.transfer_type is TransferType.COMPUTE_TO_MEM)
    placement = problem.mapping.get(gather)
    assert [[memory.name for memory in tiles] for tiles in placement.memory_allocation] == [
        ["vmem_0", "vmem_1", "vmem_2", "vmem_3"]
    ]
    assert [[(position[dim], f) for dim, f in tiling] for tiling in placement.inter_core_tiling] == [gathered]
