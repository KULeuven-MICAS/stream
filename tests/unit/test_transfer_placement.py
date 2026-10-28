"""A transfer's copy lives in the memories its reader reads it from, not in every memory the transfer reaches."""

from stream.api import evaluate_mapping
from stream.opt.allocation.constraint_optimization.space import DecisionSpace
from stream.workload.blocks import build_block
from stream.workload.node import ComputationNode


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
