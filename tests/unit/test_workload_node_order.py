"""A workload's node order must come from the graph, not from its tensor names.

Global dimension indices, the IR node list and every solver variable are numbered from this
order, so two workloads that differ only in what their tensors are called have to produce the
same order -- otherwise renaming a weight hands the solver a different problem.

The fixture's initializers are named in dataflow order, so sorting by name and following the
dataflow agree until the rename swaps them.
"""

from __future__ import annotations

import networkx as nx
import onnx
import pytest

from stream.parser.onnx.model import ONNXModelParser
from stream.workload.node import ComputationNode, InEdge
from stream.workload.workload import Workload

_FIXTURE = "stream/inputs/testing/workload/2conv_1_8_32_32_16_32_3.onnx"
_RENAME = {"weights_1": "zeta", "weights_2": "alpha"}


def _rename(name: str) -> str:
    return _RENAME.get(name, name)


def _parse(path: str) -> Workload:
    parser = ONNXModelParser(str(path))
    parser.run()
    return parser.workload


def _tiled(path: str) -> Workload:
    workload = _parse(path)
    unique_dims, _ = workload.unique_dimensions()
    halved = {dim: max(1, workload.get_dimension_size(dim) // 2) for dim in unique_dims}
    return workload.with_modified_dimension_sizes(halved)


def _names(workload: Workload) -> list[str]:
    return [node.name for node in workload.dataflow_sort()]


def _names_sorted_by_name(workload: Workload) -> list[str]:
    return [node.name for node in nx.lexicographical_topological_sort(workload, key=lambda n: n.name)]


@pytest.fixture(scope="module")
def renamed_fixture(tmp_path_factory) -> str:
    model = onnx.load(_FIXTURE)
    for initializer in model.graph.initializer:
        initializer.name = _rename(initializer.name)
    for node in model.graph.node:
        node.input[:] = [_rename(i) for i in node.input]
    path = tmp_path_factory.mktemp("renamed") / "2conv_renamed.onnx"
    onnx.save(model, str(path))
    return str(path)


def test_renaming_moves_the_tensors_in_name_order(renamed_fixture):
    """Guard the two tests below: they only mean something while the rename reorders by name."""
    renamed = _parse(renamed_fixture)
    assert _names_sorted_by_name(renamed) != _names(renamed)


def test_dataflow_order_is_unchanged_by_renaming(renamed_fixture):
    assert _names(_parse(renamed_fixture)) == [_rename(name) for name in _names(_parse(_FIXTURE))]


def test_dataflow_order_survives_tiling(renamed_fixture):
    """Tiling recreates every node; the recreated workload must keep the order it was built from."""
    assert _names(_tiled(renamed_fixture)) == [_rename(name) for name in _names(_tiled(_FIXTURE))]


def test_group_split_keeps_the_input_order(renamed_fixture):
    """Each fused group receives its inputs in the order the workload lists them, since that is
    the order the generated design takes its runtime arguments in."""
    workload = _parse(renamed_fixture)
    order = _names(workload)

    for group in workload.split_fusion_groups(cut_points=["Conv1"]):
        inherited = {node.name for node in group.nodes if isinstance(node, InEdge)} & set(order)
        assert [node.name for node in group.nodes if node.name in inherited] == [
            name for name in order if name in inherited
        ]


def test_order_is_cached_until_the_graph_changes():
    workload = _parse(_FIXTURE)
    assert workload.dataflow_sort() is workload.dataflow_sort()
    assert workload.global_idxs is workload.global_idxs

    old = workload.get_computation_nodes()[0]
    new = ComputationNode(
        type=old.type,
        name="replacement",
        inputs=old.inputs,
        outputs=old.outputs,
        operand_mapping=old.operand_mapping,
    )
    workload.replace_node(old, new)

    assert "replacement" in _names(workload)
    assert old.name not in _names(workload)
    assert new in workload.global_idxs
