"""Tests for ONNX parser completions."""

import onnx

from stream.parser.onnx.model import ONNXModelParser
from stream.workload.node import ComputationNode, FusionEdge

_RESNET18_PATH = "stream/inputs/examples/workload/resnet18.onnx"


def test_resnet18_full_parse():
    """All 49 ResNet18 ONNX nodes parse: 48 into ComputationNodes, the Flatten folded into the Gemm reading it.

    Verifies:
    - No exceptions during parsing
    - Expected op type distribution: Conv x20, Relu x17, Add x8, MaxPool x1,
      GlobalAveragePool x1, Gemm x1, and no FusionEdge
    - AffineMap rank consistency: each AffineMap result count matches its tensor's dimensionality
      (validates that e.g. no 2D map was used on a 4D tensor, which would crash downstream)
    """
    parser = ONNXModelParser(_RESNET18_PATH)
    parser.run()
    workload = parser.workload

    # Count ComputationNode and FusionEdge instances
    computation_nodes = [n for n in workload.nodes if isinstance(n, ComputationNode)]
    fusion_edges = [n for n in workload.nodes if isinstance(n, FusionEdge)]

    # 48 ComputationNodes (Conv x20 + Relu x17 + Add x8 + MaxPool x1 + GlobalAveragePool x1 + Gemm x1)
    assert len(computation_nodes) == 48, f"Expected 48 ComputationNodes, got {len(computation_nodes)}"
    assert not fusion_edges, f"Expected the Flatten folded into the Gemm, got {fusion_edges}"

    # Verify op type distribution among ComputationNodes
    op_types: dict[str, int] = {}
    for cn in computation_nodes:
        op_types[cn.type] = op_types.get(cn.type, 0) + 1

    assert op_types.get("Conv", 0) == 20, f"Expected 20 Conv nodes, got {op_types.get('Conv', 0)}"
    assert op_types.get("Relu", 0) == 17, f"Expected 17 Relu nodes, got {op_types.get('Relu', 0)}"
    assert op_types.get("Add", 0) == 8, f"Expected 8 Add nodes, got {op_types.get('Add', 0)}"
    assert op_types.get("MaxPool", 0) == 1, f"Expected 1 MaxPool node, got {op_types.get('MaxPool', 0)}"
    assert op_types.get("GlobalAveragePool", 0) == 1, (
        f"Expected 1 GlobalAveragePool node, got {op_types.get('GlobalAveragePool', 0)}"
    )
    assert op_types.get("Gemm", 0) == 1, f"Expected 1 Gemm node, got {op_types.get('Gemm', 0)}"

    # CRITICAL: Validate AffineMap rank consistency across all ComputationNodes.
    # For each (tensor, operand_mapping) pair, the AffineMap result count must equal
    # the tensor's number of dimensions. A mismatch (e.g. 2D map on 4D tensor) would
    # crash downstream operations. This catches the 2D-map-on-4D-tensor bug.
    rank_errors = []
    for node in workload.get_iteration_space_nodes():
        for tensor, mapping in zip(node.tensors, node.operand_mapping, strict=True):
            tensor_rank = len(tensor.shape)
            map_results = len(mapping.results)
            if tensor_rank != map_results:
                rank_errors.append(
                    f"{node.name}: tensor '{tensor.name}' rank={tensor_rank} but map results={map_results}"
                )
    assert not rank_errors, "AffineMap rank mismatches found:\n" + "\n".join(rank_errors)


def test_resnet18_shape_inference():
    """Shape inference runs and intermediate tensor shapes are available."""
    model = onnx.load(_RESNET18_PATH, load_external_data=False)
    inferred = onnx.shape_inference.infer_shapes(model)
    # After inference, value_info should contain intermediate tensor shapes
    assert len(inferred.graph.value_info) > 0, "Shape inference should populate value_info"


def test_resnet18_flatten_folds_into_the_classifier():
    """The classifier reads the pooled [1, 512, 1, 1] activations in place, so nothing splits the graph, and cutting
    before the Gemm gives it a group of its own whose dimensions resolve."""
    parser = ONNXModelParser(_RESNET18_PATH)
    parser.run()
    workload = parser.workload

    gemm = next(n for n in workload.get_computation_nodes() if n.type == "Gemm")
    pool = next(n for n in workload.get_computation_nodes() if n.type == "GlobalAveragePool")
    assert gemm.inputs[0] is pool.outputs[0]
    assert str(gemm.operand_mapping[0]) == "(d0, d1, d2) -> (0, d1, 0, 0)"
    assert len(workload.split_fusion_groups()) == 1

    groups = workload.split_fusion_groups(cut_points=[pool.name])
    assert [len([n for n in g.nodes if isinstance(n, ComputationNode)]) for g in groups] == [47, 1]
    assert groups[1].get_dimension_sizes(), "get_dimension_sizes() on the Gemm group should be non-empty"
