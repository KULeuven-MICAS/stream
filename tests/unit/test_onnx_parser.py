"""Tests for ONNX parser completions (PARSE-01 through PARSE-05)."""

import onnx

from stream.parser.onnx.model import ONNXModelParser
from stream.workload.node import ComputationNode, FusionEdge

_RESNET18_PATH = "stream/inputs/examples/workload/resnet18.onnx"


def test_resnet18_full_parse():
    """PARSE-05: All 49 ResNet18 ONNX nodes parse into valid ComputationNode or FusionEdge.

    Verifies:
    - No exceptions during parsing
    - All 49 nodes accounted for (as ComputationNode, FusionEdge, InEdge, or OutEdge)
    - Expected op type distribution: Conv x20, Relu x17, Add x8, MaxPool x1,
      GlobalAveragePool x1, Flatten x1 (FusionEdge), Gemm x1
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
    # 1 FusionEdge (Flatten)
    assert len(computation_nodes) == 48, f"Expected 48 ComputationNodes, got {len(computation_nodes)}"
    assert len(fusion_edges) == 1, f"Expected 1 FusionEdge (Flatten), got {len(fusion_edges)}"
    assert fusion_edges[0].op_type == "Flatten", (
        f"Expected FusionEdge op_type='Flatten', got '{fusion_edges[0].op_type}'"
    )

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
    """PARSE-02: Shape inference runs and intermediate tensor shapes are available."""
    model = onnx.load(_RESNET18_PATH, load_external_data=False)
    inferred = onnx.shape_inference.infer_shapes(model)
    # After inference, value_info should contain intermediate tensor shapes
    assert len(inferred.graph.value_info) > 0, "Shape inference should populate value_info"


def test_resnet18_split_fusion_groups():
    """Verify split_fusion_groups produces 2 sub-workloads for ResNet18 (split at Flatten)."""
    parser = ONNXModelParser(_RESNET18_PATH)
    parser.run()
    workload = parser.workload

    groups = workload.split_fusion_groups()
    assert len(groups) == 2, f"Expected 2 fusion groups (split at Flatten), got {len(groups)}"
    # First group: all conv/relu/add/maxpool/globalavgpool nodes
    # Second group: Gemm node
    group_0_comp = [n for n in groups[0].nodes if isinstance(n, ComputationNode)]
    group_1_comp = [n for n in groups[1].nodes if isinstance(n, ComputationNode)]
    assert len(group_0_comp) == 47, f"Expected 47 ComputationNodes in group 0, got {len(group_0_comp)}"
    assert len(group_1_comp) == 1, f"Expected 1 ComputationNode in group 1 (Gemm), got {len(group_1_comp)}"

    # Validate group 1 (Gemm) dimension sizes -- this group has a simple linear topology
    # and get_dimension_sizes() should succeed for it
    dim_sizes = groups[1].get_dimension_sizes()
    assert len(dim_sizes) > 0, "get_dimension_sizes() on Gemm group should return non-empty tuple"
