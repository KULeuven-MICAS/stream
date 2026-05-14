---
phase: 25-resnet18-sub-graph-patterns
plan: 01
subsystem: parser, workload
tags: [onnx, resnet18, fusion-edge, residual-block, sub-graph, reshape]

requires:
  - phase: 24-fan-out-multi-group
    provides: "FusionEdge parsing, split_fusion_groups(), GenericMappingGenerator, ZigZag fallback"
provides:
  - "FusionEdgeParser data-input-only fix for Reshape and other multi-input FusionEdge ops"
  - "Parametric ResNet18 sub-graph builder with 4 patterns (BASIC_RESIDUAL, STRIDE2_DOWNSAMPLE, FRONTEND, DUAL_RESIDUAL)"
  - "Identity Reshape as FusionEdge trigger pattern for dual-residual multi-group splitting"
affects: [25-02, resnet18-pipeline-tests, fusion-edge-parsing]

tech-stack:
  added: []
  patterns:
    - "Identity Reshape as FusionEdge trigger for multi-group splitting (Solution A)"
    - "Pattern-specific dimension overrides in config dataclass"

key-files:
  created:
    - stream/inputs/testing/workload/make_resnet_subgraph.py
  modified:
    - stream/parser/onnx/fusion_edge.py

key-decisions:
  - "FusionEdgeParser takes only self.node.input[0] as data input, ignoring metadata inputs (shape tensors)"
  - "DUAL_RESIDUAL uses identity Reshape as FusionEdge trigger (Solution A from RESEARCH.md) -- all patterns use optimize_allocation_co_generic"
  - "STRIDE2_DOWNSAMPLE overrides to height=32, width=32, out_channels=16 for valid stride-2 arithmetic"
  - "FRONTEND overrides to in_channels=3, height=32, width=32 for RGB-like input with valid spatial dims"

patterns-established:
  - "Identity Reshape between residual blocks triggers split_fusion_groups() without changing data shape"
  - "Pattern enum + config dataclass for parametric ONNX workload builders"

requirements-completed: [RNET-01, RNET-02, RNET-03]

duration: 14min
completed: 2026-05-14
---

# Phase 25 Plan 01: ResNet18 Sub-Graph Patterns Summary

**FusionEdgeParser fixed for multi-input ops (Reshape) and parametric builder created for 4 ResNet18 sub-graph patterns with identity Reshape as dual-residual FusionEdge trigger**

## Tasks Completed

### Task 1: Fix FusionEdgeParser to handle multi-input FusionEdge ops (Reshape)

**Commit:** b8f6d3b

Changed `FusionEdgeParser.generate_node()` from iterating all `self.node.input` entries to taking only `self.node.input[0]` (the data tensor). This ensures:
- Flatten (1 input) works unchanged -- data_input_name is its single input
- Reshape (2 inputs: data + shape tensor) only takes the data tensor
- `split_fusion_groups()` assertion `len(fe.inputs) == 1` passes for both op types

**Files modified:** `stream/parser/onnx/fusion_edge.py`

### Task 2: Create parametric ResNet18 sub-graph builder with 4 patterns

**Commit:** 24c9da4

Created `stream/inputs/testing/workload/make_resnet_subgraph.py` with:
- `ResNetPattern` enum: BASIC_RESIDUAL, STRIDE2_DOWNSAMPLE, FRONTEND, DUAL_RESIDUAL
- `ResNetSubgraphConfig` dataclass with pattern-specific dimension overrides
- `make_resnet_subgraph()` entry point returning `str` (onnx_path) for all patterns

Pattern details:
- **BASIC_RESIDUAL:** Conv->Relu->Conv->Add with skip connection (input fan-out to Conv1 and Add1)
- **STRIDE2_DOWNSAMPLE:** Conv(s=2)->Relu->Conv + Conv(1x1,s=2)->Add with 32x32 input, doubles channels 8->16
- **FRONTEND:** Conv(7x7,s=2)->Relu->MaxPool(s=2) with 3 RGB input channels, 32x32 spatial
- **DUAL_RESIDUAL:** Two residual blocks with identity Reshape FusionEdge boundary between blocks; reshape_shape is an initializer (not graph input)

**Files created:** `stream/inputs/testing/workload/make_resnet_subgraph.py` (453 lines)

## Deviations from Plan

None -- plan executed exactly as written.

## Verification Results

- Existing `test_pipeline_multi_group` passes (Flatten-based FusionEdge unchanged)
- All 4 ONNX patterns pass `onnx.shape_inference.infer_shapes()` without error
- BASIC_RESIDUAL: Conv1, Relu1, Conv2, Add1 nodes; input fan-out to Conv1 and Add1 verified
- STRIDE2_DOWNSAMPLE: Conv1 strides=[2,2]; Conv3 kernel=[1,1], strides=[2,2] verified
- FRONTEND: Conv1 kernel=[7,7], strides=[2,2]; MaxPool1 kernel=[3,3], strides=[2,2] verified
- DUAL_RESIDUAL: BlockBoundary Reshape present; reshape_shape is initializer; 2 Add nodes verified
- Full test suite: 204 tests passed, 0 failures

## Known Stubs

None -- all patterns produce complete ONNX models with valid shapes and node configurations.

## Self-Check: PASSED

- FOUND: stream/parser/onnx/fusion_edge.py
- FOUND: stream/inputs/testing/workload/make_resnet_subgraph.py (453 lines)
- FOUND: commit b8f6d3b (Task 1)
- FOUND: commit 24c9da4 (Task 2)
