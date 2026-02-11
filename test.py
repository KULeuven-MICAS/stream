#!/usr/bin/env python3
"""
Generate a simple mapping dict for a pruned YOLO ONNX model.

- Supports ONLY: Conv, HardSigmoid, Mul
- Allocates compute cores from the sequence:
    2,3,4,5, 8,9,10,11, 14,15,16,17, ... up to 47
  (blocks of 4, skipping 2 after each block)

- Inter-core tiling:
    Conv       -> dim D6, split = cores_per_conv
    HardSigmoid-> dim D1, split = cores_per_sigmoid
    Mul        -> dim D1, split = cores_per_mul
  Only emitted if split > 1.

- Kernel:
    {"name": "default", "kwargs": {}}

Function you asked for:
def make_yolo_pruned2_mapping(onnx_model_path, nb_rows, row_tile_size,
                             cores_per_conv, cores_per_sigmoid, cores_per_mul):
"""

from __future__ import annotations

import argparse
import json
from typing import Dict, List, Any, Iterator, Optional

import onnx
import yaml


SUPPORTED_OPS = {"Conv", "HardSigmoid", "Mul"}


def _compute_core_ids(max_core_id: int = 47) -> List[int]:
    """
    Compute core IDs: blocks of 4, skipping 2 IDs after each block.
    Example: 2-5, 8-11, 14-17, ...
    """
    cores: List[int] = []
    for base in range(2, max_core_id + 1, 6):
        for c in range(base, base + 4):
            if c <= max_core_id:
                cores.append(c)
    return cores


class CoreAllocator:
    def __init__(self, available_cores: List[int]):
        self._cores = available_cores
        self._pos = 0

    def alloc(self, n: int, *, layer_name: str) -> List[int]:
        if n <= 0:
            return []
        if self._pos + n > len(self._cores):
            raise RuntimeError(
                f"Out of compute cores while allocating {n} cores for '{layer_name}'. "
                f"Needed position {self._pos + n} of {len(self._cores)} available."
            )
        out = self._cores[self._pos : self._pos + n]
        self._pos += n
        return out

    @property
    def remaining(self) -> int:
        return len(self._cores) - self._pos


def _node_display_name(node) -> str:
    # Prefer node.name. If empty, create a stable-ish name from op_type and first output.
    if getattr(node, "name", ""):
        return node.name
    if node.output and node.output[0]:
        return f"{node.op_type}:{node.output[0]}"
    return f"{node.op_type}:unnamed"


def make_yolo_pruned2_mapping(
    onnx_model_path: str,
    cores_per_conv: int,
    cores_per_sigmoid: int,
    cores_per_mul: int,
) -> Dict[str, Dict[str, Any]]:
    """
    Returns:
      mapping: dict keyed by operator name (node.name if available) with entries:
        {
          "name": <string>,
          "op_type": <string>,
          "core_allocation": [...],
          "inter_core_tiling": [...],   # omitted (empty list) if cores == 1
          "kernel": {"name":"default","kwargs":{}},
        }
    """
    model = onnx.load(onnx_model_path)
    graph = model.graph

    # Core allocation pool (2..47 with the skip pattern)
    core_pool = _compute_core_ids(47)
    allocator = CoreAllocator(core_pool)

    # Helper: decide per-op cores + tiling dim
    def op_cores_and_dim(op_type: str) -> (int, str):
        if op_type == "Conv":
            return cores_per_conv, "D6"
        if op_type == "HardSigmoid":
            return cores_per_sigmoid, "D1"
        if op_type == "Mul":
            return cores_per_mul, "D1"
        raise RuntimeError(f"Unsupported op_type: {op_type}")

    mapping: Dict[str, Dict[str, Any]] = {}

    # ONNX node list is already topological for standard exported graphs
    for node in graph.node:
        if node.op_type not in SUPPORTED_OPS:
            nname = _node_display_name(node)
            raise RuntimeError(
                f"Unsupported operator encountered: op_type='{node.op_type}', name='{nname}'. "
                f"Expected only {sorted(SUPPORTED_OPS)}."
            )

        op_name = _node_display_name(node)
        ncores, dim = op_cores_and_dim(node.op_type)

        if ncores < 1:
            raise RuntimeError(f"Invalid core count {ncores} for node '{op_name}' ({node.op_type}).")

        core_allocation = allocator.alloc(ncores, layer_name=op_name)

        inter_core_tiling: List[Dict[str, Any]] = []
        if ncores > 1:
            inter_core_tiling = [{"dim": dim, "split": ncores}]

        entry = {
            "name": op_name,
            "op_type": node.op_type,
            "core_allocation": core_allocation,
            "inter_core_tiling": inter_core_tiling,
            "kernel": {"name": "default", "kwargs": {}},
        }

        # Key by operator name (as requested)
        if op_name in mapping:
            raise RuntimeError(f"Duplicate operator name key '{op_name}'. Use unique node names.")
        mapping[op_name] = entry

    return mapping


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate mapping dict for pruned YOLO ONNX (Conv/HardSigmoid/Mul only).")
    ap.add_argument("onnx_model_path", help="Path to ONNX model")
    ap.add_argument("--cores-per-conv", type=int, required=True)
    ap.add_argument("--cores-per-sigmoid", type=int, required=True)
    ap.add_argument("--cores-per-mul", type=int, required=True)
    ap.add_argument("--out", type=str, default=None, help="Write JSON mapping to this path (optional)")
    args = ap.parse_args()

    mapping = make_yolo_pruned2_mapping(
        args.onnx_model_path,
        args.cores_per_conv,
        args.cores_per_sigmoid,
        args.cores_per_mul,
    )

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            yaml.safe_dump(
                mapping,
                f,
                sort_keys=False,      # <-- keep insertion order
                default_flow_style=False,
            )
        print(f"Wrote mapping YAML to: {args.out}")
    else:
        print(
            yaml.safe_dump(
                mapping,
                sort_keys=False,
                default_flow_style=False,
            )
        )


if __name__ == "__main__":
    main()
