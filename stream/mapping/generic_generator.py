"""Generic mapping generator that auto-infers core allocation, inter-core tiling,
fused groups, and intra-core tiling from a Workload + Accelerator pair.

The generated mapping follows the MappingValidator schema exactly:
  - core_allocation:    nested list  [[core_id, ...]]
  - inter_core_tiling:  nested list  [[{"dim": "D{n}", "split": k}]]
  - intra_core_tiling:  flat list    [{"dim": "NodeName.D{n}", "tile": size}]

All generated mapping dicts are validated via MappingValidator before being
written to disk.  A ValueError is raised if validation fails.
"""

import logging
import os
from typing import Any

import yaml

from stream.hardware.architecture.accelerator import Accelerator
from stream.hardware.architecture.core import Core
from stream.parser.mapping_validator import MappingValidator
from stream.workload.node import ComputationNode
from stream.workload.workload import Workload

logger = logging.getLogger(__name__)


class GenericMappingGenerator:
    """Auto-generate a MappingValidator-compliant mapping dict for any Workload + Accelerator pair.

    Core selection follows the operator_types convention (D-06):
    - Cores without operator_types (None) accept all operator types.
    - Cores with operator_types only accept nodes whose type is in the list.
    - Offchip and shim cores are never used for computation.

    Inter-core tiling (D-09/D-10):
    - Specialized cores (pooling, simd) receive the node alone on a single core.
    - Generic compute cores receive the node split across all matching cores.

    Intra-core tiling (D-08):
    - Uses the first dimension of the first computation node at full tile size
      (no temporal splitting), which is always valid per MappingValidator rules.
    """

    def __init__(self, accelerator: Accelerator, workload: Workload, output_dir: str) -> None:
        self.accelerator = accelerator
        self.workload = workload
        self.output_dir = output_dir

    # ---------------------------------------------------------------------- #
    # Public API                                                              #
    # ---------------------------------------------------------------------- #

    def generate_all_groups(self, cut_points: list[str] | None = None) -> tuple[list[str], list[Workload]]:
        """Generate one mapping YAML per fusion group.

        Args:
            cut_points: Optional list of node names at which to split the workload
                in addition to FusionEdge boundaries. Passed through to
                ``split_fusion_groups(cut_points=...)``.

        Returns:
            A tuple ``(paths, sub_workloads)`` where *paths* is a list of
            absolute file paths to the written YAML files and *sub_workloads*
            is the list of sub-workloads returned by ``split_fusion_groups()``.
        """
        sub_workloads = self.workload.split_fusion_groups(cut_points=cut_points)
        paths: list[str] = []
        for i, sub_workload in enumerate(sub_workloads):
            path = self._generate_group_yaml(sub_workload, i)
            paths.append(path)
        return paths, sub_workloads

    # ---------------------------------------------------------------------- #
    # Private helpers                                                        #
    # ---------------------------------------------------------------------- #

    def _generate_group_yaml(self, sub_workload: Workload, group_idx: int) -> str:
        """Build, validate, and write the mapping YAML for one fusion group.

        Args:
            sub_workload: The sub-workload for this group.
            group_idx:    Zero-based index used for directory naming.

        Returns:
            Absolute path to the written YAML file.

        Raises:
            ValueError: If the generated mapping fails MappingValidator.
        """
        mapping_dict = self._build_mapping_dict(sub_workload)

        validator = MappingValidator(mapping_dict)
        if not validator.validate():
            raise ValueError(f"Generated mapping for group {group_idx} failed MappingValidator: {validator.errors}")

        out_dir = os.path.join(self.output_dir, f"group_{group_idx}")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "mapping.yaml")
        with open(out_path, "w") as f:
            yaml.safe_dump(mapping_dict, f, default_flow_style=False, sort_keys=False)

        logger.debug("Wrote mapping for group %d to %s", group_idx, out_path)
        return out_path

    def _build_mapping_dict(self, sub_workload: Workload) -> dict[str, Any]:
        """Build the full mapping dict for one fusion-group sub-workload.

        Returns a dict with 'layers' and 'fused_groups' keys conforming to
        the MappingValidator schema.
        """
        cns = sub_workload.get_computation_nodes()

        layers: list[dict[str, Any]] = []
        for cn in cns:
            cores = self._select_cores_for_node(cn)
            n_cores = len(cores)

            core_allocation: list[list[int]] = [[c.id for c in cores]]

            if n_cores > 1:
                split_dim_idx = self._find_split_dim(sub_workload, cn, n_cores)
                if split_dim_idx is not None:
                    inter_core_tiling: list[list[dict[str, Any]]] = [[{"dim": f"D{split_dim_idx}", "split": n_cores}]]
                else:
                    inter_core_tiling = []
            else:
                inter_core_tiling = []

            layers.append(
                {
                    "name": cn.name,
                    "core_allocation": core_allocation,
                    "inter_core_tiling": inter_core_tiling,
                }
            )

        intra_core_tiling = self._build_intra_core_tiling(sub_workload, cns)
        fused_group: dict[str, Any] = {
            "name": "Fused_Group_1",
            "layers": [cn.name for cn in cns],
            "intra_core_tiling": intra_core_tiling,
        }

        return {"layers": layers, "fused_groups": [fused_group]}

    def _select_cores_for_node(self, node: ComputationNode) -> list[Core]:
        """Select cores that can execute *node* according to operator_types.

        Excludes offchip and shim cores unconditionally.  Selection priority:
        1. Specialized cores (operator_types is not None and node.type in list).
           Per D-09: if any specialized cores match, use them exclusively.
        2. Generic cores (operator_types is None — accepts all ops).
           Per D-10: use all matching generic cores together.
        3. D-06 fallback: if nothing matches, use all cores with kind 'compute'.

        This ensures MaxPool goes to the pooling core, Add to the simd core, and
        Conv/Gemm go to all 4 generic compute cores.
        """
        _SKIP_TYPES = {"offchip", "shim"}
        node_op = node.type

        specialized_cores: list[Core] = []
        generic_cores: list[Core] = []

        for core in self.accelerator.core_list:
            if core.type in _SKIP_TYPES:
                continue
            op_types = getattr(core, "operator_types", None)
            if op_types is not None and node_op in op_types:
                # Specialized core that explicitly handles this operator type
                specialized_cores.append(core)
            elif op_types is None:
                # Unrestricted generic compute core — accepts all operators
                generic_cores.append(core)

        if specialized_cores:
            # D-09: prefer specialized core(s) over generic compute cores
            return specialized_cores

        if generic_cores:
            # D-10: use all generic compute cores together
            return generic_cores

        # D-06 fallback: no match — use all cores with kind 'compute'
        fallback = [c for c in self.accelerator.core_list if c.type == "compute"]
        logger.warning("No core found for operator '%s'; falling back to all compute cores.", node_op)
        return fallback

    def _find_split_dim(self, sub_workload: Workload, cn: ComputationNode, n_cores: int) -> int | None:
        """Find the best dimension index to split across *n_cores*.

        Iterates node dimensions sorted by size descending.  Returns the index
        of the first dimension whose size is divisible by *n_cores*.  If none
        is perfectly divisible, returns the index of the largest dimension.
        Returns None if the node has no dimensions.
        """
        dims = sub_workload.get_dims(cn)
        if not dims:
            return None

        # Pair each dim with (index, size)
        dim_sizes = [(i, sub_workload.get_dimension_size(dim)) for i, dim in enumerate(dims)]
        # Sort descending by size
        dim_sizes_sorted = sorted(dim_sizes, key=lambda x: x[1], reverse=True)

        # Try to find a perfectly divisible dimension
        for dim_idx, size in dim_sizes_sorted:
            if size >= n_cores and size % n_cores == 0:
                return dim_idx

        # Fallback: use the largest dimension (relaxed divisibility)
        return dim_sizes_sorted[0][0]

    def _build_intra_core_tiling(
        self, sub_workload: Workload, cns: tuple[ComputationNode, ...]
    ) -> list[dict[str, Any]]:
        """Build intra-core tiling for the fused group.

        Tiles the largest spatial dimension (by unique z-variable size) to
        produce multiple temporal iterations. The constraint optimizer handles
        the actual memory fitting across spatial + temporal tiling.
        """
        unique_dims = sub_workload.unique_dimensions()[0]
        if not unique_dims:
            return []

        candidates = sorted(
            [(d, sub_workload.get_dimension_size(d)) for d in unique_dims if sub_workload.get_dimension_size(d) > 1],
            key=lambda x: x[1],
            reverse=True,
        )
        if not candidates:
            ref_cn = cns[0]
            return [{"dim": f"{ref_cn.name}.D0", "tile": sub_workload.get_dimension_size(unique_dims[0])}]

        # Pick the largest dim NOT used by inter_core_tiling (avoids
        # spatial_unrolling × tile_size divisibility conflicts).
        inter_core_dims = self._get_inter_core_tiling_dims(sub_workload, cns)
        for dim, size in candidates:
            if dim not in inter_core_dims:
                tile_dim = dim
                tile_size = size
                break
        else:
            tile_dim = candidates[0][0]
            tile_size = candidates[0][1]

        ref_cn = next((cn for cn in cns if tile_dim in sub_workload.get_dims(cn)), cns[0])
        ref_dims = sub_workload.get_dims(ref_cn)
        ref_dim_idx = ref_dims.index(tile_dim) if tile_dim in ref_dims else 0
        return [{"dim": f"{ref_cn.name}.D{ref_dim_idx}", "tile": tile_size}]

    def _get_inter_core_tiling_dims(self, sub_workload: Workload, cns: tuple[ComputationNode, ...]) -> set:
        """Return unique dims used by inter_core_tiling across all nodes."""
        inter_dims = set()
        for cn in cns:
            cores = self._select_cores_for_node(cn)
            if len(cores) <= 1:
                continue
            split_dim_idx = self._find_split_dim(sub_workload, cn, len(cores))
            if split_dim_idx is not None:
                node_dims = sub_workload.get_dims(cn)
                if split_dim_idx < len(node_dims):
                    inter_dims.add(node_dims[split_dim_idx])
        return inter_dims
