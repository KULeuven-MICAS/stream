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
import math
import os
from typing import Any

import yaml

from stream.datatypes import LayerDim
from stream.hardware.architecture.accelerator import Accelerator
from stream.hardware.architecture.core import Core
from stream.parser.mapping_validator import MappingValidator
from stream.workload.affine_access import map_dim_positions
from stream.workload.iterator_type import (
    IteratorType,
    derive_iterator_types,
    is_state_operand,
    nonlinear_reduction_dims,
    sequential_dims,
)
from stream.workload.node import ComputationNode
from stream.workload.tensor import Tensor
from stream.workload.workload import Workload, determine_fusion_cut_points

logger = logging.getLogger(__name__)


def _tensor_bits(shape: tuple[int, ...], tensor: Tensor) -> int:
    """Storage (bits) of a tensor tile of the given ``shape``."""
    return math.prod(shape) * tensor.operand_type.bitwidth


class GenericMappingGenerator:
    """Auto-generate a MappingValidator-compliant mapping dict for any Workload + Accelerator pair.

    Core selection follows the operator_types convention:
    - Cores without operator_types (None) accept all operator types.
    - Cores with operator_types only accept nodes whose type is in the list.
    - Offchip and shim cores are never used for computation.

    Inter-core tiling:
    - Specialized cores (pooling, simd) receive the node alone on a single core.
    - Generic compute cores receive the node split across all matching cores.

    Intra-core tiling:
    - Uses the first dimension of the first computation node at full tile size
      (no temporal splitting), which is always valid per MappingValidator rules.
    """

    def __init__(
        self,
        accelerator: Accelerator,
        workload: Workload,
        output_dir: str,
        intra_core_tiling: list[dict[str, Any]] | None = None,
    ) -> None:
        self.accelerator = accelerator
        self.workload = workload
        self.output_dir = output_dir
        # Optional caller-supplied fused-group intra-core (layer-fusion) tiling. Entries look like
        # {"dim": "NodeName.D{n}", "tile": size}; they override the trivial default in
        # _build_intra_core_tiling, filtered per group to the nodes that group actually contains.
        self.intra_core_tiling = intra_core_tiling

    # ---------------------------------------------------------------------- #
    # Public API                                                              #
    # ---------------------------------------------------------------------- #

    def generate_all_groups(self, cut_points: list[str] | None = None) -> tuple[list[str], list[Workload]]:
        """Generate one mapping YAML per fusion group.

        Args:
            cut_points: Node names to split at, in addition to FusionEdge boundaries. Defaults to the
                affine barriers ``determine_fusion_cut_points`` derives.

        Returns:
            A tuple ``(paths, sub_workloads)`` where *paths* is a list of
            absolute file paths to the written YAML files and *sub_workloads*
            is the list of sub-workloads returned by ``split_fusion_groups()``.
        """
        sub_workloads = self.workload.split_fusion_groups(cut_points=self._cut_points(cut_points))
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
        protected = self._protected_dims(sub_workload, tuple(cns))

        layers: list[dict[str, Any]] = []
        # Cores already given to earlier layers of this fused group. Used to place each layer on a
        # DISJOINT core set where possible, so the layers pipeline across steady-state iterations
        # (TETRA inter-iteration overlap) instead of time-sharing one core set. Degrades gracefully:
        # when a layer's candidate pool cannot give it an unused block, it shares cores as before.
        allocated_ids: set[int] = set()
        for cn in cns:
            cores = self._select_cores_for_node(cn)
            n_cores = len(cores)

            core_allocation: list[list[int]] = [[c.id for c in cores]]

            if n_cores > 1:
                split_factors = self._factor_split_across_dims(sub_workload, cn, n_cores, protected)
                if split_factors:
                    inter_core_tiling: list[list[dict[str, Any]]] = [
                        [{"dim": f"D{dim_idx}", "split": factor} for dim_idx, factor in split_factors]
                    ]
                    cores_used = math.prod(factor for _, factor in split_factors)
                    if cores_used < n_cores:
                        # The workload's dimensions can't be tiled across every core; use the largest
                        # achievable subset, preferring cores not yet taken by earlier layers so the
                        # layers run on disjoint sets (fall back to the first cores_used when the pool
                        # of free cores is exhausted).
                        free = [c for c in cores if c.id not in allocated_ids]
                        block = free[:cores_used] if len(free) >= cores_used else cores[:cores_used]
                        core_allocation = [[c.id for c in block]]
                else:
                    inter_core_tiling = []
            else:
                inter_core_tiling = []

            allocated_ids.update(core_id for group in core_allocation for core_id in group)
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
           If any specialized cores match, use them exclusively.
        2. Generic cores (operator_types is None — accepts all ops).
           Use all matching generic cores together.
        3. Fallback: if nothing matches, use all cores with kind 'compute'.

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
            # prefer specialized core(s) over generic compute cores
            return specialized_cores

        if generic_cores:
            # use all generic compute cores together
            return generic_cores

        # fallback: no match — use all cores with kind 'compute'
        fallback = [c for c in self.accelerator.core_list if c.type == "compute"]
        logger.warning("No core found for operator '%s'; falling back to all compute cores.", node_op)
        return fallback

    def _factor_split_across_dims(
        self, sub_workload: Workload, cn: ComputationNode, n_cores: int, protected: set[LayerDim]
    ) -> list[tuple[int, int]]:
        """Distribute an inter-core split of *n_cores* across the node's dimensions.

        Unrolling a single dimension by ``n_cores`` fails whenever no dimension is
        divisible by it -- e.g. a 36-core mesh on a 2-conv whose dimensions are powers
        of two plus 3x3 kernels (``32 % 36 != 0``). Instead, factor ``n_cores`` across
        multiple dimensions so the per-dimension factors multiply back to ``n_cores``
        (a "dataflow-style" split). Each factor divides its dimension's size, so the
        resulting tiling is always valid.

        Dimensions are consumed largest-first, so parallel output dimensions (OY/OX/K)
        absorb the split before the small reduction/kernel dimensions, keeping
        cross-core reduction minimal. If ``n_cores`` cannot be fully factored over the
        available dimensions, the largest achievable subset is returned (product of
        factors < ``n_cores``) rather than forcing an indivisible split.

        ``protected`` are global dimensions that must never be inter-core split (a SEQUENTIAL
        recurrence carry, or a nonlinear normalization reduction) for any node in the fused group.

        Returns a list of ``(dim_index, factor)`` pairs, empty when the node has no
        splittable dimensions.
        """
        dims = sub_workload.get_dims(cn)
        if not dims:
            return []

        # (index, size) per splittable dimension, largest first (protected dims excluded).
        dim_sizes = sorted(
            ((idx, sub_workload.get_dimension_size(dim)) for idx, dim in enumerate(dims) if dim not in protected),
            key=lambda pair: pair[1],
            reverse=True,
        )

        remaining = n_cores
        split_factors: list[tuple[int, int]] = []
        for dim_idx, size in dim_sizes:
            if remaining == 1:
                break
            # Largest factor of `remaining` that also divides this dimension's size.
            factor = math.gcd(remaining, size)
            if factor > 1:
                split_factors.append((dim_idx, factor))
                remaining //= factor
        return split_factors

    def _protected_dims(self, sub_workload: Workload, cns: tuple[ComputationNode, ...]) -> set[LayerDim]:
        """Global dimensions the whole fused group must not inter-core split: a dim that is a SEQUENTIAL
        recurrence carry or a nonlinear (softmax/layernorm) reduction for ANY node. The group shares one
        spatial unrolling, so a dim illegal for one fused node is illegal for all -- e.g. the attention
        key axis is a parallel output of the scores matmul but the softmax's nonlinear reduction, so it
        stays resident, never split across cores (that would need the online-softmax rewrite)."""
        protected: set[LayerDim] = set()
        for cn in cns:
            node_dims = sub_workload.get_dims(cn)
            for pos in sequential_dims(cn) | nonlinear_reduction_dims(cn):
                if pos < len(node_dims):
                    protected.add(node_dims[pos])
        return protected

    def _recurrence_dims(self, sub_workload: Workload, cns: tuple[ComputationNode, ...]) -> set[LayerDim]:
        """Global dimensions that carry a recurrent state (SEQUENTIAL) for any node in the fused group --
        the SSM/linear-attention token axis. This is the axis a recurrence must stream *temporally* (one
        chunk of timesteps at a time, state resident); it is never split across cores (``_protected_dims``)."""
        dims: set[LayerDim] = set()
        for cn in cns:
            node_dims = sub_workload.get_dims(cn)
            for pos in sequential_dims(cn):
                if pos < len(node_dims):
                    dims.add(node_dims[pos])
        return dims

    def _streaming_axis(
        self, sub_workload: Workload, cns: tuple[ComputationNode, ...], indexed: set[LayerDim]
    ) -> LayerDim | None:
        """The single axis a fused group streams, restricted to dims that index a fused intermediate and
        have size > 1.

        A recurrence streams its **SEQUENTIAL** token axis: the carried state is then the only cross-tile
        residency (O(1) in the sequence length), whereas tiling a parallel channel axis instead would leave
        every intermediate O(L) -- the paper's reason to tile along L, not D or N. A feed-forward group has
        no sequential axis, so it streams its largest fusible **PARALLEL** axis (e.g. attention's query),
        keeping the reductions resident. Returns None when nothing is streamable."""

        def largest(dims: set[LayerDim]) -> LayerDim | None:
            candidates = [d for d in dims if sub_workload.get_dimension_size(d) > 1]
            return max(candidates, key=sub_workload.get_dimension_size) if candidates else None

        return largest(self._recurrence_dims(sub_workload, cns) & indexed) or largest(
            self._fusible_parallel_dims(sub_workload, cns) & indexed
        )

    def _indexed_by_intermediates(
        self, sub_workload: Workload, cns: tuple[ComputationNode, ...]
    ) -> tuple[list[Tensor], dict[LayerDim, set[Tensor]]]:
        """The group's fused intermediates (tensors produced and consumed inside the group) and, per global
        dimension, the set of those intermediates it indexes."""
        intermediates = [t for cn in cns for t in cn.outputs if any(t in c.inputs for c in cns)]
        indexed: dict[LayerDim, set[Tensor]] = {}
        for tensor in intermediates:
            producer = next(cn for cn in cns if tensor in cn.outputs)
            dims = sub_workload.get_dims(producer)
            for pos in map_dim_positions(producer.get_mapping(tensor)):
                if pos < len(dims):
                    indexed.setdefault(dims[pos], set()).add(tensor)
        return intermediates, indexed

    def _inter_core_unrolling(self, sub_workload: Workload, cns: tuple[ComputationNode, ...]) -> dict[LayerDim, int]:
        """Per global loop dimension, the largest inter-core split factor applied to it across the
        group. This is exactly the "spatial unrolling" ``determine_fusion_splits`` divides by (it reads
        it back from each layer's inter-core tiling), so the default intra-core tile must divide it out
        to stay a no-op. A dimension shared across nodes (e.g. self-attention's query==key==seq collapse
        to one symbol) takes the max, matching the fused-split accounting."""
        protected = self._protected_dims(sub_workload, cns)
        unroll: dict[LayerDim, int] = {}
        for cn in cns:
            cores = self._select_cores_for_node(cn)
            if len(cores) <= 1:
                continue
            node_dims = sub_workload.get_dims(cn)
            for dim_idx, factor in self._factor_split_across_dims(sub_workload, cn, len(cores), protected):
                if dim_idx < len(node_dims):
                    dim = node_dims[dim_idx]
                    unroll[dim] = max(unroll.get(dim, 1), factor)
        return unroll

    def _cut_points(self, cut_points: list[str] | None) -> list[str]:
        """The caller's fusion cuts, else the affine barriers. Defaulting here rather than at each call
        site is what keeps ``fusion_tiling_plan`` and a real run from disagreeing about what fuses."""
        return determine_fusion_cut_points(self.workload) if cut_points is None else cut_points

    def _build_intra_core_tiling(
        self, sub_workload: Workload, cns: tuple[ComputationNode, ...]
    ) -> list[dict[str, Any]]:
        """Intra-core (layer-fusion) tiling for one fused group, in precedence order: a caller-supplied
        ``intra_core_tiling`` wins outright, else automatic fusion tiling along the streaming axis, else
        the whole-layer tile. A caller who supplies entries never silently gets the automatic ones."""
        if self.intra_core_tiling is not None:
            names = {cn.name for cn in cns}
            selected = [dict(e) for e in self.intra_core_tiling if str(e["dim"]).split(".")[0] in names]
            return selected or self._whole_layer_tiling(sub_workload, cns)
        return self._auto_fusion_tiling(sub_workload, cns) or self._whole_layer_tiling(sub_workload, cns)

    def _whole_layer_tiling(self, sub_workload: Workload, cns: tuple[ComputationNode, ...]) -> list[dict[str, Any]]:
        """Tile the first node's first dimension so the group is one steady-state tile (nb_splits=1).
        The tile is ``dim_size // inter_core_unrolling`` so ``tile x unrolling == dim_size`` and
        ``determine_fusion_splits`` cannot overflow. Empty only when no node has dimensions."""
        unroll = self._inter_core_unrolling(sub_workload, cns)
        for ref_cn in cns:
            dims = sub_workload.get_dims(ref_cn)
            if dims:
                dim_size = sub_workload.get_dimension_size(dims[0])
                factor = unroll.get(dims[0], 1)
                tile = dim_size // factor if factor > 1 and dim_size % factor == 0 else dim_size
                return [{"dim": f"{ref_cn.name}.D0", "tile": tile}]
        return []

    def fusion_tiling_plan(self, cut_points: list[str] | None = None) -> list[dict[str, Any]]:
        """A serialisable description of what fuses and how it is tiled, per fused group.

        For each group: its member nodes (with the fused-kernel tag so a softmax's sub-ops can be
        collapsed for display); the STREAMED axis the fusion tiles and its tile size -- a recurrence's
        SEQUENTIAL token axis (``recurrence: true``) or a feed-forward group's parallel axis; the RESIDENT
        axes kept on-chip while it streams (the softmax key axis, a matmul contraction, or the recurrence
        state's channel/state axes); a per-tensor breakdown of each tensor's full vs on-chip-tile shape
        (the tile sizes exposed on the tensors); and the on-chip buffer (elements) of the largest streamed
        intermediate. Goes through the same cut points and the same ``_build_intra_core_tiling`` a run
        does, so what is shown is what the pipeline would map."""
        groups: list[dict[str, Any]] = []
        for sub in self.workload.split_fusion_groups(cut_points=self._cut_points(cut_points)):
            cns = sub.get_computation_nodes()
            nodes = [{"name": cn.name, "type": cn.type, "fused_kernel": cn.fused_kernel} for cn in cns]

            _, indexed = self._indexed_by_intermediates(sub, tuple(cns))
            fusion_dim = self._streaming_axis(sub, tuple(cns), set(indexed))

            streamed_axis: dict[str, Any] | None = None
            tile: int | None = None
            recurrence = False
            buffer_elements = 0
            factor = 1
            if fusion_dim is not None:
                size = sub.get_dimension_size(fusion_dim)
                tiling = self._build_intra_core_tiling(sub, tuple(cns))
                tile = self._tile_of(sub, tuple(cns), fusion_dim, tiling) or size
                recurrence = fusion_dim in self._recurrence_dims(sub, tuple(cns))
                streamed_axis = {"name": str(fusion_dim), "size": size}
                factor = size // tile if tile else 1
                buffer_elements = max(
                    (
                        math.prod(sub.get_tensor_shape_with_tiling(t, [(fusion_dim, factor)]))
                        for t in indexed[fusion_dim]
                    ),
                    default=0,
                )

            groups.append(
                {
                    "nodes": nodes,
                    "streamed_axis": streamed_axis,
                    "tile": tile,
                    "recurrence": recurrence,
                    "resident_axes": self._resident_axes(sub, tuple(cns), fusion_dim),
                    "tensors": self._tensor_tiles(sub, tuple(cns), fusion_dim, factor),
                    "buffer_elements": int(buffer_elements),
                }
            )
        return groups

    def _tensor_tiles(
        self,
        sub_workload: Workload,
        cns: tuple[ComputationNode, ...],
        fusion_dim: LayerDim | None,
        factor: int,
    ) -> list[dict[str, Any]]:
        """Per distinct tensor touched by the group's compute nodes: its full shape and its on-chip tile
        shape when the streamed axis is tiled by ``factor``. ``streamed`` marks a tensor the fusion tiles
        (its shape shrinks); ``state`` marks the recurrence carry (read at t-1). This is the tile-size view
        rendered on the tensors."""
        seen: set[str] = set()
        out: list[dict[str, Any]] = []
        for cn in cns:
            is_state = {t.name for t in cn.inputs if is_state_operand(cn, t)}
            for tensor in cn.tensors:
                if tensor.name in seen:
                    continue
                seen.add(tensor.name)
                full = tuple(tensor.shape)
                tiled = (
                    sub_workload.get_tensor_shape_with_tiling(tensor, [(fusion_dim, factor)])
                    if fusion_dim is not None and factor > 1
                    else full
                )
                out.append(
                    {
                        "name": tensor.name,
                        "full": list(full),
                        "tile": list(tiled),
                        "streamed": tuple(tiled) != full,
                        "state": tensor.name in is_state,
                    }
                )
        return out

    def _tile_of(
        self,
        sub_workload: Workload,
        cns: tuple[ComputationNode, ...],
        fusion_dim: LayerDim,
        tiling: list[dict[str, Any]],
    ) -> int | None:
        """The tile size the auto tiling assigns to ``fusion_dim`` (None when it tiles a different dim or
        does not tile -- i.e. the whole axis stays resident)."""
        for entry in tiling:
            node_name, _, pos = str(entry["dim"]).partition(".D")
            node = next((n for n in cns if n.name == node_name), None)
            if node is not None and pos.isdigit():
                dims = sub_workload.get_dims(node)
                if int(pos) < len(dims) and dims[int(pos)] == fusion_dim:
                    return int(entry["tile"])
        return None

    def _resident_axes(
        self, sub_workload: Workload, cns: tuple[ComputationNode, ...], fusion_dim: LayerDim | None
    ) -> list[dict[str, Any]]:
        """Axes kept resident while the streamed axis flows, largest first. ``softmax`` marks the axis a
        softmax reduces (monolithic NormalizationNode nonlinear axis or a decomposed ReduceMax/ReduceSum
        sub-op); other reductions are linear matmul contractions. ``state`` marks an axis of a recurrence's
        carried state (the ``[D,N]`` the scan keeps on-chip across timesteps) -- every axis of the state
        tensor other than the streamed token axis."""
        softmax_axes: dict[LayerDim, bool] = {}
        state_axes: set[LayerDim] = set()
        for cn in cns:
            node_dims = sub_workload.get_dims(cn)
            types = derive_iterator_types(cn)
            nonlinear = nonlinear_reduction_dims(cn)
            for pos, dim in enumerate(node_dims):
                if types.get(pos) == IteratorType.REDUCTION or pos in nonlinear:
                    from_softmax = pos in nonlinear or (
                        cn.fused_kernel is not None and types.get(pos) == IteratorType.REDUCTION
                    )
                    softmax_axes[dim] = softmax_axes.get(dim, False) or from_softmax
            for tensor in cn.inputs:
                if is_state_operand(cn, tensor):
                    for pos in map_dim_positions(cn.get_mapping(tensor)):
                        if pos < len(node_dims) and node_dims[pos] != fusion_dim:
                            state_axes.add(node_dims[pos])
        all_axes = set(softmax_axes) | state_axes
        return [
            {
                "name": str(dim),
                "size": sub_workload.get_dimension_size(dim),
                "softmax": softmax_axes.get(dim, False),
                "state": dim in state_axes,
            }
            for dim in sorted(all_axes, key=sub_workload.get_dimension_size, reverse=True)
        ]

    def _fusible_parallel_dims(self, sub_workload: Workload, cns: tuple[ComputationNode, ...]) -> set[LayerDim]:
        """Global dims that are a PARALLEL output axis for *every* node that indexes them.

        These are the only axes a fused group can tile so each tile produces complete outputs and every
        reduction (linear contraction or nonlinear softmax) stays resident -- e.g. attention's query
        axis, not the key axis (softmax reduction) nor the head axis (the output projection reduces it)."""
        non_parallel: set[LayerDim] = set()
        all_dims: set[LayerDim] = set()
        for cn in cns:
            node_dims = sub_workload.get_dims(cn)
            types = derive_iterator_types(cn)
            nonlinear = nonlinear_reduction_dims(cn)
            for pos, dim in enumerate(node_dims):
                all_dims.add(dim)
                if types.get(pos) != IteratorType.PARALLEL or pos in nonlinear:
                    non_parallel.add(dim)
        return all_dims - non_parallel

    def _auto_fusion_tiling(  # noqa: PLR0911 -- a sequence of early-out guards, each a distinct "no tiling" case
        self, sub_workload: Workload, cns: tuple[ComputationNode, ...]
    ) -> list[dict[str, Any]]:
        """Automatically fuse a multi-node group along its streaming axis, tiled so the largest resident
        intermediate fits on-chip.

        The streaming axis (:meth:`_streaming_axis`) is a recurrence's SEQUENTIAL token axis when the group
        has one (stream timesteps, keep the state resident -- the paper's Fuse-All), else the largest axis
        PARALLEL for every node (stream attention's query, keep the keys resident). Tiles it to the largest
        block (per core) whose resident intermediate fits a fraction of the core's memory; the reductions
        and the recurrence state are never tiled. Returns [] to fall back to the trivial whole-layer tiling."""
        if len(cns) <= 1:
            return []
        intermediates, indexed = self._indexed_by_intermediates(sub_workload, cns)
        if not intermediates:
            return []
        fusion_dim = self._streaming_axis(sub_workload, cns, set(indexed))
        if fusion_dim is None:
            return []

        full = sub_workload.get_dimension_size(fusion_dim)
        # Only the intermediates the fusion dim indexes shrink with the tile; the others (e.g. attention's
        # K/V, indexed by the key axis, not the query) are separate resident tensors the memory model
        # handles -- they are not the streamed fusion buffer and must not drive the fusion tile size.
        streamed = [
            t for t in intermediates if sub_workload.get_tensor_shape_with_tiling(t, [(fusion_dim, full)]) != t.shape
        ]
        if not streamed:
            return []
        unroll = self._inter_core_unrolling(sub_workload, cns).get(fusion_dim, 1)
        per_core = full // unroll if unroll > 1 and full % unroll == 0 else full
        capacity_bits = min(
            (cores[0].get_memory_capacity() for cn in cns if (cores := self._select_cores_for_node(cn))),
            default=0,
        )
        budget = capacity_bits // 2  # the fusion intermediate shares L1 with weights + activations

        def resident_bits(tile: int) -> int:
            factor = full // tile
            return max(
                _tensor_bits(sub_workload.get_tensor_shape_with_tiling(t, [(fusion_dim, factor)]), t) for t in streamed
            )

        # Only tile when the whole per-core slice does NOT fit; a group that already fits keeps the
        # whole-layer tiling. This is the layer-fusion trigger.
        if budget <= 0 or resident_bits(per_core) <= budget:
            return []
        divisors = sorted((t for t in range(1, per_core + 1) if per_core % t == 0), reverse=True)
        tile = next((t for t in divisors if resident_bits(t) <= budget), None)
        if tile is None:
            return []
        for cn in cns:
            dims = sub_workload.get_dims(cn)
            if fusion_dim in dims:
                return [{"dim": f"{cn.name}.D{dims.index(fusion_dim)}", "tile": tile}]
        return []
