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
from collections.abc import Callable, Iterable
from typing import Any

import yaml
from xdsl.ir.affine import AffineMap

from stream.datatypes import LayerDim
from stream.hardware.architecture.accelerator import Accelerator
from stream.hardware.architecture.core import Core
from stream.mapping.capacity_tiler import CapacityTiler
from stream.opt.allocation.constraint_optimization.hardware import namespace_classes
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
        self._activations = {t.name for cn in workload.get_computation_nodes() for t in cn.outputs}
        self._accumulates_across_cores = all(
            cls.accumulates_across_cores for cls in namespace_classes(accelerator).values()
        )

    # ---------------------------------------------------------------------- #
    # Public API                                                              #
    # ---------------------------------------------------------------------- #

    def generate_all_groups(self, cut_points: list[str] | None = None) -> tuple[list[str], list[Workload]]:
        """Generate one mapping YAML per fusion group.

        Args:
            cut_points: Node names to split at, in addition to FusionEdge boundaries. Defaults to the
                affine barriers ``determine_fusion_cut_points`` derives, cut again where a group's weights
                cannot stay on its cores (see ``_cut_points``).

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

        Excludes offchip, shim, and memory cores unconditionally.  Selection priority:
        1. Specialized cores (operator_types is not None and node.type in list).
           If any specialized cores match, use them exclusively.
        2. Generic cores (operator_types is None — accepts all ops).
           Use all matching generic cores together.
        3. Fallback: if nothing matches, use all cores with kind 'compute'.

        This ensures MaxPool goes to the pooling core, Add to the simd core, and
        Conv/Gemm go to the generic compute cores that split it fastest.
        """
        _SKIP_TYPES = {"offchip", "shim", "memory"}
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
            return self._fastest_even_split(generic_cores)

        # fallback: no match — use all cores with kind 'compute'
        fallback = [c for c in self.accelerator.core_list if c.type == "compute"]
        logger.warning("No core found for operator '%s'; falling back to all compute cores.", node_op)
        return fallback

    @staticmethod
    def _fastest_even_split(cores: list[Core]) -> list[Core]:
        """The largest-array cores whose even split finishes first: an even split runs at the pace of its
        smallest array, so ``n`` cores of at least ``u`` units each deliver ``n * u`` MACs per cycle."""

        def units(core: Core) -> int:
            return getattr(getattr(core, "operational_array", None), "total_unit_count", 0) or 0

        _, floor = max((n * u, u) for n, u in enumerate(sorted(map(units, cores), reverse=True), start=1))
        return [core for core in cores if units(core) >= floor]

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

        Parallel output dimensions (OY/OX/K) are consumed before reduction ones, so a
        contraction only absorbs what the output axes could not: splitting a reduction
        leaves every core holding a partial sum that has to be reduced across the mesh.
        A contraction indexing an activation (an input another node of the workload
        produces) at least twice the output ranks with them where the cores can add partial
        sums: reducing the output, its partial sums at accumulator precision, then moves less
        than gathering that input onto every core, as in a tensor-parallel MLP whose down
        projection contracts the hidden dimension.
        Within each group the dimension that leaves the fused group's cores the smallest
        footprint goes first (an operand it does not index stays whole on every core; weights
        that overflow a core count before activations a sliding window streams), then the one
        smallest for this node, then the largest, then the outermost output axis. If
        ``n_cores`` cannot be fully factored over the available dimensions, the largest
        achievable subset is returned (product of factors < ``n_cores``) rather than forcing
        an indivisible split.

        ``protected`` are global dimensions that must never be inter-core split (a SEQUENTIAL
        recurrence carry, or a nonlinear normalization reduction) for any node in the fused group.

        Returns a list of ``(dim_index, factor)`` pairs, empty when the node has no
        splittable dimensions.
        """
        dims = sub_workload.get_dims(cn)
        if not dims:
            return []

        # (index, size) per splittable dimension, parallel axes before reductions and each
        # group largest first (protected dims excluded).
        types = derive_iterator_types(cn)
        operands = [(_tensor_bits(t.shape, t), map_dim_positions(cn.get_mapping(t))) for t in cn.tensors]

        output = [
            map_dim_positions(AffineMap(m.num_dims, 0, (r,)))
            for t in cn.outputs
            for m in [cn.get_mapping(t)]
            for r in m.results
        ]

        output_bits = sum(_tensor_bits(t.shape, t) for t in cn.outputs)
        activations = [
            map_dim_positions(cn.get_mapping(t))
            for t in cn.inputs
            if t.name in self._activations and _tensor_bits(t.shape, t) >= 2 * output_bits
        ]

        def ranks_as_parallel(idx: int) -> bool:
            if types.get(idx) == IteratorType.PARALLEL:
                return True
            return self._accumulates_across_cores and any(idx in indexed for indexed in activations)

        def footprint(idx: int, size: int) -> float:
            factor = math.gcd(n_cores, size)
            return sum(bits / factor if idx in indexed else bits for bits, indexed in operands)

        def outer(idx: int) -> int:
            return -next((axis for axis, read in enumerate(output) if idx in read), len(output))

        # A dim a node of the group reduces over is not one all of them split, so their tiles do not line up along
        # it: it is preferred only for the weights it fits.
        group = self._group_footprints(sub_workload, n_cores)
        for reducer in sub_workload.get_computation_nodes():
            reducer_dims = sub_workload.get_dims(reducer)
            for p, kind in derive_iterator_types(reducer).items():
                lead = sub_workload.leading_dim(reducer_dims[p])[0] if p < len(reducer_dims) else None
                if kind == IteratorType.REDUCTION and lead in group:
                    group[lead] = (group[lead][0], math.inf)
        dim_sizes = sorted(
            ((idx, sub_workload.get_dimension_size(dim)) for idx, dim in enumerate(dims) if dim not in protected),
            key=lambda pair: (
                ranks_as_parallel(pair[0]),
                tuple(-x for x in group.get(sub_workload.leading_dim(dims[pair[0]])[0], (math.inf, math.inf))),
                -footprint(*pair),
                pair[1],
                outer(pair[0]),
            ),
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

    def _group_footprints(
        self, sub_workload: Workload, n_cores: int, cns: tuple[ComputationNode, ...] | None = None
    ) -> dict[LayerDim, tuple[float, float]]:
        """Per unique dim of a group a window slides through, what its multi-core nodes hold per core when each splits
        along it: first how far the operands no sliding window streams (the weights) overflow half a core, then
        everything held, so every node splits along the dim that suits the whole group and their tiles line up.
        ``cns`` narrows the group to some of its nodes."""
        cns = cns or tuple(sub_workload.get_computation_nodes())
        sliding = {sub_workload.leading_dim(d)[0] for d in self._sliding_dims(sub_workload, cns)}
        if not sliding:
            return {}
        operands: list[tuple[int, frozenset[LayerDim]]] = []
        capacity = math.inf
        for cn in cns:
            if len(cores := self._select_cores_for_node(cn)) <= 1:
                continue
            capacity = min(capacity, *(c.get_memory_capacity() for c in cores))
            dims = sub_workload.get_dims(cn)
            for t in cn.tensors:
                lead = frozenset(sub_workload.leading_dim(dims[p])[0] for p in map_dim_positions(cn.get_mapping(t)))
                operands.append((_tensor_bits(t.shape, t), lead))
        held: dict[LayerDim, tuple[float, float]] = {}
        for dim in {d for _, lead in operands for d in lead}:
            factor = math.gcd(n_cores, sub_workload.get_dimension_size(dim))
            share = [(bits / factor if dim in lead else bits, lead) for bits, lead in operands]
            resident = sum(bits for bits, lead in share if not lead & sliding)
            held[dim] = (max(0.0, resident - capacity / 2), sum(bits for bits, _ in share))
        return held

    def _global_dims_at(
        self,
        sub_workload: Workload,
        cns: tuple[ComputationNode, ...],
        positions: Callable[[ComputationNode], Iterable[int]],
    ) -> set[LayerDim]:
        """Global dims at the node-relative positions ``positions(cn)`` yields, unioned over every node."""
        out: set[LayerDim] = set()
        for cn in cns:
            node_dims = sub_workload.get_dims(cn)
            out |= {node_dims[pos] for pos in positions(cn) if pos < len(node_dims)}
        return out

    def _protected_dims(self, sub_workload: Workload, cns: tuple[ComputationNode, ...]) -> set[LayerDim]:
        """Global dims never inter-core split: a SEQUENTIAL carry or nonlinear reduction for any node."""
        return self._global_dims_at(sub_workload, cns, lambda cn: sequential_dims(cn) | nonlinear_reduction_dims(cn))

    def _recurrence_dims(self, sub_workload: Workload, cns: tuple[ComputationNode, ...]) -> set[LayerDim]:
        """Global dims carrying a recurrent state (SEQUENTIAL) for any node."""
        return self._global_dims_at(sub_workload, cns, sequential_dims)

    def _streaming_axis(
        self, sub_workload: Workload, cns: tuple[ComputationNode, ...], indexed: set[LayerDim]
    ) -> LayerDim | None:
        """The single axis a fused group streams: dims indexing a fused intermediate with size > 1."""

        def largest(dims: set[LayerDim]) -> LayerDim | None:
            # Tie-break by name so equal-sized axes are chosen deterministically across processes.
            candidates = [d for d in dims if sub_workload.get_dimension_size(d) > 1]
            return max(candidates, key=lambda d: (sub_workload.get_dimension_size(d), str(d))) if candidates else None

        fusible = self._fusible_parallel_dims(sub_workload, cns) & indexed
        split = set(self._inter_core_unrolling(sub_workload, cns))
        sliding = {
            d for d in self._sliding_dims(sub_workload, cns) & fusible - split if sub_workload.get_dimension_size(d) > 1
        }
        if recurrent := largest(self._recurrence_dims(sub_workload, cns) & indexed):
            return recurrent
        if sliding:
            axis = max(sliding, key=lambda d: (self._output_axis(sub_workload, cns, d), str(d)))
            return sub_workload.leading_dim(axis)[0]
        return largest(fusible)

    def _sliding_dims(self, sub_workload: Workload, cns: tuple[ComputationNode, ...]) -> set[LayerDim]:
        """Global dims a node slides a window along: an output axis indexing an operand with one of its other dims."""
        out: set[LayerDim] = set()
        for cn in cns:
            dims = sub_workload.get_dims(cn)
            produced = set().union(*(map_dim_positions(cn.get_mapping(t)) for t in cn.outputs))
            for t in cn.inputs:
                access = cn.get_mapping(t)
                for result in access.results:
                    if len(read := map_dim_positions(AffineMap(access.num_dims, 0, (result,)))) > 1:
                        out |= {dims[p] for p in read & produced}
        return out

    def _output_axis(self, sub_workload: Workload, cns: tuple[ComputationNode, ...], dim: LayerDim) -> int:
        """The innermost axis ``dim`` indexes in any node's output, so a row-major activation streams its last axis."""
        return max(
            (
                axis
                for cn in cns
                for t in cn.outputs
                for axis, result in enumerate(cn.get_mapping(t).results)
                if any(
                    sub_workload.get_dims(cn)[p] == dim
                    for p in map_dim_positions(AffineMap(cn.get_mapping(t).num_dims, 0, (result,)))
                )
            ),
            default=-1,
        )

    def _indexed_by_intermediates(
        self, sub_workload: Workload, cns: tuple[ComputationNode, ...]
    ) -> tuple[list[Tensor], dict[LayerDim, set[Tensor]]]:
        """The group's fused intermediates and, per global dimension, the intermediates it indexes."""
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
        """The caller's fusion cuts, else the affine barriers ``determine_fusion_cut_points`` derives, cut again before
        a node whose weights overflow its cores more fused with the nodes before it than on their own."""
        if cut_points is not None:
            return cut_points
        cuts = determine_fusion_cut_points(self.workload)
        for sub in self.workload.split_fusion_groups(cut_points=cuts):
            segment: tuple[ComputationNode, ...] = ()
            for cn in sub.get_computation_nodes():
                fused = self._weight_overflow(sub, (*segment, cn))
                if segment and fused > self._weight_overflow(sub, segment) + self._weight_overflow(sub, (cn,)):
                    cuts.append(segment[-1].name)
                    segment = ()
                segment = (*segment, cn)
        return cuts

    def _weight_overflow(self, sub_workload: Workload, cns: tuple[ComputationNode, ...]) -> float:
        """How far ``cns``' weights overflow half their cores under the split that suits them best (0 when they fit)."""
        n_cores = max(len(self._select_cores_for_node(cn)) for cn in cns)
        held = self._group_footprints(sub_workload, n_cores, cns)
        return min((overflow for overflow, _ in held.values()), default=0.0)

    def _build_intra_core_tiling(
        self, sub_workload: Workload, cns: tuple[ComputationNode, ...]
    ) -> list[dict[str, Any]]:
        """Intra-core (layer-fusion) tiling: caller-supplied, else automatic fusion tiling, else whole-layer."""
        if self.intra_core_tiling is not None:
            names = {cn.name for cn in cns}
            selected = [dict(e) for e in self.intra_core_tiling if str(e["dim"]).split(".")[0] in names]
            return selected or self._whole_layer_tiling(sub_workload, cns)
        automatic = self._auto_fusion_tiling(sub_workload, cns) or self._whole_layer_tiling(sub_workload, cns)
        return self._capacity_refine(sub_workload, cns, automatic)

    def _capacity_refine(
        self, sub_workload: Workload, cns: tuple[ComputationNode, ...], seed_tiling: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Stream extra axes (e.g. a matmul contraction) when ``seed_tiling`` still overflows the operand buffer."""
        cores_per_node = {cn: self._select_cores_for_node(cn) for cn in cns}
        if not any(cores_per_node.values()):
            return seed_tiling
        unroll = self._inter_core_unrolling(sub_workload, cns)
        # Only nonlinear (softmax/layernorm) reductions are off-limits temporally; the contraction is streamed.
        protected = self._global_dims_at(sub_workload, cns, nonlinear_reduction_dims)
        split = self._protected_dims(sub_workload, cns)
        node_unroll = {
            cn: {
                sub_workload.get_dims(cn)[idx]: factor
                for idx, factor in self._factor_split_across_dims(sub_workload, cn, len(cores), split)
            }
            for cn, cores in cores_per_node.items()
            if len(cores) > 1
        }
        refined = CapacityTiler(sub_workload, self.accelerator).plan(
            cns, cores_per_node, unroll, protected, seed_tiling, node_unroll
        )
        return refined or seed_tiling

    def _whole_layer_tiling(self, sub_workload: Workload, cns: tuple[ComputationNode, ...]) -> list[dict[str, Any]]:
        """Tile the first node's first dimension so the group is one steady-state tile (nb_splits=1)."""
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
        """A serialisable per-group description of what fuses and how it is tiled."""
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
                        math.prod(sub.get_tensor_shape_with_tiling(t, [(fusion_dim, factor)], readers=True))
                        for dim, tensors in indexed.items()
                        if sub.leading_dim(dim)[0] == fusion_dim
                        for t in tensors
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
        """Per distinct tensor: full shape and on-chip tile shape when the streamed axis is tiled by ``factor``."""
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
                    sub_workload.get_tensor_shape_with_tiling(tensor, [(fusion_dim, factor)], readers=True)
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
        """The tile size the auto tiling assigns to ``fusion_dim`` (None when it does not tile that dim)."""
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
        """Axes kept resident while the streamed axis flows, largest first."""
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
        """Global dims that are a PARALLEL output axis for every node indexing them (e.g. attention's query axis)."""
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
        """Fuse a multi-node group along its streaming axis; [] falls back to the whole-layer tiling."""
        if len(cns) <= 1:
            return []
        intermediates, indexed = self._indexed_by_intermediates(sub_workload, cns)
        if not intermediates:
            return []
        fusion_dim = self._streaming_axis(sub_workload, cns, set(indexed))
        if fusion_dim is None:
            return []

        full = sub_workload.get_dimension_size(fusion_dim)
        # Only the intermediates the fusion dim indexes shrink with the tile; the others stay resident.
        streamed = [
            t
            for t in intermediates
            if sub_workload.get_tensor_shape_with_tiling(t, [(fusion_dim, full)], readers=True) != t.shape
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
                _tensor_bits(sub_workload.get_tensor_shape_with_tiling(t, [(fusion_dim, factor)], readers=True), t)
                for t in streamed
            )

        # Only tile when the whole per-core slice does not fit (the layer-fusion trigger).
        if budget <= 0 or resident_bits(per_core) <= budget:
            return []
        divisors = sorted((t for t in range(1, per_core + 1) if per_core % t == 0), reverse=True)
        tile = next((t for t in divisors if resident_bits(t) <= budget), divisors[-1])
        for cn in cns:
            dims = sub_workload.get_dims(cn)
            if fusion_dim in dims:
                return [{"dim": f"{cn.name}.D{dims.index(fusion_dim)}", "tile": tile}]
        return []
