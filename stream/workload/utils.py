from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from math import ceil
from typing import TYPE_CHECKING

import networkx as nx
import sympy as sp
from xdsl.ir.affine import AffineConstantExpr, AffineDimExpr, AffineExpr

from stream.datatypes import InterCoreTiling, LayerDim
from stream.workload.affine_access import dims_in_expr
from stream.workload.iterator_type import (
    IteratorType,
    NonlinearReductionUnrollError,
    check_spatial_unroll_accumulation_free,
    check_spatial_unroll_legal,
    derive_iterator_types,
    nonlinear_reduction_dims,
    sequential_dims,
)
from stream.workload.node import ComputationNode, HasIterationSpace, Node, NormalizationNode, Tensor, TransferNode
from stream.workload.steady_state.iteration_space import (
    IterationVariable,
    IterationVariableType,
    LoopEffect,
    SteadyStateIterationSpace,
)

if TYPE_CHECKING:
    from stream.mapping.mapping import Mapping
    from stream.workload.workload import ComputationNode, HasIterationSpace, TransferNode, UncoupledRead, Workload


#: Operator-type substrings that denote multiply-accumulate work (matched case-insensitively as a substring).
MAC_OPERATOR_TYPES: tuple[str, ...] = ("conv", "gemm", "matmul", "linear", "einsum")


def is_mac_operator_type(op_type: object) -> bool:
    """Whether ``op_type`` names multiply-accumulate work (the matmul/conv family)."""
    lowered = str(op_type).lower()
    return any(k in lowered for k in MAC_OPERATOR_TYPES)


def operand_inputs(node: ComputationNode) -> tuple[Tensor, ...]:
    """The inputs that are a core's operands ``I1``, ``I2``, ... in order. A MAC has two and a normalization one, its
    data; a further input such as a bias or a layer normalization's scale and shift is read with the output instead."""
    if isinstance(node, NormalizationNode):
        return node.inputs[:1]
    return node.inputs[:2] if is_mac_operator_type(node.type) else node.inputs


class UnfinishedReductionUnrollError(NonlinearReductionUnrollError):
    """Raised when a fused group tiles a dim a node reduces over before a node computed from its output walks it."""


@dataclass(frozen=True)
class UnfinishedReduction:
    """Why a dim stays whole in a fused group: ``node`` reduces over it at its dim ``position``, nonlinearly, or
    before ``walkers``, computed from its output, walk it."""

    node: str
    position: int
    nonlinear: bool
    walkers: tuple[str, ...] = ()


def determine_fusion_splits(workload: "Workload", mapping: "Mapping") -> dict[LayerDim, int]:
    """The number of blocks the fused group's intra-core tiling cuts each of its dims into."""
    assert len(mapping.fused_groups) == 1, "Only single fused group mappings are supported currently."
    fused_group = mapping.fused_groups[0]
    _, largest = collect_spatial_unrollings(workload, mapping)
    unfinished = unfinished_reductions(workload)
    uncoupled = {
        workload.leading_dim(workload.get_dims(read.reader)[read.position])[0]: read
        for read in workload.uncoupled_reads
        if {read.producer.name, read.reader.name} <= set(fused_group.layers)
    }
    result = {}
    for dim, tile_size in fused_group.intra_core_tiling:
        nb_splits = fused_tile_splits(workload, fused_group.name, dim, tile_size, largest.get(dim))
        if nb_splits > 1 and dim in unfinished:
            raise _unfinished_reduction_error(fused_group.name, dim, nb_splits, unfinished[dim])
        if nb_splits > 1 and dim in uncoupled:
            raise _uncoupled_read_error(fused_group.name, dim, nb_splits, uncoupled[dim])
        # A tile the cores' split already covers leaves no loop to step: one of a single trip is none.
        if nb_splits > 1 or largest.get(dim) is None:
            result[dim] = nb_splits
    return result


def _uncoupled_read_error(group: str, dim: LayerDim, blocks: int, read: "UncoupledRead") -> ValueError:
    """The refusal of a fused group cutting ``dim`` into ``blocks`` though ``read`` leaves its producer's axis whole."""
    return ValueError(
        f"Fused group {group} tiles {dim} into {blocks} blocks, but {read.reader.name} reads axis {read.axis} of "
        f"{read.producer.name}'s output, of extent {read.extent}, along it, which fuses the two only for an extent "
        f"from {read.low} to {read.high}: {read.producer.name} computes that axis whole. Keep {dim} whole or cut the "
        f"group between {read.producer.name} and {read.reader.name}."
    )


def _unfinished_reduction_error(group: str, dim: LayerDim, blocks: int, why: UnfinishedReduction) -> ValueError:
    """The refusal of a fused group cutting ``dim`` into ``blocks`` though ``why`` says it must stay whole."""
    tiles = f"Fused group {group} tiles {dim} into {blocks} blocks, but {why.node}.D{why.position} reduces"
    if why.nonlinear:
        return NonlinearReductionUnrollError(
            f"{tiles} nonlinearly over it (softmax/normalization): keep {dim} whole or apply the online-softmax "
            "rewrite."
        )
    return UnfinishedReductionUnrollError(
        f"{tiles} over it and {', '.join(why.walkers)}, computed from its output, walk it: each block would read "
        f"{why.node}'s unfinished reduction. Keep {dim} whole or cut the group after {why.node}."
    )


def checked_core_splits(workload: "Workload", node: "HasIterationSpace", mapping: "Mapping") -> InterCoreTiling:
    """``node``'s core splits over unique dims; a ValueError names the node, dim, extent and factor of one that does
    not divide its dim (the steady state has no remainder tiles)."""
    node_tiling = mapping.get(node).inter_core_tiling
    splits = workload.get_unique_dims_inter_core_tiling(node, mapping)
    for (named, factor), (dim, _) in zip(node_tiling[0] if node_tiling else (), splits, strict=True):
        if (extent := workload.get_dimension_size(dim)) % factor:
            raise ValueError(
                f"{node.name} splits {_split_axis(workload, node, named)} over {factor} cores, but the axis it splits, "
                f"{dim}, has extent {extent}, which {factor} does not divide."
            )
    return splits


def _split_axis(workload: "Workload", node: "HasIterationSpace", named: LayerDim) -> str:
    """The axis a mapping names ``node``'s split by, with its own extent and stride where it strides its unique dim."""
    if named.prefix != "d":
        return str(named)
    node_dim = workload.get_dims(node)[named.position]
    step = abs(workload.leading_dim(node_dim)[1])
    if step == 1:
        return f"D{named.position}"
    return f"D{named.position} (extent {workload.get_dimension_size(node_dim)}, stride {step})"


def _check_splits_nest(
    workload: "Workload",
    mapping: "Mapping",
    spatial_unrollings: dict["HasIterationSpace", InterCoreTiling],
    largest: dict[LayerDim, tuple["HasIterationSpace", int]],
) -> None:
    """Refuse a node's core split of a dim that does not divide the largest split of that dim: the smaller split's
    cores must each take whole groups of the larger's."""
    for node, splits in spatial_unrollings.items():
        for dim, factor in splits:
            splitter, cores = largest[dim]
            if cores % factor:
                raise ValueError(
                    f"{node.name} splits {_named_split(workload, mapping, node, dim)} over {factor} cores and "
                    f"{splitter.name} splits {_named_split(workload, mapping, splitter, dim)} over {cores}, but "
                    f"{factor} does not divide {cores}: splits of one dim ({dim}) must nest."
                )


def _named_split(workload: "Workload", mapping: "Mapping", node: "HasIterationSpace", dim: LayerDim) -> str:
    """The axis the mapping names ``node``'s split of unique ``dim`` by."""
    named = mapping.get(node).inter_core_tiling[0]
    unique = workload.get_unique_dims_inter_core_tiling(node, mapping)
    return next(_split_axis(workload, node, n) for (n, _), (u, _) in zip(named, unique, strict=True) if u == dim)


def fused_tile_splits(
    workload: "Workload", group: str, dim: LayerDim, tile: int, core_split: tuple["HasIterationSpace", int] | None
) -> int:
    """The blocks a fused ``tile`` of ``dim``, times its largest ``core_split``, cuts the dim into; a ValueError names
    the group, dim, tile, splitting node and extent when the product does not divide it."""
    splitter, cores = core_split or (None, 1)
    extent = workload.get_dimension_size(dim)
    blocks, rem = divmod(extent, tile * cores)
    if rem:
        split = f" and {splitter.name} splits it over {cores} cores" if splitter is not None else ""
        raise ValueError(
            f"Fused group {group} tiles {dim} by {tile}{split}, but its extent {extent} is not a multiple of "
            f"{tile * cores}."
        )
    return blocks


def unfinished_reduction_dims(workload: "Workload") -> set[LayerDim]:
    """Global dims a node reduces over that a node computed from its output walks, or a normalization reduces."""
    return set(unfinished_reductions(workload))


def unfinished_reductions(workload: "Workload") -> dict[LayerDim, UnfinishedReduction]:
    """Each of :func:`unfinished_reduction_dims` with why it stays whole: a nonlinear reduction wins over a walked
    one, else the first node found reducing it."""
    found: dict[LayerDim, UnfinishedReduction] = {}
    for node in workload.get_iteration_space_nodes():
        node_dims = workload.get_dims(node)
        reduced = {p for p, kind in derive_iterator_types(node).items() if kind is IteratorType.REDUCTION}
        later = sorted(
            (n for n in nx.descendants(workload, node) if isinstance(n, ComputationNode)), key=lambda n: n.name
        )
        for p in sorted(p for p in reduced if p < len(node_dims)):
            if walkers := tuple(n.name for n in later if node_dims[p] in workload.get_dims(n)):
                found.setdefault(node_dims[p], UnfinishedReduction(node.name, p, False, walkers))
        for p in sorted(p for p in nonlinear_reduction_dims(node) if p < len(node_dims)):
            if node_dims[p] not in found or not found[node_dims[p]].nonlinear:
                found[node_dims[p]] = UnfinishedReduction(node.name, p, True)
    return found


def get_equivalent_dimension(old_workload: "Workload", new_workload: "Workload", dim: LayerDim):
    node_with_dim = None
    for n in old_workload.get_computation_nodes():
        if dim in old_workload.get_dims(n):
            node_with_dim = n
            break
    assert node_with_dim is not None, f"Dimension {dim} not found in any computation node."
    # Find the position of the dim in the old workload node
    dim_idx = old_workload.get_dims(node_with_dim).index(dim)
    # Find equivalent nod ein the new workload based on name
    new_node = next(n for n in new_workload.get_computation_nodes() if n.name == node_with_dim.name)
    new_dim = new_workload.get_dims(new_node)[dim_idx]
    return new_dim


def generate_steady_state_iteration_spaces(
    workload: "Workload", mapping: "Mapping", fusion_splits: dict[LayerDim, int]
) -> dict["HasIterationSpace", SteadyStateIterationSpace]:
    spatial_unrollings, largest = collect_spatial_unrollings(workload, mapping)
    unique_spatial_unrollings = [(dim, factor) for dim, (_, factor) in largest.items()]
    iteration_variables = _create_spatial_iteration_variables(workload, spatial_unrollings, unique_spatial_unrollings)
    iteration_variables = _add_temporal_iteration_variables(iteration_variables, fusion_splits, workload)
    iteration_variables = _insert_kernel_iteration_variables(iteration_variables, workload, unique_spatial_unrollings)
    ssis_dict = _create_steady_state_iteration_spaces(iteration_variables, workload)
    return ssis_dict


def generate_tensor_ssis(
    workload: "Workload",
    tensor: Tensor,
    node: "HasIterationSpace",
    ssis: dict["HasIterationSpace | Tensor", SteadyStateIterationSpace],
) -> SteadyStateIterationSpace:
    """The iteration space of ``tensor`` as ``node`` produces or reads it: ``node``'s loops, each varying
    where it indexes the tensor."""
    producer_ssis = ssis.get(node, None)
    if producer_ssis is None:
        raise KeyError(f"Node {node.name} does not have a valid producer SSIS.")
    tensor_dims = workload.get_tensor_dimensions(tensor)
    tensor_ivs = []
    for prod_iv in producer_ssis.variables:
        prod_iv_dim = prod_iv.dimension
        if prod_iv_dim in tensor_dims:
            tensor_effect = LoopEffect.VARYING
        else:
            tensor_effect = LoopEffect.ABSENT if prod_iv.effect == LoopEffect.ABSENT else LoopEffect.INVARIANT
        tensor_ivs.append(
            IterationVariable(
                dimension=prod_iv_dim,
                size=prod_iv.size,
                type=prod_iv.type,
                effect=tensor_effect,
            )
        )
    return SteadyStateIterationSpace(variables=tuple(tensor_ivs))


def _create_steady_state_iteration_spaces(iteration_variables, workload: "Workload"):
    """Create the steady state iteration spaces for each computation node."""
    ssis_dict: dict[ComputationNode, SteadyStateIterationSpace] = {}
    for node in workload.get_iteration_space_nodes():
        ssis_dict[node] = SteadyStateIterationSpace(iteration_variables[node])
    return ssis_dict


def _add_temporal_iteration_variables(
    iteration_variables: dict["HasIterationSpace", list[IterationVariable]],
    fusion_splits: dict[LayerDim, int],
    workload: "Workload",
) -> dict["HasIterationSpace", list[IterationVariable]]:
    """Iterate through all computation nodes and add the temporal iteration variables: a node varies along a fused dim
    it steps along, even through a strided reader's ``s*d + r``."""

    def steps(node: "HasIterationSpace") -> set[LayerDim]:
        return {workload.leading_dim(d)[0] for d in workload.get_dims(node)}

    for node in workload.get_iteration_space_nodes():
        for dim, size in fusion_splits.items():
            if isinstance(node, ComputationNode):
                effect = LoopEffect.VARYING if dim in steps(node) else LoopEffect.ABSENT
            elif isinstance(node, TransferNode):
                compute_preds_succs = get_compute_predecessors_successors(node, workload)
                if dim in steps(node):
                    effect = LoopEffect.VARYING
                elif any(dim in steps(n) for n in compute_preds_succs):
                    effect = LoopEffect.INVARIANT
                else:
                    effect = LoopEffect.ABSENT
            else:
                raise TypeError(f"Unexpected node type for iteration-variable effect: {type(node).__name__}")
            iteration_variables[node].append(
                IterationVariable(dimension=dim, size=size, effect=effect, type=IterationVariableType.TEMPORAL)
            )
    return iteration_variables


def _insert_kernel_iteration_variables(
    iteration_variables: dict["HasIterationSpace", list[IterationVariable]],
    workload: "Workload",
    unique_spatial_unrollings: set[tuple[LayerDim, int]],
):
    """Iterate through all computation nodes and add the kernel iteration variables for the unique dimensions."""
    unique_dims, _ = workload.unique_dimensions()
    for node in workload.get_iteration_space_nodes():
        for dim in reversed(unique_dims):
            spatial_unrolling = next((su[1] for su in unique_spatial_unrollings if su[0] == dim), 1)
            dim_size = workload.get_dimension_size(dim)
            size, rem = divmod(dim_size, spatial_unrolling)
            assert rem == 0, f"Dim size {dim_size} not divisible by spatial unrolling {spatial_unrolling}"
            effect = LoopEffect.VARYING if dim in workload.get_dims(node) else LoopEffect.INVARIANT
            iteration_variables[node].insert(
                0,
                IterationVariable(
                    dimension=dim,
                    size=size,
                    effect=effect,
                    type=IterationVariableType.KERNEL,
                ),
            )
    return iteration_variables


def _create_spatial_iteration_variables(workload: "Workload", spatial_unrollings, unique_spatial_unrollings):
    """Iterate through all computation nodes and add the spatial or
    replacement temporal iteration variables if it doesn't have that spatial unrolling."""
    iteration_variables: dict[ComputationNode, list[IterationVariable]] = defaultdict(list)
    for node in workload.get_iteration_space_nodes():
        for spatial_unrolling in unique_spatial_unrollings:
            dim, unrolling = spatial_unrolling
            if spatial_unrolling in spatial_unrollings[node]:
                # Create a spatial iteration variable
                effect = LoopEffect.VARYING if dim in workload.get_dims(node) else LoopEffect.INVARIANT
                iteration_variables[node].append(
                    IterationVariable(
                        dimension=dim,
                        size=unrolling,
                        effect=effect,
                        type=IterationVariableType.SPATIAL,
                    )
                )
            elif any(dim == su[0] for su in spatial_unrollings[node]):
                # This node has a different unrolling size for the unique dim
                # Create a hybrid of both spatial and temporal iteration variables
                spatial_size = next(su[1] for su in spatial_unrollings[node] if su[0] == dim)
                remaining_size, rem = divmod(unrolling, spatial_size)
                assert rem == 0, f"Unrolling size {unrolling} not divisible by spatial size {spatial_size}"
                own = dim in workload.get_dims(node)
                effect = LoopEffect.VARYING if own else LoopEffect.INVARIANT
                # First add the spatiotemporal variable. A dimension outside this operand's
                # index space addresses nothing over time, so the remainder the cores iterate
                # is absent rather than invariant: the tensor is already where it is needed
                # and no step of that loop asks for it again. The spatial half stays, so the
                # operand still reaches every core the dimension is spread over.
                iteration_variables[node].append(
                    IterationVariable(
                        dimension=dim,
                        size=remaining_size,
                        effect=effect if own else LoopEffect.ABSENT,
                        type=IterationVariableType.SPATIOTEMPORAL,
                    )
                )
                # Then add the spatial variable
                iteration_variables[node].append(
                    IterationVariable(
                        dimension=dim,
                        size=spatial_size,
                        effect=effect,
                        type=IterationVariableType.SPATIAL,
                    )
                )
            elif dim not in workload.get_dims(node):
                # Not this node's dimension and no unrolling of its own: one absent extent
                # over the whole split.
                iteration_variables[node].append(
                    IterationVariable(
                        dimension=dim,
                        size=unrolling,
                        effect=LoopEffect.ABSENT,
                        type=IterationVariableType.SPATIOTEMPORAL,
                    )
                )
            else:
                type = IterationVariableType.SPATIOTEMPORAL
                # Create a replacement temporal variable
                effect = LoopEffect.VARYING if dim in workload.get_dims(node) else LoopEffect.INVARIANT
                iteration_variables[node].append(
                    IterationVariable(
                        dimension=dim,
                        size=unrolling,
                        effect=effect,
                        type=type,
                    )
                )

    return iteration_variables


def _reject_illegal_spatial_unroll(
    workload: "Workload", node: "HasIterationSpace", unrollings: InterCoreTiling, code_generated: bool
) -> None:
    """Raise if a mapping spatially unrolls a SEQUENTIAL dimension or a nonlinear reduction of ``node``,
    or, when ``node`` is code-generated, any reduction at all."""
    if not (code_generated or sequential_dims(node) or nonlinear_reduction_dims(node)):
        return
    node_dims = workload.get_dims(node)
    spatial_positions = [node_dims.index(dim) for dim, factor in unrollings if factor > 1 and dim in node_dims]
    check_spatial_unroll_legal(node, spatial_positions)
    if code_generated:
        check_spatial_unroll_accumulation_free(node, spatial_positions)


def collect_spatial_unrollings(
    workload: "Workload", mapping: "Mapping"
) -> tuple[dict["HasIterationSpace", InterCoreTiling], dict[LayerDim, tuple["HasIterationSpace", int]]]:
    """Each node's core splits above one, and the largest split of each dim with the first node splitting it so;
    a ValueError names a split that does not divide its dim or the dim's largest split."""
    spatial_unrollings: dict[HasIterationSpace, InterCoreTiling] = {}
    for node in workload.get_iteration_space_nodes():
        node_mapping = mapping.get(node)
        assert node_mapping is not None, f"No mapping found for node {node.name}"
        # A factor of one spreads nothing: the node then has no spatial variable of that dimension.
        unrollings = tuple(u for u in checked_core_splits(workload, node, mapping) if u[1] > 1)
        _reject_illegal_spatial_unroll(workload, node, unrollings, node_mapping.kernel is not None)
        spatial_unrollings[node] = unrollings
    largest: dict[LayerDim, tuple[HasIterationSpace, int]] = {}
    for node, unrollings in spatial_unrollings.items():
        for dim, factor in unrollings:
            if dim not in largest or factor > largest[dim][1]:
                largest[dim] = (node, factor)
    _check_splits_nest(workload, mapping, spatial_unrollings, largest)
    return spatial_unrollings, largest


def get_compute_predecessors_successors(tr: TransferNode, workload: "Workload") -> list[ComputationNode]:
    """Walk preds/succs until no transfer nodes remain, returning all compute nodes found."""
    compute_nodes: list[ComputationNode] = []
    visited: set[Node] = set()
    to_visit: list[Node] = [tr]
    while to_visit:
        current = to_visit.pop()
        if current in visited:
            continue
        visited.add(current)
        if isinstance(current, ComputationNode):
            compute_nodes.append(current)
        else:
            to_visit.extend(workload.predecessors(current))
            to_visit.extend(workload.successors(current))
    return compute_nodes


def get_node_with_largest_resource_allocation(nodes: list[ComputationNode], mapping: "Mapping") -> ComputationNode:
    """Return the node with the largest number of resource allocations in the mapping."""
    max_tiling = -1
    node_with_resource_allocation = None
    for node in nodes:
        node_mapping = mapping.get(node)
        assert node_mapping is not None, f"No mapping found for node {node.name}"
        allocation = node_mapping.resource_allocation
        assert len(allocation) == 1, "Multiple possible allocations not supported currently."
        allocation = allocation[0]
        allocation_size = len(allocation)
        if allocation_size > max_tiling:
            max_tiling = allocation_size
            node_with_resource_allocation = node
    assert node_with_resource_allocation is not None, "No node with resource allocation found."
    return node_with_resource_allocation


def sympy_to_xdsl(expr: sp.Expr) -> AffineExpr:
    """
    Convert SymPy expression over z0,z1,... into an xdsl AffineExpr.
    z* symbols become LayerDim(*), not AffineDimExpr(*).
    """
    expr = sp.expand(expr)
    # Constant term
    if expr.is_Number:
        return AffineConstantExpr(int(expr))
    # Single symbol: z3
    if isinstance(expr, sp.Symbol):
        name = expr.name
        if not name:
            raise ValueError("Encountered SymPy symbol without a name")
        idx = int(name[1:])

        if name.startswith("z"):
            return LayerDim(idx)
        elif name.startswith("d"):
            return AffineDimExpr(idx)
        else:
            raise ValueError(f"Unsupported symbol name: {name}")
    # Addition: z0 + 2*z1 - 3
    if isinstance(expr, sp.Add):
        affine_expr: AffineExpr = AffineConstantExpr(0)
        for term in expr.args:
            affine_expr += sympy_to_xdsl(term)
        return affine_expr
    # Multiplication: 2*z1, -3*z0
    if isinstance(expr, sp.Mul):
        coeff, rest = expr.as_coeff_Mul()
        if rest == 1:
            return AffineConstantExpr(int(coeff))
        base = sympy_to_xdsl(rest)
        return base * int(coeff)
    raise ValueError(f"Unsupported sympy expression type: {type(expr)} ({expr})")


def sliding_window(
    sizes: Sequence[int],
    kernel: Sequence[int],
    strides: Sequence[int] | None = None,
    dilations: Sequence[int] | None = None,
    pads: Sequence[int] | None = None,
    auto_pad: str = "NOTSET",
) -> tuple[list[int], ...]:
    """A sliding window's strides, dilations and leading padding per spatial axis: ``pads``, or what ONNX's
    ``auto_pad`` (``SAME_UPPER``, ``SAME_LOWER``, ``VALID``) derives from the input ``sizes`` and the ``kernel``."""
    strides, dilations = list(strides or [1] * len(sizes)), list(dilations or [1] * len(sizes))
    if auto_pad == "NOTSET":
        return strides, dilations, list(pads or [0] * len(sizes))[: len(sizes)]
    totals = [
        max(0, (ceil(n / s) - 1) * s + (f - 1) * d + 1 - n) if auto_pad != "VALID" else 0
        for n, f, s, d in zip(sizes, kernel, strides, dilations, strict=True)
    ]
    return strides, dilations, [t // 2 if auto_pad == "SAME_UPPER" else t - t // 2 for t in totals]


def window_index(
    outputs: Sequence[AffineExpr], kernels: Sequence[AffineExpr], strides, dilations, pads
) -> tuple[AffineExpr, ...]:
    """The index a sliding window reads along each spatial axis: ``s*o + d*f - p`` of its output and kernel dims."""
    return tuple(s * o + d * f - p for o, f, s, d, p in zip(outputs, kernels, strides, dilations, pads, strict=True))


def affine_coefficients(expr: AffineExpr, n: int) -> tuple[int, list[int]]:
    """The constant of an affine-linear ``expr`` over ``n`` dims and the coefficient of each dim."""
    c = int(expr.eval([0] * n, []))
    coefficients = [0] * n
    for d in dims_in_expr(expr):
        coefficients[d] = int(expr.eval([int(i == d) for i in range(n)], [])) - c
    return c, coefficients


def affine_bounds(expr: AffineExpr, dim_sizes: list[int]) -> tuple[int, int]:
    """
    Compute min/max value of expr when each dim i is in [0, dim_sizes[i]-1].
    Assumes expr is affine-linear in dims (no mod/floordiv).
    """
    c, coeffs = affine_coefficients(expr, len(dim_sizes))
    low = c + sum(a * (S - 1) for a, S in zip(coeffs, dim_sizes, strict=True) if a < 0)
    high = c + sum(a * (S - 1) for a, S in zip(coeffs, dim_sizes, strict=True) if a > 0)
    return low, high
