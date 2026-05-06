from collections import defaultdict
from typing import TYPE_CHECKING

import sympy as sp
from xdsl.ir.affine import AffineConstantExpr, AffineDimExpr, AffineExpr

from stream.datatypes import InterCoreTiling, LayerDim
from stream.workload.node import ComputationNode, Node, TransferNode
from stream.workload.steady_state.iteration_space import (
    IterationVariable,
    IterationVariableType,
    LoopEffect,
    SteadyStateIterationSpace,
)

if TYPE_CHECKING:
    from stream.mapping.mapping import Mapping
    from stream.workload.workload import ComputationNode, HasIterationSpace, TransferNode, Workload


def determine_fusion_splits(workload: "Workload", mapping: "Mapping") -> dict[LayerDim, int]:
    """
    Determine the best dimension to fuse the layers on.
    Currently, we fuse on the dimension with the smallest total size across all layers.
    """
    # Go through the defined mapping intra_core_tiling dims and check that they occur for all layers
    assert len(mapping.fused_groups) == 1, "Only single fused group mappings are supported currently."
    fused_group = mapping.fused_groups[0]
    _, unique_spatial_unrollings = collect_spatial_unrollings(workload, mapping)
    unique_unrollings_dict = dict(unique_spatial_unrollings)
    result = {}
    for dim, tile_size in fused_group.intra_core_tiling:
        # assert dim in max_dims, f"Fused group intra_core_tiling dimension {dim} not present for all layers."
        nb_splits, rem = divmod(workload.get_dimension_size(dim), int(tile_size * unique_unrollings_dict.get(dim, 1)))
        assert rem == 0, (
            f"Dimension size {workload.get_dimension_size(dim)} not divisible by "
            f"desired tile size {tile_size * unique_unrollings_dict.get(dim, 1)}"
        )
        result[dim] = nb_splits
    return result


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
    spatial_unrollings, unique_spatial_unrollings = collect_spatial_unrollings(workload, mapping)
    iteration_variables = _create_spatial_iteration_variables(workload, spatial_unrollings, unique_spatial_unrollings)
    iteration_variables = _add_temporal_iteration_variables(iteration_variables, fusion_splits, workload)
    iteration_variables = _insert_kernel_iteration_variables(iteration_variables, workload, unique_spatial_unrollings)
    ssis_dict = _create_steady_state_iteration_spaces(iteration_variables, workload)
    return ssis_dict


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
    """Iterate through all computation nodes and add the temporal iteration variables."""
    for node in workload.get_iteration_space_nodes():
        for dim, size in fusion_splits.items():
            if isinstance(node, ComputationNode):
                effect = LoopEffect.VARYING if dim in workload.get_dims(node) else LoopEffect.ABSENT
            elif isinstance(node, TransferNode):
                compute_preds_succs = get_compute_predecessors_successors(node, workload)
                if dim in workload.get_dims(node):
                    effect = LoopEffect.VARYING
                elif any(dim in workload.get_dims(n) for n in compute_preds_succs):
                    effect = LoopEffect.INVARIANT
                else:
                    effect = LoopEffect.ABSENT
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
            size, rem = divmod(workload.get_dimension_size(dim), spatial_unrolling)
            assert rem == 0, (
                f"Dim size {workload.get_dimension_size(dim)} not divisible by spatial unrolling {spatial_unrolling}"
            )
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
                iteration_variables[node].append(
                    IterationVariable(
                        dimension=dim,
                        size=unrolling,
                        effect=LoopEffect.VARYING,
                        type=IterationVariableType.SPATIAL,
                    )
                )
            elif dim not in workload.get_dims(node):
                if isinstance(node, ComputationNode):
                    effect = LoopEffect.ABSENT
                elif isinstance(node, TransferNode):
                    compute_preds_succs = get_compute_predecessors_successors(node, workload)
                    dim_not_in_any_compute = all(dim not in workload.get_dims(n) for n in compute_preds_succs)
                    effect = LoopEffect.ABSENT if dim_not_in_any_compute else LoopEffect.INVARIANT
                # This dimension is not present, so add an absent spatial var
                iteration_variables[node].append(
                    IterationVariable(
                        dimension=dim,
                        size=unrolling,
                        effect=effect,
                        type=IterationVariableType.SPATIOTEMPORAL,
                    )
                )
            elif any(dim == su[0] for su in spatial_unrollings[node]):
                # This node has a different unrolling size for the unique dim
                # Create a hybrid of both spatial and temporal iteration variables
                spatial_size = next(su[1] for su in spatial_unrollings[node] if su[0] == dim)
                remaining_size, rem = divmod(unrolling, spatial_size)
                assert rem == 0, f"Unrolling size {unrolling} not divisible by spatial size {spatial_size}"
                # First add the spatiotemporal variable
                effect = LoopEffect.VARYING if dim in workload.get_dims(node) else LoopEffect.INVARIANT
                iteration_variables[node].append(
                    IterationVariable(
                        dimension=dim,
                        size=remaining_size,
                        effect=effect,
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


def collect_spatial_unrollings(workload: "Workload", mapping: "Mapping"):
    spatial_unrollings: dict[ComputationNode, InterCoreTiling] = {}
    for node in workload.get_iteration_space_nodes():
        node_mapping = mapping.get(node)
        assert node_mapping is not None, f"No mapping found for node {node.name}"
        spatial_unrollings[node] = workload.get_unique_dims_inter_core_tiling(node, mapping)

    unique_spatial_unrollings: list[tuple[LayerDim, int]] = []
    for unrollings in spatial_unrollings.values():
        for unrolling in unrollings:
            # Keep the largest unrolling size for each dimension
            dim, size = unrolling
            existing = next((u for u in unique_spatial_unrollings if u[0] == dim), None)
            if existing is None:
                unique_spatial_unrollings.append((dim, size))
            elif size > existing[1]:
                idx = unique_spatial_unrollings.index(existing)
                unique_spatial_unrollings.pop(idx)
                unique_spatial_unrollings.insert(idx, (dim, size))
    return spatial_unrollings, unique_spatial_unrollings


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


def affine_bounds(expr: AffineExpr, dim_sizes: list[int]) -> tuple[int, int]:
    """
    Compute min/max value of expr when each dim i is in [0, dim_sizes[i]-1].
    Assumes expr is affine-linear in dims (no mod/floordiv).
    """
    # Extract coefficients by probing basis vectors (works with your AffineExpr.eval)
    n = len(dim_sizes)

    zero = [0] * n
    c = int(expr.eval(zero, []))  # constant term

    # coeff[i] = expr(e_i) - expr(0)
    coeffs: list[int] = []
    for i in range(n):
        e = [0] * n
        e[i] = 1
        coeffs.append(int(expr.eval(e, [])) - c)

    # Now compute min/max over a box.
    # For each coeff a:
    # - if a >= 0, min uses 0, max uses (S-1)
    # - if a < 0, min uses (S-1), max uses 0
    mins = c
    maxs = c
    for a, S in zip(coeffs, dim_sizes, strict=True):
        lo = 0
        hi = S - 1
        if a >= 0:
            mins += a * lo
            maxs += a * hi
        else:
            mins += a * hi
            maxs += a * lo

    return mins, maxs
