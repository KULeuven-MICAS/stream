from collections import defaultdict
from math import prod
from typing import TYPE_CHECKING

import sympy as sp
from xdsl.ir.affine import AffineConstantExpr, AffineDimExpr, AffineExpr

from stream.datatypes import InterCoreTiling, LayerDim
from stream.workload.node import TransferNode
from stream.workload.steady_state.iteration_space import (
    IterationVariable,
    IterationVariableType,
    SteadyStateIterationSpace,
)

if TYPE_CHECKING:
    from stream.mapping.mapping import Mapping
    from stream.workload.workload import ComputationNode, HasIterationSpace, TransferNode, Workload


def determine_fusion_dimensions(workload: "Workload") -> dict[LayerDim, int]:
    """
    Determine the best dimension to fuse the layers on.
    Currently, we fuse on the dimension with the smallest total size across all layers.
    """
    dim_occurrence_count = defaultdict(int)
    for node in workload.get_iteration_space_nodes():
        for expr in workload.get_dims(node):
            for dim in expr.used_dims():
                dim_occurrence_count[LayerDim(dim)] += 1
    max_dim_count = max(dim_occurrence_count.values())
    max_dims = tuple(k for k, v in dim_occurrence_count.items() if v == max_dim_count)
    for max_dim in max_dims:
        assert dim_occurrence_count[max_dim] == len(workload.get_computation_nodes()), (
            "Not all layers share the same dimension for fusion."
        )
    return {
        max_dims[0]: workload.get_dimension_size(max_dims[0]),
        max_dims[1]: 16,  # fixed for testing
    }


def generate_steady_state_iteration_spaces(
    workload: "Workload", mapping: "Mapping", fuse_dimensions: dict[LayerDim, int]
) -> dict["HasIterationSpace", SteadyStateIterationSpace]:
    spatial_unrollings, unique_spatial_unrollings = collect_spatial_unrollings(workload, mapping)
    iteration_variables = _create_spatial_iteration_variables(workload, spatial_unrollings, unique_spatial_unrollings)
    temporal_unrollings = _derive_temporal_unrollings(workload, unique_spatial_unrollings, fuse_dimensions)
    iteration_variables = _add_temporal_iteration_variables(iteration_variables, temporal_unrollings, workload)
    iteration_variables = _insert_kernel_iteration_variables(iteration_variables, workload, unique_spatial_unrollings)
    ssis_dict = _create_steady_state_iteration_spaces(iteration_variables, workload)
    return ssis_dict


def _create_steady_state_iteration_spaces(iteration_variables, workload: "Workload"):
    """Create the steady state iteration spaces for each computation node."""
    ssis_dict: dict[ComputationNode, SteadyStateIterationSpace] = {}
    for node in workload.get_iteration_space_nodes():
        ssis_dict[node] = SteadyStateIterationSpace(iteration_variables[node])
        print(node.name, ssis_dict[node])
    return ssis_dict


def _add_temporal_iteration_variables(
    iteration_variables: dict["HasIterationSpace", list[IterationVariable]], temporal_unrollings, workload: "Workload"
) -> dict["HasIterationSpace", list[IterationVariable]]:
    """Iterate through all computation nodes and add the temporal iteration variables."""
    for node in workload.get_iteration_space_nodes():
        for temporal_unrolling in temporal_unrollings:
            dim, size = temporal_unrolling
            relevant = dim in workload.get_dims(node)
            iteration_variables[node].append(
                IterationVariable(dim, size, relevant, type=IterationVariableType.TEMPORAL)
            )
    return iteration_variables


def _derive_temporal_unrollings(workload: "Workload", unique_spatial_unrollings, fuse_dimensions: dict[LayerDim, int]):
    """Iterate through the unique workload dimensions and get temporal unrollings"""
    temporal_unrollings: list[tuple[LayerDim, int]] = []  # list because order matters
    unique_dims, _ = workload.unique_dimensions()
    for dim in unique_dims:  # iterate in different order here if needed
        if dim not in fuse_dimensions:
            size = 1
        else:
            size, rem = divmod(
                fuse_dimensions[dim],
                _get_total_spatial_unrolling_for_dim(dim, unique_spatial_unrollings),
            )
            assert rem == 0, (
                f"Dimension size {fuse_dimensions[dim]} not divisible by spatial unrolling "
                f"{_get_total_spatial_unrolling_for_dim(dim, unique_spatial_unrollings)}"
            )
        temporal_unrollings.append((dim, size))
    return temporal_unrollings


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
            relevant = dim in workload.get_dims(node)
            iteration_variables[node].insert(
                0,
                IterationVariable(
                    dimension=dim,
                    size=size,
                    relevant=relevant,
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
                        relevant=True,
                        type=IterationVariableType.SPATIAL,
                    )
                )
            elif any(dim == su[0] for su in spatial_unrollings[node]):
                # This node has a different unrolling size for the unique dim
                # Create a hybrid of both spatial and temporal iteration variables
                spatial_size = next(su[1] for su in spatial_unrollings[node] if su[0] == dim)
                iteration_variables[node].append(
                    IterationVariable(
                        dimension=dim,
                        size=spatial_size,
                        relevant=True,
                        type=IterationVariableType.SPATIAL,
                    )
                )
                remaining_size, rem = divmod(unrolling, spatial_size)
                assert rem == 0, f"Unrolling size {unrolling} not divisible by spatial size {spatial_size}"
                iteration_variables[node].append(
                    IterationVariable(
                        dimension=dim,
                        size=remaining_size,
                        relevant=True,
                        type=IterationVariableType.TEMPORAL,
                    )
                )
            else:
                if isinstance(node, TransferNode):
                    type = IterationVariableType.SPATIAL
                else:
                    type = IterationVariableType.TEMPORAL
                # Create a replacement temporal variable
                relevant = dim in workload.get_dims(node)
                iteration_variables[node].append(
                    IterationVariable(
                        dimension=dim,
                        size=unrolling,
                        relevant=relevant,
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

    unique_spatial_unrollings: set[tuple[LayerDim, int]] = set()
    for unrollings in spatial_unrollings.values():
        for unrolling in unrollings:
            # Keep the largest unrolling size for each dimension
            dim, size = unrolling
            existing = next((u for u in unique_spatial_unrollings if u[0] == dim), None)
            if existing is None:
                unique_spatial_unrollings.add((dim, size))
            elif size > existing[1]:
                unique_spatial_unrollings.discard(existing)
                unique_spatial_unrollings.add((dim, size))
    return spatial_unrollings, unique_spatial_unrollings


def _get_total_spatial_unrolling_for_dim(
    dim: LayerDim,
    spatial_unrollings: set[tuple[LayerDim, int]],
) -> int:
    total_unrolling = prod(su[1] for su in spatial_unrollings if su[0] == dim)
    return total_unrolling


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
