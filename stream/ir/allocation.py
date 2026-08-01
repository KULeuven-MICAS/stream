"""AllocationIR Pydantic model with per-persona view methods.

Wraps the output of SteadyStateScheduler.get_ir() in a typed, versioned Pydantic model.
Construction is always via the from_internal() classmethod.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from stream.cost_model.steady_state_scheduler import SteadyStateScheduler


class LatencyInfo(BaseModel):
    """Latency metrics from a solved SteadyStateScheduler."""

    total: int = Field(description="Total schedule latency in cycles across all iterations")
    per_iteration: int = Field(description="Latency of a single steady-state iteration in cycles")
    overlap_between_iterations: int = Field(
        description="Overlap cycles between consecutive iterations (pipeline depth)"
    )


class CostModelsIR(BaseModel):
    """Which cost models produced this result -- surfaced so the end user knows exactly what was
    modelled, not just the final number."""

    intra_core: str = Field(description="Per-core compute/energy cost model (the intra-core estimator)")
    scheduler: str = Field(description="Inter-core latency/schedule model")
    solver: str = Field(description="MILP solver backend used for tensor/transfer allocation")

    @classmethod
    def for_backend(cls, backend: str) -> CostModelsIR:
        return cls(
            intra_core="ZigZag analytical (per-node latency & energy, MAC-array spatial utilization)",
            scheduler="SteadyStateScheduler (steady-state pipeline latency, compute vs transfer bottleneck)",
            solver=backend,
        )


class ConstraintSelectionIR(BaseModel):
    """IR representation of the ConstraintSelection configuration used during the solve."""

    memory_capacity: bool = Field(description="Whether memory capacity constraints were active during solve")
    object_fifo_depth: bool = Field(description="Whether object FIFO depth constraints were active during solve")
    buffer_descriptors: bool = Field(description="Whether buffer descriptor constraints were active during solve")
    dma_channels: bool = Field(description="Whether DMA channel constraints were active during solve")


class NodeAllocationIR(BaseModel):
    """IR representation of the allocation result for a single workload node."""

    resource_allocation: list[list[dict[str, Any]]] = Field(
        description="Per-slot list of resource dicts: {'type': 'core', 'id': N} or {'type': 'path', ...}"
    )
    inter_core_tiling: list[list[list[Any]]] = Field(
        description="Per-slot tiling as [[dim_str, factor], ...] specifying how the node is split across cores"
    )
    memory_allocation: list[list[int]] = Field(
        description="Per-slot list of core IDs indicating where tensors are placed in memory"
    )


class FusedGroupIR(BaseModel):
    """IR representation of a fused group of workload layers."""

    name: str = Field(description="Fused group identifier")
    layers: list[str] = Field(description="Names of the workload layers fused together in this group")
    intra_core_tiling: list[list[Any]] = Field(
        description="Tiling factors within a single core as [[dim_str, factor], ...]"
    )


# A tiling pair is [dim, factor]; anything shorter is not a decision we can type.
_TILE_PAIR_LEN = 2


class TileFactorIR(BaseModel):
    """One tiling decision: a loop dimension split by an integer factor.

    `dim` is the global loop dimension name (e.g. 'z1'); `axis` is the semantic
    tensor axis it indexes (e.g. 'seq', 'embedding') when the affine IR can
    resolve it, else None. Typing tiling as `{dim, factor, axis}` replaces the
    stringly-typed `[dim_str, factor]` pairs that stages 2/3 were read out of.
    """

    dim: str = Field(description="The loop dimension being tiled")
    factor: int = Field(description="The integer split/tile factor applied to this dimension")
    axis: str | None = Field(
        default=None, description="Semantic tensor axis this dim indexes (when resolvable), else None"
    )


class FusionIR(BaseModel):
    """Stage-2 (Fuse) typed artifact: which layers share on-chip residency.

    A dedicated typed sub-object for the fusion decision, rather than reading it
    out of `fused_groups` strings.
    """

    n_groups: int = Field(description="Number of fused groups the workload was partitioned into")
    groups: list[FusedGroupIR] = Field(description="The fused groups: their layers and intra-core tiling")


class TilingIR(BaseModel):
    """Stage-3 (Tile) typed artifact: the spatial (inter-core) and temporal (intra-core) tiling.

    A dedicated typed sub-object so the Tile decision is inspectable directly,
    rather than reconstructed from `fusion_splits`/`inter_core_tiling` strings.
    """

    fusion_splits: list[TileFactorIR] = Field(
        description="Per-dimension fusion split factors applied before scheduling"
    )
    inter_core: dict[str, list[TileFactorIR]] = Field(
        description="Per-node spatial split across cores (the first slot's inter-core tiling)"
    )
    intra_core: dict[str, list[TileFactorIR]] = Field(
        description="Per-fused-group temporal tile factors within a single core"
    )


class SteadyStateOperatorIR(BaseModel):
    """One original (un-tiled) operator of a fused group and the sizes of its tensors."""

    name: str = Field(description="Operator name")
    op: str = Field(description="Operator type, e.g. 'MatMul', 'SelectiveScan', 'Softmax'")
    tensors: list[dict[str, Any]] = Field(description="Its operand tensors as [{'name', 'shape': [..]}, ...]")


class SteadyStateLoopIR(BaseModel):
    """One loop of the steady-state iteration space: a for-loop the fused schedule iterates."""

    dim: str = Field(description="The tiled loop dimension")
    size: int = Field(description="Trip count within a single steady-state slice")
    type: str = Field(description="Loop kind: 'temporal' (a for-loop), 'spatial' (unrolled across cores), 'kernel'")


class SteadyStateIR(BaseModel):
    """The tiled / steady-state view of a fused group: the original operators with their tensor sizes, the
    for-loop nest over the steady-state iteration space, and the tiled workload graph with the transfer
    nodes (the tensor copies kept on-chip between cores). Best-effort inspection view; None if unavailable."""

    operators: list[SteadyStateOperatorIR] = Field(description="Original operators + tensor sizes")
    loops: list[SteadyStateLoopIR] = Field(description="The steady-state iteration-space for-loop nest")
    tiled_graph: dict[str, Any] = Field(
        description="The tiled workload with transfers: {'nodes': [{name,kind,...}], 'edges': [{source,target}]}"
    )


class AllocationAlgorithmicView(BaseModel):
    """Algorithmic-persona projection of AllocationIR.

    Contains latency totals, solver backend, constraint configuration, and fusion splits.
    Suitable for algorithmic engineers reasoning about schedule quality and solver behaviour.
    """

    schema_version: Literal["1.0"] = "1.0"
    latency: LatencyInfo = Field(description="Latency metrics: total, per-iteration, and overlap cycles")
    backend: str = Field(description="Solver backend used: e.g. 'ORTOOLS_GSCIP' or 'ORTOOLS_HIGHS'")
    constraint_selection: ConstraintSelectionIR | None = Field(
        description="Constraint groups active during solve, or None if no selection was specified"
    )
    fusion_splits: dict[str, int] = Field(description="Fusion split factors per dimension applied before scheduling")


class AllocationHardwareView(BaseModel):
    """Hardware-persona projection of AllocationIR.

    Contains per-node resource and memory allocation. Suitable for hardware engineers
    reasoning about physical resource usage and memory placement per node.
    """

    schema_version: Literal["1.0"] = "1.0"
    mapping_nodes: dict[str, NodeAllocationIR] = Field(
        description="Per-node resource and memory allocation: use resource_allocation and memory_allocation fields"
    )


class AllocationCompilerView(BaseModel):
    """Compiler-persona projection of AllocationIR.

    Contains node-to-core mapping (inter_core_tiling), fused groups, and runtime args.
    Suitable for compiler engineers performing code generation and transfer routing.
    """

    schema_version: Literal["1.0"] = "1.0"
    mapping_nodes: dict[str, NodeAllocationIR] = Field(
        description="Per-node tiling and core mapping: use inter_core_tiling and resource_allocation fields"
    )
    fused_groups: list[FusedGroupIR] = Field(
        description="Groups of layers fused together with their intra-core tiling factors"
    )
    runtime_args: dict[str, str] = Field(description="Runtime arguments for code generation (e.g. buffer depths)")


class NodePerformanceIR(BaseModel):
    """Per-node utilization/efficiency summary for the performance view."""

    kind: str = Field(description="Node kind, e.g. 'compute'")
    n_cores: int = Field(description="Number of cores the node is inter-core-tiled across")
    latency_cycles: int = Field(description="The node's latency contribution to one steady-state iteration")
    ideal_compute_cycles: float | None = Field(
        default=None, description="Cycles at perfect MAC spatial utilization (the compute-ideal floor)"
    )
    mac_spatial_utilization: float | None = Field(
        default=None, description="Fraction of the core's MAC array used spatially (1.0 = full PE array)"
    )
    compute_efficiency: float | None = Field(
        default=None, description="ideal_compute_cycles / latency_cycles; how close to the compute-ideal this node runs"
    )


class BottleneckIR(BaseModel):
    """Per-iteration latency split by the resource class that sets each slot's latency."""

    compute_bound_cycles: int = Field(description="Per-iteration cycles in slots whose latency is set by compute")
    transfer_bound_cycles: int = Field(
        description="Per-iteration cycles in slots whose latency is set by data transfer/DMA"
    )
    compute_bound_pct: float | None = Field(
        default=None, description="Percent of per-iteration latency that is compute-bound"
    )
    transfer_bound_pct: float | None = Field(
        default=None, description="Percent of per-iteration latency that is transfer/DMA-bound"
    )


class PerformanceAggregateIR(BaseModel):
    """Accelerator-wide utilization aggregates."""

    compute_cores_available: int = Field(description="Non-offchip cores in the accelerator")
    compute_cores_used: int = Field(description="Distinct cores any computation node is mapped to")
    latency_weighted_mac_spatial_utilization: float | None = Field(
        default=None,
        description="Latency-weighted mean MAC spatial utilization across compute nodes (1.0 = full PE arrays)",
    )
    min_mac_spatial_utilization: float | None = Field(
        default=None, description="Worst per-node MAC spatial utilization"
    )


class AllocationPerformanceView(BaseModel):
    """Performance-persona projection of AllocationIR.

    Exposes WHERE the schedule's latency goes, so a reader can tell whether a schedule is
    compute-bound, transfer/DMA-bound, or simply under-utilized -- instead of reading
    total latency alone. Look here first when a result is surprising (e.g. adding cores
    doesn't change latency): check `bottleneck` (compute vs transfer split),
    `aggregate.latency_weighted_mac_spatial_utilization` and `compute_cores_used` vs
    `compute_cores_available`, and per-node `mac_spatial_utilization` / `compute_efficiency`.
    """

    schema_version: Literal["1.0"] = "1.0"
    latency: LatencyInfo = Field(description="Latency metrics: total, per-iteration, and overlap cycles")
    bottleneck: BottleneckIR = Field(description="Per-iteration compute-bound vs transfer/DMA-bound cycle split")
    aggregate: PerformanceAggregateIR = Field(description="Accelerator-wide core usage and MAC utilization")
    nodes: dict[str, NodePerformanceIR] = Field(description="Per-node utilization and compute efficiency")


class AllocationIR(BaseModel):
    """Typed Pydantic model wrapping SteadyStateScheduler.get_ir() output.

    schema_version '1.0': minor bumps (1.1) for additive fields, major bumps (2.0) for
    removed/renamed fields. Construction is always via from_internal().

    Note: from_internal() raises ValueError if called on a pre-solve scheduler
    (latency_total == -1 sentinel).
    """

    model_config = ConfigDict(
        json_schema_extra={
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": "stream/allocation_ir/v1",
        }
    )

    # 1.1 (additive): typed `fusion` (stage 2) and `tiling` (stage 3) sub-objects.
    schema_version: Literal["1.1"] = "1.1"
    latency: LatencyInfo = Field(description="Latency metrics from the solved scheduler")
    backend: str = Field(description="Solver backend used: e.g. 'ORTOOLS_GSCIP' or 'ORTOOLS_HIGHS'")
    cost_models: CostModelsIR | None = Field(
        default=None, description="Which cost models produced this result (transparency); always set by from_internal"
    )
    constraint_selection: ConstraintSelectionIR | None = Field(
        description="Constraint groups active during solve, or None if no selection was specified"
    )
    fusion_splits: dict[str, int] = Field(description="Fusion split factors per dimension applied before scheduling")
    mapping_nodes: dict[str, NodeAllocationIR] = Field(
        description="Per-node allocation result: resource, tiling, and memory allocation"
    )
    fused_groups: list[FusedGroupIR] = Field(description="Groups of fused layers with their intra-core tiling factors")
    runtime_args: dict[str, str] = Field(description="Runtime arguments for code generation (e.g. buffer depths)")
    performance: AllocationPerformanceView | None = Field(
        default=None,
        description="Read-only utilization/bottleneck summary; None if stats were unavailable for this solve",
    )
    steady_state: SteadyStateIR | None = Field(
        default=None,
        description="Tiled/steady-state inspection view (operators+tensor sizes, loop nest, transfer graph)",
    )
    fusion: FusionIR | None = Field(
        default=None,
        description="Stage-2 (Fuse) typed artifact: which layers share on-chip residency",
    )
    tiling: TilingIR | None = Field(
        default=None,
        description="Stage-3 (Tile) typed artifact: spatial (inter-core) + temporal (intra-core) tiling",
    )

    @classmethod
    def from_internal(cls, scheduler: SteadyStateScheduler) -> AllocationIR:
        """Construct AllocationIR from a post-solve SteadyStateScheduler.

        Calls scheduler.get_ir() once, maps the resulting dict fields to Pydantic types,
        and validates on construction. Raises ValueError if the scheduler has not been solved
        (latency_total == -1 sentinel from SteadyStateScheduler.__init__).
        """

        if scheduler.latency_total == -1:
            raise ValueError("Cannot build AllocationIR from unsolved SteadyStateScheduler")

        raw = scheduler.get_ir()
        cs_raw = raw.get("constraint_selection")
        constraint_selection = ConstraintSelectionIR(**cs_raw) if cs_raw else None

        mapping = raw["mapping"]
        mapping_nodes = {
            name: NodeAllocationIR(
                resource_allocation=node["resource_allocation"],
                inter_core_tiling=node["inter_core_tiling"],
                memory_allocation=node["memory_allocation"],
            )
            for name, node in mapping["nodes"].items()
        }
        fused_groups = [
            FusedGroupIR(
                name=fg["name"],
                layers=fg["layers"],
                intra_core_tiling=fg["intra_core_tiling"],
            )
            for fg in mapping["fused_groups"]
        ]

        perf_raw = raw.get("performance")
        performance = (
            AllocationPerformanceView(
                latency=LatencyInfo(**raw["latency"]),
                bottleneck=BottleneckIR(**perf_raw["bottleneck"]),
                aggregate=PerformanceAggregateIR(**perf_raw["aggregate"]),
                nodes={name: NodePerformanceIR(**d) for name, d in perf_raw["per_node"].items()},
            )
            if perf_raw
            else None
        )

        ss_raw = raw.get("steady_state")
        steady_state = SteadyStateIR(**ss_raw) if ss_raw else None

        # Stage-2 (Fuse) and stage-3 (Tile) typed artifacts, derived from the
        # same mapping dict — so the fuse/tile decisions are inspectable as
        # typed objects rather than reconstructed from strings downstream.
        def _factors(pairs: list) -> list[TileFactorIR]:
            out: list[TileFactorIR] = []
            for pair in pairs or []:
                if isinstance(pair, (list, tuple)) and len(pair) >= _TILE_PAIR_LEN:
                    out.append(TileFactorIR(dim=str(pair[0]), factor=int(pair[1])))
            return out

        fusion = FusionIR(n_groups=len(fused_groups), groups=fused_groups)
        tiling = TilingIR(
            fusion_splits=[TileFactorIR(dim=str(d), factor=int(f)) for d, f in raw["fusion_splits"].items()],
            inter_core={
                name: _factors(node["inter_core_tiling"][0] if node["inter_core_tiling"] else [])
                for name, node in mapping["nodes"].items()
            },
            intra_core={fg["name"]: _factors(fg["intra_core_tiling"]) for fg in mapping["fused_groups"]},
        )

        return cls(
            latency=LatencyInfo(**raw["latency"]),
            backend=raw["backend"],
            cost_models=CostModelsIR.for_backend(raw["backend"]),
            constraint_selection=constraint_selection,
            fusion_splits=raw["fusion_splits"],
            mapping_nodes=mapping_nodes,
            fused_groups=fused_groups,
            runtime_args={k: str(v) for k, v in mapping["runtime_args"].items()},
            performance=performance,
            steady_state=steady_state,
            fusion=fusion,
            tiling=tiling,
        )

    def algorithmic_view(self) -> AllocationAlgorithmicView:
        """Return algorithmic-persona projection: latency, backend, constraint selection, fusion splits."""
        return AllocationAlgorithmicView(
            latency=self.latency,
            backend=self.backend,
            constraint_selection=self.constraint_selection,
            fusion_splits=self.fusion_splits,
        )

    def hardware_view(self) -> AllocationHardwareView:
        """Return hardware-persona projection: per-node resource and memory allocation."""
        return AllocationHardwareView(
            mapping_nodes=self.mapping_nodes,
        )

    def compiler_view(self) -> AllocationCompilerView:
        """Return compiler-persona projection: node-to-core tiling, fused groups, runtime args."""
        return AllocationCompilerView(
            mapping_nodes=self.mapping_nodes,
            fused_groups=self.fused_groups,
            runtime_args=self.runtime_args,
        )

    def performance_view(self) -> AllocationPerformanceView | None:
        """Return performance-persona projection: bottleneck split + per-node/aggregate utilization.

        Returns None if performance stats were not captured for this solve.
        """
        return self.performance
