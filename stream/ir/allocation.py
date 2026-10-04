"""AllocationIR Pydantic model with per-persona view methods.

Wraps the output of SteadyStateSchedule.get_ir() in a typed, versioned Pydantic model.
Construction is always via the from_internal() classmethod.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from stream.plugins import loaded_overlays

if TYPE_CHECKING:
    from stream.allocation.schedule import SteadyStateSchedule


class LatencyInfo(BaseModel):
    """Latency metrics from a solved SteadyStateSchedule."""

    total: int = Field(description="Total schedule latency in cycles across all iterations")
    per_iteration: int = Field(description="Latency of a single steady-state iteration in cycles")
    overlap_between_iterations: int = Field(
        description="Overlap cycles between consecutive iterations (pipeline depth)"
    )
    fill: int = Field(
        default=0,
        description="Cycles each run waits before its first iteration for the tensors it holds in one buffer",
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


class SolveStatsIR(BaseModel):
    """What the MILP solver reported about the solve itself."""

    status: str = Field(description="Solve status, e.g. 'OPTIMAL', 'TIME_LIMIT'")
    solver: str = Field(description="Underlying solver, e.g. 'gurobi', 'gscip', 'highs'")
    mip_gap: float | None = Field(
        default=None, description="Relative optimality gap; None = the backend defines none, i.e. floor unknown"
    )
    objective: float | None = Field(default=None, description="Objective value of the best solution found")
    solve_time_s: float | None = Field(default=None, description="Wall-clock solve time in seconds")
    node_count: int | None = Field(default=None, description="Branch-and-bound nodes explored")
    iteration_count: int | None = Field(default=None, description="Simplex iterations")


class ConstraintFamilyIR(BaseModel):
    """One constraint family the allocation model was built from."""

    name: str = Field(description="The family's name, as SolveOptions.families selects it")
    options: dict[str, Any] = Field(default_factory=dict, description="The options the family was built with")


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


class SplitIR(BaseModel):
    """A loop dimension cut into `factor` parts -- a count, so tile extent is `dim_size // factor`."""

    dim: str = Field(description="The loop dimension being split")
    factor: int = Field(description="Number of parts the dimension is cut into")


class TileIR(BaseModel):
    """Walked in blocks of `tile` elements -- an extent (steps = `dim_size // tile`), not a count like SplitIR."""

    dim: str = Field(description="The loop dimension being tiled")
    tile: int = Field(description="Block extent in elements")


class FusionIR(BaseModel):
    """Stage-2 (Fuse) typed artifact: which layers share on-chip residency."""

    n_groups: int = Field(description="Number of fused groups the workload was partitioned into")
    groups: list[FusedGroupIR] = Field(description="The fused groups: their layers and intra-core tiling")


class TilingIR(BaseModel):
    """Stage-3 (Tile) typed artifact: the spatial (inter-core) and temporal (intra-core) tiling."""

    fusion_splits: list[SplitIR] = Field(
        description="Per-dimension fusion split counts before scheduling; global dim names ('z1')"
    )
    inter_core: dict[str, list[SplitIR]] = Field(
        description=(
            "Per-node spatial split across cores (first slot). The dim namespace follows whatever the "
            "mapping recorded: global ('z0') from the generic mapper, node-local ('D0') from a "
            "hand-written mapping. Do not join it to the other two by dim without checking."
        )
    )
    intra_core: dict[str, list[TileIR]] = Field(
        description="Per-fused-group temporal block extents within one core; global dim names ('z1')"
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
    node: str | None = Field(
        default=None,
        description=(
            "The computation node this loop belongs to, for loops below the tile. A fused group "
            "holds several nodes, each with its own intra-core nest; without the owner they "
            "concatenate into one flat list that reads as a nest that never existed. None for the "
            "loops above the tile, which the whole group shares."
        ),
    )


class SteadyStateIR(BaseModel):
    """Tiled/steady-state view of a fused group: operators + tensor sizes, loop nest, and tiled transfer graph."""

    operators: list[SteadyStateOperatorIR] = Field(description="Original operators + tensor sizes")
    loops: list[SteadyStateLoopIR] = Field(description="The steady-state iteration-space for-loop nest")
    tiled_graph: dict[str, Any] = Field(
        description="The tiled workload with transfers: {'nodes': [{name,kind,...}], 'edges': [{source,target}]}"
    )


class AllocationAlgorithmicView(BaseModel):
    """Algorithmic-persona projection of AllocationIR.

    Contains latency totals, solver backend, constraint families, and fusion splits.
    Suitable for algorithmic engineers reasoning about schedule quality and solver behaviour.
    """

    schema_version: Literal["2.0"] = "2.0"
    latency: LatencyInfo = Field(description="Latency metrics: total, per-iteration, and overlap cycles")
    backend: str = Field(description="Solver backend used: e.g. 'ORTOOLS_GSCIP' or 'ORTOOLS_HIGHS'")
    solve: SolveStatsIR | None = Field(
        default=None, description="Solver status and optimality gap: the noise floor for any latency comparison"
    )
    families: list[ConstraintFamilyIR] = Field(
        description="The constraint families the allocation model was built from"
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
    fallback: bool = Field(
        default=False,
        description=(
            "True when a matmul/conv node's ZigZag estimate fell back to the 1-MAC/cycle scalar cost "
            "(no CME): the spatial array was not modelled, so this node's latency is untrustworthy"
        ),
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
    total_mac_ops: float | None = Field(
        default=None, description="Useful MAC operations in the workload (matmul/conv family only)"
    )
    peak_macs_per_cycle: float | None = Field(
        default=None,
        description=(
            "Summed operational-array size over the on-chip cores whose operator_types admit the "
            "matmul/conv work total_mac_ops counts -- a vector core that may never run a GEMM is excluded"
        ),
    )
    mac_capable_cores: int | None = Field(
        default=None, description="How many on-chip cores contribute to peak_macs_per_cycle"
    )
    end_to_end_mac_utilization: float | None = Field(
        default=None,
        description=(
            "total_mac_ops / (peak_macs_per_cycle x total_latency): the fraction of the MAC roofline "
            "actually used, folding in spatial fill, stalls, idle MAC cores and transfer overhead. Both "
            "terms cover the matmul/conv family only, so 1.0 means the matrix engines are saturated -- "
            "not that the whole chip is; elementwise work appears in neither term"
        ),
    )
    degenerate: bool = Field(
        default=False, description="True iff a matmul/conv node fell back to the scalar cost (latency untrustworthy)"
    )
    degenerate_nodes: list[str] = Field(default_factory=list, description="Names of the fallback nodes")


class ResourceSlackIR(BaseModel):
    """One resource's steady-state boundary idle within a single iteration."""

    resource: str = Field(description="Resource key, e.g. a core or link identifier")
    kind: str = Field(description="'core' or 'link'")
    slack_cycles: int = Field(description="Reclaimable boundary idle in one iteration")


class OverlapIR(BaseModel):
    """Why the inter-iteration overlap is what it is (the solver's own slack breakdown)."""

    overlap_cycles: int | None = Field(default=None, description="Solved overlap between consecutive iterations")
    binding_resources: list[str] = Field(
        default_factory=list,
        description=(
            "Resources at the minimum slack, i.e. the resource-side cap on the overlap. The solved "
            "overlap can sit strictly below this cap, so compare it against per_resource_slack rather "
            "than assuming equality"
        ),
    )
    per_resource_slack: list[ResourceSlackIR] = Field(
        default_factory=list, description="Per-resource slack, ascending (the binding ones first)"
    )
    recurrence_bound_cycles: int = Field(
        default=0, description="Cycles a loop-carried state forbids overlapping (RecMII); 0 when feed-forward"
    )


class TensorReuseIR(BaseModel):
    """One tensor's on-chip residency as the solver chose it."""

    tensor: str
    size_bits: int | None = Field(default=None)
    reuse_factor: int | None = Field(
        default=None, description="Steady-state iterations it stays resident; 1 = re-fetched every iteration"
    )
    reuse_stop_level: int | None = Field(default=None, description="Loop level reuse stops at; -1 = none")
    on_chip_tiles: int | None = Field(default=None, description="Tile buffers that residency needs")
    loop_nest_out_to_in: list[str] = Field(default_factory=list, description="Its steady-state loop nest")


class ResidentTensorIR(BaseModel):
    """One tensor's contribution to a core's solved on-chip residency."""

    tensor: str
    bits: int


class MemoryOccupancyIR(BaseModel):
    """How full one core's memory actually is under the solved placement."""

    core_id: int
    core_name: str
    resident_bits: int = Field(description="Bits the solved placement keeps on this core in the steady state")
    capacity_bits: int = Field(description="The core's declared top-level memory capacity")
    utilization: float | None = Field(default=None, description="resident_bits / capacity_bits")
    tensors: list[ResidentTensorIR] = Field(
        default_factory=list, description="Largest resident tensors first: what sets the floor on a shrink"
    )


class ResourceActivityIR(BaseModel):
    """One resource under the memory_ports family, in ZigZag's port-activity terms."""

    kind: str = Field(default="memory_port", description="'memory_port', 'shared_bandwidth' or 'link'")
    resource: str = Field(description="'<memory>.<port>', 'measured', or '<sender>-><receiver>' for a link")
    core_ids: list[int] = Field(description="Cores sharing the resource")
    core_types: list[str] = Field(default_factory=list, description="Core type of each of core_ids")
    bw_bits_per_cycle: float = Field(description="Port bandwidth")
    bits_per_iteration: float = Field(description="Bits the solved schedule moves through the port per iteration")
    req_bw_aver: float | None = Field(description="Average bandwidth required over the initiation interval")
    real_cycle: float = Field(description="Cycles the port needs for one iteration's bits")
    allowed_cycle: float = Field(description="Initiation interval: the cycles one iteration gives the port")
    stall_or_slack: float = Field(description="real_cycle - allowed_cycle; negative is slack")
    utilization: float | None = Field(description="real_cycle / allowed_cycle; 1 when the port sets the interval")
    burst_utilization: float | None = Field(description="Highest per-slot bits / (bandwidth * slot latency)")
    burst_slot: int | None = Field(description="Slot where burst_utilization occurs")


class AllocationPerformanceView(BaseModel):
    """Performance-persona projection of AllocationIR.

    Exposes WHERE the schedule's latency goes, so a reader can tell whether a schedule is
    compute-bound, transfer/DMA-bound, or simply under-utilized -- instead of reading
    total latency alone. Look here first when a result is surprising (e.g. adding cores
    doesn't change latency): check `bottleneck` (compute vs transfer split),
    `aggregate.latency_weighted_mac_spatial_utilization` and `compute_cores_used` vs
    `compute_cores_available`, and per-node `mac_spatial_utilization` / `compute_efficiency`.
    """

    schema_version: Literal["1.1"] = "1.1"
    latency: LatencyInfo = Field(description="Latency metrics: total, per-iteration, and overlap cycles")
    bottleneck: BottleneckIR = Field(description="Per-iteration compute-bound vs transfer/DMA-bound cycle split")
    aggregate: PerformanceAggregateIR = Field(description="Accelerator-wide core usage and MAC utilization")
    nodes: dict[str, NodePerformanceIR] = Field(description="Per-node utilization and compute efficiency")
    overlap: OverlapIR | None = Field(
        default=None, description="What binds the inter-iteration overlap (the solver's own slack breakdown)"
    )
    tensor_reuse: list[TensorReuseIR] = Field(
        default_factory=list, description="Per-tensor on-chip residency the solver chose, largest first"
    )
    memory_occupancy: list[MemoryOccupancyIR] = Field(
        default_factory=list,
        description="Per-core solved residency vs declared capacity: the floor on any capacity reduction",
    )
    memory_ports: list[ResourceActivityIR] = Field(
        default_factory=list,
        description="Activity of each memory port, shared-bandwidth core and link, busiest first; empty unless "
        "memory_ports is on",
    )


class AllocationIR(BaseModel):
    """Typed Pydantic model wrapping SteadyStateSchedule.get_ir() output.

    schema_version '2.0': minor bumps for additive fields, major bumps (2.0) for
    removed/renamed fields. Construction is always via from_internal().
    """

    model_config = ConfigDict(
        json_schema_extra={
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": "stream/allocation_ir/v1",
        }
    )

    schema_version: Literal["2.0"] = "2.0"
    latency: LatencyInfo = Field(description="Latency metrics from the solved scheduler")
    backend: str = Field(description="Solver backend used: e.g. 'ORTOOLS_GSCIP' or 'ORTOOLS_HIGHS'")
    solve: SolveStatsIR | None = Field(
        default=None,
        description="Solver status and optimality gap; the gap is the noise floor for comparing two results",
    )
    cost_models: CostModelsIR | None = Field(
        default=None, description="Which cost models produced this result (transparency); always set by from_internal"
    )
    families: list[ConstraintFamilyIR] = Field(
        description="The constraint families the allocation model was built from"
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
    overlays: list[str] = Field(
        default_factory=list,
        description=(
            "Out-of-tree overlay distributions loaded for this run. Two results are only comparable "
            "when they were produced with the same set: an overlay can supply operators, hardware "
            "namespaces or constraints that change the answer."
        ),
    )

    @classmethod
    def from_internal(cls, schedule: SteadyStateSchedule) -> AllocationIR:
        """Construct AllocationIR from a solved SteadyStateSchedule.

        Calls schedule.get_ir() once, maps the resulting dict fields to Pydantic types,
        and validates on construction.
        """
        raw = schedule.get_ir()

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
                overlap=OverlapIR(**perf_raw["overlap"]) if perf_raw.get("overlap") else None,
                tensor_reuse=[TensorReuseIR(**d) for d in perf_raw.get("tensor_reuse") or []],
                memory_occupancy=[MemoryOccupancyIR(**d) for d in perf_raw.get("memory_occupancy") or []],
                memory_ports=[ResourceActivityIR(**d) for d in perf_raw.get("memory_ports") or []],
            )
            if perf_raw
            else None
        )

        solve_raw = raw.get("solve")
        solve = SolveStatsIR(**solve_raw) if solve_raw else None

        ss_raw = raw.get("steady_state")
        steady_state = SteadyStateIR(**ss_raw) if ss_raw else None

        # Stage-2 (Fuse) and stage-3 (Tile) typed artifacts, derived from the mapping dict.
        def _pairs(pairs: list) -> list[tuple[str, int]]:
            return [
                (str(p[0]), int(p[1])) for p in pairs or [] if isinstance(p, (list, tuple)) and len(p) >= _TILE_PAIR_LEN
            ]

        fusion = FusionIR(n_groups=len(fused_groups), groups=fused_groups)
        tiling = TilingIR(
            fusion_splits=[SplitIR(dim=str(d), factor=int(f)) for d, f in raw["fusion_splits"].items()],
            inter_core={
                name: [
                    SplitIR(dim=d, factor=f)
                    for d, f in _pairs(node["inter_core_tiling"][0] if node["inter_core_tiling"] else [])
                ]
                for name, node in mapping["nodes"].items()
            },
            intra_core={
                fg["name"]: [TileIR(dim=d, tile=t) for d, t in _pairs(fg["intra_core_tiling"])]
                for fg in mapping["fused_groups"]
            },
        )

        return cls(
            overlays=list(loaded_overlays()),
            latency=LatencyInfo(**raw["latency"]),
            backend=raw["backend"],
            solve=solve,
            cost_models=CostModelsIR.for_backend(raw["backend"]),
            families=[ConstraintFamilyIR(**family) for family in raw["families"]],
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
        """Return algorithmic-persona projection: latency, backend, constraint families, fusion splits."""
        return AllocationAlgorithmicView(
            latency=self.latency,
            backend=self.backend,
            solve=self.solve,
            families=self.families,
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
