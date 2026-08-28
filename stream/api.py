import json
import logging as _logging
import os
import tempfile
from typing import Any

import yaml
from onnx import ModelProto
from zigzag.mapping.temporal_mapping import TemporalMappingType
from zigzag.utils import open_yaml, pickle_load

from stream.hardware.bundle import HardwareBundle
from stream.hardware.cost import HardwareBudget, assert_within_budget, evaluate_bundle_cost
from stream.instrumentation import build_instrumentation, fail_instrumentation, finish_instrumentation, instrument
from stream.ir.graph_view import WorkloadGraphView
from stream.opt.solver import ConstraintSelection, GurobiBackend, SolverBackend
from stream.stages.allocation.constraint_optimization_allocation import ConstraintOptimizationAllocationStage
from stream.stages.context import StageContext
from stream.stages.estimation.core_cost_estimation import CoreCostEstimationStage
from stream.stages.estimation.memory_accesses_estimation import MemoryAccessesEstimationStage
from stream.stages.generation.fusion_group_iteration import FusionGroupIterationStage
from stream.stages.generation.generic_mapping_generation import GenericMappingGenerationStage
from stream.stages.generation.kernel_state import KernelStateStage
from stream.stages.generation.mapping_generation import MappingGenerationStage
from stream.stages.generation.mapping_generation_multi import MappingGenerationMultiThreadedStage
from stream.stages.generation.normalization_expansion import ExpandNormalizationStage
from stream.stages.generation.tiling_generation import TilingGenerationStage
from stream.stages.parsing.accelerator_parser import AcceleratorParserStage
from stream.stages.parsing.mapping_parser import MappingParserStage
from stream.stages.parsing.onnx_model_parser import ONNXModelParserStage as StreamONNXModelParserStage
from stream.stages.stage import LeafStage, MainStage, StageCallable
from stream.workload.workload import Workload

_logging_level = _logging.INFO
_logging_format = "%(asctime)s - %(funcName)s +%(lineno)s - %(levelname)s - %(message)s"


def configure_logging(level: int = _logging_level, fmt: str = _logging_format) -> None:
    """Configure root logging. Called by CLI scripts; MCP server manages its own logging."""
    _logging.basicConfig(level=level, format=fmt)


def _sanity_check_inputs(hardware: str, workload: str, mapping: str, output_path: str):
    assert os.path.exists(hardware), f"Hardware file {hardware} does not exist"
    assert isinstance(workload, ModelProto) or os.path.exists(workload), f"Workload file {workload} does not exist"
    assert os.path.exists(mapping), f"Mapping file {mapping} does not exist"
    if not os.path.exists(output_path):
        os.makedirs(output_path)


def _sanity_check_gurobi_license():
    GurobiBackend.check_license()


def _as_bool(value: Any) -> bool:
    """Coerce a possibly-stringy flag to a real bool -- JSON callers pass "false"/"true" as strings, and
    a non-empty "false" is otherwise truthy."""
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def optimize_allocation_co_with_mapping(  # noqa: PLR0913, PLR0912
    hardware: str,
    workload: str,
    mapping: str,
    experiment_id: str,
    output_path: str,
    skip_if_exists: bool = False,
    temporal_mapping_type: str = "uneven",
    enable_codegen: bool = False,
    trace_size: int = 0,
    trace_max_tiles: int = 31,
    trace_tiles: tuple[tuple[int, int], ...] = (),
    trace_group: int | None = None,
    nb_cols_to_use: int = 4,
    npu: str = "npu2",
    backend: str = "ortools_gscip",
    constraint_selection: ConstraintSelection | None = None,
    kernels: dict[str, Any] | None = None,
    instrumentation: dict[str, Any] | None = None,
) -> StageContext:
    # Callers (e.g. the web runner) may pass JSON-sourced strings for the booleans; coerce them so a
    # literal "false" cannot read as True and silently pull in the optional AIE code-gen path (snaxc).
    enable_codegen = _as_bool(enable_codegen)
    skip_if_exists = _as_bool(skip_if_exists)
    _sanity_check_inputs(hardware, workload, mapping, output_path)
    _backend_enum = SolverBackend[backend.upper()]
    if _backend_enum in (SolverBackend.GUROBI, SolverBackend.ORTOOLS_GUROBI):
        _sanity_check_gurobi_license()

    # Create experiment_id path
    output_path = f"{output_path}/{experiment_id}"
    os.makedirs(output_path, exist_ok=True)

    # Get logger
    logger = _logging.getLogger(__name__)

    # Determine temporal mapping type for ZigZag
    if temporal_mapping_type == "uneven":
        temporal_mapping_type = TemporalMappingType.UNEVEN
    elif temporal_mapping_type == "even":
        temporal_mapping_type = TemporalMappingType.EVEN
    else:
        raise ValueError(f"Invalid temporal mapping type: {temporal_mapping_type}. Must be 'uneven' or 'even'.")

    # Load final resulting context if it exists and skip_if_exists is True
    ctx_path = f"{output_path}/ctx.pickle"
    if os.path.exists(ctx_path) and skip_if_exists:
        ctx = pickle_load(ctx_path)
        logger.info(f"Loaded context from {ctx_path}")
    else:
        stages: list[StageCallable] = [  # Initializes the MainStage as entry point
            AcceleratorParserStage,  # Parses the accelerator
            StreamONNXModelParserStage,  # Parses the ONNX Model into the workload
            MappingParserStage,
            KernelStateStage,  # the state a kernel carries, before the iteration space is read
            TilingGenerationStage,
            CoreCostEstimationStage,
            ConstraintOptimizationAllocationStage,
            MemoryAccessesEstimationStage,
        ]
        ctx = StageContext.from_kwargs(
            accelerator=hardware,  # required by AcceleratorParserStage
            workload_path=workload,  # required by ModelParserStage
            mapping_path=mapping,  # required by ModelParserStage
            loma_lpf_limit=6,  # required by LomaEngine
            output_path=output_path,
            temporal_mapping_type=temporal_mapping_type,  # required by CoreCostEstimationStage
            trace_size=trace_size,
            trace_max_tiles=trace_max_tiles,
            trace_tiles=trace_tiles,
            trace_group=trace_group,
            nb_cols_to_use=nb_cols_to_use,  # required by ConstraintOptimizationAllocationStage
            backend=_backend_enum.value,
            constraint_selection=constraint_selection,
            kernels=kernels,  # optional caller-supplied kernel factory overrides
        )
        # optionally add code generation stage
        if enable_codegen:
            from stream.stages.codegen.aie_code_generation import AIECodeGenerationStage  # noqa: PLC0415

            n_fused_groups = len(open_yaml(mapping)["fused_groups"]) if isinstance(mapping, str) else 1
            if n_fused_groups > 1:
                # Multi-group fixed mapping: split the workload at the mapping's
                # fused-group boundaries and run the allocation + codegen inner
                # pipeline once per group, writing each group's MLIR under
                # <output_path>/group_i/. Mirrors the generic multi-group pipeline
                # but driven by the hand-written (fixed) mapping.
                from stream.stages.generation.fixed_mapping_generation import (  # noqa: PLC0415
                    FixedMappingGenerationStage,
                )

                stages = [
                    AcceleratorParserStage,
                    StreamONNXModelParserStage,
                    FixedMappingGenerationStage,  # split workload + build per-group mappings (in-memory)
                    FusionGroupIterationStage,  # outer loop over groups; sets the per-group mapping
                    AIECodeGenerationStage,  # codegen each group (inner pipeline)
                    # No MappingParserStage: FixedMappingGenerationStage supplies the
                    # per-group Mapping objects in-memory via FusionGroupIterationStage.
                    KernelStateStage,  # the state a kernel carries, before the iteration space is read
                    TilingGenerationStage,
                    CoreCostEstimationStage,
                    ConstraintOptimizationAllocationStage,
                    MemoryAccessesEstimationStage,
                ]
            else:
                stages = [AIECodeGenerationStage] + stages
            ctx.set(
                npu=npu,  # required by AIECodeGenerationStage
            )

        observers = build_instrumentation("optimize_allocation_co_with_mapping", instrumentation)
        stages = instrument(stages, observers)

        mainstage = MainStage(stages, ctx)
        # Launch the MainStage
        try:
            answers = mainstage.run()
        except BaseException as exc:  # noqa: BLE001 -- record where the solve stopped, then re-raise unchanged
            fail_instrumentation(observers, str(exc) or exc.__class__.__name__)
            raise
        assert len(answers) == 1, "Expected a single result from the optimization."
        finish_instrumentation(observers)
        ctx = answers[0]
    return ctx


# Backward-compatible alias: old name -> new name
optimize_allocation_co = optimize_allocation_co_with_mapping


def _build_generic_co_stages(parse_stages: list[StageCallable]) -> list[StageCallable]:
    """The generic CO stage list."""
    return [
        AcceleratorParserStage,  # Parses the accelerator
        *parse_stages,
        ExpandNormalizationStage,  # expand softmax/norm into affine sub-ops (two reduction passes)
        GenericMappingGenerationStage,  # generates per-group YAMLs + sub_workloads
        FusionGroupIterationStage,  # outer loop over groups (reads sub_workloads from ctx)
        MappingParserStage,  # inner pipeline starts here
        KernelStateStage,  # the state a kernel carries, before the iteration space is read
        TilingGenerationStage,
        CoreCostEstimationStage,
        ConstraintOptimizationAllocationStage,
        MemoryAccessesEstimationStage,
    ]


def _run_generic_co(  # noqa: PLR0913
    hardware: str,
    experiment_id: str,
    output_path: str,
    *,
    workload_path: str | ModelProto | None = None,
    workload_obj: Workload | None = None,
    skip_if_exists: bool = False,
    temporal_mapping_type: str = "uneven",
    nb_cols_to_use: int = 4,
    backend: str = "ortools_gscip",
    constraint_selection: ConstraintSelection | None = None,
    intra_core_tiling: list[dict] | None = None,
    fusion_cut_points: list[str] | None = None,
    instrumentation: dict[str, Any] | None = None,
    hardware_budget: HardwareBudget | None = None,
) -> StageContext:
    """Shared generic CO pipeline. Feeds either an ONNX ``workload_path`` or a prebuilt in-memory
    ``workload_obj`` (the ONNX stage is skipped).

    ``instrumentation`` names out-of-tree observers to wrap the stage list with ({name: options});
    see :mod:`stream.instrumentation`. ``hardware_budget`` rejects an over-budget hardware variant
    before the solve.
    """
    assert os.path.exists(hardware), f"Hardware file {hardware} does not exist"
    if hardware_budget is not None:
        assert_within_budget(HardwareBundle.from_yaml(hardware), hardware_budget)
    assert (workload_path is None) != (workload_obj is None), "Provide exactly one of workload_path / workload_obj"
    if workload_path is not None:
        assert isinstance(workload_path, ModelProto) or os.path.exists(workload_path), (
            f"Workload file {workload_path} does not exist"
        )
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    _backend_enum = SolverBackend[backend.upper()]
    if _backend_enum in (SolverBackend.GUROBI, SolverBackend.ORTOOLS_GUROBI):
        _sanity_check_gurobi_license()

    # Create experiment_id path
    output_path = f"{output_path}/{experiment_id}"
    os.makedirs(output_path, exist_ok=True)

    # Get logger
    logger = _logging.getLogger(__name__)

    # Determine temporal mapping type for ZigZag
    if temporal_mapping_type == "uneven":
        temporal_mapping_type = TemporalMappingType.UNEVEN
    elif temporal_mapping_type == "even":
        temporal_mapping_type = TemporalMappingType.EVEN
    else:
        raise ValueError(f"Invalid temporal mapping type: {temporal_mapping_type}. Must be 'uneven' or 'even'.")

    # Load final resulting context if it exists and skip_if_exists is True
    ctx_path = f"{output_path}/ctx.pickle"
    if os.path.exists(ctx_path) and skip_if_exists:
        ctx = pickle_load(ctx_path)
        logger.info(f"Loaded context from {ctx_path}")
    else:
        # The ONNX parser stage is only needed when a file/proto workload is given.
        parse_stages: list[StageCallable] = [StreamONNXModelParserStage] if workload_path is not None else []
        stages = _build_generic_co_stages(parse_stages)
        observers = build_instrumentation("optimize_allocation_co_generic", instrumentation)
        stages = instrument(stages, observers)
        workload_kwargs = {"workload_path": workload_path} if workload_path is not None else {"workload": workload_obj}
        ctx = StageContext.from_kwargs(
            accelerator=hardware,  # required by AcceleratorParserStage
            **workload_kwargs,  # workload_path (ONNX) or workload (in-memory), required downstream
            loma_lpf_limit=6,  # required by LomaEngine
            output_path=output_path,
            temporal_mapping_type=temporal_mapping_type,  # required by CoreCostEstimationStage
            nb_cols_to_use=nb_cols_to_use,  # required by ConstraintOptimizationAllocationStage
            backend=_backend_enum.value,
            constraint_selection=constraint_selection,
            intra_core_tiling=intra_core_tiling,  # optional layer-fusion tiling for GenericMappingGenerationStage
            fusion_cut_points=fusion_cut_points,  # None -> derive from affine barriers
        )

        mainstage = MainStage(stages, ctx)
        try:
            answers = mainstage.run()
        except BaseException as exc:  # noqa: BLE001 -- record where the solve stopped, then re-raise unchanged
            fail_instrumentation(observers, str(exc) or exc.__class__.__name__)
            raise
        assert len(answers) == 1, "Expected a single result from the optimization."
        finish_instrumentation(observers)
        ctx = answers[0]
    return ctx


def optimize_allocation_co_generic(  # noqa: PLR0913
    hardware: str,
    workload: str,
    experiment_id: str,
    output_path: str,
    skip_if_exists: bool = False,
    temporal_mapping_type: str = "uneven",
    nb_cols_to_use: int = 4,
    backend: str = "ortools_gscip",
    constraint_selection: ConstraintSelection | None = None,
    intra_core_tiling: list[dict] | None = None,
    fusion_cut_points: list[str] | None = None,
    instrumentation: dict[str, Any] | None = None,
    hardware_budget: HardwareBudget | None = None,
) -> StageContext:
    """Run the CO pipeline with auto-generated mapping from workload+hardware.

    Unlike optimize_allocation_co, this does not require a hand-written mapping YAML.
    GenericMappingGenerationStage infers the mapping, then FusionGroupIterationStage
    runs the inner pipeline once per fusion group. Every Softmax/normalization is expanded into its
    affine sub-ops (max/exp/sum/div) so its two reduction passes are cost-modelled explicitly.

    Args:
        intra_core_tiling: Optional fused-group intra-core (layer-fusion) tiling, e.g.
            ``[{"dim": "Gemm_Left.D0", "tile": 16}, ...]``, so the solver costs one steady-state tile
            instead of the full layer. Entries are filtered per fusion group to the nodes that group
            contains; a group with no matching entry keeps the whole-layer tile. Supplying this
            disables automatic fusion tiling outright. When None, a group whose streamed intermediate
            does not fit on-chip is tiled automatically along its streaming axis; the rest keep the
            whole-layer tile.

    Returns the final StageContext with total_latency aggregated across all groups.
    """
    return _run_generic_co(
        hardware,
        experiment_id,
        output_path,
        workload_path=workload,
        skip_if_exists=skip_if_exists,
        temporal_mapping_type=temporal_mapping_type,
        nb_cols_to_use=nb_cols_to_use,
        backend=backend,
        constraint_selection=constraint_selection,
        intra_core_tiling=intra_core_tiling,
        fusion_cut_points=fusion_cut_points,
        instrumentation=instrumentation,
        hardware_budget=hardware_budget,
    )


def optimize_allocation_co_generic_workload(  # noqa: PLR0913
    hardware: str,
    workload: Workload,
    experiment_id: str,
    output_path: str,
    skip_if_exists: bool = False,
    temporal_mapping_type: str = "uneven",
    nb_cols_to_use: int = 4,
    backend: str = "ortools_gscip",
    constraint_selection: ConstraintSelection | None = None,
    intra_core_tiling: list[dict] | None = None,
    fusion_cut_points: list[str] | None = None,
    instrumentation: dict[str, Any] | None = None,
    hardware_budget: HardwareBudget | None = None,
) -> StageContext:
    """Run the generic CO pipeline on an in-memory ``Workload`` (e.g. a ``stream.workload.models``
    catalog block), skipping ONNX parsing. This is the end-to-end entry point for the affine-IR
    model blocks (MHA / GQA / linear-attention / Mamba) whose Scan/StateUpdate/Softmax node types
    have no ONNX round-trip. Only a data-dependent read cuts a fusion group; a reduction (including the
    softmax) is kept resident, never a barrier. Every softmax is decomposed into its affine sub-ops so
    its two reduction passes are cost-modelled explicitly.

    Returns the final StageContext with total_latency aggregated across all groups.
    """
    return _run_generic_co(
        hardware,
        experiment_id,
        output_path,
        workload_obj=workload,
        skip_if_exists=skip_if_exists,
        temporal_mapping_type=temporal_mapping_type,
        nb_cols_to_use=nb_cols_to_use,
        backend=backend,
        constraint_selection=constraint_selection,
        intra_core_tiling=intra_core_tiling,
        fusion_cut_points=fusion_cut_points,
        instrumentation=instrumentation,
        hardware_budget=hardware_budget,
    )


def optimize_mapping(  # noqa: PLR0913
    hardware: str,
    workload: str,
    experiment_id: str,
    output_path: str,
    max_nb_mappings: int = 20,
    skip_if_exists: bool = False,
    temporal_mapping_type: str = "uneven",
    enable_codegen: bool = False,
    trace_size: int = 0,
    trace_max_tiles: int = 31,
    trace_tiles: tuple[tuple[int, int], ...] = (),
    trace_group: int | None = None,
    nb_cols_to_use: int = 8,
    nb_rows_to_use: int = 4,
    seq_len_tile_size: int = 32,
    embedding_tile_size: int = 128,
    hidden_tile_size: int = 64,
    last_gemm_down: bool = False,
    npu: str = "npu2",
    nb_workers: int = 1,
    backend: str = "ortools_gscip",
    constraint_selection: ConstraintSelection | None = None,
    instrumentation: dict[str, Any] | None = None,
) -> StageContext:
    """Search generated mappings for the lowest-latency one and return the winning variant's context.

    ``instrumentation`` names out-of-tree observers to wrap the stage list with ({name: options});
    see :mod:`stream.instrumentation`. The inner pipeline runs once per variant, so an observer sees
    every variant it evaluates, not just the winner.
    """
    _backend_enum = SolverBackend[backend.upper()]
    if _backend_enum in (SolverBackend.GUROBI, SolverBackend.ORTOOLS_GUROBI):
        _sanity_check_gurobi_license()

    # Create experiment_id path
    output_path = f"{output_path}/{experiment_id}"
    os.makedirs(output_path, exist_ok=True)

    # Get logger
    logger = _logging.getLogger(__name__)

    # Determine temporal mapping type for ZigZag
    if temporal_mapping_type == "uneven":
        temporal_mapping_type = TemporalMappingType.UNEVEN
    elif temporal_mapping_type == "even":
        temporal_mapping_type = TemporalMappingType.EVEN
    else:
        raise ValueError(f"Invalid temporal mapping type: {temporal_mapping_type}. Must be 'uneven' or 'even'.")

    if nb_workers > 1:
        mapping_generation_stage = MappingGenerationMultiThreadedStage
    else:
        mapping_generation_stage = MappingGenerationStage

    # Load final resulting context if it exists and skip_if_exists is True
    ctx_path = f"{output_path}/ctx.pickle"
    if os.path.exists(ctx_path) and skip_if_exists:
        ctx = pickle_load(ctx_path)
        logger.info(f"Loaded context from {ctx_path}")
    else:
        stages: list[StageCallable] = [  # Initializes the MainStage as entry point
            AcceleratorParserStage,  # Parses the accelerator
            StreamONNXModelParserStage,  # Parses the ONNX Model into the workload
            mapping_generation_stage,
            MappingParserStage,
            KernelStateStage,  # the state a kernel carries, before the iteration space is read
            TilingGenerationStage,
            CoreCostEstimationStage,
            ConstraintOptimizationAllocationStage,
            MemoryAccessesEstimationStage,
        ]
        ctx = StageContext.from_kwargs(
            accelerator=hardware,  # required by AcceleratorParserStage
            workload_path=workload,  # required by ModelParserStage
            loma_lpf_limit=6,  # required by LomaEngine
            output_path=output_path,
            temporal_mapping_type=temporal_mapping_type,  # required by CoreCostEstimationStage
            trace_size=trace_size,
            trace_max_tiles=trace_max_tiles,
            trace_tiles=trace_tiles,
            trace_group=trace_group,
            nb_cols_to_use=nb_cols_to_use,  # required by ConstraintOptimizationAllocationStage
            nb_rows_to_use=nb_rows_to_use,  # used by MappingGenerator for shape-aware tiling
            seq_len_tile_size=seq_len_tile_size,
            embedding_tile_size=embedding_tile_size,
            hidden_tile_size=hidden_tile_size,
            last_gemm_down=last_gemm_down,
            max_nb_mappings=max_nb_mappings,
            backend=_backend_enum.value,
            constraint_selection=constraint_selection,
        )
        # optionally add code generation stage
        if enable_codegen:
            from stream.stages.codegen.aie_code_generation import AIECodeGenerationStage  # noqa: PLC0415

            stages = [AIECodeGenerationStage] + stages
            ctx.set(
                npu=npu,  # required by AIECodeGenerationStage
            )
        if nb_workers > 1:
            ctx.set(
                max_workers=nb_workers,
            )

        observers = build_instrumentation("optimize_mapping", instrumentation)
        stages = instrument(stages, observers)

        mainstage = MainStage(stages, ctx)
        # Launch the MainStage
        try:
            answers = mainstage.run()
        except BaseException as exc:  # noqa: BLE001 -- record where the search stopped, then re-raise unchanged
            fail_instrumentation(observers, str(exc) or exc.__class__.__name__)
            raise
        assert len(answers) == 1, "Expected a single result from the optimization."
        finish_instrumentation(observers)
        ctx = answers[0]
    return ctx


def parse_accelerator_ir(
    hardware: str,
    arch_ir_path: str,
) -> str:
    """Parse a hardware definition into an accelerator IR YAML file.

    Instantiates the :class:`~stream.hardware.architecture.accelerator.Accelerator`
    from the given hardware YAML, calls its :meth:`~stream.hardware.architecture.accelerator.Accelerator.get_ir`
    method, and writes the result to *arch_ir_path*.

    Args:
        hardware: Path to the hardware definition YAML file.
        arch_ir_path: Destination path for the accelerator IR YAML.

    Returns:
        The path to the saved IR file (*arch_ir_path*).
    """
    base_output_path = os.path.dirname(os.path.abspath(arch_ir_path))
    os.makedirs(base_output_path, exist_ok=True)
    ctx = StageContext.from_kwargs(
        accelerator=hardware,
        output_path=base_output_path,
    )
    stages: list[StageCallable] = [
        AcceleratorParserStage,
        LeafStage,
    ]
    mainstage = MainStage(stages, ctx)
    ctxs = mainstage.run()
    assert len(ctxs) == 1, "Expected a single result from the accelerator parsing"
    ctx: StageContext = ctxs[0]  # type: ignore[no-redef]
    accelerator = ctx.get("accelerator")
    arch_ir = accelerator.get_ir()
    with open(arch_ir_path, "w") as f:
        yaml.dump(arch_ir, f, sort_keys=False)
    return arch_ir_path


def hardware_cost_report(hardware: str, output_path: str | None = None) -> dict[str, Any]:
    """Price a hardware YAML: area in mm², peak access energy in pJ/cycle, plus the breakdown.

    Everything is derived from the declared capacities, widths, port counts and array dimensions,
    so a mutated variant reports a different cost. See :mod:`stream.hardware.cost` for the model and
    the technology assumptions. Optionally written to *output_path* as JSON.
    """
    report = evaluate_bundle_cost(HardwareBundle.from_yaml(hardware)).to_dict()
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
    return report


def parse_workload_ir(
    workload_path: str,
    arch_ir_path: str,
) -> str:
    """Parse a workload to the arch_ir.

    Args:
        workload_path: Path to the workload file (ONNX model).
        output_path: Path where output files should be saved.

    Returns:
        The parsed workload context.
    """
    base_output_path = os.path.dirname(os.path.abspath(arch_ir_path))
    os.makedirs(base_output_path, exist_ok=True)
    ctx = StageContext.from_kwargs(
        workload_path=workload_path,
        output_path=base_output_path,
    )
    stages: list[StageCallable] = [
        StreamONNXModelParserStage,
        LeafStage,
    ]
    mainstage = MainStage(
        stages,
        ctx,
    )
    ctxs = mainstage.run()
    assert len(ctxs) == 1, "Expected a single result from the workload parsing"
    ctx: StageContext = ctxs[0]
    workload = ctx.get("workload")
    arch_ir = workload.get_ir()
    with open(arch_ir_path, "w") as f:
        yaml.dump(arch_ir, f, sort_keys=False)
    return arch_ir_path


def workload_graph_view(workload_path: str, output_path: str | None = None, fusion_capacity: int | None = None) -> dict:
    """Parse a workload (ONNX) and return the unified :class:`~stream.ir.graph_view.WorkloadGraphView`
    as a JSON-able dict.

    The one smart graph view a consumer renders: a proper node/edge graph plus repeated-block collapse
    (draw one representative, mark the rest ``×N``), fusable regions (zoom), and the derived affine
    metadata per node. Works for any parsed workload; the same view serializes a tiled/steady-state
    graph identically.

    When ``fusion_capacity`` (a near-memory budget in elements) is given, the view also carries the
    auto-proposed fusion regions -- the greedy dataflow chains that fit that budget, each legal by
    construction. Left ``None`` (default) the ``proposed_regions`` list is empty, preserving the
    read-only, capacity-free view.

    The parser stage writes a debug ``workload_graph.png`` into ``output_path``; default it to a temp
    dir so this read-only view never litters the workload's own directory.
    """
    base_output_path = output_path or tempfile.mkdtemp(prefix="stream_graph_view_")
    os.makedirs(base_output_path, exist_ok=True)
    ctx = StageContext.from_kwargs(workload_path=workload_path, output_path=base_output_path)
    ctxs = MainStage([StreamONNXModelParserStage, LeafStage], ctx).run()
    assert len(ctxs) == 1, "Expected a single result from the workload parsing"
    workload = ctxs[0].get("workload")
    return WorkloadGraphView.from_workload(workload, fusion_capacity=fusion_capacity).model_dump()
