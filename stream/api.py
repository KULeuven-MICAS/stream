import logging as _logging
import os
from typing import Literal

import yaml
from onnx import ModelProto
from zigzag.mapping.temporal_mapping import TemporalMappingType
from zigzag.utils import pickle_load, pickle_save

from stream.cost_model.cost_model import StreamCostModelEvaluation
from stream.opt.solver import ConstraintSelection, GurobiBackend, SolverBackend
from stream.stages.allocation.constraint_optimization_allocation import ConstraintOptimizationAllocationStage
from stream.stages.allocation.genetic_algorithm_allocation import GeneticAlgorithmAllocationStage
from stream.stages.context import StageContext
from stream.stages.estimation.core_cost_estimation import CoreCostEstimationStage
from stream.stages.estimation.memory_accesses_estimation import MemoryAccessesEstimationStage
from stream.stages.generation.mapping_generation import MappingGenerationStage
from stream.stages.generation.mapping_generation_multi import MappingGenerationMultiThreadedStage
from stream.stages.generation.tiling_generation import TilingGenerationStage
from stream.stages.parsing.accelerator_parser import AcceleratorParserStage
from stream.stages.parsing.mapping_parser import MappingParserStage
from stream.stages.parsing.onnx_model_parser import ONNXModelParserStage as StreamONNXModelParserStage
from stream.stages.stage import LeafStage, MainStage, StageCallable

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


def optimize_allocation_ga(  # noqa: PLR0913
    hardware: str,
    workload: str,
    mapping: str,
    mode: Literal["lbl"] | Literal["fused"],
    layer_stacks: list[tuple[int, ...]],
    nb_ga_generations: int,
    nb_ga_individuals: int,
    experiment_id: str,
    output_path: str,
    skip_if_exists: bool = False,
    temporal_mapping_type: str = "uneven",
) -> StreamCostModelEvaluation:
    _sanity_check_inputs(hardware, workload, mapping, output_path)

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

    # Load SCME if it exists and skip_if_exists is True
    scme_path = f"{output_path}/scme.pickle"
    if os.path.exists(scme_path) and skip_if_exists:
        scme = pickle_load(scme_path)
        logger.info(f"Loaded SCME from {scme_path}")
    else:
        ctx = StageContext.from_kwargs(
            accelerator=hardware,  # required by AcceleratorParserStage
            workload_path=workload,  # required by ModelParserStage
            mapping_path=mapping,  # required by ModelParserStage
            loma_lpf_limit=6,  # required by LomaEngine
            nb_ga_generations=nb_ga_generations,  # number of genetic algorithm (ga) generations
            nb_ga_individuals=nb_ga_individuals,  # number of individuals in each ga generation
            mode=mode,
            layer_stacks=layer_stacks,
            output_path=output_path,
            temporal_mapping_type=temporal_mapping_type,  # required by CoreCostEstimationStage
            operands_to_prefetch=[],  # required by GeneticAlgorithmAllocationStage
        )
        mainstage = MainStage(
            [  # Initializes the MainStage as entry point
                AcceleratorParserStage,  # Parses the accelerator
                StreamONNXModelParserStage,  # Parses the ONNX Model into the workload
                MappingParserStage,
                # LayerStacksGenerationStage,
                # TilingGenerationStage,
                # CoreCostEstimationStage,
                # SetFixedAllocationPerformanceStage,
                # SchedulingOrderGenerationStage,
                GeneticAlgorithmAllocationStage,
            ],
            ctx,
        )
        # Launch the MainStage
        answers = mainstage.run()
        scme = answers[0][0]
        pickle_save(scme, scme_path)  # type: ignore
    return scme


def optimize_allocation_co(  # noqa: PLR0913
    hardware: str,
    workload: str,
    mapping: str,
    experiment_id: str,
    output_path: str,
    skip_if_exists: bool = False,
    temporal_mapping_type: str = "uneven",
    enable_codegen: bool = False,
    trace_size: int = 1048576,
    nb_cols_to_use: int = 4,
    npu: str = "npu2",
    backend: str = "ortools_gscip",
    constraint_selection: ConstraintSelection | None = None,
) -> StreamCostModelEvaluation:
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
            nb_cols_to_use=nb_cols_to_use,  # required by ConstraintOptimizationAllocationStage
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

        mainstage = MainStage(stages, ctx)
        # Launch the MainStage
        answers = mainstage.run()
        assert len(answers) == 1, "Expected a single result from the optimization."
        ctx = answers[0]
    return ctx


def optimize_mapping(  # noqa: PLR0913
    hardware: str,
    workload: str,
    experiment_id: str,
    output_path: str,
    max_nb_mappings: int = 20,
    skip_if_exists: bool = False,
    temporal_mapping_type: str = "uneven",
    enable_codegen: bool = False,
    trace_size: int = 1048576,
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
) -> StageContext:
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

        mainstage = MainStage(stages, ctx)
        # Launch the MainStage
        answers = mainstage.run()
        assert len(answers) == 1, "Expected a single result from the optimization."
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
