"""Stream's entry points: price a mapping of a workload on an accelerator, choose among candidate
mappings, and generate the code of one.

A workload is anything a registered frontend loads, such as an ONNX path, or a
:class:`~stream.workload.workload.Workload`; an accelerator is a hardware YAML path or an
:class:`~stream.hardware.architecture.accelerator.Accelerator`. Everything that depends on the
hardware is found through plugins: the namespace constraints and core-cost backends of the solve, the
mapping generator that proposes a mapping when none is given, and the code generation backend.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from zigzag.mapping.temporal_mapping import TemporalMappingType

from stream.compiler.kernels.library import KernelLibrary
from stream.frontends import load_workload
from stream.hardware.architecture.accelerator import Accelerator
from stream.instrumentation import build_instrumentation, fail_instrumentation, finish_instrumentation, instrument
from stream.opt.allocation.constraint_optimization.context import build_transfer_context
from stream.opt.allocation.constraint_optimization.families import FamilySpec, drop_families, load_families
from stream.opt.solver import GurobiBackend, SolverBackend
from stream.profiling import span
from stream.stages.allocation.steady_state_allocation import DEFAULT_TIME_LIMIT_S, AllocationStage
from stream.stages.allocation.steady_state_lowering import SteadyStateLoweringStage
from stream.stages.codegen.backends import codegen_backend_for
from stream.stages.context import StageContext
from stream.stages.estimation.core_cost_estimation import CoreCostEstimationStage
from stream.stages.estimation.memory_accesses_estimation import MemoryAccessesEstimationStage
from stream.stages.generation.fixed_mapping_generation import FixedMappingGenerationStage
from stream.stages.generation.fusion_group_iteration import FusionGroupIterationStage
from stream.stages.generation.kernel_state import KernelStateStage
from stream.stages.generation.mapping_generators import mapping_generator_for
from stream.stages.generation.placement_generation import PlacementGenerationStage
from stream.stages.generation.tile_search import TileSearchStage
from stream.stages.generation.tiling_generation import TilingGenerationStage
from stream.stages.parsing.accelerator_parser import AcceleratorParserStage, parse_accelerator
from stream.stages.stage import MainStage, StageCallable
from stream.workload.workload import Workload

__all__ = ["MappingEstimate", "SolveOptions", "default_families", "evaluate_mapping", "generate_code", "select_mapping"]

logger = logging.getLogger(__name__)

# What the solve runs on each fused group's workload and mapping.
_ALLOCATION_STAGES: list[StageCallable] = [
    PlacementGenerationStage,
    KernelStateStage,
    TileSearchStage,
    TilingGenerationStage,
    CoreCostEstimationStage,
    SteadyStateLoweringStage,
    AllocationStage,
    MemoryAccessesEstimationStage,
]


@dataclass(frozen=True)
class SolveOptions:
    """How the allocation is solved.

    ``stage_options`` carries what a plugin's stages read from the context, such as the generic mapping
    generator's ``fusion_cut_points`` and ``intra_core_tiling``, or the AIE code generator's ``npu``,
    ``trace_size`` and ``trace_max_tiles``. ``instrumentation`` names observers to wrap the stages with, such as
    ``timing`` or ``allocation_artifacts`` (the traces, plots and reports of each solve). ``solver_log`` prints the
    solver's log, and ``time_limit_s`` bounds each allocation solve, after which its best incumbent is taken.
    ``families`` are the constraint families the allocation model is built from, by name or ``{name: options}``;
    None builds :func:`default_families`.
    """

    backend: str = "ortools_gscip"
    nb_cols_to_use: int = 4
    temporal_mapping_type: str = "uneven"
    kernel_library: KernelLibrary | str | Mapping[str, Any] | None = None
    tile_search: bool = False
    time_limit_s: float = DEFAULT_TIME_LIMIT_S
    solver_log: bool = False
    instrumentation: Mapping[str, Any] | None = None
    stage_options: Mapping[str, Any] = field(default_factory=dict)
    families: Sequence[FamilySpec] | None = None


def default_families(hardware: str | Accelerator, without: Iterable[str] = ()) -> tuple[FamilySpec, ...]:
    """The constraint families a solve on ``hardware`` builds by default: Stream's own and those of each namespace
    it has cores of. ``without`` leaves out the families it names and those that need what only they provide."""
    accelerator = hardware if isinstance(hardware, Accelerator) else parse_accelerator(hardware)
    return drop_families(build_transfer_context(accelerator).default_families, without)


@dataclass(frozen=True)
class MappingEstimate:
    """A mapping priced by the allocation solve of each of its fused groups; ``mapping`` is None for a
    mapping the accelerator's mapping generator proposed."""

    mapping: str | None
    group_cycles: tuple[float, ...]
    dispatch_cycles: float
    context: StageContext = field(repr=False, compare=False)

    @property
    def cycles(self) -> float:
        return sum(self.group_cycles) + self.dispatch_cycles


def evaluate_mapping(
    hardware: str | Accelerator,
    workload: Any,
    output_path: str,
    mapping: str | None = None,
    options: SolveOptions | None = None,
) -> MappingEstimate:
    """Solve the allocation of each fused group of ``mapping`` and price it, without generating code.

    Each group costs its scheduler's estimate, and a dispatch of several groups adds the reconfiguration
    the accelerator's namespace constraints charge. Without a mapping, the mapping generator that claims
    the accelerator proposes one.
    """
    return _solve(hardware, workload, output_path, mapping, options or SolveOptions(), codegen=False)


def select_mapping(
    hardware: str | Accelerator,
    workload: Any,
    output_path: str,
    candidates: Sequence[str],
    options: SolveOptions | None = None,
) -> MappingEstimate:
    """The candidate mapping estimated to take the fewest cycles; one that cannot be allocated loses.

    Each candidate is solved under ``output_path/candidate_<index>``.
    """
    estimates: list[MappingEstimate] = []
    failures: list[str] = []
    for index, candidate in enumerate(candidates):
        try:
            estimate = evaluate_mapping(hardware, workload, f"{output_path}/candidate_{index}", candidate, options)
        except (RuntimeError, ValueError) as error:
            logger.info("Mapping %s cannot be allocated: %s", candidate, error)
            failures.append(f"{candidate}: {error}")
            continue
        logger.info("Mapping %s: %.0f cycles", candidate, estimate.cycles)
        estimates.append(estimate)
    if not estimates:
        raise RuntimeError(f"No candidate mapping could be allocated: {failures}")
    return min(estimates, key=lambda estimate: estimate.cycles)


def generate_code(
    hardware: str | Accelerator,
    workload: Any,
    output_path: str,
    mapping: str | None = None,
    options: SolveOptions | None = None,
) -> MappingEstimate:
    """:func:`evaluate_mapping`, with the code generation backend that claims the accelerator writing
    each fused group's design under ``output_path/group_<index>``."""
    return _solve(hardware, workload, output_path, mapping, options or SolveOptions(), codegen=True)


def _solve(
    hardware: str | Accelerator,
    workload: Any,
    output_path: str,
    mapping: str | None,
    options: SolveOptions,
    codegen: bool,
) -> MappingEstimate:
    observers = build_instrumentation("generate_code" if codegen else "evaluate_mapping", options.instrumentation)
    try:
        with span("parse_accelerator"):
            accelerator = hardware if isinstance(hardware, Accelerator) else parse_accelerator(hardware)
        backend = SolverBackend[options.backend.upper()]
        if backend in (SolverBackend.GUROBI, SolverBackend.ORTOOLS_GUROBI):
            with span("solver_license"):
                GurobiBackend.check_license()
        with span("load_families"):
            transfer_context = build_transfer_context(accelerator)
            families = load_families(
                options.families if options.families is not None else transfer_context.default_families
            )
        proposal = [FixedMappingGenerationStage] if mapping is not None else mapping_generator_for(accelerator).stages()
        emission = [codegen_backend_for(accelerator).stage()] if codegen else []
        stages = [AcceleratorParserStage, *proposal, FusionGroupIterationStage, *emission, *_ALLOCATION_STAGES]
        with span("load_workload"):
            loaded = workload if isinstance(workload, Workload) else load_workload(workload)
        ctx = StageContext.from_kwargs(
            accelerator=accelerator,
            workload=loaded,
            mapping_path=mapping,
            output_path=output_path,
            loma_lpf_limit=6,
            temporal_mapping_type=TemporalMappingType[options.temporal_mapping_type.upper()],
            nb_cols_to_use=options.nb_cols_to_use,
            backend=backend.value,
            families=families,
            kernel_library=options.kernel_library,
            tile_search=options.tile_search,
            time_limit_s=options.time_limit_s,
            solver_log=options.solver_log,
            **options.stage_options,
        )
        (ctx,) = MainStage(instrument(stages, observers), ctx).run()
    except BaseException as exc:
        fail_instrumentation(observers, str(exc) or exc.__class__.__name__)
        raise
    finish_instrumentation(observers)
    groups = sorted(ctx.get("group_cycles"))
    columns = [len(ctx.get("group_columns")[i]) for i in groups]
    return MappingEstimate(
        mapping=mapping,
        group_cycles=tuple(ctx.get("group_cycles")[i] for i in groups),
        dispatch_cycles=transfer_context.dispatch_overhead_cycles(columns),
        context=ctx,
    )
