import logging as _logging
import os
from typing import Literal

import gurobipy as gp
from onnx import ModelProto
from zigzag.mapping.temporal_mapping import TemporalMappingType
from zigzag.utils import pickle_load, pickle_save

from stream.cost_model.cost_model import StreamCostModelEvaluation
from stream.stages.allocation.constraint_optimization_allocation import ConstraintOptimizationAllocationStage
from stream.stages.allocation.genetic_algorithm_allocation import GeneticAlgorithmAllocationStage
from stream.stages.estimation.zigzag_core_mapping_estimation import ZigZagCoreMappingEstimationStage
from stream.stages.generation.layer_stacks_generation import LayerStacksGenerationStage
from stream.stages.generation.scheduling_order_generation import SchedulingOrderGenerationStage
from stream.stages.generation.tiled_workload_generation import TiledWorkloadGenerationStage
from stream.stages.generation.tiling_generation import TilingGenerationStage
from stream.stages.parsing.accelerator_parser import AcceleratorParserStage
from stream.stages.parsing.onnx_model_parser import ONNXModelParserStage as StreamONNXModelParserStage
from stream.stages.set_fixed_allocation_performance import SetFixedAllocationPerformanceStage
from stream.stages.stage import MainStage

_logging_level = _logging.INFO
_logging_format = "%(asctime)s - %(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
_logging.basicConfig(level=_logging_level, format=_logging_format)


def _sanity_check_inputs(
    hardware: str, workload: str, mapping: str, mode: Literal["lbl"] | Literal["fused"], output_path: str
):
    assert os.path.exists(hardware), f"Hardware file {hardware} does not exist"
    assert isinstance(workload, ModelProto) or os.path.exists(workload), f"Workload file {workload} does not exist"
    assert os.path.exists(mapping), f"Mapping file {mapping} does not exist"
    assert mode in ["lbl", "fused"], "Mode must be either 'lbl' or 'fused'"
    if not os.path.exists(output_path):
        os.makedirs(output_path)


def _sanity_check_gurobi_license():
    try:
        # Try to create a simple optimization model
        model = gp.Model()
        model.setParam("OutputFlag", 0)
        # Check if the model was successfully created (license check)
        model.optimize()
        # If model.optimize() runs without a license issue, return
        return
    except gp.GurobiError as exc:
        # Catch any Gurobi errors, especially licensing errors
        if exc.errno == gp.GRB.Error.NO_LICENSE:
            error_message = "No valid Gurobi license found. Get an academic WLS license at https://www.gurobi.com/academia/academic-program-and-licenses/"
        else:
            error_message = f"An unexpected Gurobi error occurred: {exc.message}"
        raise ValueError(error_message) from exc


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
    _sanity_check_inputs(hardware, workload, mapping, mode, output_path)

    # Create experiment_id path
    os.makedirs(f"{output_path}/{experiment_id}", exist_ok=True)

    # Output paths
    tiled_workload_path = f"{output_path}/{experiment_id}/tiled_workload.pickle"
    cost_lut_path = f"{output_path}/{experiment_id}/cost_lut.pickle"
    scme_path = f"{output_path}/{experiment_id}/scme.pickle"

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
    if os.path.exists(scme_path) and skip_if_exists:
        scme = pickle_load(scme_path)
        logger.info(f"Loaded SCME from {scme_path}")
    else:
        mainstage = MainStage(
            [  # Initializes the MainStage as entry point
                AcceleratorParserStage,  # Parses the accelerator
                StreamONNXModelParserStage,  # Parses the ONNX Model into the workload
                LayerStacksGenerationStage,
                TilingGenerationStage,
                TiledWorkloadGenerationStage,
                ZigZagCoreMappingEstimationStage,
                SetFixedAllocationPerformanceStage,
                SchedulingOrderGenerationStage,
                GeneticAlgorithmAllocationStage,
            ],
            accelerator=hardware,  # required by AcceleratorParserStage
            workload_path=workload,  # required by ModelParserStage
            mapping_path=mapping,  # required by ModelParserStage
            loma_lpf_limit=6,  # required by LomaEngine
            nb_ga_generations=nb_ga_generations,  # number of genetic algorithm (ga) generations
            nb_ga_individuals=nb_ga_individuals,  # number of individuals in each ga generation
            mode=mode,
            layer_stacks=layer_stacks,
            tiled_workload_path=tiled_workload_path,
            cost_lut_path=cost_lut_path,
            temporal_mapping_type=temporal_mapping_type,  # required by ZigZagCoreMappingEstimationStage
            operands_to_prefetch=[],  # required by GeneticAlgorithmAllocationStage
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
    mode: Literal["lbl"] | Literal["fused"],
    layer_stacks: list[tuple[int, ...]],
    experiment_id: str,
    output_path: str,
    skip_if_exists: bool = False,
    temporal_mapping_type: str = "uneven",
) -> StreamCostModelEvaluation:
    _sanity_check_inputs(hardware, workload, mapping, mode, output_path)
    _sanity_check_gurobi_license()

    # Create experiment_id path
    os.makedirs(f"{output_path}/{experiment_id}", exist_ok=True)

    # Output paths
    tiled_workload_path = f"{output_path}/{experiment_id}/tiled_workload.pickle"
    cost_lut_path = f"{output_path}/{experiment_id}/cost_lut.pickle"
    allocations_path = f"{output_path}/{experiment_id}/waco/"
    tiled_workload_post_co_path = f"{output_path}/{experiment_id}/tiled_workload_post_co.pickle"
    cost_lut_post_co_path = f"outputs/{experiment_id}/cost_lut_post_co.pickle"
    scme_path = f"{output_path}/{experiment_id}/scme.pickle"

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
    if os.path.exists(scme_path) and skip_if_exists:
        scme = pickle_load(scme_path)
        logger.info(f"Loaded SCME from {scme_path}")
    else:
        mainstage = MainStage(
            [  # Initializes the MainStage as entry point
                AcceleratorParserStage,  # Parses the accelerator
                StreamONNXModelParserStage,  # Parses the ONNX Model into the workload
                LayerStacksGenerationStage,
                TilingGenerationStage,
                TiledWorkloadGenerationStage,
                ZigZagCoreMappingEstimationStage,
                ConstraintOptimizationAllocationStage,
            ],
            accelerator=hardware,  # required by AcceleratorParserStage
            workload_path=workload,  # required by ModelParserStage
            mapping_path=mapping,  # required by ModelParserStage
            loma_lpf_limit=6,  # required by LomaEngine
            mode=mode,
            layer_stacks=layer_stacks,
            tiled_workload_path=tiled_workload_path,
            cost_lut_path=cost_lut_path,
            allocations_path=allocations_path,
            tiled_workload_post_co_path=tiled_workload_post_co_path,
            cost_lut_post_co_path=cost_lut_post_co_path,
            temporal_mapping_type=temporal_mapping_type,  # required by ZigZagCoreMappingEstimationStage
            operands_to_prefetch=[],  # required by ConstraintOptimizationAllocationStage
        )
        # Launch the MainStage
        answers = mainstage.run()
        scme = answers[0][0]
        pickle_save(scme, scme_path)  # type: ignore
    return scme
