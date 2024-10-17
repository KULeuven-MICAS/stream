import logging as _logging
import os
from typing import Literal

import gurobipy as gp
from zigzag.utils import pickle_load, pickle_save

from stream.cost_model.cost_model import StreamCostModelEvaluation
from stream.stages.allocation.constraint_optimization_allocation import ConstraintOptimizationAllocationStage
from stream.stages.allocation.genetic_algorithm_allocation import GeneticAlgorithmAllocationStage
from stream.stages.estimation.zigzag_core_mapping_estimation import ZigZagCoreMappingEstimationStage
from stream.stages.generation.layer_stacks_generation import LayerStacksGenerationStage
from stream.stages.generation.scheduling_order_generation import SchedulingOrderGenerationStage
from stream.stages.generation.tiled_workload_generation import (
    TiledWorkloadGenerationStage,
)
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
    assert os.path.exists(workload), f"Workload file {workload} does not exist"
    assert os.path.exists(mapping), f"Mapping file {mapping} does not exist"
    assert mode in ["lbl", "fused"], "Mode must be either 'lbl' or 'fused'"
    if not os.path.exists(output_path):
        os.makedirs(output_path)


def _sanity_check_gurobi_license():
    try:
        # Try to create a simple optimization model
        model = gp.Model()
        # Check if the model was successfully created (license check)
        model.optimize()
        # If model.optimize() runs without a license issue, return
        return
    except gp.GurobiError as e:
        # Catch any Gurobi errors, especially licensing errors
        if e.errno == gp.GRB.Error.NO_LICENSE:
            error_message = "No valid Gurobi license found. Get an academic WLS license at https://www.gurobi.com/academia/academic-program-and-licenses/"
        else:
            error_message = f"An unexpected Gurobi error occurred: {e.message}"
        raise ValueError(error_message)


def optimize_allocation_ga(
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
) -> StreamCostModelEvaluation:
    _sanity_check_inputs(hardware, workload, mapping, mode, output_path)

    logger = _logging.getLogger(__name__)

    # Output paths
    node_hw_performances_path = f"{output_path}/{experiment_id}-saved_cn_hw_cost.pickle"
    scme_path = f"{output_path}/{experiment_id}-scme.pickle"

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
            nb_ga_generations=nb_ga_generations,  # number of genetic algorithm generations
            nb_ga_individuals=nb_ga_individuals,  # number of individuals in each genetic algorithm generation
            mode=mode,
            layer_stacks=layer_stacks,
            node_hw_performances_path=node_hw_performances_path,
            operands_to_prefetch=[],  # required by GeneticAlgorithmAllocationStage
        )
        # Launch the MainStage
        answers = mainstage.run()
        scme = answers[0][0]
        pickle_save(scme, scme_path)
    return scme


def optimize_allocation_co(
    hardware: str,
    workload: str,
    mapping: str,
    mode: Literal["lbl"] | Literal["fused"],
    layer_stacks: list[tuple[int, ...]],
    experiment_id: str,
    output_path: str,
    skip_if_exists: bool = False,
) -> StreamCostModelEvaluation:
    _sanity_check_inputs(hardware, workload, mapping, mode, output_path)
    _sanity_check_gurobi_license()

    # Output paths
    node_hw_performances_path = f"{output_path}/{experiment_id}-saved_cn_hw_cost.pickle"
    scme_path = f"{output_path}/{experiment_id}-scme.pickle"
    # After constraint optimization paths
    node_hw_performances_path_with_split = f"outputs/{experiment_id}-saved_cn_hw_cost-with_split.pickle"

    logger = _logging.getLogger(__name__)

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
                ConstraintOptimizationAllocationStage,
            ],
            accelerator=hardware,  # required by AcceleratorParserStage
            workload_path=workload,  # required by ModelParserStage
            mapping_path=mapping,  # required by ModelParserStage
            loma_lpf_limit=6,  # required by LomaEngine
            mode=mode,
            layer_stacks=layer_stacks,
            node_hw_performances_path=node_hw_performances_path,
            node_hw_performances_path_with_split=node_hw_performances_path_with_split,
            operands_to_prefetch=[],  # required by ConstraintOptimizationAllocationStage
        )
        # Launch the MainStage
        answers = mainstage.run()
        scme = answers[0][0]
        pickle_save(scme, scme_path)
    return scme


if __name__ == "__main__":
    from stream.visualization.memory_usage import plot_memory_usage
    from stream.visualization.schedule import visualize_timeline_plotly

    accelerator = "stream/inputs/examples/hardware/tpu_like_quad_core.yaml"
    workload = "stream/inputs/examples/workload/resnet18.yaml"
    mapping = "stream/inputs/examples/mapping/tpu_like_quad_core.yaml"

    hw_name = "tpu_like_quad_core"
    wl_name = "resnet18"
    mode = "fused"
    experiment_id = f"{hw_name}-{wl_name}"
    output_path = "outputs"
    layer_stacks = [tuple(range(0, 11)), tuple(range(11, 22))] + list((i,) for i in range(22, 49))

    scme, _ = optimize_allocation_ga(
        accelerator,
        workload,
        mapping,
        mode,
        layer_stacks,
        experiment_id,
        output_path,
    )

    plot_full_schedule = True
    draw_dependencies = True
    plot_data_transfer = True
    section_start_percent = (0,)
    percent_shown = (100,)
    schedule_fig_path = f"{output_path}/schedule_plot.png"
    memory_fig_path = f"{output_path}/memory_plot.png"
    energy_fig_path = f"{output_path}/energy_plot.png"
    visualize_timeline_plotly(
        scme=scme,
        draw_dependencies=draw_dependencies,
        draw_communication=True,
        fig_path=schedule_fig_path,
    )
    plot_memory_usage(scme.accelerator.memory_manager, fig_path=memory_fig_path)
    # bar_plot_stream_cost_model_evaluations_breakdown([scme], fig_path=energy_fig_path)
