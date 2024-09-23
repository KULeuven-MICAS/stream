import os
from zigzag.stages.main import MainStage
from zigzag.utils import pickle_load

from stream.stages.allocation.genetic_algorithm_allocation import GeneticAlgorithmAllocationStage
from stream.stages.estimation.zigzag_core_mapping_estimation import ZigZagCoreMappingEstimationStage
from stream.stages.generation.hint_loops_generation import HintLoopsGenerationStage
from stream.stages.generation.hint_loops_partitioned_workload_generation import (
    HintLoopsPartitionedWorkloadGenerationStage,
)
from stream.stages.generation.layer_stacks_generation import LayerStacksGenerationStage
from stream.stages.generation.scheduling_order_generation import SchedulingOrderGenerationStage
from stream.stages.parsing.accelerator_parser import AcceleratorParserStage
from stream.stages.parsing.onnx_model_parser import ONNXModelParserStage as StreamONNXModelParserStage
from stream.stages.set_fixed_allocation_performance import SetFixedAllocationPerformanceStage

import logging as _logging

def optimize_allocation_ga(hardware, workload, mapping, mode, experiment_id, output_path):

    _logging_level = _logging.INFO
    # _logging_format = '%(asctime)s - %(name)s.%(funcName)s +%(lineno)s - %(levelname)s - %(message)s'
    _logging_format = "%(asctime)s - %(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
    _logging.basicConfig(level=_logging_level, format=_logging_format)

    # Output paths
    plot_file_name = f"-{experiment_id}-ga"
    node_hw_performances_path = f"{output_path}/{experiment_id}-saved_cn_hw_cost.pickle"
    scme_path = f"{output_path}/{experiment_id}-scme.pickle"

    if os.path.exists(scme_path):
        scme = pickle_load(scme_path)
    else:
        mainstage = MainStage(
            [  # Initializes the MainStage as entry point
                AcceleratorParserStage,  # Parses the accelerator
                StreamONNXModelParserStage,  # Parses the ONNX Model into the workload
                LayerStacksGenerationStage,
                HintLoopsGenerationStage,
                HintLoopsPartitionedWorkloadGenerationStage,
                ZigZagCoreMappingEstimationStage,
                SetFixedAllocationPerformanceStage,
                SchedulingOrderGenerationStage,
                GeneticAlgorithmAllocationStage,
            ],
            accelerator=hardware,  # required by AcceleratorParserStage
            workload_path=workload,  # required by ModelParserStage
            mapping_path=mapping,  # required by ModelParserStage
            loma_lpf_limit=6,  # required by LomaEngine
            nb_ga_individuals=16,  # number of individuals in each genetic algorithm generation
            nb_ga_generations=16,  # number of genetic algorithm generations
            node_hw_performances_path=node_hw_performances_path,
            plot_file_name=plot_file_name,
            mode=mode,
        )
        # Launch the MainStage
        answers = mainstage.run()
        scme = answers[0]
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

    scme, _ = optimize_allocation_ga(
        accelerator,
        workload,
        mapping,
        mode,
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
