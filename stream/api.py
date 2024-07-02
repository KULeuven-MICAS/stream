from zigzag.stages.MainStage import MainStage
from stream.classes.stages import *
import re


def get_hardware_performance_stream(
    hardware, workload, mapping, CN_define_mode, hint_loops, node_hw_cost_pkl_name
):
    # Initialize the logger
    import logging as _logging

    _logging_level = _logging.INFO
    # _logging_format = '%(asctime)s - %(name)s.%(funcName)s +%(lineno)s - %(levelname)s - %(message)s'
    _logging_format = (
        "%(asctime)s - %(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
    )
    _logging.basicConfig(level=_logging_level, format=_logging_format)

    mainstage = MainStage(
        [  # Initializes the MainStage as entry point
            AcceleratorParserStage,  # Parses the accelerator
            # StreamONNXModelParserStage,  # Parses the ONNX Model into the workload
            UserDefinedModelParserStage,  # Parses the user-defined Model into the workload
            GenerateCNWorkloadHybridStage,
            IntraCoreMappingStage,
            InterCoreMappingStage,
        ],
        accelerator=hardware,  # required by AcceleratorParserStage
        workload_path=workload,  # required by ModelParserStage
        mapping_path=mapping,  # required by ModelParserStage
        loma_lpf_limit=6,  # required by LomaStage
        nb_ga_individuals=128,  # number of individuals in each genetic algorithm generation
        nb_ga_generations=100,  # number of genetic algorithm generations
        node_hw_performances_path=f"outputs/{node_hw_cost_pkl_name}.pickle",  # saved node_hw_performances to skip re-computation
        plot_hof=True,  # Save schedule and memory usage plot of each individual in the Genetic Algorithm hall of fame
        plot_file_name=True,
        plot_full_schedule=True,
        plot_data_transfer=True,
        cn_define_mode=CN_define_mode,
        hint_loops=hint_loops,
        scheduler_candidate_selection="memory",
    )

    # Launch the MainStage
    answers = mainstage.run()
    return answers


if __name__ == "__main__":
    CN_define_mode = 1
    hint_loops = [("OY", "all")]

    accelerator = "stream/inputs/examples/hardware/tpu_like_quad_core.yaml"
    workload = "stream/inputs/examples/workload/resnet18.yaml"
    mapping = "stream/inputs/examples/mapping/tpu_like_quad_core.yaml"

    hw_name = "tpu_like_quad_core"
    wl_name = "resnet18"
    experiment_id = (
        f"{hw_name}-{wl_name}-CNmode_{CN_define_mode}-hintloop_{str(hint_loops)}"
    )
    node_hw_cost_pkl_name = f"saved_CN_HW_cost-{experiment_id}"

    scme, _ = get_hardware_performance_stream(
        accelerator,
        workload,
        mapping,
        CN_define_mode,
        hint_loops,
        node_hw_cost_pkl_name,
    )

    from stream.visualization.schedule import plot_timeline_brokenaxes
    from stream.visualization.memory_usage import plot_memory_usage
    from stream.visualization.plot_scme import (
        bar_plot_stream_cost_model_evaluations_breakdown,
    )

    plot_full_schedule = True
    draw_dependencies = True
    plot_data_transfer = True
    section_start_percent = (0,)
    percent_shown = (100,)
    timeline_fig_path = "outputs/schedule_plot.png"
    memory_fig_path = "outputs/memory_plot.png"
    energy_fig_path = "outputs/energy_plot.png"
    plot_timeline_brokenaxes(
        scme[0].workload,
        scme[0].accelerator,
        draw_dependencies,
        section_start_percent,
        percent_shown,
        plot_data_transfer,
        fig_path=timeline_fig_path,
    )
    plot_memory_usage(scme[0].accelerator.memory_manager, fig_path=memory_fig_path)
    # bar_plot_stream_cost_model_evaluations_breakdown([scme], fig_path=energy_fig_path)
