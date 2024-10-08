import logging as _logging
import re

from zigzag.stages.main import MainStage

from stream.cost_model.cost_model import StreamCostModelEvaluation
from stream.stages.allocation.genetic_algorithm_allocation import GeneticAlgorithmAllocationStage
from stream.stages.estimation.zigzag_core_mapping_estimation import ZigZagCoreMappingEstimationStage
from stream.stages.generation.hint_loops_partitioned_workload_generation import (
    HintLoopsPartitionedWorkloadGenerationStage,
)
from stream.stages.parsing.accelerator_parser import AcceleratorParserStage as AcceleratorParserStage_
from stream.stages.parsing.onnx_model_parser import UserDefinedModelParserStage
from stream.visualization.memory_usage import plot_memory_usage
from stream.visualization.schedule import visualize_timeline_plotly

_logging_level = _logging.INFO
_logging_format = "%(asctime)s - %(name)s.%(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
_logging.basicConfig(level=_logging_level, format=_logging_format)

#################################
accelerator = "stream/inputs/testing/hardware/single_testing_core_offchip.yaml"
workload_path = "stream/inputs/testing/workload/testing_workload_for_1_core.yaml"
mapping_path = "stream/inputs/testing/mapping/testing_mapping.yaml"

CN_define_mode = 1  # manually define outer CN size for all cores and all layers
hint_loops = [("OY", "all")]  # outer CN loops, with error in resnet18 plotting

hw_name = accelerator.split("/")[-1].split(".")[0]
wl_name = re.split(r"/|\.", workload_path)[-1]
experiment_id = f"{hw_name}-{wl_name}-CNmode_{CN_define_mode}-hintloop_{str(hint_loops)}"
node_hw_cost_pkl_name = f"saved_CN_HW_cost-{experiment_id}"
plot_file_name = f"-{experiment_id}-"
plot_full_schedule = True
plot_data_transfer = True
#################################


mainstage = MainStage(
    [  # Initializes the MainStage as entry point
        AcceleratorParserStage_,  # Parses the accelerator
        # StreamONNXModelParserStage,  # Parses the ONNX Model into the workload
        UserDefinedModelParserStage,  # Parses the user-defined Model into the workload
        HintLoopsPartitionedWorkloadGenerationStage,
        ZigZagCoreMappingEstimationStage,
        GeneticAlgorithmAllocationStage,
    ],
    accelerator=accelerator,  # required by AcceleratorParserStage
    workload_path=workload_path,  # required by ModelParserStage
    mapping_path=mapping_path,  # required by ModelParserStage
    loma_lpf_limit=6,  # required by LomaEngine
    nb_ga_individuals=4,  # number of individuals in each genetic algorithm generation
    nb_ga_generations=1,  # number of genetic algorithm generations
    node_hw_performances_path=f"outputs/{node_hw_cost_pkl_name}.pickle",  # saved results to skip re-computation
    plot_hof=True,  # Save schedule and memory usage plot of each individual in the Genetic Algorithm hall of fame
    plot_file_name=plot_file_name,
    plot_full_schedule=plot_full_schedule,
    plot_data_transfer=plot_data_transfer,
    cn_define_mode=CN_define_mode,
    hint_loops=hint_loops,
    scheduler_candidate_selection="latency",
    operands_to_prefetch=["W"],
)

# Launch the MainStage
scme, _ = mainstage.run()
scme: StreamCostModelEvaluation = scme[0]

# Ploting Results
plot_full_schedule = True
draw_dependencies = True
draw_communication = True
plot_data_transfer = True
section_start_percent = (0,)
percent_shown = (100,)
timeline_fig_path = "outputs/schedule_plot.html"
memory_fig_path = "outputs/memory_plot.png"

visualize_timeline_plotly(
    scme,
    draw_dependencies,
    draw_communication,
    fig_path=timeline_fig_path,
)

print(f"Total network energy = {scme.energy:.2e} pJ")
print(f"Total network latency = {scme.latency:.2e} cycles")
plot_memory_usage(scme, fig_path=memory_fig_path)
