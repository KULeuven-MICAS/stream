from stream.classes.stages.AcceleratorParserStage import AcceleratorParserStage as AcceleratorParserStage_
from zigzag.stages.MainStage import MainStage
from stream.classes.stages import *
from stream.visualization.schedule import visualize_timeline_plotly
from stream.visualization.memory_usage import plot_memory_usage

import re

# Initialize the logger
import logging as _logging

_logging_level = _logging.INFO
_logging_format = "%(asctime)s - %(name)s.%(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
_logging.basicConfig(level=_logging_level, format=_logging_format)

#################################
accelerator = "stream/inputs/testing/shared/dual_testing_core_offchip.yaml"
workload_path = "stream/inputs/testing/shared/testing_workload_for_2_cores.yaml"
mapping_path = "stream/inputs/testing/mapping/testing_mapping.yaml"

CN_define_mode = 1  # manually define outer CN size for all cores and all layers
# hint_loops = [('OY', 'all')]  # outer CN loops, with error in resnet18 plotting
hint_loops = []

hw_name = accelerator.split(".")[-1]
wl_name = re.split(r"/|\.", workload_path)[-1]
if wl_name == "onnx":
    wl_name = re.split(r"/|\.", workload_path)[-2]
hint_loops_str_list = []
for dim, size in hint_loops:
    hint_loops_str_list.extend([str(dim).lower(), str(size)])
hint_loops_str = "_".join(hint_loops_str_list)
experiment_id = f"{hw_name}-{wl_name}-hintloop_{hint_loops_str}"
node_hw_cost_pkl_name = f"saved_cn_hw_cost-{experiment_id}"
plot_file_name = f"-{experiment_id}-"
plot_full_schedule = True
plot_data_transfer = True
nb_ga_individuals = 16  # number of individuals in each genetic algorithm generation
nb_ga_generations = 16  # number of genetic algorithm generations
node_hw_performances_path = f"outputs/{node_hw_cost_pkl_name}.pickle"
#################################


mainstage = MainStage(
    [  # Initializes the MainStage as entry point
        AcceleratorParserStage_,  # Parses the accelerator
        # StreamONNXModelParserStage,  # Parses the ONNX Model into the workload
        UserDefinedModelParserStage,  # Parses the user-defined Model into the workload
        GenerateCNWorkloadHybridStage,
        IntraCoreMappingStage,
        InterCoreMappingStage,
    ],
    accelerator=accelerator,  # required by AcceleratorParserStage
    workload_path=workload_path,  # required by ModelParserStage
    mapping_path=mapping_path,  # required by ModelParserStage
    loma_lpf_limit=6,  # required by LomaStage
    nb_ga_individuals=nb_ga_individuals,  # number of individuals in each genetic algorithm generation
    nb_ga_generations=nb_ga_generations,  # number of genetic algorithm generations
    node_hw_performances_path=node_hw_performances_path,  # save node_hw_performances to skip re-computation
    plot_hof=True,  # Save schedule and memory usage plot of each individual in the Genetic Algorithm hall of fame
    plot_file_name=plot_file_name,
    plot_full_schedule=plot_full_schedule,
    plot_data_transfer=plot_data_transfer,
    cn_define_mode=CN_define_mode,
    hint_loops=hint_loops,
    scheduler_candidate_selection="memory",
    operands_to_prefetch=["W"],
)

# Launch the MainStage
scme, _ = mainstage.run()

# Ploting Results
plot_full_schedule = True
draw_dependencies = True
draw_communication = True
plot_data_transfer = True
section_start_percent = (0,)
percent_shown = (100,)
timeline_fig_path = f"outputs/{experiment_id}-schedule.png"
memory_fig_path = f"outputs/{experiment_id}-memory.png"

visualize_timeline_plotly(
    scme[0],
    draw_dependencies,
    draw_communication,
    fig_path=timeline_fig_path,
)

print(f"Total network energy = {scme[0].energy:.2e} pJ")
print(f"Total network latency = {scme[0].latency:.2e} cycles")
plot_memory_usage(scme[0], fig_path=memory_fig_path)
