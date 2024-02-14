import sys
import os

sys.path.insert(0, os.getcwd())
from zigzag.classes.stages import *
from stream.classes.stages import *
from stream.visualization.schedule import (
    plot_timeline_brokenaxes,
    visualize_timeline_plotly,
)
from stream.visualization.memory_usage import plot_memory_usage
import re

############################## Initialize the logger ##############################
import logging as _logging

_logging_level = _logging.INFO
_logging_format = "%(asctime)s - %(levelname)s - %(message)s"
_logging.basicConfig(level=_logging_level, format=_logging_format)
####################################################################################

############################## Provide inputs ######################################
workload_path = "lab4.inputs.workload.resnet18_4_layers_fixed_parallel"
accelerator = "lab4.inputs.hardware.heterogeneous_quadcore"
mapping_path = "lab4.inputs.mapping.mapping_fixed_parallel"
####################################################################################

############################## Define variables for run ############################
CN_define_mode = 1  # manually define outer CN size for all cores and all layers
hint_loops = [("K", 4)]  # outer CN loops
hw_name = accelerator.split(".")[-1]
wl_name = re.split(r"/|\.", workload_path)[-1]
if wl_name == "onnx":
    wl_name = re.split(r"/|\.", workload_path)[-2]
hint_loops_str_list = []
for dim, size in hint_loops:
    hint_loops_str_list.extend([str(dim).lower(), str(size)])
hint_loops_str = "_".join(hint_loops_str_list)
experiment_id = f"{hw_name}-{wl_name}-hintloop_{hint_loops_str}-parallel"
node_hw_cost_pkl_name = f"saved_cn_hw_cost-{experiment_id}"
plot_file_name = f"-{experiment_id}-"
plot_full_schedule = True
plot_data_transfer = True
nb_ga_individuals = 16  # number of individuals in each genetic algorithm generation
nb_ga_generations = 16  # number of genetic algorithm generations
node_hw_performances_path = f"lab4/outputs/{node_hw_cost_pkl_name}.pickle"
visualize_node_hw_performances_path = (
    f"lab4/outputs/{node_hw_cost_pkl_name}_visualization.png"
)
####################################################################################


mainstage = MainStage(
    [  # Initializes the MainStage as entry point
        AcceleratorParserStage,  # Parses the accelerator
        # StreamONNXModelParserStage,  # Parses the ONNX Model into the workload
        UserDefinedModelParserStage,  # Parses the user-defined Model into the workload
        # ProfileWorkloadStage,
        GenerateCNWorkloadHybridStage,
        IntraCoreMappingStage,
        InterCoreMappingStage,
    ],
    accelerator=accelerator,  # required by AcceleratorParserStage
    workload_path=workload_path,  # required by ModelParserStage
    mapping_path=mapping_path,  # required by ModelParserStage
    loma_lpf_limit=6,  # required by LomaStage
    nb_ga_individuals=nb_ga_individuals,
    nb_ga_generations=nb_ga_generations,
    node_hw_performances_path=node_hw_performances_path,  # saved node_hw_performances to skip re-computation
    plot_hof=True,  # Save schedule and memory usage plot of each individual in the Genetic Algorithm hall of fame
    plot_file_name=plot_file_name,
    plot_full_schedule=plot_full_schedule,
    plot_data_transfer=plot_data_transfer,
    cn_define_mode=CN_define_mode,
    hint_loops=hint_loops,
    scheduler_candidate_selection="latency",
    visualize_node_hw_performances_path=visualize_node_hw_performances_path,
    operands_to_prefetch=[],
)

# Launch the MainStage
scme, _ = mainstage.run()
scme = scme[0]

# Ploting Results
plot_full_schedule = True
draw_dependencies = True
plot_data_transfer = True
section_start_percent = (0,)
percent_shown = (100,)
fig_path = f"lab4/outputs/timeline-{experiment_id}.png"
timeline_fig_path_plotly = f"lab4/outputs/timeline-{experiment_id}.html"

# Plotting results using Plotly
visualize_timeline_plotly(
    scme,
    draw_dependencies=draw_dependencies,
    draw_communication=plot_data_transfer,
    fig_path=timeline_fig_path_plotly,
)

# Plotting results using brokenaxes
plot_timeline_brokenaxes(
    scme,
    draw_dependencies,
    section_start_percent,
    percent_shown,
    plot_data_transfer,
    fig_path=fig_path,
)
