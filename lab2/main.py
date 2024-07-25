import os
import sys
import re
import logging as _logging

sys.path.insert(0, os.getcwd())  # Insert main folder in path

from stream.visualization.memory_usage import plot_memory_usage
from stream.visualization.schedule import (
    visualize_timeline_plotly,
)
from stream.classes.stages import *
from zigzag.stages.MainStage import MainStage


############################## Initialize the logger ##############################

_logging_level = _logging.INFO
_logging_format = "%(asctime)s - %(levelname)s - %(message)s"
_logging.basicConfig(level=_logging_level, format=_logging_format)
logger = _logging.getLogger(__name__)
####################################################################################

############################## Provide inputs ######################################
workload_path = "lab2/inputs/workload/resnet18_3_residuals.onnx"
accelerator = "lab2/inputs/hardware/hda_bus.yaml"
mapping_path = "lab2/inputs/mapping/mapping.yaml"
timeline_fig_path_plotly = f"lab2/outputs/timeline.html"
memory_fig_path = f"lab2/outputs/memory_usage.png"
####################################################################################

############################## Define variables for run ############################
CN_define_mode = 1  # manually define outer CN size for all cores and all layers
hint_loops = []  # outer CN loops
hw_name = accelerator.split(".")[-1]
wl_name = re.split(r"/|\.", workload_path)[-1]
if wl_name == "onnx":
    wl_name = re.split(r"/|\.", workload_path)[-2]
experiment_id = "lab2"
node_hw_cost_pkl_name = f"saved_cn_hw_cost-{experiment_id}"
plot_file_name = f"-{experiment_id}-"
plot_full_schedule = True
plot_data_transfer = True
node_hw_performances_path = f"lab2/outputs/{node_hw_cost_pkl_name}.pickle"
visualize_node_hw_performances_path = (
    f"lab2/outputs/{node_hw_cost_pkl_name}_visualization.png"
)
nb_ga_generations = 4
nb_ga_individuals = 4
####################################################################################


mainstage = MainStage(
    [  # Initializes the MainStage as entry point
        AcceleratorParserStage,  # Parses the accelerator
        StreamONNXModelParserStage,  # Parses the ONNX Model into the workload
        # UserDefinedModelParserStage,  # Parses the user-defined Model into the workload
        # ProfileWorkloadStage,
        GenerateCNWorkloadHybridStage,
        IntraCoreMappingStage,
        InterCoreMappingStage,
    ],
    accelerator=accelerator,  # required by AcceleratorParserStage
    workload_path=workload_path,  # required by ModelParserStage
    mapping_path=mapping_path,  # required by ModelParserStage
    loma_lpf_limit=6,  # required by LomaStage
    node_hw_performances_path=node_hw_performances_path,  # saved node_hw_performances to skip re-computation
    plot_hof=True,  # Save schedule and memory usage plot of each individual in the Genetic Algorithm hall of fame
    plot_file_name=plot_file_name,
    cn_define_mode=CN_define_mode,
    hint_loops=hint_loops,
    scheduler_candidate_selection="latency",
    visualize_node_hw_performances_path=visualize_node_hw_performances_path,
    operands_to_prefetch=[],
    nb_ga_generations=nb_ga_generations,
    nb_ga_individuals=nb_ga_individuals,
)

# Launch the MainStage
scme, _ = mainstage.run()
scme = scme[0]

# Log total energy and latency
logger.info(f"Total energy: {scme.energy:.3e}; Total latency: {scme.latency:.3e}")

# Ploting Results
plot_full_schedule = True
draw_dependencies = True
plot_data_transfer = True
section_start_percent = (0,)
percent_shown = (100,)


# Plotting results using Plotly
visualize_timeline_plotly(
    scme,
    draw_dependencies=draw_dependencies,
    draw_communication=plot_data_transfer,
    fig_path=timeline_fig_path_plotly,
)

plot_memory_usage(scme, section_start_percent, percent_shown, fig_path=memory_fig_path)
