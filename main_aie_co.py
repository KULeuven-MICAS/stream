import logging as _logging
import re

from stream.api import optimize_allocation_co
from stream.visualization.memory_usage import plot_memory_usage
from stream.visualization.schedule import (
    visualize_timeline_plotly,
)

_logging_level = _logging.INFO
_logging_format = "%(asctime)s - %(name)s.%(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
_logging.basicConfig(level=_logging_level, format=_logging_format)

############################################INPUTS############################################
workload_path = "stream/inputs/aie/workload/conv1x1_64_64_32_32.onnx"
accelerator = "stream/inputs/aie/hardware/single_aie_tile.yaml"
mapping_path = "stream/inputs/aie/mapping/single_aie_tile.yaml"
mode = "lbl"
layer_stacks = [(0,),]
# mode = "fused"
# layer_stacks = [(0, 1)]
nb_ga_generations = 16
nb_ga_individuals = 16
##############################################################################################

################################PARSING###############################
hw_name = accelerator.split("/")[-1].split(".")[0]
wl_name = re.split(r"/|\.", workload_path)[-1]
if wl_name == "onnx":
    wl_name = re.split(r"/|\.", workload_path)[-2]
experiment_id = f"{hw_name}-{wl_name}-{mode}-constraint-optimization"
######################################################################

##############PLOTTING###############
plot_file_name = f"-{experiment_id}-"
plot_full_schedule = True
draw_dependencies = True
plot_data_transfer = True
section_start_percent = (0,)
percent_shown = (100,)
#####################################


################################PATHS################################
timeline_fig_path_plotly = f"outputs/{experiment_id}-schedule.html"
memory_fig_path = f"outputs/{experiment_id}-memory.png"
#####################################################################

scme = optimize_allocation_co(
    hardware=accelerator,
    workload=workload_path,
    mapping=mapping_path,
    mode=mode,
    layer_stacks=layer_stacks,
    experiment_id=experiment_id,
    output_path="outputs",
)

# Plotting schedule timeline of best SCME
visualize_timeline_plotly(
    scme,
    draw_dependencies=draw_dependencies,
    draw_communication=plot_data_transfer,
    fig_path=timeline_fig_path_plotly,
)
# Plotting memory usage of best SCME
plot_memory_usage(scme, section_start_percent, percent_shown, fig_path=memory_fig_path)
