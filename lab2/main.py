import logging as _logging
import os
import re
import sys

sys.path.insert(0, os.getcwd())  # Insert main folder in path
from zigzag.utils import pickle_load

from stream.api import optimize_allocation_ga
from stream.utils import CostModelEvaluationLUT
from stream.visualization.memory_usage import plot_memory_usage
from stream.visualization.perfetto import convert_scme_to_perfetto_json

############################## Initialize the logger ##############################
_logging_level = _logging.INFO
_logging_format = "%(asctime)s - %(levelname)s - %(message)s"
_logging.basicConfig(level=_logging_level, format=_logging_format)
logger = _logging.getLogger(__name__)
####################################################################################

############################## Provide inputs ######################################
workload_path = "lab2/inputs/workload/4_convs.onnx"
accelerator = "lab2/inputs/hardware/hda_bus.yaml"
mapping_path = "lab2/inputs/mapping/mapping.yaml"
mode = "lbl"
nb_ga_generations = 4
nb_ga_individuals = 4
####################################################################################

################################## Parsing #########################################
hw_name = accelerator.split("/")[-1].split(".")[0]
wl_name = re.split(r"/|\.", workload_path)[-1]
if wl_name == "onnx":
    wl_name = re.split(r"/|\.", workload_path)[-2]
experiment_id = f"{hw_name}-{wl_name}-{mode}-genetic_algorithm"
####################################################################################

##############PLOTTING###############
plot_file_name = f"-{experiment_id}-"
plot_full_schedule = True
draw_dependencies = True
plot_data_transfer = True
section_start_percent = (0,)
percent_shown = (100,)
#####################################


################################PATHS################################
output_folder = f"lab2/outputs/{experiment_id}"
timeline_fig_path_plotly = f"{output_folder}/schedule.html"
memory_fig_path = f"{output_folder}/memory.png"
json_path = f"{output_folder}/scme.json"
scme_path = f"{output_folder}/scme.pickle"
#####################################################################

if not os.path.exists(scme_path):
    scme = optimize_allocation_ga(
        hardware=accelerator,
        workload=workload_path,
        mapping=mapping_path,
        mode=mode,
        layer_stacks=None,
        nb_ga_generations=nb_ga_generations,
        nb_ga_individuals=nb_ga_individuals,
        experiment_id=experiment_id,
        output_path="lab2/outputs",
        skip_if_exists=True,
    )
else:
    scme = pickle_load(scme_path)

# Load in the CostModelEvaluationLUT from the run
cost_lut_path = f"{output_folder}/cost_lut.pickle"
cost_lut = CostModelEvaluationLUT(cost_lut_path)

# Plotting memory usage of best SCME
plot_memory_usage(scme, section_start_percent, percent_shown, fig_path=memory_fig_path, show_dram=True)

# Save json for perfetto visualization (Visualize at http://ui.perfetto.dev/)
convert_scme_to_perfetto_json(scme, cost_lut, json_path=json_path)
