import argparse
import re
import os

from zigzag.classes.stages import *
from stream.classes.stages import *
from stream.utils import load_scme, save_scme
from stream.visualization.schedule import (
    plot_timeline_brokenaxes,
    visualize_timeline_plotly,
)
from stream.visualization.memory_usage import plot_memory_usage

# Initialize the logger
import logging as _logging

_logging_level = _logging.INFO
_logging_format = (
    "%(asctime)s - %(name)s.%(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
)
_logging.basicConfig(level=_logging_level, format=_logging_format)


parser = argparse.ArgumentParser(description="Setup zigzag-v2 inputs")
parser.add_argument(
    "--workload_path",
    metavar="path",
    required=True,
    help="module path to workload, e.g. inputs.examples.workloads.resnet18",
)
parser.add_argument(
    "--mapping_path",
    metavar="path",
    required=True,
    help="path to mapping file, e.g., inputs.examples.mapping.tpu_like",
)
parser.add_argument(
    "--accelerator",
    metavar="path",
    required=True,
    help="module path to the accelerator, e.g. inputs.examples.hardware.TPU_like",
)
parser.add_argument(
    "--headname", metavar="path", required=True, help="experiment number",
)

parser.add_argument(
    "--results_path",
    metavar="path",
    required=True,
    help="root path to save the results to, e.g. '/users/asymons/results'",
)
parser.add_argument(
    "--suffix",
    metavar="path",
    required=True,
    help="suffix to be added to the end of the experiment id, e.g. fused",
)
args = parser.parse_args()

################################### HYPERPARAMETERS ##################################
CN_define_mode = 1  # manually define outer CN size for all cores and all layers
hint_loops = [("OY", "all")]
nb_ga_individuals = 64  # number of individuals in each genetic algorithm generation
nb_ga_generations = 10  # number of genetic algorithm generations
######################################################################################

################################# INPUT NAME PARSING #################################
hw_name = args.accelerator.split(".")[-1]
wl_name = re.split(r"/|\.", args.workload_path)[-1]
if wl_name == "onnx":
    wl_name = re.split(r"/|\.", args.workload_path)[-2]
hint_loops_str_list = []
for dim, size in hint_loops:
    hint_loops_str_list.extend([str(dim).lower(), str(size)])
hint_loops_str = "_".join(hint_loops_str_list)
experiment_id = (
    f"{args.headname}-{hw_name}-{wl_name}-hintloop_{hint_loops_str}-{args.suffix}"
)
######################################################################################

######################################## PATHS #######################################
results_path = args.results_path
intra_result_path = os.path.join(results_path, "intra_result/")
inter_result_path = os.path.join(results_path, "inter_result/")
plot_path = os.path.join(results_path, "plot/")
node_hw_performances_path = (
    os.path.join(intra_result_path, f"{experiment_id}-saved_cn_hw_cost.pickle")
)
visualize_node_hw_performances_path = (
    os.path.join(intra_result_path, f"{experiment_id}-saved_cn_hw_cost.png")
)
scme_path = os.path.join(inter_result_path, f"{experiment_id}-scme.pickle")
timeline_fig_path = os.path.join(plot_path, f"{experiment_id}-schedule.png")
memory_fig_path = os.path.join(plot_path, f"{experiment_id}-memory.png")
timeline_fig_path_plotly = os.path.join(plot_path, f"{experiment_id}-schedule.html")
######################################################################################

######################################## SETUP #######################################
mainstage = MainStage(
    [  # Initializes the MainStage as entry point
        AcceleratorParserStage,  # Parses the accelerator
        StreamONNXModelParserStage,  # Parses the ONNX Model into the workload
        # UserDefinedModelParserStage,  # Parses the user-defined Model into the workload
        GenerateCNWorkloadHybridStage,
        IntraCoreMappingStage,
        InterCoreMappingStage,
    ],
    accelerator=args.accelerator,  # required by AcceleratorParserStage
    workload_path=args.workload_path,  # required by ModelParserStage
    mapping_path=args.mapping_path,  # required by ModelParserStage
    loma_lpf_limit=7,  # required by LomaStage
    nb_ga_individuals=nb_ga_individuals,  # number of individuals in each genetic algorithm generation
    nb_ga_generations=nb_ga_generations,  # number of genetic algorithm generations
    node_hw_performances_path=node_hw_performances_path,  # saved node_hw_performances to skip re-computation
    plot_hof=True,  # Save schedule and memory usage plot of each individual in the Genetic Algorithm hall of fame
    plot_file_name=None,
    cn_define_mode=CN_define_mode,
    hint_loops=hint_loops,
    operands_to_prefetch=[],
    scheduler_candidate_selection="latency",
    visualize_node_hw_performances_path=visualize_node_hw_performances_path,
)
######################################################################################

######################################## LAUNCH #######################################
try:
    scme = load_scme(scme_path)
except:
    scme, _ = mainstage.run()
    scme = scme[0]
    save_scme(scme, scme_path)
######################################################################################

######################################### PLOT #######################################
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

plot_timeline_brokenaxes(
    scme,
    draw_dependencies,
    section_start_percent,
    percent_shown,
    plot_data_transfer,
    fig_path=timeline_fig_path,
)
plot_memory_usage(scme, section_start_percent, percent_shown, fig_path=memory_fig_path)
######################################################################################

######################################### PRINT ######################################
logger = _logging.getLogger(__name__)
latency = scme.latency
energy = scme.energy
edp = latency * energy
logger.info(
    f"Experiment {args.headname} Results: Latency = {int(latency):.4e} Cycles   Energy = {energy:.4e} pJ   EDP = {edp:.4e}"
)
######################################################################################
