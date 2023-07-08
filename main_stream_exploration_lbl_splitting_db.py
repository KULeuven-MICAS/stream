from zigzag.classes.stages import *
from stream.classes.stages import *
from stream.visualization.schedule import plot_timeline_brokenaxes
from stream.visualization.memory_usage import plot_memory_usage
import argparse
import pickle
import re
import os


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
    "--headname",
    metavar="path",
    required=True,
    help="module path to the accelerator, e.g. inputs.examples.hardware.TPU_like",
)
args = parser.parse_args()


# Initialize the logger
import logging as _logging

_logging_level = _logging.INFO
_logging_format = (
    "%(asctime)s - %(name)s.%(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
)
_logging.basicConfig(level=_logging_level, format=_logging_format)

#################################
# accelerator = "stream.inputs.exploration.hardware.HW1_1bigcore"
# accelerator = "stream.inputs.exploration.hardware.HW2_4homo"
# accelerator = "stream.inputs.exploration.hardware.HW3_4hetero"


# workload_path = "stream/inputs/exploration/workload/resnet18.onnx"
# mapping_path = "stream.inputs.exploration.mapping.HW2_4homo"

CN_define_mode = 1  # manually define outer CN size for all cores and all layers
hint_loops = []

hw_name = args.accelerator.split(".")[-1]
wl_name = re.split(r"/|\.", args.workload_path)[-1]
if wl_name == "onnx":
    wl_name = re.split(r"/|\.", args.workload_path)[-2]
hint_loops_str_list = []
for dim, size in hint_loops:
    hint_loops_str_list.extend([str(dim).lower(), str(size)])
hint_loops_str = "_".join(hint_loops_str_list)
experiment_id = f"{hw_name}-{wl_name}-hintloop_{hint_loops_str}"
node_hw_cost_pkl_name = f"saved_cn_hw_cost-{experiment_id}"
plot_file_name = f"-{experiment_id}-"
plot_full_schedule = True
plot_data_transfer = True
nb_ga_individuals = 96  # number of individuals in each genetic algorithm generation
nb_ga_generations = 24  # number of genetic algorithm generations

root_path = f"/esat/prometheus1/users/lmei/Stream_2023_TC_exploration_results6_latency"
intra_result_path = f"{root_path}/intra_result/"
inter_result_path = f"{root_path}/inter_result/"
plot_path = f"{root_path}/plot/"
filename = f"{args.headname}-lbl-{node_hw_cost_pkl_name}"
node_hw_performances_path = f"{intra_result_path}{filename}.pickle"
split_onnx_model_path = f"{root_path}/split_models/split_model-lbl-{experiment_id}.onnx"
split_W_double_buffered = True
#################################


mainstage = MainStage(
    [  # Initializes the MainStage as entry point
        AcceleratorParserStage,  # Parses the accelerator
        StreamONNXModelParserStage,  # Parses the ONNX Model into the workload
        LayerSplittingStage,
        StreamONNXModelParserStage,  # Parses the potentially split ONNX model into the workload
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
    plot_file_name=plot_file_name,
    plot_full_schedule=plot_full_schedule,
    plot_data_transfer=plot_data_transfer,
    cn_define_mode=CN_define_mode,
    hint_loops=hint_loops,
    operands_to_prefetch=[],
    scheduler_candidate_selection="latency",
    visualize_node_hw_performances_path=f"{intra_result_path}{filename}.png",
    split_onnx_model_path=split_onnx_model_path,
    split_W_double_buffered=split_W_double_buffered,
)

# Launch the MainStage
scme, _ = mainstage.run()
scme = scme[0]

# Save result
result_filename = f"{inter_result_path}{filename}.pickle"
# Create subfolders for result if they don't exist
dir_name = os.path.dirname(os.path.abspath(result_filename))
if not os.path.isdir(dir_name):
    os.makedirs(dir_name)
with open(result_filename, "wb") as f:
    pickle.dump(scme, f, -1)

# Ploting Results
plot_full_schedule = True
draw_dependencies = True
plot_data_transfer = True
section_start_percent = (0,)
percent_shown = (100,)
timeline_fig_path = f"{plot_path}{filename}-schedule.png"
memory_fig_path = f"{plot_path}{filename}-memory.png"

plot_timeline_brokenaxes(
    scme,
    draw_dependencies,
    section_start_percent,
    percent_shown,
    plot_data_transfer,
    fig_path=timeline_fig_path,
)
plot_memory_usage(scme, section_start_percent, percent_shown, fig_path=memory_fig_path)

# Print Results
import logging

logger = logging.getLogger(__name__)
latency = scme.latency
energy = scme.energy
edp = latency * energy
logger.info(
    f"Experiment {args.headname} Results: Latency = {int(latency):.4e} Cycles   Energy = {energy:.4e} pJ   EDP = {edp:.4e}"
)
