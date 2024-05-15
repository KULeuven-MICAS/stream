from stream.classes.stages.AcceleratorParserStage import AcceleratorParserStage as AcceleratorParserStage_
from zigzag.stages.MainStage import MainStage
from stream.classes.stages import *
from stream.visualization.schedule import (
    plot_timeline_brokenaxes,
    visualize_timeline_plotly,
)
from stream.visualization.memory_usage import plot_memory_usage
import re
import pickle

# Initialize the logger
import logging as _logging

_logging_level = _logging.INFO
_logging_format = (
    # "%(asctime)s - %(name)s.%(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
    "%(asctime)s - %(levelname)s - %(message)s"
)
_logging.basicConfig(level=_logging_level, format=_logging_format)

################################INPUTS################################
accelerator = "stream/inputs/examples/hardware/tpu_like_quad_core.yaml"
workload_path = "stream/inputs/examples/workload/resnet18.onnx"
mapping_path = "stream/inputs/examples/mapping/tpu_like_quad_core.yaml"
CN_define_mode = 4  # manually define outer-CN loops
hint_loops = [("OY", "all")]
split_W_percentage = 0.5
nb_ga_individuals = 16  # number of individuals in each generation
nb_ga_generations = 16  # number of genetic algorithm generations
######################################################################

################################PARSING###############################
hw_name = accelerator.split(".")[-1]
wl_name = re.split(r"/|\.", workload_path)[-1]
if wl_name == "onnx":
    wl_name = re.split(r"/|\.", workload_path)[-2]
hint_loops_str_list = []
if isinstance(hint_loops[0], list):
    hint_loops_nested = hint_loops
else:
    hint_loops_nested = [hint_loops]
for sublist in hint_loops_nested:
    for dim, size in sublist:
        hint_loops_str_list.extend([str(dim).lower(), str(size)])
hint_loops_str = "_".join(hint_loops_str_list)
experiment_id = f"{hw_name}-{wl_name}-hintloop_{hint_loops_str}"
node_hw_cost_pkl_name = f"{experiment_id}-saved_cn_hw_cost"
scme_pkl_name = f"{experiment_id}-scme"
######################################################################

############PLOTTING#############
plot_file_name = f"-{experiment_id}-"
plot_full_schedule = True
draw_dependencies = True
plot_data_transfer = True
section_start_percent = (0,)
percent_shown = (100,)
#################################


################################PATHS################################
node_hw_performances_path = f"outputs/{node_hw_cost_pkl_name}.pickle"
scme_path = f"outputs/{scme_pkl_name}.pickle"
timeline_fig_path_plotly = f"outputs/{experiment_id}-schedule.html"
timeline_fig_path_matplotlib = f"outputs/{experiment_id}-schedule.png"
memory_fig_path = f"outputs/{experiment_id}-memory.png"
#####################################################################

mainstage = MainStage(
    [  # Initializes the MainStage as entry point
        AcceleratorParserStage_,  # Parses the accelerator
        StreamONNXModelParserStage,  # Parses the ONNX Model into the workload
        # UserDefinedModelParserStage,  # Parses the user-defined Model into the workload
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
    node_hw_performances_path=node_hw_performances_path,  # saved node_hw_performances to skip re-computation
    plot_hof=True,  # Save schedule and memory usage plot of each individual in the Genetic Algorithm hall of fame
    plot_file_name=plot_file_name,
    plot_full_schedule=plot_full_schedule,
    plot_data_transfer=plot_data_transfer,
    cn_define_mode=CN_define_mode,
    hint_loops=hint_loops,
    split_W_percentage=split_W_percentage,
    scheduler_candidate_selection="memory",
    operands_to_prefetch=[],
)

# Launch the MainStage

scme, _ = mainstage.run()
scme = scme[0]

# Save the scme to a pickle file
with open(scme_path, "wb") as fp:
    pickle.dump(scme, fp)

# Plotting results using Plotly
visualize_timeline_plotly(
    scme,
    draw_dependencies=draw_dependencies,
    draw_communication=plot_data_transfer,
    fig_path=timeline_fig_path_plotly,
)

# Ploting results using Matplotlib
plot_timeline_brokenaxes(
    scme,
    draw_dependencies,
    section_start_percent,
    percent_shown,
    plot_data_transfer,
    fig_path=timeline_fig_path_matplotlib,
)
plot_memory_usage(scme, section_start_percent, percent_shown, fig_path=memory_fig_path)
