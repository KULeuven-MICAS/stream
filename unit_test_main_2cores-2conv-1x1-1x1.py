# This file is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
 
#
# Copyright (C) 2024, Advanced Micro Devices, Inc.
#
#===----------------------------------------------------------------------===

from zigzag.classes.stages import *
from stream.classes.stages import *
from stream.visualization.schedule import (
    plot_timeline_brokenaxes,
    visualize_timeline_plotly,
)
from stream.visualization.memory_usage import plot_memory_usage
import re
import pickle

# Aya added this
# from zigzag.visualization.results import (
#     print_mapping,
# )
from stream.utils import save_core_allocation

# Initialize the logger
import logging as _logging

_logging_level = _logging.INFO
_logging_format = (
    "%(asctime)s - %(name)s.%(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
)
_logging.basicConfig(level=_logging_level, format=_logging_format)

#################################
accelerator = "unit_tests_accelerators.two_cores_accelerator"
workload_path = "unit_tests_workloads/conv2_1x1_C_512_K_256-1x1_C_256_K_256_workload.onnx"
mapping_path = "unit_tests_accelerators.two_cores_mapping"

# Aya: added this to customize the path to the output
example_name = "2cores-2conv-1x1-1x1"
results_path = "unit_tests_results/" + example_name


# Parameters determining the granularity of the layers splitting
CN_define_mode = 1 # automatically split layers if too big to fit: # manually define outer CN size for all cores and all layers
split_W_percentage = 0.5 # max percentage of capacity a single node's weights can be
hint_loops = [("OY", "all")] # outer CN loops, with error in resnet18 plotting

nb_ga_individuals = 16  # number of individuals in each generation
nb_ga_generations = 16  # number of genetic algorithm generations
######################################################################

hw_name = accelerator.split(".")[-1]
wl_name = re.split(r"/|\.", workload_path)[-1]
if wl_name == "onnx":
    wl_name = re.split(r"/|\.", workload_path)[-2]
hint_loops_str_list = []
for dim, size in hint_loops:
    hint_loops_str_list.extend([str(dim).lower(), str(size)])
hint_loops_str = "_".join(hint_loops_str_list)
experiment_id = f"{hw_name}-{wl_name}-hintloop_{hint_loops_str}-fixed-split"
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
node_hw_performances_path = f"unit_tests_results/{example_name}/{node_hw_cost_pkl_name}-CN_{CN_define_mode}.pickle"
scme_path = f"unit_tests_results/{example_name}/{scme_pkl_name}-CN_{CN_define_mode}.pickle"
timeline_fig_path_plotly = f"unit_tests_results/{example_name}/{experiment_id}-schedule-CN_{CN_define_mode}.html"
timeline_fig_path_matplotlib = f"unit_tests_results/{example_name}/{experiment_id}-schedule-CN_{CN_define_mode}.png"
memory_fig_path = f"unit_tests_results/{example_name}/{experiment_id}-memory-CN_{CN_define_mode}.png"
#####################################################################

mainstage = MainStage(
    [  # Initializes the MainStage as entry point
        AcceleratorParserStage,  # Parses the accelerator
        StreamONNXModelParserStage,  # Parses the ONNX Model into the workload
        #UserDefinedModelParserStage,  # Parses the user-defined Model into the workload
        GenerateCNWorkloadHybridStage,
        IntraCoreMappingStage,
        DetermineSchedulingOrderStage,
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
    scheduler_candidate_selection="memory",  # Aya: used by the scheduler in the IntraCoreMapping stage
    operands_to_prefetch=[],
    split_W_percentage=split_W_percentage,
    results_path=results_path, # Aya: added this to define the path to the results
    memTile_flag = False,  # Aya: added this to make it easy to add or remove memTiles
    memTile_prefetch_flag=False,
    memTile_prefetch_count=4,
    memTile_eviction_flag=False,
    idle_num_for_mem_tile=2, # Aya: represents the number of idle offchip links after which we start to go through the memTile
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