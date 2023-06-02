from zigzag.classes.stages import *
from stream.classes.stages import *
from stream.visualization.schedule import plot_timeline_brokenaxes
from stream.visualization.memory_usage import plot_memory_usage
from stream.visualization.plot_scme import (
    bar_plot_stream_cost_model_evaluations_breakdown,
)
from stream.visualization.memory_usage import humanbytes
import re

# Initialize the logger
import logging as _logging

_logging_level = _logging.INFO
_logging_format = (
    "%(asctime)s - %(name)s.%(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
)
_logging.basicConfig(level=_logging_level, format=_logging_format)

#################################
CN_define_mode = 1
hint_loops = [("OY", "all")]

accelerator = "stream.inputs.validation.depfin.hardware.depfin"
workload_path = "stream.inputs.validation.depfin.workload.fsrcnn"
mapping_path = "stream.inputs.validation.depfin.mapping.mapping"


hw_name = accelerator.split(".")[-1]
wl_name = re.split(r"/|\.", workload_path)[-1]
if wl_name == "onnx":
    wl_name = re.split(r"/|\.", workload_path)[-2]
hint_loops_str_list = []
for dim, size in hint_loops:
    hint_loops_str_list.extend([str(dim).lower(), str(size)])
hint_loops_str = "_".join(hint_loops_str_list)
experiment_id = f"{hw_name}-{wl_name}-hintloop_{hint_loops_str}-new-all"
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
        AcceleratorParserStage,  # Parses the accelerator
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
    nb_ga_individuals=nb_ga_generations,  # number of individuals in each genetic algorithm generation
    nb_ga_generations=nb_ga_individuals,  # number of genetic algorithm generations
    node_hw_performances_path=node_hw_performances_path,  # saved node_hw_performances to skip re-computation
    plot_hof=True,  # Save schedule and memory usage plot of each individual in the Genetic Algorithm hall of fame
    plot_file_name=plot_file_name,
    plot_full_schedule=plot_full_schedule,
    plot_data_transfer=plot_data_transfer,
    cn_define_mode=CN_define_mode,
    hint_loops=hint_loops,
    scheduler_candidate_selection="memory",
    visualize_node_hw_performances_path=None,
)

# Launch the MainStage
scme, _ = mainstage.run()
scme = scme[0]

# Ploting Results
plot_full_schedule = True
draw_dependencies = False
plot_data_transfer = True
section_start_percent = (0,)
percent_shown = (2,)
timeline_fig_path = f"outputs/{experiment_id}_schedule.png"
memory_fig_path = f"outputs/{experiment_id}_memory.png"
breakdown_fig_path = "outputs/breakdown_plot.png"

plot_timeline_brokenaxes(
    scme,
    draw_dependencies,
    section_start_percent,
    percent_shown,
    plot_data_transfer,
    fig_path=timeline_fig_path,
)
plot_memory_usage(scme, section_start_percent, percent_shown, fig_path=memory_fig_path)

# list_scme = []
# list_scme.append(scme)
# list_scme.append(scme)

# bar_plot_stream_cost_model_evaluations_breakdown(list_scme, fig_path=breakdown_fig_path)
