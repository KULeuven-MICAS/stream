import logging as _logging
import pickle
import re

from zigzag.stages.main import MainStage

from stream.stages.allocation.genetic_algorithm_allocation import GeneticAlgorithmAllocationStage
from stream.stages.estimation.zigzag_core_mapping_estimation import ZigZagCoreMappingEstimationStage
from stream.stages.generation.hint_loops_generation import HintLoopsGenerationStage
from stream.stages.generation.hint_loops_partitioned_workload_generation import (
    HintLoopsPartitionedWorkloadGenerationStage,
)
from stream.stages.generation.layer_stacks_generation import LayerStacksGenerationStage
from stream.stages.generation.scheduling_order_generation import SchedulingOrderGenerationStage
from stream.stages.parsing.accelerator_parser import (
    AcceleratorParserStage as AcceleratorParserStage_,
)
from stream.stages.parsing.onnx_model_parser import ONNXModelParserStage as StreamONNXModelParserStage
from stream.stages.set_fixed_allocation_performance import SetFixedAllocationPerformanceStage
from stream.visualization.memory_usage import plot_memory_usage
from stream.visualization.schedule import (
    plot_timeline_brokenaxes,
    visualize_timeline_plotly,
)

_logging_level = _logging.INFO
_logging_format = "%(asctime)s - %(name)s.%(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
_logging.basicConfig(level=_logging_level, format=_logging_format)

############################################INPUTS############################################
accelerator = "stream/inputs/examples/hardware/tpu_like_quad_core.yaml"
workload_path = "stream/inputs/examples/workload/resnet18.onnx"
mapping_path = "stream/inputs/examples/mapping/tpu_like_quad_core.yaml"
mode = "fused"
layer_stacks = [tuple(range(0, 11)), tuple(range(11, 22))] + list((i,) for i in range(22, 49))
nb_ga_generations = 16
nb_ga_individuals = 16
##############################################################################################

################################PARSING###############################
hw_name = accelerator.split("/")[-1].split(".")[0]
wl_name = re.split(r"/|\.", workload_path)[-1]
if wl_name == "onnx":
    wl_name = re.split(r"/|\.", workload_path)[-2]
experiment_id = f"{hw_name}-{wl_name}-{mode}-genetic_algorithm"
node_hw_cost_pkl_name = f"{experiment_id}-saved_cn_hw_cost"
scme_pkl_name = f"{experiment_id}-scme"
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
        LayerStacksGenerationStage,
        HintLoopsGenerationStage,
        HintLoopsPartitionedWorkloadGenerationStage,
        ZigZagCoreMappingEstimationStage,
        SetFixedAllocationPerformanceStage,
        SchedulingOrderGenerationStage,
        GeneticAlgorithmAllocationStage,
    ],
    accelerator=accelerator,  # required by AcceleratorParserStage
    workload_path=workload_path,  # required by ModelParserStage
    mapping_path=mapping_path,  # required by ModelParserStage
    loma_lpf_limit=6,  # required by LomaEngine
    node_hw_performances_path=node_hw_performances_path,  # saved node_hw_performances to skip re-computation
    operands_to_prefetch=[],
    mode=mode,
    layer_stacks=layer_stacks,
    plot_file_name=plot_file_name,
    nb_ga_generations=nb_ga_generations,
    nb_ga_individuals=nb_ga_individuals,
)

# if os.path.exists(scme_path):
#     scme = pickle_load(scme_path)
# else:
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
