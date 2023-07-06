from zigzag.classes.stages import *
from stream.classes.stages import *
from stream.visualization.schedule import plot_timeline_brokenaxes
from stream.visualization.memory_usage import plot_memory_usage
import argparse
import pickle
import re

# Initialize the logger
import logging as _logging

_logging_level = _logging.INFO
_logging_format = (
    "%(asctime)s - %(name)s.%(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
)
_logging.basicConfig(level=_logging_level, format=_logging_format)


headname = 4
root_path = f"/volume1/users/lmei/Stream_2023_TC_exploration_results_test_bigger_W_GB"
filename = f"{headname}-fused-saved_cn_hw_cost-HW2p5_4homo_mesh_dpDRAM_big_onchip_W_GB-resnet18-hintloop_oy_all"
path = f"{root_path}/inter_result/{filename}"
intra_result_path = f"{root_path}/intra_result/"
inter_result_path = f"{root_path}/inter_result/"
plot_path = f"{root_path}/plot/"
node_hw_performances_path = f"{intra_result_path}{filename}.pickle"

# Save result
with open(f"{path}.pickle", "rb") as f:
    scme = pickle.load(f)

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
logger = _logging.getLogger(__name__)
latency = scme.latency
energy = scme.energy
edp = latency * energy
logger.info(f"Experiment {headname} Results: Latency = {int(latency):.4e} Cycles   Energy = {energy:.4e} pJ   EDP = {edp:.4e}")
