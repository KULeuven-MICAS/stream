import logging as _logging
import re

from stream.api import optimize_allocation_co
from stream.cost_model.steady_state_scheduler import SteadyStateScheduler
from stream.inputs.testing.mapping.make_2_conv_mapping import make_2_conv_mapping
from stream.inputs.testing.workload.make_2_conv import TwoConvWorkloadConfig, make_2_conv_workload

_logging_level = _logging.INFO
_logging_format = "%(asctime)s - %(name)s.%(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
_logging.basicConfig(level=_logging_level, format=_logging_format)

############################################INPUTS############################################
accelerator = "stream/inputs/examples/hardware/tpu_like_quad_core.yaml"
workload_config = TwoConvWorkloadConfig(
    batch_size=1,
    in_channels=8,
    height=32,
    width=32,
    out_channels_1=16,
    out_channels_2=32,
    kernel_size=3,
    in_dtype="bf16",
    weight_dtype="bf16",
)
workload_path = make_2_conv_workload(workload_config)
mapping_path = make_2_conv_mapping(workload_config)
##############################################################################################

################################PARSING###############################
hw_name = accelerator.rsplit("/", maxsplit=1)[-1].split(".", maxsplit=1)[0]
wl_name = re.split(r"/|\.", workload_path)[-1]
if wl_name == "onnx":
    wl_name = re.split(r"/|\.", workload_path)[-2]
experiment_id = f"{hw_name}-{wl_name}-constraint_optimization"
######################################################################

ctx = optimize_allocation_co(
    hardware=accelerator,
    workload=workload_path,
    mapping=mapping_path,
    experiment_id=experiment_id,
    output_path="outputs",
    skip_if_exists=False,
)

# Access final scheduler from context
scheduler: SteadyStateScheduler = ctx.get("scheduler")
print(f"Total latency: {scheduler.latency_total:.3e} cycles")
print(f"Latency per iteration: {scheduler.latency_per_iteration:.3e} cycles")
print(f"Iterations: {scheduler.iterations}")
print(f"Overlap between iterations: {scheduler.overlap_between_iterations:.3e} cycles")
