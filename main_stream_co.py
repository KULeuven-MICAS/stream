import logging as _logging
import re

from stream.api import configure_logging, optimize_allocation_co

_logging_level = _logging.INFO
_logging_format = "%(asctime)s - %(name)s.%(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
configure_logging(level=_logging_level, fmt=_logging_format)

############################################INPUTS############################################
accelerator = "stream/inputs/examples/hardware/tpu_like_quad_core.yaml"
workload_path = "stream/inputs/examples/workload/resnet18.onnx"
mapping_path = "stream/inputs/examples/mapping/tpu_like_quad_core.yaml"
##############################################################################################

################################PARSING###############################
hw_name = accelerator.rsplit("/", maxsplit=1)[-1].split(".", maxsplit=1)[0]
wl_name = re.split(r"/|\.", workload_path)[-1]
if wl_name == "onnx":
    wl_name = re.split(r"/|\.", workload_path)[-2]
experiment_id = f"{hw_name}-{wl_name}-constraint_optimization"
######################################################################

sss = optimize_allocation_co(
    hardware=accelerator,
    workload=workload_path,
    mapping=mapping_path,
    experiment_id=experiment_id,
    output_path="outputs",
    skip_if_exists=True,
)
