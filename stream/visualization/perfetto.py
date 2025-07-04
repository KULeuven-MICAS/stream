import json
import logging

import numpy as np
import pandas as pd
from zigzag.utils import pickle_load

from stream.cost_model.cost_model import StreamCostModelEvaluation
from stream.utils import CostModelEvaluationLUT
from stream.visualization.utils import get_dataframe_from_scme

PROCESS_NAME = "scme"
UNKNOWN_STRING = "Unknown"

logger = logging.getLogger(__name__)


def parse_non_base_attrs(row: pd.Series, base_attrs: list[str]) -> dict:
    """
    Parse attributes that are not in the base_attrs list
    """
    args = {}
    for k, v in row.items():
        if k in base_attrs:
            continue
        if v is None:
            continue
        if isinstance(v, float) and np.isnan(v):
            continue
        if isinstance(v, list):
            v_new = [str(i) for i in v]
        if isinstance(v, dict):
            new_v = {}
            for k2, v2 in v.items():
                if isinstance(v2, list):
                    new_v[str(k2)] = [str(i) for i in v2]
                else:
                    new_v[str(k2)] = str(v2)
            v_new = new_v
        else:
            v_new = str(v)
        args[k] = v_new
    return args


def add_preamble_to_events(df: pd.DataFrame, events: list, process_name: str):
    # Extract and sort unique compute resources
    compute_resources = sorted(
        set(row["Resource"] for _, row in df.iterrows() if row["Type"] == "compute"),
        key=lambda x: int(x.split()[1]) if x.startswith("Core") else x,
    )

    # Extract and sort unique communication channel resources
    comm_resources = sorted(set(row["Resource"] for _, row in df.iterrows() if row["Type"] in ["transfer", "block"]))

    # Combine resources with compute resources first
    resources = compute_resources + comm_resources

    # Create metadata events for each resource
    tids = {}
    colors = {}
    for idx, resource in enumerate(resources):
        color = "blue" if idx < len(compute_resources) else "green"
        colors[resource] = color
        tids[resource] = idx
        metadata_event = {
            "name": "thread_name",
            "ph": "M",
            "pid": process_name,
            "tid": idx,
            "args": {"name": resource},
            "cname": color,
        }
        events.append(metadata_event)

    return events, tids, colors


def get_task_name(row: pd.Series, unknown_string: str) -> str:
    task_name = row["Task"]
    if pd.isna(task_name):
        return unknown_string
    if row["Type"] == "compute":
        return f"{row['Task']} (Id: {row['Id']}, Sub_id: {row['Sub_id']})"
    if row["Type"] == "transfer":
        return f"Transfer ({row['Tensors']})"
    if row["Type"] == "block":
        return f"Block ({row['Tensors']})"
    return task_name


def convert_scme_to_perfetto_json(
    scme: "StreamCostModelEvaluation",
    cost_lut: CostModelEvaluationLUT,
    json_path: str,
    layer_ids: list[int] | None = None,
    process_name: str = PROCESS_NAME,
    unknown_string: str = UNKNOWN_STRING,
) -> str:
    # Get the layer ids from the scme if not provided
    if layer_ids is None:
        layer_ids = sorted(set(n.id for n in scme.workload.node_list))

    # Define the base attributes that are common to all events
    base_attrs = ["Start", "End", "Task", "Type", "Resource", "Layer", "Id", "Sub_id"]

    # Get the DataFrame from the SCME
    df = get_dataframe_from_scme(scme, layer_ids, add_communication=True, cost_lut=cost_lut)

    # Initialize the list to store events
    events = []

    # Add metadata events for each resource
    events, tids, colors = add_preamble_to_events(df, events, process_name)

    # Iterate through each row in the DataFrame
    for _, row in df.iterrows():
        # Extract relevant data, handling np.nan values
        start = row["Start"]
        end = row["End"]
        if np.isnan(start) or np.isnan(end):
            raise ValueError(f"Row {row} has NaN start or end time")

        task_name = get_task_name(row, unknown_string)
        category = row["Type"]
        pid = process_name
        tid = tids[row["Resource"]]
        color = colors[row["Resource"]]

        # Collect remaining attributes that are not NaN or None
        args = parse_non_base_attrs(row, base_attrs)

        event = {
            "name": task_name,
            "cat": category,
            "ph": "X",
            "ts": int(start),
            "dur": int(end - start),
            "pid": pid,
            "tid": tid,
            "cname": color,
            "args": args,
        }

        # Append the event to the list
        events.append(event)

    # Sort the events based on thread names
    events.sort(key=lambda e: int(e["tid"]))

    # Convert the list of events to a JSON string
    perfetto_json = json.dumps(events, indent=2)

    with open(json_path, "w") as f:
        f.write(perfetto_json)

    logger.info(f"Saved Perfetto JSON to {json_path}")

    return perfetto_json


if __name__ == "__main__":
    # Example usage
    scme = pickle_load("outputs/tpu_like_quad_core-resnet18-fused-genetic_algorithm/scme.pickle")
    cost_lut = CostModelEvaluationLUT("outputs/tpu_like_quad_core-resnet18-fused-genetic_algorithm/cost_lut.pickle")
    layer_ids = sorted(set(n.id for n in scme.workload.node_list))
    json_path = "outputs/tpu_like_quad_core-resnet18-fused-genetic_algorithm/scme.json"
    perfetto_json = convert_scme_to_perfetto_json(scme, cost_lut, json_path=json_path, layer_ids=layer_ids)
    print(perfetto_json)
