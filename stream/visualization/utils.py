from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from stream.hardware.architecture.noc.communication_link import CommunicationLink
from stream.utils import CostModelEvaluationLUT
from stream.workload.computation.computation_node import ComputationNode

if TYPE_CHECKING:
    from stream.cost_model.cost_model import StreamCostModelEvaluation
    from stream.hardware.architecture.accelerator import Accelerator


def get_communication_dicts(scme: "StreamCostModelEvaluation"):
    dicts = []
    accelerator: Accelerator = scme.accelerator

    active_links: set[CommunicationLink] = set()
    for all_links_pairs in accelerator.communication_manager.all_pair_links.values():
        for link_pair in all_links_pairs:
            for link in link_pair:
                if link.events:
                    active_links.add(link)

    for cl in active_links:
        for cl_event in cl.events:
            task_type = cl_event.type
            start = cl_event.start
            end = cl_event.end
            runtime = end - start
            energy = cl_event.energy
            tensor = cl_event.tensor
            node = tensor.origin
            layer_id = node.id
            activity = cl_event.activity
            sender = cl_event.sender
            receiver = cl_event.receiver
            if runtime == 0:
                continue
            d = dict(
                Task=task_type.capitalize(),
                Id=np.nan,
                Sub_id=np.nan,
                Start=start,
                End=end,
                Resource=cl.get_name_for_schedule_plot(),
                Layer=layer_id,
                Runtime=runtime,
                Tensors={tensor: tensor.size},
                Type=task_type,
                Activity=activity,
                Energy=energy,
                LinkBandwidth=cl.bandwidth,
                Sender=sender,
                Receiver=receiver,
            )
            dicts.append(d)
    return dicts


def get_real_input_tensors(n, g):
    preds = list(g.predecessors(n))
    inputs = [pred.operand_tensors[pred.output_operand] for pred in preds if pred.id != n.id]
    inputs += [n.operand_tensors[op] for op in n.constant_operands]
    return inputs


def get_spatial_utilizations(
    scme: "StreamCostModelEvaluation", node: "ComputationNode", cost_lut: "CostModelEvaluationLUT | None"
):
    if cost_lut:
        equal_node = cost_lut.get_equal_node(node)
        assert equal_node, (
            f"No equal node for {node} found in CostModelEvaluationLUT. Check if pre/post LUT path is correct."
        )
        assert isinstance(node.chosen_core_allocation, int), (
            f"Chosen core allocation for {node} should be an integer, got {type(node.chosen_core_allocation)}."
        )
        core = scme.accelerator.get_core(node.chosen_core_allocation)
        cme = cost_lut.get_cme(equal_node, core)
        return cme.mac_spatial_utilization, cme.mac_utilization1
    return np.nan, np.nan


def get_energy_breakdown(
    scme: "StreamCostModelEvaluation", node: "ComputationNode", cost_lut: "CostModelEvaluationLUT | None"
):
    if cost_lut:
        equal_node = cost_lut.get_equal_node(node)
        assert equal_node, (
            f"No equal node for {node} found in CostModelEvaluationLUT. Check if pre/post LUT path is correct."
        )
        assert isinstance(node.chosen_core_allocation, int), (
            f"Chosen core allocation for {node} should be an integer, got {type(node.chosen_core_allocation)}."
        )
        core = scme.accelerator.get_core(node.chosen_core_allocation)
        cme = cost_lut.get_cme(equal_node, core)
        total_ops = cme.layer.total_mac_count
        en_total_per_op = cme.energy_total / total_ops
        en_breakdown = cme.mem_energy_breakdown
        en_breakdown_per_op = {}
        energy_sum_check = 0
        for layer_op, energies_for_all_levels in en_breakdown.items():
            d = {}
            mem_op = cme.layer.memory_operand_links[layer_op]
            for mem_level_idx, en in enumerate(energies_for_all_levels):
                mem_name = cme.accelerator.get_memory_level(mem_op, mem_level_idx).name
                d[mem_name] = en / total_ops
                energy_sum_check += en
            en_breakdown_per_op[layer_op] = d
        assert np.isclose(energy_sum_check, cme.mem_energy)
        return en_total_per_op, en_breakdown_per_op
    return np.nan, np.nan


def get_dataframe_from_scme(
    scme: "StreamCostModelEvaluation",
    layer_ids: list[int],
    add_communication: bool = False,
    cost_lut: "CostModelEvaluationLUT | None" = None,
):
    nodes = scme.workload.topological_sort()
    dicts = []
    for node in nodes:
        id = node.id
        layer = id
        if layer not in layer_ids:
            continue
        core_id = node.chosen_core_allocation
        start = node.start
        end = node.end
        runtime = node.runtime
        su_perfect_temporal, su_nonperfect_temporal = get_spatial_utilizations(scme, node, cost_lut)
        en_total_per_op, en_breakdown_per_op = get_energy_breakdown(scme, node, cost_lut)
        energy = node.onchip_energy
        tensors = get_real_input_tensors(node, scme.workload)
        task_type = "compute"
        d = dict(
            Task=node.short_name,
            Id=str(int(node.id)),
            Sub_id=str(int(node.sub_id)),
            Start=start,
            End=end,
            Resource=f"Core {core_id}",
            Layer=layer,
            Runtime=runtime,
            SpatialUtilization=su_perfect_temporal,
            SpatialUtilizationWithTemporal=su_nonperfect_temporal,
            Tensors={tensor: tensor.size for tensor in tensors},
            Type=task_type,
            Activity=np.nan,
            Energy=energy,
            EnergyTotalPerOp=en_total_per_op,
            EnergyBreakdownPerOp=en_breakdown_per_op,
        )
        dicts.append(d)
    if add_communication:
        communication_dicts = get_communication_dicts(scme)
        dicts += communication_dicts
    df = pd.DataFrame(dicts)
    return df
