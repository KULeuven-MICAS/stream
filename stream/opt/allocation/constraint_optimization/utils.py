from math import ceil, prod

from stream.cost_model.communication_manager import MulticastPathPlan
from stream.workload.node import Node, TransferNode
from stream.workload.steady_state.iteration_space import LoopEffect, SteadyStateIterationSpace
from stream.workload.workload import ComputationNode


def get_transfer_latency_for_path(tr: TransferNode, path: MulticastPathPlan) -> int:
    """Cycles one firing of this transfer costs on this path.

    The bytes over the narrowest link, spread over the chains that carry them: sources and
    targets that pair up one to one take a slice each over disjoint chains and move at once,
    where a transfer that fans out of or into a single core shares that core's link.
    """
    if not path or not path.links_used:
        return 0
    min_bw = min(link.bandwidth for link in path.links_used)
    assert len(tr.inputs) == 1, "Only single-input transfers are supported for latency calculation."
    tensor = tr.inputs[0]
    chains = len(path.targets) if 1 < len(path.sources) == len(path.targets) else 1
    return ceil(tensor.size_bits() / (min_bw * chains))


def get_active_transfer_latency_for_path(tr: TransferNode, choice: MulticastPathPlan, reuse_factor, ssis) -> int:
    latency_constant = float(get_transfer_latency_for_path(tr, choice))
    active_latency_absent_loops = get_active_latency(tr, latency_constant, ssis)
    active_latency = ceil(active_latency_absent_loops / reuse_factor)
    return active_latency


def get_active_latency(n: Node, runtime_constant: float, ssis: dict[ComputationNode, SteadyStateIterationSpace]) -> int:
    # Get the temporal steady state fraction of 'ABSENT' loops
    ssis_t = ssis.get(n).get_temporal_variables()
    total_product = prod([ssis_var.size for ssis_var in ssis_t])
    product_without_absent = prod([ssis_var.size for ssis_var in ssis_t if ssis_var.effect != LoopEffect.ABSENT])
    fraction = product_without_absent / total_product if total_product > 0 else 1.0
    # Scale the runtime constant by the fraction to get the effective latency
    active_latency = int(round(runtime_constant * fraction))
    return active_latency
