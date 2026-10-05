from math import ceil, prod

from stream.cost_model.communication_manager import MulticastPathPlan
from stream.hardware.architecture.core import Core
from stream.hardware.architecture.noc.communication_link import CommunicationLink
from stream.workload.node import Node, TransferNode
from stream.workload.steady_state.iteration_space import LoopEffect, SteadyStateIterationSpace
from stream.workload.workload import ComputationNode

MAX_KEY_LENGTH = 255


def resource_key(res: Core | CommunicationLink | tuple[CommunicationLink, ...] | None) -> str:
    """A core's, link's or path's name in the model: ``Core <id>``, the link, or ``Path[...]`` cut to fit."""
    if isinstance(res, Core):
        return f"Core {res.id}"
    if isinstance(res, CommunicationLink):
        return str(res)
    if isinstance(res, tuple):
        path_str = "Path[" + "→".join(resource_key(link) for link in res) + "]"
        return path_str[: MAX_KEY_LENGTH - 55] + "..." if len(path_str) > MAX_KEY_LENGTH else path_str
    return str(res)


def get_transfer_latency_for_path(tr: TransferNode, path: MulticastPathPlan, bits: int | None = None) -> int:
    """Cycles one firing of this transfer costs on this path, moving its tensor or ``bits``.

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
    return ceil((tensor.size_bits() if bits is None else bits) / (min_bw * chains))


def get_active_transfer_latency_for_path(tr: TransferNode, choice: MulticastPathPlan, reuse_factor, ssis) -> int:
    latency_constant = float(get_transfer_latency_for_path(tr, choice))
    active_latency_absent_loops = get_active_latency(tr, latency_constant, ssis)
    active_latency = ceil(active_latency_absent_loops / reuse_factor)
    return active_latency


def active_fraction(n: Node, ssis: dict[ComputationNode, SteadyStateIterationSpace]) -> float:
    """Fraction of the steady-state iterations in which ``n`` is not idle on an absent loop."""
    temporal = ssis.get(n).get_temporal_variables()
    total = prod(v.size for v in temporal)
    return prod(v.size for v in temporal if v.effect != LoopEffect.ABSENT) / total if total else 1.0


def get_active_latency(n: Node, runtime_constant: float, ssis: dict[ComputationNode, SteadyStateIterationSpace]) -> int:
    return int(round(runtime_constant * active_fraction(n, ssis)))
