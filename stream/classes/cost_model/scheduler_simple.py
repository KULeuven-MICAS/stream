from operator import attrgetter, itemgetter
from networkx import DiGraph
from stream.classes.hardware.architecture.accelerator import Accelerator
from stream.classes.workload.tensor import Tensor
import logging

logger = logging.getLogger(__name__)


def schedule_graph(
    G: DiGraph,
    accelerator: Accelerator,
    cores_idle_from=None,
    candidate_selection="latency",
    operands_to_prefetch=[],
):
    """Simple scheduler that doesn't do any memory management.

    Args:
        G (DiGraph): The workload graph with nodes to be scheduled
        accelerator (Accelerator): The hardware accelerator onto which the nodes will be scheduled
        cores_idle_from (dict, optional): Dict of core start times. Defaults to None.
        candidate_selection (str, optional): Next node selection criterion. Defaults to "latency".
        operands_to_prefetch (list, optional): This is unused for this simple scheduler. Defaults to [].
    """
    if candidate_selection not in ["latency", "memory"]:
        raise ValueError(
            f"Scheduler's CN candidate_selection criterion '{candidate_selection}' is not supported."
        )
    all_core_ids = sorted(list(set(n.core_allocation for n in G.nodes())))
    if cores_idle_from is None:
        # Make it 0 for all cores
        cores_idle_from = {core_allocation: 0 for core_allocation in all_core_ids}

    nb_graph_nodes = G.number_of_nodes()
    nb_scheduled_nodes = 0
    scheduled_nodes = set()

    # List that keeps all possible candidate nodes for each core.
    candidates = []

    # Put the very first nodes of a layer that doesn't have any incoming edges as the first candidates
    for source_node in (n for n, d in G.in_degree() if d == 0):
        core_allocation = source_node.core_allocation
        # core_candidates[core_allocation].append((cores_idle_from[core_allocation], source_node))
        candidates.append((cores_idle_from[core_allocation], source_node))

    done = False
    while not done:
        if not candidates:
            raise ValueError(
                f"There are no candidates to schedule and only {nb_scheduled_nodes}/{nb_graph_nodes} nodes have been scheduled."
            )
        if candidate_selection == "latency":
            # Get the best candidate: the one with the earliest possible start time
            (preds_end, best_candidate) = min(candidates)
        elif candidate_selection == "memory":
            # Get the best candidate: the one with the highest layer_id
            preds_ends, cn_candidates = zip(*candidates)
            best_candidate = max(cn_candidates, key=attrgetter("id"))
            preds_end = preds_ends[cn_candidates.index(best_candidate)]
        candidates.remove((preds_end, best_candidate))

        core_id = best_candidate.core_allocation
        start = max(cores_idle_from[core_id], preds_end)
        end = start + best_candidate.get_runtime()
        best_candidate.set_start(start)
        best_candidate.set_end(end)
        cores_idle_from[core_id] = end

        scheduled_nodes.add(best_candidate)
        # For each successor of this node, check if all of its predecessors have been scheduled
        for successor in sorted(G.successors(best_candidate)):
            if all((pred in scheduled_nodes for pred in G.predecessors(successor))):
                preds_end = max(
                    (predecessor.end for predecessor in G.predecessors(successor)),
                    default=0,
                )
                # core_candidates[successor.core_allocation].append((preds_end, successor))
                candidates.append((preds_end, successor))
        nb_scheduled_nodes += 1
        done = nb_scheduled_nodes == nb_graph_nodes

    latency = max((n.end for n in G.nodes()))
    total_onchip_energy = sum((n.get_onchip_energy() for n in G.nodes()))
    total_offchip_energy = sum((n.get_offchip_energy() for n in G.nodes()))

    return latency, total_onchip_energy, total_offchip_energy
