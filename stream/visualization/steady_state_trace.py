"""stream/visualization/steady_state_trace.py
-----------------------------------------------
Export a solved steady-state schedule (from TransferAndTensorAllocator) to a
Perfetto / Chrome-Trace-Viewer JSON file.

Usage
-----
    from stream.visualization.steady_state_trace import export_steady_state_trace

    trace_path = export_steady_state_trace(
        tta=tta,
        iterations=iterations,
        overlap=overlap,
        latency_per_iteration=latency_per_iteration,
        output_path="outputs/my_run",
    )
    # Open the resulting JSON at https://ui.perfetto.dev/

Timeline model
--------------
  Let P  = latency_per_iteration   (= Σ slot_latency[s]  for all slots s)
  Let V  = overlap                  (cycles by which iteration i+1 may start
                                     before iteration i finishes)
  Let S  = P - V                    (iteration step / pipeline period)

  The file always shows exactly three consecutive representative iterations:
      iteration  i-1  starts at  t = -S
      iteration  i    starts at  t =  0
      iteration  i+1  starts at  t = +S

  Slot s within iteration k begins at:
      t_{k,s} = k * S + Σ_{j<s} slot_lat[j]

  These three iterations are sufficient to see:
    - the full pipeline pattern (all slots visible in i)
    - how i+1 starts before i finishes  (overlap V at the right edge)
    - how i started after i-1           (same overlap at the left edge)

  All global metadata (true total iterations, total latency, etc.) is stored
  in the ``otherData`` field of the trace for reference.

Track layout
------------
  • One Perfetto "thread" per hardware resource, all sharing one "process".
  • Computation nodes  → appear on each of their mapped Core tracks in
    parallel (they execute simultaneously due to inter-core tiling).
  • Transfer nodes     → appear on each CommunicationLink of the chosen path.
  • Tracks are sorted: compute cores first (by col then row), then links.
"""

from __future__ import annotations

import json
import logging
import os
from math import ceil
from typing import TYPE_CHECKING

from stream.cost_model.communication_manager import MulticastPathPlan
from stream.hardware.architecture.core import Core
from stream.hardware.architecture.noc.communication_link import CommunicationLink
from stream.opt.allocation.constraint_optimization.transfer_and_tensor_allocation import TransferAlloc
from stream.workload.node import ComputationNode as _ComputationNode

if TYPE_CHECKING:
    from stream.hardware.architecture.core import Core
    from stream.hardware.architecture.noc.communication_link import CommunicationLink
    from stream.opt.allocation.constraint_optimization.transfer_and_tensor_allocation import (
        TransferAndTensorAllocator,
    )

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────── colours ─────────────── #
# Perfetto "cname" tokens (from the Chrome Trace Viewer colour palette).
_CNAME_COMPUTE = "good"  # teal / blue-green
_CNAME_TRANSFER = "thread_state_running"  # green
_CNAME_TRANSFER_CONST = "bad"  # orange  (constant-tensor / weight DMA)

# ─────────────────────────────────────────────── resource helpers ─────────── #


def _core_label(core: Core) -> str:
    col = getattr(core, "col_id", "?")
    row = getattr(core, "row_id", "?")
    return f"Core {core.id} [{core.core_type}] (col={col}, row={row})"


def _link_label(link: CommunicationLink) -> str:
    bw = getattr(link, "bandwidth", "?")
    sender_id = getattr(link.sender, "id", "?")
    receiver_id = getattr(link.receiver, "id", "?")
    return f"Link {sender_id}→{receiver_id} (bw={bw})"


def _path_label(path: MulticastPathPlan) -> str:
    """Short but complete label for a path of CommunicationLinks used as a track name."""
    hops = "→".join(str(getattr(lnk.receiver, "id", "?")) for lnk in path.links_used)
    src = getattr(path.sources[0], "id", "?") if path.links_used else "?"
    return f"Path {src}→{hops}"


def _resource_label(res) -> str:
    if isinstance(res, Core):
        return _core_label(res)
    if isinstance(res, CommunicationLink):
        return _link_label(res)
    return str(res)


def _resource_sort_key(res) -> int:
    """
    Cores before links; within cores, sort by (col, row) so the grid layout
    matches the chip topology left-to-right, bottom-to-top.
    """

    if isinstance(res, Core):
        col = getattr(res, "col_id", 0) or 0
        row = getattr(res, "row_id", 0) or 0
        return col * 1000 + row
    # Links sorted after all cores, then by sender id
    return (
        1_000_000
        + getattr(getattr(res, "sender", None), "id", 0) * 1000
        + getattr(getattr(res, "receiver", None), "id", 0)
    )


# ─────────────────────────────────────────────────── public API ──────────── #


def export_steady_state_trace(  # noqa: PLR0912, PLR0915
    tta: TransferAndTensorAllocator,
    iterations: int,
    overlap: int,
    latency_per_iteration: float,
    output_path: str,
    transfer_allocations: TransferAlloc,
    *,
    filename: str = "steady_state_trace.json",
) -> str:
    """
    Produce a Perfetto JSON trace showing three representative iterations
    (i-1, i, i+1) of the solved steady-state schedule.

    Only three iterations are rendered regardless of how many real iterations
    exist, because the pattern is fully periodic.  The true total count and
    total latency are stored in ``otherData`` for reference.

    Parameters
    ----------
    tta:
        A *solved* ``TransferAndTensorAllocator``.
    iterations:
        Total number of steady-state iterations (metadata only).
    overlap:
        Overlap in cycles between consecutive iterations (from the solver).
    latency_per_iteration:
        Pipeline period in cycles (= sum of all solved slot latencies).
    output_path:
        Directory where the JSON file will be written.
    filename:
        Name of the output file (default ``steady_state_trace.json``).

    Returns
    -------
    str
        Absolute path of the written JSON file.
    """
    # ── 1.  Slot timings from solver ─────────────────────────────────── #
    max_slot = tta.max_slot
    slot_lat: dict[int, float] = {s: float(tta.slot_latency[s].X) for s in range(max_slot + 1)}

    # Cumulative start of each slot *within* one iteration
    slot_starts: dict[int, float] = {}
    cumulative = 0.0
    for s in range(max_slot + 1):
        slot_starts[s] = cumulative
        cumulative += slot_lat[s]

    # Distance between the start of successive iterations
    iter_step: float = latency_per_iteration - overlap

    # ── 2.  One TID per physical resource ────────────────────────────── #
    # Cores for computation nodes, link-tuples (chosen paths) for transfers.
    # One row per physical resource is correct: the solver's constraint
    #   V ≤ idle_lat[res] = idle_start_lat[res] + idle_end_lat[res]
    # guarantees that no two consecutive iterations ever have overlapping
    # events on the same resource track.

    all_resources: list = []

    # Cores (computation nodes)
    for node in tta.ssc_nodes:
        for core in tta.mapping.get(node).resource_allocation:
            if core not in all_resources:
                all_resources.append(core)

    # Chosen transfer paths — collect (path, label) pairs so the row name
    # lists every link in the path.
    path_of: dict = {}  # transfer node → chosen path tuple
    for tr, path in transfer_allocations.items():
            path_of[tr] = path
            for link in path.links_used:
                if link not in all_resources:
                    all_resources.append(link)

    all_resources.sort(key=_resource_sort_key)
    tid_of: dict[object, int] = {res: tid for tid, res in enumerate(all_resources)}

    # Timestamp offset so that iteration i-1 starts at t=0 (Perfetto rejects
    # negative timestamps).
    #   rel_iter = -1  →  abs_ts = 0            + slot_starts[s]
    #   rel_iter =  0  →  abs_ts = iter_step     + slot_starts[s]
    #   rel_iter = +1  →  abs_ts = 2*iter_step   + slot_starts[s]
    shown_iterations = [-1, 0, 1]
    _ITER_LABEL = {-1: "i-1", 0: "i", 1: "i+1"}
    ts_offset: float = iter_step  # shift so i-1 starts at t=0

    # ── 3.  Build trace events ───────────────────────────────────────── #
    pid = "steady_state_schedule"
    trace_events: list[dict] = []

    # One thread_name / sort_index metadata event per physical resource
    for res, tid in tid_of.items():
        trace_events.append(
            {
                "name": "thread_name",
                "ph": "M",
                "pid": pid,
                "tid": tid,
                "args": {"name": _resource_label(res)},
            }
        )
        trace_events.append(
            {
                "name": "thread_sort_index",
                "ph": "M",
                "pid": pid,
                "tid": tid,
                "args": {"sort_index": _resource_sort_key(res)},
            }
        )

    # ── 4.  Computation node events ──────────────────────────────────── #
    for node in tta.ssc_nodes:
        slot = tta.slot_of[node]
        eq_node = tta.cost_lut.get_equal_node(node)
        if eq_node is not None:
            lut_cores = tta.cost_lut.get_cores(eq_node)
            node_dur: float = (
                float(ceil(max(tta.cost_lut.get_cost(eq_node, c).latency_total for c in lut_cores)))
                if lut_cores
                else slot_lat[slot]
            )
        else:
            node_dur = slot_lat[slot]

        alloc_cores = list(tta.mapping.get(node).resource_allocation)
        assert len(alloc_cores) == 1, "Should have a single chosen resource allocation"
        compute_allocation = alloc_cores[0]
        for rel_iter in shown_iterations:
            label = _ITER_LABEL[rel_iter]
            abs_start = (rel_iter + 1) * ts_offset + slot_starts[slot]
            for core in compute_allocation:
                if core not in tid_of:
                    continue
                trace_events.append(
                    {
                        "name": f"{node.name} [{label}]",
                        "cat": "computation",
                        "ph": "X",
                        "ts": abs_start,
                        "dur": max(node_dur, 1.0),
                        "pid": pid,
                        "tid": tid_of[core],
                        "cname": _CNAME_COMPUTE,
                        "args": {
                            "iteration": label,
                            "slot": slot,
                            "slot_latency_cycles": slot_lat[slot],
                            "node_latency_cycles": node_dur,
                            "core_id": core.id,
                            "core_type": core.core_type,
                            "col": getattr(core, "col_id", None),
                            "row": getattr(core, "row_id", None),
                        },
                    }
                )

    # ── 5.  Transfer node events ─────────────────────────────────────── #
    # The SSIS (Steady State Iteration Space) is a multi-dimensional loop nest
    # with N_total = prod(sizes) iterations total.  firesC is the number of
    # those iterations that need a fresh DMA transfer; the rest reuse cached
    # data.  Each steady-state iteration in the trace corresponds to ONE
    # iteration of that loop nest — it either fires (needs a transfer) or
    # reuses.  The slot is allocated at slot_lat[s] >= one_transfer_lat to
    # cover the case where a transfer does fire; in reuse iterations the link
    # is idle during that slot.  firesC only appears in the objective as an
    # energy/bandwidth cost across the full run, not as a timing multiplier.
    for node in tta.transfer_nodes:
        slot = tta.slot_of[node]
        chosen_path = path_of.get(node)
        if chosen_path is None:
            continue

        one_transfer_lat = float(tta._transfer_latency_cache[(node, chosen_path)].X)
        ssis = tta.ssis[node]
        reuse_summary = ssis.reuse_summary()  # set by update_transfer_reuse_levels after solve

        is_const_io = tta._is_const_io(node)
        is_const_i = tta._is_const_i(node)
        is_const_o = tta._is_const_o(node)
        cname = _CNAME_TRANSFER_CONST if is_const_io else _CNAME_TRANSFER

        # Computation nodes directly connected to this transfer in the workload
        # graph.  An InEdge (constant weight/bias source) has no computation
        # node predecessor; an OutEdge has no computation node successor.
        input_of: list[str] = [n.name for n in tta.workload.successors(node) if isinstance(n, _ComputationNode)]
        output_of: list[str] = [n.name for n in tta.workload.predecessors(node) if isinstance(n, _ComputationNode)]

        for rel_iter in shown_iterations:
            label = _ITER_LABEL[rel_iter]
            abs_start = (rel_iter + 1) * ts_offset + slot_starts[slot]
            for link in chosen_path.links_used:
                if link not in tid_of:
                    continue
                trace_events.append(
                    {
                        "name": f"{node.name} [{label}]",
                        "cat": "transfer_const" if is_const_io else "transfer",
                        "ph": "X",
                        "ts": abs_start,
                        "dur": max(slot_lat[slot], 1.0),
                        "pid": pid,
                        "tid": tid_of[link],
                        "cname": cname,
                        "args": {
                            "iteration": label,
                            "slot": slot,
                            "slot_latency_cycles": slot_lat[slot],
                            "one_transfer_latency_cycles": one_transfer_lat,
                            "is_const_io": is_const_io,
                            "is_const_input": is_const_i,
                            "is_const_output": is_const_o,
                            "input_of": input_of,
                            "output_of": output_of,
                            "full_path": _path_label(chosen_path),
                            "this_link": _link_label(link),
                            "reuse": reuse_summary,
                        },
                    }
                )

    # ── 6.  Write JSON ───────────────────────────────────────────────── #
    os.makedirs(output_path, exist_ok=True)
    out_file = os.path.join(output_path, filename)

    trace = {
        "traceEvents": trace_events,
        "displayTimeUnit": "ns",  # cycles treated as "ns" for Perfetto labelling
        "otherData": {
            "note": (
                "Timestamps and durations are in hardware clock cycles. "
                "Three representative iterations are shown: "
                "i-1 at t=0, i at t=iter_step, i+1 at t=2*iter_step. "
                "displayTimeUnit='ns' is used as a Perfetto label approximation."
            ),
            "total_iterations": iterations,
            "latency_per_iteration_cycles": latency_per_iteration,
            "overlap_cycles": overlap,
            "iter_step_cycles": iter_step,
            "total_latency_cycles": (iterations * latency_per_iteration - (iterations - 1) * overlap),
            "slot_latencies_cycles": slot_lat,
        },
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(trace, f, indent=2)

    logger.info("Steady-state trace written to %s", out_file)
    return out_file
