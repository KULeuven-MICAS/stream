# time_slot_allocation.py
import json
import sys
from enum import Enum, auto
from io import StringIO
from itertools import count

from pyparsing import Any

from stream.hardware.architecture.core import Core
from stream.hardware.architecture.noc.communication_link import CommunicationLink
from stream.workload.steady_state.computation import SteadyStateComputation
from stream.workload.steady_state.node import SteadyStateNode
from stream.workload.steady_state.tensor import SteadyStateTensor
from stream.workload.steady_state.transfer import SteadyStateTransfer

# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
Resource = Core | tuple[CommunicationLink] | CommunicationLink | None


def _resource_key(res: Resource) -> str:
    if isinstance(res, Core):  # unchanged
        return f"Core {res.id}"
    if isinstance(res, CommunicationLink):
        return str(res)
    if isinstance(res, tuple):  # pretty-print full paths
        return "Path[" + "→".join(_resource_key(link) for link in res) + "]"
    return str(res)


# --------------------------------------------------------------------------- #
# Public enumeration                                                          #
# --------------------------------------------------------------------------- #
class NodeType(Enum):
    WARMUP = auto()
    STEADY_STATE = auto()
    COOLDOWN = auto()


# --------------------------------------------------------------------------- #
# The new generic TimeSlotAllocation                                          #
# --------------------------------------------------------------------------- #
class TimeSlotAllocation:
    """
    A **generic** slot-table:

        * rows   - any resources (`Core`, `CommunicationLink`, …)
        * cols   - integer time-slots
        * cells  - *one* `SteadyStateNode` (or subclass) or empty
    """

    # .................................................... nested exception ...
    class AllocationConflictError(Exception):
        """Raised when the caller tries to double-book a resource/slot."""

    # ........................................................... constructor
    def __init__(
        self,
        allocations: list[tuple[int, Resource, SteadyStateNode]] | None = None,
        node_type: NodeType = NodeType.STEADY_STATE,
    ):
        self.allocations: list[tuple[int, Resource, SteadyStateNode]] = []
        self._slot_res_to_node: dict[int, dict[Resource, SteadyStateNode]] = {}
        self._node_to_res: dict[SteadyStateNode, set[Resource]] = {}
        self._res_max_slot: dict[Resource, int] = {}
        self._node_types: dict[SteadyStateNode, NodeType] = {}

        if allocations:
            for slot, res, node in allocations:
                self._add_alloc(slot, res, node, node_type)

    # ............................................................ properties
    @property
    def slots(self) -> list[int]:
        return sorted(self._slot_res_to_node)

    @property
    def slot_min(self) -> int:
        return min(self._slot_res_to_node, default=0)

    @property
    def slot_max(self) -> int:
        return max(self._slot_res_to_node, default=0)

    @property
    def resources(self) -> list[Resource]:
        return sorted(self._res_max_slot, key=_resource_key)

    @property
    def nodes(self) -> list[SteadyStateNode]:
        return sorted(self._node_to_res, key=lambda n: n.id)

    # .............................................................. low-lvl
    def _add_alloc(
        self,
        slot: int,
        res: Resource,
        node: SteadyStateNode,
        node_type: NodeType = NodeType.STEADY_STATE,
    ):
        # double-booking guard
        if node in self._node_to_res and res in self._node_to_res[node]:
            return  # already booked exactly like this
        if slot in self._slot_res_to_node and res in self._slot_res_to_node[slot]:
            raise self.AllocationConflictError(
                f"slot {slot} / {_resource_key(res)} already holds {self._slot_res_to_node[slot][res].node_name}"
            )

        self.allocations.append((slot, res, node))
        self._slot_res_to_node.setdefault(slot, {})[res] = node
        self._node_to_res.setdefault(node, set()).add(res)
        self._res_max_slot[res] = max(self._res_max_slot.get(res, 0), slot)
        self._node_types[node] = node_type

    # ................................................. public slot helpers
    def add_node_to_next_slot(
        self,
        node: SteadyStateNode,
        res: Resource,
        *,
        min_slot: int = 0,
        node_type: NodeType = NodeType.STEADY_STATE,
    ) -> int:
        """Place *node* on *res* at the first free slot ≥ *min_slot*."""
        next_slot = max(self._res_max_slot.get(res, -1) + 1, min_slot)
        while next_slot in self._slot_res_to_node and res in self._slot_res_to_node[next_slot]:
            next_slot += 1
        self._add_alloc(next_slot, res, node, node_type)
        return next_slot

    def get_allocations_in_slot(self, slot: int) -> dict[Resource, SteadyStateNode]:
        return dict(self._slot_res_to_node.get(slot, {}))

    def get_resources_for_node(self, node: SteadyStateNode) -> set[Resource]:
        return set(self._node_to_res.get(node, set()))

    def get_resources_for_node_id(self, node_id: int) -> set[Resource]:
        nodes = [n for n in self._node_to_res if n.id == node_id]
        return set(res for n in nodes for res in self._node_to_res[n])

    def get_timeslot_of_node_on_resource(self, node: SteadyStateNode, res: Resource) -> int:
        for slot, res_map in self._slot_res_to_node.items():
            if res_map.get(res) is node:
                return slot
        raise ValueError(f"{node.node_name} not on {_resource_key(res)}.")

    def get_timeslot_of_node(self, node: SteadyStateNode) -> int:
        slots = [slot for slot, res_map in self._slot_res_to_node.items() if node in res_map.values()]
        if not slots:
            raise ValueError(f"{node.node_name} not scheduled.")
        return max(slots)

    # ...................................................... visualization
    def visualize_allocation(self, cols_per_row: int = 10):
        """
        Lightweight ASCII dump. Colours show NodeType.
        """
        color = {
            NodeType.WARMUP: "\033[93m",
            NodeType.STEADY_STATE: "\033[92m",
            NodeType.COOLDOWN: "\033[94m",
            "END": "\033[0m",
        }
        col_w = 12
        slots = self.slots or [0]
        rows = self.resources

        for row_slots in [slots[i : i + cols_per_row] for i in range(0, len(slots), cols_per_row)]:
            header = " " * 15 + "".join(f"|{s:^{col_w}}" for s in row_slots) + "|"
            print(header)
            print("-" * len(header))
            for res in rows:
                label = _resource_key(res)
                row = f"{label:<14}"
                for s in row_slots:
                    node = self._slot_res_to_node.get(s, {}).get(res)
                    if node:
                        nt = self._node_types.get(node, NodeType.STEADY_STATE)
                        txt = f"{node.node_name}"
                        row += f"|{color[nt]}{txt:^{col_w}}{color['END']}"
                    else:
                        row += f"|{'':^{col_w}}"
                row += "|"
                print(row)
            print("-" * len(header))

    # ------------------------------------------------------------------ #
    # Perfetto / Chrome-trace export                                     #
    # ------------------------------------------------------------------ #
    def to_perfetto(
        self,
        filepath: str,
        *,
        slot_length: float = 1.0,
        time_unit: str = "us",
        default_runtime: float = 1.0,
    ):
        """
        Convert the allocation to a Perfetto-compatible JSON trace.

        Parameters
        ----------
        slot_length : float, default 1.0
            Wall-clock duration that one timeslot represents **in the chosen
            time unit** ( e.g. 1.0 µs ).
        time_unit : {"us","ms","ns"}, default "us"
            Unit used for *ts* / *dur* in the resulting trace.
        default_runtime : float, default 1.0
            Duration to use for nodes whose ``runtime`` attribute is ``None``.
        as_string : bool, default **True**
            If *True* return a JSON **string**; otherwise return the underlying
            Python ``dict`` so caller can further tweak / dump it.

        Returns
        -------
        str | dict
            Chrome-trace (Perfetto) representation.
        """

        # ---------- helper IDs ----------------------------------------
        pid_gen = count(0)
        pid_of: dict[Resource, int] = {r: next(pid_gen) for r in self.resources}

        # ---------- trace events list ---------------------------------
        trace_events: list[dict[str, Any]] = []

        # ---------- metadata (one thread per resource) ----------------
        for res, pid in pid_of.items():
            trace_events.append(
                {
                    "name": "thread_name",
                    "ph": "M",
                    "pid": pid,
                    "tid": 0,
                    "args": {"name": _resource_key(res)},
                }
            )

        # ---------- main "X" (complete) events ------------------------
        for slot, res, node in self.allocations:
            pid = pid_of[res]
            cat = self._node_types.get(node, NodeType.STEADY_STATE).name
            ts = slot * slot_length
            dur = node.runtime if getattr(node, "runtime", None) not in (None, 0) else default_runtime
            trace_events.append(
                {
                    "name": node.node_name,
                    "cat": cat,
                    "ph": "X",
                    "ts": ts,
                    "dur": dur,
                    "pid": pid,
                    "tid": 0,
                    "args": {
                        "id": node.id,
                        "type": node.type,
                        "slot": slot,
                        "resource": _resource_key(res),
                    },
                }
            )

        # ---------- top-level wrapper --------------------------------
        out = {
            "traceEvents": trace_events,
            "displayTimeUnit": time_unit,
            "systemTraceConfig": {},
            "otherData": {"exported_by": "TimeSlotAllocation.to_perfetto"},
        }
        json_out = json.dumps(out, indent=2)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(json_out)

    # ------------------------------------------------------------------ #
    # latency / overlap utility                                          #
    # ------------------------------------------------------------------ #
    def compute_latency(self, iterations: int = 1) -> tuple[int, int, int]:
        """
        Parameters
        ----------
        iterations : int
            Number of steady-state iterations that will run back-to-back.

        Returns
        -------
        total_latency     : int
            Total latency for *iterations* consecutive iterations.
        latency_per_iter  : int
            Sum of slot-latencies of **this** allocation (single iteration).
        overlap           : int
            Part that can overlap between iterations (= min idle-latency
            across *all* resources).

        Notes
        -----
        ▸ *Start-idle* slots = slots **before** the first busy slot
        on a resource.
        ▸ *End-idle* slots   = slots **after**  the  last busy slot
        on that resource.
        Only those contribute to overlap.
        """

        # ------------------ 1. slot latency (max runtime over resources) ----
        slot_latency: dict[int, int] = {s: 0 for s in range(self.slot_min, self.slot_max + 1)}
        for slot, _, node in self.allocations:
            rt = getattr(node, "runtime", 0) or 0
            slot_latency[slot] = max(slot_latency.get(slot, 0), rt)

        if not slot_latency:  # empty schedule
            return 0, 0, 0

        latency_per_iter = sum(slot_latency[s] for s in range(self.slot_min, self.slot_max + 1))

        # ------------------ 2. idle-latency per resource --------------------
        idle_lat_per_res: dict[Resource, int] = {}

        for res in self.resources:
            # list of busy slots for this resource
            busy = sorted(slot for slot, res_map in self._slot_res_to_node.items() if res in res_map)
            if not busy:
                continue  # never used → ignore

            first_busy, last_busy = busy[0], busy[-1]

            idle_lat = sum(
                slot_latency[s] for s in range(self.slot_min, self.slot_max + 1) if s < first_busy or s > last_busy
            )
            idle_lat_per_res[res] = idle_lat

        if not idle_lat_per_res:
            # should not happen – means no resource ever used
            return latency_per_iter * iterations, latency_per_iter, 0

        overlap = min(idle_lat_per_res.values())

        total_latency = iterations * latency_per_iter - (iterations - 1) * overlap
        return total_latency, latency_per_iter, overlap

    # ............................................................... dunder
    def __repr__(self):
        return f"TimeSlotAllocation({len(self.allocations)} allocs, {len(self.resources)} resources)"

    def __str__(self):
        buff = StringIO()
        stdout, sys.stdout = sys.stdout, buff
        try:
            self.visualize_allocation()
        finally:
            sys.stdout = stdout
        return buff.getvalue()

    # .................................................... node type helpers
    def get_computation_nodes(self):
        """Return all SteadyStateComputation nodes in the allocation."""
        return [n for n in self.nodes if isinstance(n, SteadyStateComputation)]

    def get_transfer_nodes(self):
        """Return all SteadyStateTransfer nodes in the allocation."""
        return [n for n in self.nodes if isinstance(n, SteadyStateTransfer)]

    def get_tensor_nodes(self):
        """Return all SteadyStateTensor nodes in the allocation."""
        return [n for n in self.nodes if isinstance(n, SteadyStateTensor)]
