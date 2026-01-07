from itertools import combinations, product
from typing import TYPE_CHECKING, Literal

import numpy as np

from stream.cost_model.communication_manager import CommunicationLinkEvent

if TYPE_CHECKING:
    from stream.hardware.architecture.core import Core
    from stream.workload.tensor import SubviewTensor

ENABLE_BROADCASTING = False


def get_bidirectional_edges(
    core_a: "Core",
    core_b: "Core",
    bandwidth: float,
    unit_energy_cost: float,
    link_type: Literal["bus"] | Literal["link"],
    bus_instance: "CommunicationLink | None" = None,
) -> list[tuple["Core", "Core", dict[str, "CommunicationLink"]]]:
    """Create a list with two edges: from A to B and B to A."""
    bus = bus_instance or CommunicationLink("Any", "Any", bandwidth, unit_energy_cost, bidirectional=True)
    link_a_to_b = CommunicationLink(core_a, core_b, bandwidth, unit_energy_cost)
    link_b_to_a = CommunicationLink(core_b, core_a, bandwidth, unit_energy_cost)

    # if have_shared_memory(core_a, core_b):
    #     # No edge if the cores have a shared memory
    #     return []

    return [
        #  A -> B
        (
            core_a,
            core_b,
            {"cl": bus if link_type == "bus" else link_a_to_b},
        ),
        # B -> A
        (
            core_b,
            core_a,
            {"cl": bus if link_type == "bus" else link_b_to_a},
        ),
    ]


class CommunicationLink:
    """Represents a fixed-bandwidth communication link used to communicate between two cores."""

    def __init__(
        self,
        sender: "Core | Literal['Any']",
        receiver: "Core | Literal['Any']",
        bandwidth: int | float,
        unit_energy_cost: float,
        bidirectional: bool = False,
    ) -> None:
        self.sender = sender
        self.receiver = receiver
        self.bandwidth = bandwidth
        self.unit_energy_cost = unit_energy_cost
        self.bidirectional = bidirectional  # TODO this property is not in use?

        self.events: list[CommunicationLinkEvent] = []
        self.active_periods = [(0, float("inf"), 0)]
        self.active_ts = np.array([0, float("inf")])
        self.active_deltas = np.array([0, 0])
        self.previously_seen_tensors: dict[SubviewTensor, list[CommunicationLinkEvent]] = {}

    def __str__(self) -> str:
        return f"CL({self.sender}, {self.receiver}, bw={self.bandwidth})"

    def __repr__(self) -> str:
        return str(self)

    def __hash__(self) -> int:
        return hash((self.sender, self.receiver, self.bandwidth, self.unit_energy_cost, self.bidirectional))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, CommunicationLink) and (self.sender, self.receiver, self.bandwidth) == (
            other.sender,
            other.receiver,
            other.bandwidth,
        )

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, CommunicationLink):
            return NotImplemented
        sender_self_id = self.sender.id
        sender_other_id = other.sender.id
        if sender_self_id != sender_other_id:
            return sender_self_id < sender_other_id

        receiver_self_id = self.receiver.id
        receiver_other_id = other.receiver.id
        return receiver_self_id < receiver_other_id

    def get_name_for_schedule_plot(self) -> str:
        if self.bidirectional:
            return f"{self.sender} <-> {self.receiver}"
        else:
            return f"{self.sender} -> {self.receiver}"

    def transfer(self, link_event: CommunicationLinkEvent) -> float:
        """Transfer data on this communication link at timestep.
        The transfer can take longer than necessary for this link if another lower-bandwidth link is involved.

        Args:
            tensor : The tensor to be transferred.
            start : The timestep in clock cyles to start the transfer.
            duration : The duration of the transfer.

        Returns:
            int: The end time when communication on this link is finished
        """
        energy_cost = link_event.energy
        self.update_activity(link_event)
        return energy_cost

    def block(
        self,
        start: int,
        duration: int,
        tensors: list["SubviewTensor"],
        activity: int = 100,
        source: "Core" = None,
        destinations: "Core" = None,
    ):
        """Block this communication link from start timestep for a given duration.

        Args:
            start: The timestep at which the blocking starts.
            duration: The duration of the blocking.
            tensors: A list of tensors for which we are blocking the link.
            activity: The percentage of the link bandwidth used
        """
        end = start + duration
        # Create a CLEvent
        event = CommunicationLinkEvent(
            type="block",
            start=start,
            end=end,
            tensors=tensors,
            energy=tensors[0].cn_source.get_offchip_energy(),
            activity=activity,
            source=source,
            destinations=destinations,
        )
        self.update_activity(event)

    def update_activity(self, event: CommunicationLinkEvent):
        start = event.start
        end = event.end
        activity = event.activity
        if start == end:
            return
        # Check if this is a duplicate event for broadcast
        previous_events = (
            self.previously_seen_tensors[event.tensors] if event.tensors in self.previously_seen_tensors else []
        )
        if any(previous_event.start == event.start for previous_event in previous_events):
            return False
        idx_start = np.searchsorted(self.active_ts, start)
        if self.active_ts[idx_start] == start:
            self.active_deltas[idx_start] += activity
        else:
            self.active_ts = np.insert(self.active_ts, idx_start, start)
            self.active_deltas = np.insert(self.active_deltas, idx_start, activity)
        idx_end = np.searchsorted(self.active_ts, end)
        if self.active_ts[idx_end] == end:
            self.active_deltas[idx_end] -= activity
        else:
            self.active_ts = np.insert(self.active_ts, idx_end, end)
            self.active_deltas = np.insert(self.active_deltas, idx_end, -activity)
        # Track that this link has transferred the tensors of this event for future broadcasts
        self.previously_seen_tensors[event.tensors] = previous_events + [event]
        self.events.append(event)

    def get_idle_window(
        self,
        bandwidth_per_tensor: list[tuple["SubviewTensor", int]],
        duration: int,
        earliest_t: int,
    ):
        """
        Get the earliest time window of duration `duration` from `earliest_t` with at least `activity` percent
        available.
        """

        def find_valid_window_for_given_bw(required_bandwidth: int):
            valid_windows: list[tuple[int, int]] = []

            # Check other possible periods given the activity
            activities = np.cumsum(self.active_deltas)
            earliest_t_index = np.searchsorted(self.active_ts, earliest_t, side="right")
            relevant_ts = self.active_ts[earliest_t_index:]
            updated_ts = relevant_ts.copy()
            relevant_activities = activities[earliest_t_index:]
            # Insert the earliest timestep and the activity at that timestep
            updated_ts = np.insert(updated_ts, 0, earliest_t)
            updated_activities = np.insert(relevant_activities, 0, activities[earliest_t_index - 1])
            updated_activities = updated_activities + required_bandwidth
            idxs = np.argwhere(updated_activities > self.bandwidth)
            idxs = [idx[0] for idx in idxs]
            idxs.append(len(updated_ts) - 1)
            start = earliest_t
            for idx in idxs:
                end: int = updated_ts[idx]  # type: ignore
                if end - start >= duration:
                    valid_windows.append((start, end))
                try:
                    start: int = updated_ts[idx + 1]  # type: ignore
                except IndexError:
                    break

            if not valid_windows:
                raise ValueError(
                    f"There are no valid windows of activity {required_bandwidth} and duration {duration} for {self}."
                )
            return valid_windows

        def get_previous_valid_windows(tensor: "SubviewTensor"):
            windows: list[tuple[int, int]] = []
            if tensor in self.previously_seen_tensors:
                previous_events = self.previously_seen_tensors[tensor]
                for previous_event in previous_events:
                    # Previous event needs to be long enough
                    duration_valid = previous_event.duration >= duration
                    # Previous event needs to have happened at late enough time
                    earliest_t_valid = previous_event.start >= earliest_t
                    if duration_valid and earliest_t_valid:
                        windows.append((previous_event.start, previous_event.end))
            return windows

        def window_has_bandwidth_left(window: tuple[int, int], remaining_req_bw: int):
            if remaining_req_bw == 0:
                return True

            start, end = window
            assert start in self.active_ts and end in self.active_ts
            start_idx = np.where(self.active_ts == start)[0]
            end_idx = np.where(self.active_ts == end)[0]
            activities = np.cumsum(self.active_deltas)
            activities_in_window = activities[start_idx:end_idx]
            return all(activities_in_window + remaining_req_bw <= self.bandwidth)

        tensors = [tensor for tensor, _ in bandwidth_per_tensor]
        valid_windows_per_tensor = {
            tensor: get_previous_valid_windows(tensor) for tensor in tensors if get_previous_valid_windows(tensor) != []
        }

        # Check all previously seen window combinations:
        all_valid_windows: list[tuple[int, int]] = []
        for r in range(1, len(valid_windows_per_tensor) + 1, -1)[::-1]:
            # e.g. if 3 tensors have been seen before, check the windows for tensors (1,2,3), (1,2), (2,3), (1,3), ...
            for tensor_combination in combinations(valid_windows_per_tensor, r):
                # Bandwidth that needs to be allocated, for tensors not in the previously registered window
                remaining_req_bw = sum([bw for tensor, bw in bandwidth_per_tensor if tensor not in tensor_combination])
                all_window_combinations = product(*[valid_windows_per_tensor[tensor] for tensor in tensor_combination])
                for window_combination in all_window_combinations:
                    curr_window = window_combination[0]
                    # Windows must overlap exactly and have bandwidth left
                    if all(window == curr_window for window in window_combination[1::]) and window_has_bandwidth_left(
                        curr_window, remaining_req_bw
                    ):
                        all_valid_windows.append(curr_window)

        # If valid windows have been found in previously registered windows, return those
        if all_valid_windows:
            return all_valid_windows

        # Base case: don't assume previous transfers and find new window for all tensors
        total_req_bw = sum([bw for _, bw in bandwidth_per_tensor])
        return find_valid_window_for_given_bw(total_req_bw)
