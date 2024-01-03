import numpy as np

from stream.classes.cost_model.communication_manager import CommunicationLinkEvent


class CommunicationLink:
    """Represents a fixed-bandwidth communication link used to communicate between two cores."""

    def __init__(
        self, sender, receiver, bandwidth, unit_energy_cost, bidirectional=False
    ) -> None:
        self.sender = sender
        self.receiver = receiver
        self.bandwidth = bandwidth
        self.unit_energy_cost = unit_energy_cost
        self.bidirectional = bidirectional

        self.events = []
        self.active_periods = [(0, float("inf"), 0)]
        self.active_ts = np.array([0, float("inf")])
        self.active_deltas = np.array([0, 0])
        self.tensors = {}

    def __str__(self) -> str:
        return f"CommunicationLink({self.sender}, {self.receiver}, bw={self.bandwidth})"

    def __repr__(self) -> str:
        return str(self)

    def __hash__(self) -> int:
        return hash(
            (
                self.sender,
                self.receiver,
                self.bandwidth,
                self.unit_energy_cost,
                self.bidirectional,
            )
        )

    def __eq__(self, other) -> bool:
        return (self.sender, self.receiver, self.bandwidth) == (
            other.sender,
            other.receiver,
            other.bandwidth,
        )

    def get_name_for_schedule_plot(self) -> str:
        if self.bidirectional:
            return f"{self.sender} <-> {self.receiver}"
        else:
            return f"{self.sender} -> {self.receiver}"

    def transfer(self, cle: CommunicationLinkEvent) -> float:
        """Transfer data on this communication link at timestep.
        The transfer can take longer than necessary for this link if another lower-bandwidth link is involved.

        Args:
            tensor (Tensor): The tensor to be transferred.
            start (int): The timestep in clock cyles to start the transfer.
            duration (int): The duration of the transfer.

        Returns:
            int: The end time when communication on this link is finished
        """
        energy_cost = cle.energy
        self.update_activity(cle)
        return energy_cost

    def block(
        self,
        start: int,
        duration: int,
        tensors: list,
        activity: int = 100,
    ):
        """Block this communication link from start timestep for a given duration.

        Args:
            start (int): The timestep at which the blocking starts.
            duration (int): The duration of the blocking.
            tensors (list): A list of tensors for which we are blocking the link.
            activity (int): The bandwidth activity in bits/cc.
        """
        end = start + duration
        # Create a CLEvent
        event = CommunicationLinkEvent(
            type="block",
            start=start,
            end=end,
            tensors=tensors,
            energy=tensors[0].origin.get_offchip_energy(),
            activity=activity,
        )
        self.update_activity(event)
        return

    def update_activity(self, event: CommunicationLinkEvent):
        start = event.start
        end = event.end
        activity = event.activity
        if start == end:
            return
        # Check if this is a duplicate event for broadcast
        for tensor in event.tensors:
            previous_events = self.tensors.get(tensor, [])
            if any((previous_event.start == event.start for previous_event in previous_events)):
                return
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
        for tensor in event.tensors:
            self.tensors[tensor] = self.tensors.get(tensor, []) + [event]
        self.events.append(event)

    def get_idle_window(self, activity, duration, earliest_t, tensors):
        """
        Get the earliest time window of duration 'duration' from 'earliest_t'
        with atleast 'activity' percent available.
        """
        valid_windows = []
        ## Check if this tensor has already been transferred on this link before
        # If so, check duration and earliest timestep requirements of this call
        for tensor in tensors:
            if tensor in self.tensors:
                previous_events = self.tensors[tensor]
                for previous_event in previous_events:
                    # Previous event needs to be long enough
                    duration_valid = previous_event.duration >= duration
                    # Previous event needs to have happened at late enough time
                    earliest_t_valid = previous_event.start >= earliest_t
                    if duration_valid and earliest_t_valid:
                        valid_windows.append((previous_event.start, previous_event.end))
        ## Check other possible periods given the activity
        activities = np.cumsum(self.active_deltas)
        earliest_t_index = np.searchsorted(self.active_ts, earliest_t, side="right")
        relevant_ts = self.active_ts[earliest_t_index:]
        updated_ts = relevant_ts.copy()
        relevant_activities = activities[earliest_t_index:]
        # Insert the earliest timestep and the activity at that timestep
        updated_ts = np.insert(updated_ts, 0, earliest_t)
        updated_activities = np.insert(relevant_activities, 0, activities[earliest_t_index - 1])
        updated_activities = updated_activities + activity
        idxs = np.argwhere(updated_activities > self.bandwidth)
        idxs = [idx[0] for idx in idxs]
        idxs.append(len(updated_ts) - 1)
        start = earliest_t
        for idx in idxs:
            end = updated_ts[idx]
            if end - start >= duration:
                valid_windows.append((start, end))
            try:
                start = updated_ts[idx+1]
            except:
                break
        if not valid_windows:
            raise ValueError(f"There are no valid windows of activity {activity} and duration {duration} for {self}.")
        return valid_windows