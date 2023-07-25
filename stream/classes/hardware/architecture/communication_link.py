from math import ceil
import numpy as np

from stream.classes.workload.tensor import Tensor
from stream.classes.cost_model.communication_manager import CommunicationLinkEvent


class BusyTimeViolationException(Exception):
    pass


class IdleTimeViolationException(Exception):
    pass


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
        self.busy_periods = []
        self.idle_periods = [(0, float("inf"))]

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
        # TODO Check when we can actually do the transfer based on start and duration at higher level
        # duration = ceil(tensor.size / self.bandwidth)
        energy_cost = cle.energy

        self.update_busy_periods(cle)
        self.update_idle_periods(cle)
        self.events.append(cle)
        return energy_cost

    def block(
        self,
        start: int,
        duration: int,
        tensors: list,
    ):
        """Block this communication link from start timestep for a given duration.

        Args:
            start (int): The timestep at which the blocking starts.
            duration (int): The duration of the blocking.
            tensors (list): A list of tensors for which we are blocking the link.
        """
        end = start + duration
        # Create a CLEvent
        event = CommunicationLinkEvent(
            type="block",
            start=start,
            end=end,
            tensors=tensors,
            energy=tensors[0].origin.get_offchip_energy(),
        )
        self.update_busy_periods(event)
        self.update_idle_periods(event)
        self.events.append(event)
        return

    def update_busy_periods(self, event: CommunicationLinkEvent):
        start = event.start
        end = event.end
        if start == end:
            return
        busy_starts = [start for (start, _) in self.busy_periods]
        idx = np.searchsorted(busy_starts, start, side="right")
        if idx > 0:
            previous_end = self.busy_periods[idx - 1][1]
            if previous_end > start:
                raise BusyTimeViolationException()
        # Sanity checks
        if idx <= len(self.busy_periods) - 1:
            next_start = self.busy_periods[idx][0]
            if next_start < end:
                raise BusyTimeViolationException()
        # Insert the new busy period
        self.busy_periods.insert(idx, (start, end))

    def update_idle_periods(self, event: CommunicationLinkEvent):
        busy_start = event.start
        busy_end = event.end
        busy_duration = event.duration
        idle_starts = [start for (start, _) in self.idle_periods]
        if busy_duration == 0:
            return
        idx = np.searchsorted(idle_starts, busy_start, side="right") - 1
        assert idx >= 0
        idle_start, idle_end = self.idle_periods[idx]
        if idle_start > busy_start or busy_end > idle_end:
            raise IdleTimeViolationException(
                "Busy period must fall within idle period."
            )
        if idle_end - idle_start < busy_duration:
            raise IdleTimeViolationException(
                "Busy period must fall within idle period."
            )
        if idle_start == busy_start:
            new_idle_period = (busy_end, idle_end)
            self.idle_periods[idx] = new_idle_period
        elif idle_end == busy_end:
            new_idle_period = (idle_start, busy_start)
            self.idle_periods[idx] = new_idle_period
        else:
            new_idle_periods = [(idle_start, busy_start), (busy_end, idle_end)]
            del self.idle_periods[idx]
            self.idle_periods.insert(idx, new_idle_periods[1])
            self.idle_periods.insert(idx, new_idle_periods[0])
