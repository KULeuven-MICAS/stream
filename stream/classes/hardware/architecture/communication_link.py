from math import ceil

from stream.classes.workload.tensor import Tensor

SAVE_ACTIVE_PERIODS = True
SAVE_BLOCKED_PERIODS = True

class CommunicationLink:
    """Represents a fixed-bandwidth communication link used to communicate between two cores.
    """
    def __init__(self, sender, receiver, bandwidth, unit_energy_cost, bidirectional=False) -> None:
        self.sender = sender
        self.receiver = receiver
        self.bandwidth = bandwidth
        self.unit_energy_cost = unit_energy_cost
        self.bidirectional = bidirectional
        self.available_from = 0  # from which timestep this link is available
        if SAVE_ACTIVE_PERIODS:
            self.active_periods = []  # will contain from when to when this link was active
        if SAVE_BLOCKED_PERIODS:
            self.blocked_periods = []  # will contain the ranges the port is blocked due toe execution of node with too large operand

    def __str__(self) -> str:
        return f"CommunicationLink({self.sender}, {self.receiver}, bw={self.bandwidth})"

    def __repr__(self) -> str:
        return str(self)

    def __hash__(self) -> int:
        return hash((self.sender, self.receiver, self.bandwidth, self.unit_energy_cost, self.bidirectional))

    def __eq__(self, other) -> bool:
        return str(self) == str(other)

    def get_name_for_schedule_plot(self) -> str:
        if self.bidirectional:
            return f"{self.sender} <-> {self.receiver}"
        else:
            return f"{self.sender} -> {self.receiver}"

    def is_available(self, timestep):
        if timestep < self.available_from:
            return False
        return True

    def put(self, data: Tensor, timestep: int) -> tuple[int, float]:
        """Put data on this communication link at timestep.

        Args:
            data_size (int): The size of the packet in number of bits
            timestep (int): The timestep in clock cyles to start the data transfer

        Returns:
            int: The end time when communication on this link is finished
        """
        start_timestep = max(self.available_from, timestep)
        duration = ceil(data.size/self.bandwidth)
        energy_cost = self.unit_energy_cost * duration
        end_timestep = start_timestep + duration
        self.update_available_time(end_timestep)
        if SAVE_ACTIVE_PERIODS:
            self.active_periods.append((start_timestep, end_timestep, data.layer_operand, data.origin.id))
        return end_timestep, energy_cost

    def update_available_time(self, new_available_time: int) -> None:
        """Update communication link available time.

        Args:
            new_available_time (int): The updated available time for the communication link
        """
        self.available_from = new_available_time

    def block(self, start_timestep: int, blocking_time: int, cn_id: tuple):
        """Block this communication link from start timestep for a given duration.

        Args:
            start_timestep (int): The timestep at which the port ideally starts being blocked.
            blocking_time (int): The duration for which the link should be blocked

        Returns:
            int: The start time at which we can effectively start blocking the port.
            int: The end time at which the blocking ends.
        """
        effective_blocking_start_timestep = max(self.available_from, start_timestep)
        effective_blocking_end_timestep = effective_blocking_start_timestep + blocking_time
        self.update_available_time(effective_blocking_end_timestep)
        if SAVE_BLOCKED_PERIODS:
            self.blocked_periods.append((effective_blocking_start_timestep, effective_blocking_end_timestep, cn_id))
        return effective_blocking_start_timestep, effective_blocking_end_timestep
