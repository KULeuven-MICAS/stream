from zigzag.datatypes import LayerOperand

from stream.hardware.architecture.accelerator import Accelerator
from stream.workload.workload import Workload


class StreamCostModelEvaluation:
    """
    Evaluates the cost model for a given workload and accelerator using the Schedule class.
    Computes latency and various energy metrics by running the schedule.
    """

    def __init__(
        self,
        workload: Workload,
        accelerator: Accelerator,
        operands_to_prefetch: list[LayerOperand],
        scheduling_order: list[tuple[int, int]],
    ) -> None:
        # Initialize the SCME by setting the workload graph to be scheduled
        self.workload = workload
        self.accelerator = accelerator
        self.energy: float | None = None
        self.total_cn_onchip_energy: float | None = None
        self.total_cn_offchip_link_energy: float | None = None
        self.total_cn_offchip_memory_energy: float | None = None
        self.total_eviction_to_offchip_link_energy: float | None = None
        self.total_eviction_to_offchip_memory_energy: float | None = None
        self.total_sink_layer_output_offchip_link_energy: float | None = None
        self.total_sink_layer_output_offchip_memory_energy: float | None = None
        self.total_core_to_core_link_energy: float | None = None
        self.total_core_to_core_memory_energy: float | None = None

        self.latency: int | None = None
        self.max_memory_usage = None
        self.core_timesteps_delta_cumsums = None
        self.operands_to_prefetch = operands_to_prefetch
        self.scheduling_order = scheduling_order

    def __str__(self):
        return f"SCME(energy={self.energy:.2e}, latency={self.latency:.2e})"

    def evaluate(self):
        """
        Runs the scheduling and cost model evaluation, updating latency and energy attributes.
        Uses the Schedule class for modular scheduling and result extraction.
        """
        # schedule = CoalaScheduler(
        #     g=self.workload,
        #     accelerator=self.accelerator,
        #     scheduling_order=self.scheduling_order,
        #     operands_to_prefetch=self.operands_to_prefetch,
        # )
        # schedule.run()

        # self.latency = schedule.latency
        # self.total_cn_onchip_energy = schedule.total_cn_onchip_energy
        # self.total_cn_offchip_link_energy = schedule.total_cn_offchip_link_energy
        # self.total_cn_offchip_memory_energy = schedule.total_cn_offchip_memory_energy
        # self.total_eviction_to_offchip_link_energy = schedule.total_eviction_to_offchip_link_energy
        # self.total_eviction_to_offchip_memory_energy = schedule.total_eviction_to_offchip_memory_energy
        # self.total_sink_layer_output_offchip_link_energy = schedule.total_sink_layer_output_offchip_link_energy
        # self.total_sink_layer_output_offchip_memory_energy = schedule.total_sink_layer_output_offchip_memory_energy
        # self.total_core_to_core_link_energy = schedule.total_core_to_core_link_energy
        # self.total_core_to_core_memory_energy = schedule.total_core_to_core_memory_energy

        # self.energy = (
        #     self.total_cn_onchip_energy
        #     + self.total_cn_offchip_link_energy
        #     + self.total_cn_offchip_memory_energy
        #     + self.total_eviction_to_offchip_link_energy
        #     + self.total_eviction_to_offchip_memory_energy
        #     + self.total_sink_layer_output_offchip_link_energy
        #     + self.total_sink_layer_output_offchip_memory_energy
        #     + self.total_core_to_core_link_energy
        #     + self.total_core_to_core_memory_energy
        # )
