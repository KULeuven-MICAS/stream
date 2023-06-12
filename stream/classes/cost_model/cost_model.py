from networkx import DiGraph

from stream.classes.hardware.architecture.accelerator import Accelerator
from stream.classes.cost_model.scheduler import schedule_graph
from stream.visualization.memory_usage import plot_memory_usage
from stream.visualization.schedule import plot_timeline_brokenaxes


class StreamCostModelEvaluation:
    """Stream's cost model evaluation class which includes a scheduler and memory utilization tracer.
    Throughout SCME will be used as abbreviation.
    This evaluation computes the total latency and activation memory utilization throughout the inference.
    """

    def __init__(
        self,
        workload: DiGraph,
        accelerator: Accelerator,
        scheduler_candidate_selection: str,
        operands_to_prefetch: list,
    ) -> None:
        # Initialize the SCME by setting the workload graph to be scheduled
        self.workload = workload
        self.accelerator = accelerator
        self.energy = None
        self.total_cn_onchip_energy = None
        self.total_cn_offchip_link_energy = None
        self.total_cn_offchip_memory_energy = None
        self.total_eviction_to_offchip_link_energy = None
        self.total_eviction_to_offchip_memory_energy = None
        self.total_sink_layer_output_offchip_link_energy = None
        self.total_sink_layer_output_offchip_memory_energy = None
        self.total_core_to_core_link_energy = None
        self.total_core_to_core_memory_energy = None

        self.latency = None
        self.max_memory_usage = None
        self.core_timesteps_delta_cumsums = None
        self.scheduler_candidate_selection = scheduler_candidate_selection
        self.operands_to_prefetch = operands_to_prefetch

    def __str__(self):
        return f"SCME(energy={self.energy:.2e}, latency={self.latency:.2e})"

    def run(self):
        """Run the SCME by scheduling the graph through time. The scheduler takes into account inter-core data movement and also tracks energy and memory through the memory manager.
        This assumes each node in the graph has an energy and runtime of the core to which they are allocated to.
        """
        results = schedule_graph(
            self.workload,
            self.accelerator,
            candidate_selection=self.scheduler_candidate_selection,
            operands_to_prefetch=self.operands_to_prefetch,
        )
        self.latency = results[0]
        self.total_cn_onchip_energy = results[1]
        self.total_cn_offchip_link_energy, self.total_cn_offchip_memory_energy = (
            results[2],
            results[3],
        )
        (
            self.total_eviction_to_offchip_link_energy,
            self.total_eviction_to_offchip_memory_energy,
        ) = (results[4], results[5])
        (
            self.total_sink_layer_output_offchip_link_energy,
            self.total_sink_layer_output_offchip_memory_energy,
        ) = (results[6], results[7])
        self.total_core_to_core_link_energy, self.total_core_to_core_memory_energy = (
            results[8],
            results[9],
        )

        self.energy = (
            self.total_cn_onchip_energy
            + self.total_cn_offchip_link_energy
            + self.total_cn_offchip_memory_energy
            + self.total_eviction_to_offchip_link_energy
            + self.total_eviction_to_offchip_memory_energy
            + self.total_sink_layer_output_offchip_link_energy
            + self.total_sink_layer_output_offchip_memory_energy
            + self.total_core_to_core_link_energy
            + self.total_core_to_core_memory_energy
        )

    def plot_schedule(
        self,
        plot_full_schedule=False,
        draw_dependencies=True,
        plot_data_transfer=False,
        section_start_percent=(0, 50, 95),
        percent_shown=(5, 5, 5),
        fig_path="outputs/schedule_plot.png",
    ):
        """Plot the schedule of this SCME."""
        if plot_full_schedule:
            section_start_percent = (0,)
            percent_shown = (100,)
        plot_timeline_brokenaxes(
            self.workload,
            self.accelerator,
            draw_dependencies,
            section_start_percent,
            percent_shown,
            plot_data_transfer,
            fig_path,
        )

    def plot_memory_usage(self, fig_path="outputs/memory_usage_plot.png"):
        """Plot the memory usage of this SCME."""
        plot_memory_usage(self.accelerator.memory_manager, fig_path)
