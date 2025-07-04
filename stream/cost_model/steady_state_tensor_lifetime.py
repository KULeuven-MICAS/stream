from dataclasses import dataclass
from typing import Any

import numpy as np
import seaborn
from matplotlib import patches
from matplotlib import pyplot as plt

from stream.workload.steady_state.node import SteadyStateNode
from stream.workload.steady_state.tensor import SteadyStateTensor
from stream.workload.steady_state.workload import SteadyStateWorkload


@dataclass(frozen=True)
class TensorLifetime:
    tensor: SteadyStateTensor
    memory: Any  # Typically the chosen_resource_allocation (e.g., core or memory id)
    start: int  # Time when tensor is produced (on its memory)
    end: int  # Time after last consumption (on its memory)


class TensorLifetimeAnalyzer:
    """
    Analyzes tensor lifetimes and memory usage in a SteadyStateWorkload, accounting for parallelism across resources.
    """

    def __init__(self, workload: SteadyStateWorkload):
        self.workload = workload
        self.tensor_nodes: list[SteadyStateTensor] = workload.tensor_nodes
        self.lifetimes: list[TensorLifetime] = []
        self.node_start_times: dict[SteadyStateNode, int] = {}
        self.node_finish_times: dict[SteadyStateNode, int] = {}
        self.resource_times: dict[Any, int] = {}
        self._schedule_nodes_per_resource()
        self._analyze_lifetimes()

    def _schedule_nodes_per_resource(self):
        """
        Assigns a finish time to each node, per resource, simulating parallel execution.
        Assumes each node has a .runtime property (default 1 if missing).
        """
        # Group nodes by their chosen_resource_allocation
        resource_to_nodes: dict[Any, list[SteadyStateNode]] = {}
        for node in self.workload.nodes():
            resource = getattr(node, "chosen_resource_allocation", None)
            resource_to_nodes.setdefault(resource, []).append(node)
        # For each resource, schedule nodes in topological order (local to that resource)
        for resource in resource_to_nodes.keys():
            self.resource_times[resource] = 0
        for node in self.workload.topological_sort():
            runtime = getattr(node, "runtime", None)
            if runtime is None:
                runtime = 1
            # Start time is max of finish times of all predecessors (on any resource)
            preds = list(self.workload.predecessors(node))
            if preds:
                start_time = max(self.node_finish_times.get(pred, 0) for pred in preds)
            else:
                start_time = 0
            start_time = max(start_time, self.resource_times[resource])
            finish_time = start_time + runtime
            self.node_start_times[node] = start_time
            self.node_finish_times[node] = finish_time
            self.resource_times[resource] = finish_time

    def _analyze_lifetimes(self):
        """
        For each tensor node, determine its production and last consumption time based on actual node finish times.
        """
        for tensor in self.tensor_nodes:
            # Produced by its predecessor (or itself if source)
            preds = list(self.workload.predecessors(tensor))
            if preds:
                prod_time = min(self.node_start_times.get(pred, 0) for pred in preds)
            else:
                prod_time = 0
            # Last consumed by its successors
            succs = list(self.workload.successors(tensor))
            if succs:
                cons_time = max(self.node_finish_times.get(succ, prod_time) for succ in succs)
            else:
                cons_time = prod_time
            mem = tensor.chosen_resource_allocation
            self.lifetimes.append(TensorLifetime(tensor, mem, prod_time, cons_time))

    def get_lifetimes_by_memory(self) -> dict[Any, list[TensorLifetime]]:
        """Group tensor lifetimes by their memory (resource)."""
        mem_map: dict[Any, list[TensorLifetime]] = {}
        for lt in self.lifetimes:
            mem_map.setdefault(lt.memory, []).append(lt)
        return mem_map

    def get_memory_usage_over_time(self, memory: Any) -> tuple[list[int], list[int]]:
        """
        For a given memory, return (times, memory_usage) lists.
        """
        events = []  # (time, size_change)
        for lt in self.get_lifetimes_by_memory().get(memory, []):
            size = getattr(lt.tensor, "size", 0)
            events.append((lt.start, size))
            events.append((lt.end + 1, -size))  # +1: dead after last use
        # Aggregate events
        events.sort()
        times = []
        mem_usage = []
        current = 0
        for t, delta in events:
            if times and t == times[-1]:
                mem_usage[-1] += delta
            else:
                times.append(t)
                mem_usage.append(current + delta)
            current = mem_usage[-1]
        return times, mem_usage

    def get_max_memory_usage(self, memory: Any) -> tuple[int, int]:
        times, usage = self.get_memory_usage_over_time(memory)
        if not usage:
            return 0, 0
        max_idx = int(np.argmax(usage))
        return times[max_idx], usage[max_idx]

    def get_alive_tensors_at(self, memory: Any, time: int) -> list[SteadyStateTensor]:
        return [lt.tensor for lt in self.get_lifetimes_by_memory().get(memory, []) if lt.start <= time <= lt.end]

    def summary(self) -> None:
        print("Tensor lifetimes by memory:")
        for mem, lifetimes in self.get_lifetimes_by_memory().items():
            print(f"\nMemory {mem}:")
            for lt in lifetimes:
                print(f"  Tensor {lt.tensor.node_name}: [{lt.start}, {lt.end}] size={getattr(lt.tensor, 'size', 0)}")
            t, m = self.get_max_memory_usage(mem)
            print(f"  Max usage: {m} at time {t}")

    def visualize(self, figsize: tuple[int, int] = (10, 6), filename_prefix: str = "tensor_lifetime") -> None:
        """
        For each memory/resource, plot a Gantt chart:
        - Top: computation/transfer nodes scheduled on that resource (not tensor nodes)
        - Bottom: tensor lifetimes (rectangles) for tensors stored in that memory
        """
        color_palette = seaborn.color_palette("pastel")
        tensor_color_map = {}
        node_color_map = {}
        all_tensors = [lt.tensor for lt in self.lifetimes]
        all_nodes = [n for n in self.workload.nodes() if n not in all_tensors]
        for i, tensor in enumerate(all_tensors):
            tensor_color_map[tensor] = color_palette[i % len(color_palette)]
        for i, node in enumerate(all_nodes):
            node_color_map[node] = color_palette[i % len(color_palette)]

        for mem, lifetimes in self.get_lifetimes_by_memory().items():
            fig, ax = plt.subplots(figsize=figsize)
            # 1. Plot tensor lifetimes (bottom): rectangles for each tensor
            y_base = 0
            y_gap = 0
            for lt in sorted(lifetimes, key=lambda i: i.start):
                start = lt.start
                end = lt.end
                color = tensor_color_map.get(lt.tensor, "#bbbbbb")
                height = lt.tensor.size
                ax.add_patch(
                    patches.Rectangle((start, y_base), end - start, height, facecolor=color, edgecolor="black")
                )
                text_x = (start + end) / 2
                text_y = y_base + height / 2
                text = lt.tensor.node_name
                ax.text(text_x, text_y, text, ha="center", va="center", fontsize=9)
                y_base += height + y_gap
            # 2. Plot schedule (top): all computation/transfer nodes on this resource
            nodes_on_mem = [
                n
                for n in self.workload.nodes()
                if getattr(n, "chosen_resource_allocation", None) == mem and n not in all_tensors
            ]
            nodes_on_mem = sorted(
                nodes_on_mem, key=lambda n: (self.node_start_times.get(n, 0), self.node_finish_times.get(n, 0))
            )
            y_sched = y_base
            height_sched = y_base / 6  # Fixed height for schedule relative to the tensors
            for node in nodes_on_mem:
                start = self.node_start_times.get(node, 0)
                end = self.node_finish_times.get(node, 0)
                color = node_color_map.get(node, "#cccccc")
                ax.add_patch(
                    patches.Rectangle((start, y_sched), end - start, height_sched, facecolor=color, edgecolor="black")
                )
                text = node.plot_name
                text_x = (start + end) / 2
                text_y = y_sched + height_sched / 2
                ax.text(text_x, text_y, text, ha="center", va="bottom", fontsize=9)
            # 3. Formatting
            max_time = max(self.node_finish_times.values(), default=1)
            ax.set_xlim(0, max_time + 1)
            ax.set_ylim(0, y_base + height_sched)
            ax.set_yticks([])
            ax.set_xlabel("Time")
            ax.set_ylabel("Tensors (bottom) / Schedule (top)")
            ax.grid(True, axis="x", linestyle="--", alpha=0.5)
            # 4. Save
            fname = f"{filename_prefix}_mem{mem}.png"
            fig.tight_layout()
            fig.savefig(fname, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved tensor lifetime visualization for memory {mem} to {fname}")
