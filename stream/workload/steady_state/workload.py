from typing import Any

import networkx as nx
from networkx.drawing.nx_pydot import to_pydot  # type: ignore
from zigzag.utils import DiGraphWrapper

from stream.opt.allocation.constraint_optimization.timeslot_allocation import Resource, TimeSlotAllocation
from stream.workload.steady_state.computation import SteadyStateComputation
from stream.workload.steady_state.node import SteadyStateNode
from stream.workload.steady_state.rolling_buffer import SteadyStateRollingBuffer
from stream.workload.steady_state.tensor import SteadyStateTensor, TensorFlag
from stream.workload.steady_state.transfer import SteadyStateTransfer


class SteadyStateWorkload(DiGraphWrapper[SteadyStateNode]):
    """Workload graph for steady state scheduling, supporting multiple node types."""

    def __init__(self, **attr: Any):
        super().__init__(**attr)

    def __str__(self) -> str:
        return (
            f"SteadyStateWorkload("
            f"computations={len(self.computation_nodes)}, "
            f"transfers={len(self.transfer_nodes)}, "
            f"tensors={len(self.tensor_nodes)}, "
            f"edges={self.number_of_edges()})"  # type: ignore
        )

    def __repr__(self) -> str:
        return str(self)

    def add(self, node_obj: SteadyStateNode):
        self.add_node(node_obj)
        # Edges should be added externally as needed

    def add_edge(self, u: SteadyStateNode, v: SteadyStateNode, **attrs: Any):
        """Add an edge between two nodes in the workload graph."""
        super().add_edges_from(
            [
                (u, v, attrs),
            ]
        )

    def get_edges(
        self, data: bool = False
    ) -> list[tuple[SteadyStateNode, SteadyStateNode]] | list[tuple[SteadyStateNode, SteadyStateNode, dict[Any, Any]]]:
        """Return all edges in the workload graph."""
        return list(self.edges(data=data))  # type: ignore

    @property
    def computation_nodes(self) -> list[SteadyStateComputation]:
        """Return a list of all computation nodes in the workload."""
        return [node for node in self.nodes() if isinstance(node, SteadyStateComputation)]

    @property
    def transfer_nodes(self) -> list[SteadyStateTransfer]:
        """Return a list of all transfer nodes in the workload."""
        return [node for node in self.nodes() if isinstance(node, SteadyStateTransfer)]

    @property
    def tensor_nodes(self) -> list[SteadyStateTensor]:
        """Return a list of all tensor nodes in the workload."""
        return [node for node in self.nodes() if isinstance(node, SteadyStateTensor)]

    def get_subgraph(self, nodes: list[SteadyStateNode]) -> "SteadyStateWorkload":
        return self.subgraph(nodes)  # type: ignore

    def to_timeslotallocation(self) -> TimeSlotAllocation:
        """
        Deterministically map the steady-state DAG to a *fixed* slot table.

        Rules
        -----
        1.  All **SteadyStateComputation** nodes of a topological generation *g*
            share the same base-slot *slot0* (one per core → parallel execution).
        2.  **SteadyStateTensor** nodes
            * with a **known** `chosen_resource_allocation`
                are *packed per core* starting at *slot0*.
            * with an **unknown** allocation (``chosen_resource_allocation is None``)
                each receives its **own** successive slot (they cannot be packed
                per core yet).
            Runtime of tensors is 0 ⇒ the exact slot spacing has no impact
            on the final latency but keeps the allocation table unambiguous.
        3.  Every **SteadyStateTransfer** node of the generation is placed
            in a fresh slot directly **after** the last tensor slot.
            The *resource* stored in the table is the transfer's *source* core
            (possibly ``None`` when this is still a decision variable).

        The function does **not** invent core allocations - when a node has
        `chosen_resource_allocation is None` we simply record ``None``
        (the generic ``TimeSlotAllocation`` already accepts that).
        """

        allocations: list[tuple[int, Resource, SteadyStateNode]] = []
        global_slot = 0  # next free slot number

        for generation_iter in nx.topological_generations(self):
            generation = list(generation_iter)

            # ------------------------------------------------ computations --
            comp_nodes = [n for n in generation if isinstance(n, SteadyStateComputation)]

            next_free_slot_per_core: dict[Resource, int] = {}  # Track next free slot per core
            highest_slot_used = global_slot

            for n in comp_nodes:
                core = n.chosen_resource_allocation
                assert core is not None, f"{n.node_name} has no core assigned."
                # Determine the next available slot for this core
                slot = next_free_slot_per_core.get(core, global_slot)
                # Set the helper var and update the allocations
                next_free_slot_per_core[core] = slot + 1
                highest_slot_used = max(highest_slot_used, slot)
                allocations.append((slot, core, n))

            # --------------------------------------------------- tensors ----
            tensor_nodes = [n for n in generation if isinstance(n, SteadyStateTensor)]

            next_free_slot_per_core: dict[Resource, int] = {}  # only for *known* cores
            highest_slot_used = global_slot

            for t in tensor_nodes:
                core = t.chosen_resource_allocation

                if core is None:
                    # unknown placement → give the tensor its own slot
                    slot = highest_slot_used + 1
                    highest_slot_used = slot
                    allocations.append((slot, None, t))
                    continue

                # ---- core is known → try to pack on that core -------------
                slot = next_free_slot_per_core.get(core, global_slot)

                # find next empty slot for this core
                while any(s == slot and c == core for s, c, _ in allocations):
                    slot += 1

                next_free_slot_per_core[core] = slot + 1
                highest_slot_used = max(highest_slot_used, slot)
                allocations.append((slot, core, t))

            # ------------------------------------------------- transfers ----
            transfer_nodes = [n for n in generation if isinstance(n, SteadyStateTransfer)]
            transfer_slot = highest_slot_used + 1

            for tr in transfer_nodes:
                # the *source* core – may legitimately be None
                resource = tr.chosen_resource_allocation
                allocations.append((transfer_slot, resource, tr))
                transfer_slot += 1

            # prepare *slot0* for the next generation
            global_slot = max(slot for slot, _, _ in allocations) + 1

        # -------------------------------------------------------------------
        return TimeSlotAllocation(
            allocations=allocations,
        )  # every node in SS gets this tag

    def visualize_to_file(self, filepath: str = "workload_graph.png"):
        """Visualize the graph using Graphviz and save it to an image file.

        Nodes are laid out horizontally by topological generation,
        and vertically stacked to avoid overlap. The resource is displayed
        below each node in a clean and readable way.
        """
        dot = to_pydot(self)

        # Set global graph layout left to right
        dot.set_rankdir("LR")
        dot.set_concentrate(True)

        # Determine node positions based on topological generations
        generation_to_nodes = {}
        for gen_idx, generation in enumerate(nx.topological_generations(self)):
            generation_to_nodes[gen_idx] = list(generation)

        # Assign nodes to horizontal positions based on their generation
        for gen_idx, nodes in generation_to_nodes.items():
            for idx, node in enumerate(nodes):
                n = dot.get_node(str(node))[0]
                n.set_pos(f"{gen_idx},{-idx}!")  # Horizontal by generation, vertical by index

        # Customize node appearances and add resource labels
        for node in self.nodes():
            n = dot.get_node(str(node))[0]
            if isinstance(node, SteadyStateComputation):
                n.set_shape("ellipse")
                n.set_label(
                    f"{node.node_name}\n"
                    f"Resource: {getattr(node, 'chosen_resource_allocation', 'None')}\n"
                    f"{node.steady_state_iteration_space}"
                )
                n.set_style("filled")
                n.set_fillcolor("#60bcf0")
            elif isinstance(node, SteadyStateRollingBuffer):
                n.set_shape("box")
                n.set_label(
                    f"{node.node_name}\n{node.loop_ranges_per_dim}\n"
                    f"Resource: {getattr(node, 'chosen_resource_allocation', 'None')}"
                )
                n.set_style("filled")
                n.set_fillcolor("#eaff76e1")
            elif isinstance(node, SteadyStateTensor):
                n.set_shape("box")
                n.set_label(
                    f"{node.node_name}\n{node.loop_ranges_per_dim}\n"
                    f"Resource: {getattr(node, 'chosen_resource_allocation', 'None')}"
                )
                n.set_style("filled")
                n.set_fillcolor("#c2f0c2")
            elif isinstance(node, SteadyStateTransfer):
                if TensorFlag.CONSTANT in node.tensor.tensor_flag:
                    chosen_mem_core_label = f"Chosen Mem Core: {getattr(node, 'chosen_memory_core', 'None')}\n"
                else:
                    chosen_mem_core_label = ""
                n.set_shape("box")
                n.set_label(
                    f"{node.node_name}\n"
                    f"Resource: {getattr(node, 'chosen_resource_allocation', 'None')}\n"
                    f"Type: {node.transfer_type.value.capitalize()}\n"
                    f"{chosen_mem_core_label}"
                    f"{node.steady_state_iteration_space}"
                )
                n.set_style("filled")
                n.set_fillcolor("#ffcb9a")
            else:
                raise ValueError(f"Unknown node type: {type(node)}")

        # Save to file
        dot.write_png(filepath)
        print(f"Graph saved to {filepath}")
