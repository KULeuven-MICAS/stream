from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field, replace
from fractions import Fraction
from math import prod
from typing import TYPE_CHECKING, Any, TypeAlias

from xdsl.ir.affine import AffineMap

from stream.cost_model.communication_manager import MulticastPathPlan
from stream.datatypes import InterCoreTiling, LayerDim
from stream.hardware.architecture.accelerator import Accelerator
from stream.hardware.architecture.core import Core
from stream.hardware.architecture.memory import Memory, size_text
from stream.hardware.ports import OUTPUT, input_role
from stream.ir.infeasibility import InfeasibleAllocationError
from stream.opt.allocation.constraint_optimization.diagnosis import ResourceKind, overflow_infeasibility
from stream.workload.affine_access import map_dim_positions
from stream.workload.iterator_type import is_state_operand
from stream.workload.node import ComputationNode, InEdge, OutEdge, TransferNode
from stream.workload.tensor import Tensor
from stream.workload.utils import get_equivalent_dimension, operand_inputs
from stream.workload.workload import Node, Workload, position_of

if TYPE_CHECKING:
    # AIEKernel lives in stream.compiler (AIE codegen), an optional install. Only used as an
    # annotation here and validated at runtime when a kernel is actually set (AIE mappings),
    # so importing it eagerly would force the AIE toolchain onto the base install.
    from stream.compiler.kernels.aie_kernel import AIEKernel

Resource: TypeAlias = Core | MulticastPathPlan


@dataclass(slots=True)
class FusedGroup:
    name: str
    layers: tuple[str, ...] = field(default_factory=tuple)
    intra_core_tiling: tuple[tuple[LayerDim, int], ...] = field(default_factory=tuple)
    runtime_dims: tuple[LayerDim, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("FusedGroup name must be provided")
        for dim, tile in self.intra_core_tiling:
            if not isinstance(dim, LayerDim) or not dim:
                raise TypeError("intra_core_tiling dim must be a non-empty string")
            if not isinstance(tile, int) or tile <= 0:
                raise ValueError(f"Tile for '{dim}' must be a positive int, got {tile!r}.")


@dataclass(frozen=True, slots=True)
class EdgePlacement:
    """Which off-chip memories hold an edge's tensor: block ``s`` of the whole tensor along axis ``split`` in
    ``memories[s]``, or, without a split, all of it in each."""

    memories: tuple[Memory, ...]
    split: int | None = None

    def block(self, position: int, extent: int) -> tuple[int, int]:
        """The inclusive index range along the axis the memory at ``position`` holds, of an axis ``extent`` long."""
        if extent % len(self.memories):
            names = [memory.name for memory in self.memories]
            raise ValueError(f"axis {self.split} of extent {extent} does not split evenly over {names}")
        size = extent // len(self.memories)
        return position * size, (position + 1) * size - 1


@dataclass(slots=True)
class NodeMapping:
    """Mapping attributes for a single workload Node. A Node can be either a ComputationNode or a TransferNode."""

    resource_allocation: tuple[tuple[Resource, ...], ...] = field(default_factory=tuple)
    inter_core_tiling: tuple[InterCoreTiling, ...] = field(default_factory=tuple)
    memory_allocation: tuple[tuple[Memory, ...], ...] = field(default_factory=tuple)
    kernel: AIEKernel | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.resource_allocation, tuple):
            raise TypeError("core_allocation must be a tuple of Resource objects")

        if not isinstance(self.inter_core_tiling, tuple):
            raise TypeError("inter_core_tiling must be a tuple of (dimension, factor) tuples")

        for entry in self.inter_core_tiling:
            for item in entry:
                required_length = 2
                if not (isinstance(item, tuple) and len(item) == required_length):
                    raise TypeError("inter_core_tiling entries must be tuples of length 2: (dim, factor)")
                dim, factor = item
                if not isinstance(factor, int) or factor <= 0:
                    raise ValueError(f"Tiling factor for '{dim}' must be a positive int, got {factor!r}.")

        if self.kernel is not None:
            from stream.compiler.kernels.aie_kernel import AIEKernel  # noqa: PLC0415

            if not isinstance(self.kernel, AIEKernel):
                raise TypeError("Kernel must be an AIEKernel instance. Other kernel types are not supported yet.")

    __hash__ = None  # type: ignore[assignment]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, NodeMapping):
            return NotImplemented
        return self.inter_core_tiling == other.inter_core_tiling and self.kernel == other.kernel


class Mapping:
    """Holds per-Node mapping information across multiple layers."""

    def __init__(
        self,
        initial: dict[Node, NodeMapping] | None = None,
        fused_groups: Iterable[FusedGroup] | None = None,
        runtime_args: dict[str, str] | None = None,
        edge_placements: dict[str, EdgePlacement] | None = None,
    ) -> None:
        self._by_node: dict[Node, NodeMapping] = {}
        self._fused_groups: tuple[FusedGroup, ...] = tuple(fused_groups or ())
        for fused_group in self._fused_groups:
            if not isinstance(fused_group, FusedGroup):
                raise TypeError("fused_groups must contain FusedGroup instances")
        if initial:
            for node, layer_mapping in initial.items():
                self.set(node, layer_mapping)
        self._runtime_args = runtime_args or {}
        self._edge_placements = edge_placements or {}

    def set(self, node: Node, layer_mapping: NodeMapping) -> None:
        if node is None:
            raise ValueError("node must not be None")
        if not isinstance(layer_mapping, NodeMapping):
            raise TypeError("layer_mapping must be a NodeMapping instance")
        self._by_node[node] = layer_mapping

    def update_inter_core_tiling(self, node: Node, new_inter_core_tiling: tuple[InterCoreTiling, ...]) -> None:
        self._by_node[node].inter_core_tiling = new_inter_core_tiling

    def set_for_node(
        self,
        node: Node,
        resource_allocation: tuple[tuple[Resource, ...], ...],
        inter_core_tiling: tuple[InterCoreTiling, ...],
        memory_allocation: tuple[tuple[Memory, ...], ...] = (),
        kernel: AIEKernel | None = None,
    ) -> None:
        self.set(
            node,
            NodeMapping(
                resource_allocation=resource_allocation,
                inter_core_tiling=inter_core_tiling,
                kernel=kernel,
                memory_allocation=memory_allocation,
            ),
        )

    def get(self, node: Node) -> NodeMapping:
        layer_mapping = self._by_node.get(node)
        if layer_mapping is None:
            raise KeyError(f"Node {node} not found in mapping")
        return layer_mapping

    def get_equal_computation_node(self, node: ComputationNode) -> ComputationNode | None:
        for n in self._by_node:
            if isinstance(n, ComputationNode) and n.name == node.name and n.has_same_performance(node):
                return n
        return None

    def remove(self, node: Node) -> None:
        """Remove a node entry from the mapping."""
        if node not in self._by_node:
            raise KeyError(f"Node {node} not found in mapping")
        del self._by_node[node]

    def __getitem__(self, node: Node) -> NodeMapping:
        return self._by_node[node]

    def __setitem__(self, node: Node, layer_mapping: NodeMapping) -> None:
        self.set(node, layer_mapping)

    def __contains__(self, node: object) -> bool:
        return node in self._by_node

    def __len__(self) -> int:
        return len(self._by_node)

    def __iter__(self) -> Iterator[Node]:
        return iter(self._by_node)

    def items(self) -> Iterable[tuple[Node, NodeMapping]]:
        return self._by_node.items()

    def nodes(self) -> Iterable[Node]:
        return self._by_node.keys()

    def values(self) -> Iterable[NodeMapping]:
        return self._by_node.values()

    def get_ir(self) -> dict:
        """Return a dictionary representation of the mapping for serialization/inspection.

        This captures:
        - All nodes with their resource allocations (as core IDs or path descriptors)
        - Inter-core tiling as dimension/factor pairs
        - Memory allocation as lists of core IDs
        - Fused groups with their layers and intra-core tiling
        - Runtime arguments
        """
        nodes_ir: dict[str, dict] = {}
        for node, nm in self._by_node.items():
            node_key = str(getattr(node, "name", None) or repr(node))

            # Serialize resource_allocation -> list of lists of dicts.
            # A slot is usually a tuple of Resources (compute nodes), but transfer nodes may
            # carry a bare Resource (e.g. a MulticastPathPlan) per slot; normalize to an iterable.
            resource_allocation_ir = []
            for slot in nm.resource_allocation:
                slot_ir = []
                resources = slot if isinstance(slot, tuple | list) else (slot,)
                for resource in resources:
                    if isinstance(resource, Core):
                        slot_ir.append({"type": "core", "id": resource.id})
                    elif isinstance(resource, MulticastPathPlan):
                        slot_ir.append(
                            {
                                "type": "path",
                                "sources": [m.name for m in resource.sources],
                                "targets": [m.name for m in resource.targets],
                                "hops": resource.total_hops_objective,
                            }
                        )
                    else:
                        slot_ir.append({"type": "unknown", "repr": repr(resource)})
                resource_allocation_ir.append(slot_ir)

            # Serialize inter_core_tiling: tuple[InterCoreTiling, ...] -> list of lists of [dim_str, factor]
            inter_core_tiling_ir = []
            for slot in nm.inter_core_tiling:
                slot_ir = [[str(dim), factor] for dim, factor in slot]
                inter_core_tiling_ir.append(slot_ir)

            memory_allocation_ir = [[memory.name for memory in slot] for slot in nm.memory_allocation]

            nodes_ir[node_key] = {
                "resource_allocation": resource_allocation_ir,
                "inter_core_tiling": inter_core_tiling_ir,
                "memory_allocation": memory_allocation_ir,
            }

        # Serialize fused_groups
        fused_groups_ir = []
        for fg in self._fused_groups:
            fused_groups_ir.append(
                {
                    "name": fg.name,
                    "layers": list(fg.layers),
                    "intra_core_tiling": [[str(dim), factor] for dim, factor in fg.intra_core_tiling],
                }
            )

        return {
            "nodes": nodes_ir,
            "fused_groups": fused_groups_ir,
            "runtime_args": dict(self._runtime_args),
            "edge_placements": {
                name: {"memories": [memory.name for memory in placement.memories], "split": placement.split}
                for name, placement in self._edge_placements.items()
            },
        }

    def to_dict(self) -> dict[str, dict[str, Any]]:
        """
        Serialize to a JSON-friendly dict keyed by a stable node identifier.

        Uses node.name if present, else repr(node).
        """
        out: dict[str, dict[str, Any]] = {}
        for node, lm in self._by_node.items():
            node_key = getattr(node, "name", None) or repr(node)
            out[str(node_key)] = {
                "resource_allocation": [repr(c) for c in lm.resource_allocation],
                "inter_core_tiling": list(lm.inter_core_tiling),
                "kernel": dict(lm.kernel) if lm.kernel is not None else None,
            }
        return out

    def with_updated_workload(self, new_workload: Workload, old_workload: Workload) -> Mapping:
        """Return a new Mapping instance with the same LayerMappings but updated to a new Workload.
        This is useful when the Workload has been modified (e.g., tiled) and we want to keep the same
        mapping information.
        This assumes the names of the nodes are unique identifiers that remain unchanged between workloads.

        Args:
            workload (Workload): The new Workload instance.

        Returns:
            Mapping: A new Mapping instance with updated workload reference.
        """
        # Update the fused_groups to reference the new workload's dimension names
        new_fused_groups = []
        for fused_group in self.fused_groups:
            new_tilings = []
            for dim, tile in fused_group.intra_core_tiling:
                new_dim = get_equivalent_dimension(old_workload, new_workload, dim)
                new_tilings.append((new_dim, tile))
            new_fused_group = FusedGroup(
                name=fused_group.name,
                layers=fused_group.layers,
                intra_core_tiling=tuple(new_tilings),
            )
            new_fused_groups.append(new_fused_group)
        new_mapping = Mapping(
            fused_groups=new_fused_groups, runtime_args=self.runtime_args, edge_placements=self._edge_placements
        )
        for node in self.nodes():
            updated_node = new_workload.get_node_by_name(node.name)
            new_mapping.set(updated_node, self.get(node))
        return new_mapping

    @property
    def fused_groups(self) -> tuple[FusedGroup, ...]:
        return self._fused_groups

    def with_fused_groups(self, fused_groups: Iterable[FusedGroup]) -> Mapping:
        """The same per-node mapping under different fused groups (e.g. another intra-core tiling)."""
        return Mapping(
            initial=dict(self._by_node),
            fused_groups=fused_groups,
            runtime_args=self._runtime_args,
            edge_placements=self._edge_placements,
        )

    def with_edge_placements(self, edge_placements: dict[str, EdgePlacement]) -> Mapping:
        """The same mapping with ``edge_placements`` added to the placements it declares."""
        return Mapping(
            initial=dict(self._by_node),
            fused_groups=self._fused_groups,
            runtime_args=self._runtime_args,
            edge_placements=self._edge_placements | edge_placements,
        )

    def copy(self) -> Mapping:
        """A detached mapping a consumer may rewrite in place."""
        cloned = {node: replace(nm) for node, nm in self._by_node.items()}
        return Mapping(
            initial=cloned,
            fused_groups=self._fused_groups,
            runtime_args=self._runtime_args,
            edge_placements=self._edge_placements,
        )

    @property
    def runtime_args(self) -> dict[str, str]:
        return self._runtime_args

    @property
    def edge_placements(self) -> dict[str, EdgePlacement]:
        """The placement of each edge the mapping declares one for, by edge name."""
        return self._edge_placements


def role_of(node: ComputationNode, tensor: Tensor) -> str:
    """The operand role ``tensor`` has on ``node``: the k-th input role for its k-th operand (:func:`operand_inputs`),
    else the output role. The output role also covers state updated in place (an accumulator) and extra inputs such
    as a bias."""
    if tensor in node.outputs or (tensor in node.inputs and is_state_operand(node, tensor)):
        return OUTPUT
    operands = operand_inputs(node)
    if tensor in operands:
        return input_role(operands.index(tensor) + 1)
    if tensor in node.inputs:
        return OUTPUT
    raise ValueError(f"{tensor.name} is no operand of {node.name}")


def offchip_memory(accelerator: Accelerator) -> Memory:
    """The memory the workload's edge tensors live in."""
    memory = accelerator.offchip_memory
    if memory is None:
        raise ValueError(f"{accelerator.name} has no off-chip memory to hold the workload's inputs and outputs")
    return memory


def the_one[T](options: tuple[T, ...], node: Node, what: str) -> T:
    """The single entry of ``options``, the candidate ``what`` of ``node``. Choosing between several is not supported
    yet."""
    if len(options) != 1:
        raise NotImplementedError(
            f"{node.name} has {len(options)} candidate {what}; one is supported, so the mapping must settle on it"
        )
    return options[0]


def placements_of(accelerator: Accelerator, mapping: Mapping, node: Node, role: str) -> tuple[tuple[Memory, ...], ...]:
    """The candidate memories ``node`` holds its ``role`` operand in, one per position: for an input edge the memories
    of its placement, else the off-chip memory; the memory allocation for a transfer, and for a computation the memory
    each core keeps that operand in. Cores sharing a memory repeat it. An input needs a placement when there are several
    off-chip memories. Where an output lands is :func:`landing_of`."""
    if isinstance(node, InEdge):
        if placement := mapping.edge_placements.get(node.name):
            return (placement.memories,)
        if len(offchip := accelerator.offchip_memories) > 1:
            names = ", ".join(memory.name for memory in offchip)
            raise ValueError(
                f"input {node.name} has no placement over the off-chip memories [{names}]; give it one in the mapping "
                f"file: runtime_args: {{{node.name}: {{placement: {{memories: [{names}], split: <axis>}}}}}}"
            )
        return ((offchip_memory(accelerator),),)
    if isinstance(node, TransferNode):
        return mapping.get(node).memory_allocation
    if isinstance(node, ComputationNode):
        return tuple(
            tuple(accelerator.memory_of(core, role) for core in cores)
            for cores in mapping.get(node).resource_allocation
        )
    raise ValueError(f"{node} holds no operand to place")


def edge_tensor(edge: InEdge | OutEdge) -> Tensor:
    """The workload input ``edge`` brings in or the output it takes out."""
    return edge.outputs[0] if isinstance(edge, InEdge) else edge.inputs[0]


def runtime_args_per_group(runtime_args: dict[str, Any], workloads: Sequence[Workload]) -> list[dict[str, Any]]:
    """The runtime args of each group's input and output edges: those declared for its tensor, which the two ends of
    a fusion cut share and must not declare differently, else an empty entry for the standard layout."""
    declared: dict[Tensor, dict[str, tuple[str, Any]]] = {}
    edges = [[node for node in workload.nodes if isinstance(node, InEdge | OutEdge)] for workload in workloads]
    for edge in (edge for group in edges for edge in group if edge.name in runtime_args):
        merged = declared.setdefault(edge_tensor(edge), {})
        for key, value in runtime_args[edge.name].items():
            name, first = merged.setdefault(key, (edge.name, value))
            if first != value:
                raise ValueError(
                    f"{name} and {edge.name} are the two ends of one fusion cut but declare {key} {first} and {value}"
                )
    return [
        {edge.name: {key: value for key, (_, value) in declared.get(edge_tensor(edge), {}).items()} for edge in group}
        for group in edges
    ]


def placement_split_bits(node: ComputationNode, split_axes: dict[str, int]) -> dict[int, int]:
    """Per position of ``node``'s dims, the bits of its operands whose placement splits an axis that dim indexes, by the
    axis ``split_axes`` gives each tensor name."""
    bits: dict[int, int] = {}
    for tensor in node.tensors:
        if (axis := split_axes.get(tensor.name)) is not None:
            access = node.get_mapping(tensor)
            for p in map_dim_positions(AffineMap(access.num_dims, 0, (access.results[axis],))):
                bits[p] = bits.get(p, 0) + tensor.size_bits()
    return bits


def landing_of(accelerator: Accelerator, mapping: Mapping, edge: OutEdge | None, src: Node) -> tuple[Memory, ...]:
    """The off-chip memories an output from ``src`` lands in, one per position: those of the placement of ``edge``, the
    output edge reading it, else the off-chip memory, or of several the one nearest each memory ``src`` holds it in."""
    if edge is not None and (placement := mapping.edge_placements.get(edge.name)):
        return placement.memories
    if len(accelerator.offchip_memories) <= 1:
        return (offchip_memory(accelerator),)
    held_in = the_one(placements_of(accelerator, mapping, src, OUTPUT), src, "placements")
    return tuple(map(accelerator.nearest_offchip, held_in))


def landed_placement(accelerator: Accelerator, mapping: Mapping, src: ComputationNode) -> EdgePlacement | None:
    """Where ``src``'s output landed without a placement (:func:`landing_of`), as one: a block along the axis its first
    split dim indexing the output indexes, in the memory the cores holding that block land it in, or all of it in each
    memory it landed in where no split dim indexes it, as the cores of a split reduction land it; None if the cores
    holding one block land it in several."""
    positions = landing_of(accelerator, mapping, None, src)
    if len(set(positions)) == 1:
        return EdgePlacement(positions[:1])
    tilings = mapping.get(src).inter_core_tiling
    tiling = tilings[0] if tilings else ()
    access = src.get_mapping(src.outputs[0])
    indexed = (
        (loop, axis)
        for loop, (dim, _) in enumerate(tiling)
        for axis, result in enumerate(access.results)
        if dim.position in map_dim_positions(AffineMap(access.num_dims, 0, (result,)))
    )
    if (found := next(indexed, None)) is None:
        return EdgePlacement(tuple(dict.fromkeys(positions)))
    loop, axis = found
    factor, inner = tiling[loop][1], prod(f for _, f in tiling[loop + 1 :])
    blocks: dict[int, set[Memory]] = defaultdict(set)
    for core, memory in enumerate(positions):
        blocks[position_of(core, len(positions), tiling) // inner % factor].add(memory)
    if any(len(memories) > 1 for memories in blocks.values()):
        return None
    return EdgePlacement(tuple(next(iter(blocks[b])) for b in range(factor)), axis)


def with_cut_landings(
    accelerator: Accelerator, workloads: Sequence[Workload], mappings: Sequence[Mapping]
) -> list[Mapping]:
    """``mappings`` with the input end of each fusion cut declaring no placement reading where its output end landed
    (:func:`landed_placement`), when there are several off-chip memories."""
    if len(accelerator.offchip_memories) <= 1:
        return list(mappings)
    landed: dict[Tensor, EdgePlacement] = {}
    for workload, mapping in zip(workloads, mappings, strict=True):
        for edge in workload.nodes:
            src = next(iter(workload.predecessors(edge)), None) if isinstance(edge, OutEdge) else None
            if isinstance(src, ComputationNode) and edge.name not in mapping.edge_placements:
                if (placement := landed_placement(accelerator, mapping, src)) is not None:
                    landed[edge_tensor(edge)] = placement
    return [
        mapping.with_edge_placements(
            {
                edge.name: landed[edge_tensor(edge)]
                for edge in workload.nodes
                if isinstance(edge, InEdge) and edge.name not in mapping.edge_placements and edge_tensor(edge) in landed
            }
        )
        for workload, mapping in zip(workloads, mappings, strict=True)
    ]


def offchip_holdings(
    accelerator: Accelerator, workloads: Sequence[Workload], mappings: Sequence[Mapping]
) -> dict[Memory, dict[Tensor, Fraction]]:
    """The share of each workload input and output each off-chip memory holds, a tensor once over the groups and the
    two ends of a fusion cut: a block per position it lives in, all of it in each memory of a copy."""
    held: dict[Memory, dict[Tensor, Fraction]] = defaultdict(dict)
    counted: set[Tensor] = set()
    for workload, mapping in zip(workloads, mappings, strict=True):
        for edge in workload.nodes:
            if not isinstance(edge, InEdge | OutEdge) or (tensor := edge_tensor(edge)) in counted:
                continue
            counted.add(tensor)
            placement = mapping.edge_placements.get(edge.name) or _landed_copy(accelerator, workload, mapping, edge)
            if placement is not None and placement.split is None:
                shares = dict.fromkeys(placement.memories, Fraction(1))
            else:
                positions = (
                    landing_of(accelerator, mapping, edge, next(iter(workload.predecessors(edge))))
                    if isinstance(edge, OutEdge)
                    else the_one(placements_of(accelerator, mapping, edge, OUTPUT), edge, "placements")
                )
                shares = {memory: Fraction(n, len(positions)) for memory, n in Counter(positions).items()}
            for memory, share in shares.items():
                held[memory][tensor] = share
    return held


def _landed_copy(accelerator: Accelerator, workload: Workload, mapping: Mapping, edge: Node) -> EdgePlacement | None:
    """Where an output without a placement landed whole in several memories, as a split reduction's does."""
    src = next(iter(workload.predecessors(edge)), None) if isinstance(edge, OutEdge) else None
    if not isinstance(src, ComputationNode) or len(accelerator.offchip_memories) <= 1:
        return None
    placement = landed_placement(accelerator, mapping, src)
    return placement if placement is not None and placement.split is None else None


OFFCHIP_CAPACITY = ResourceKind(
    "offchip_capacity",
    "off-chip memory capacity exceeded",
    unit="bytes",
    demand_label="workload inputs and outputs living in",
    bound_label="capacity of",
    demand_input="workload tensor sizes × mapping runtime_args placements",
    bound_input="hardware spec: memory size",
    term_detail="share it holds",
    levers=(
        "Increase {resource} (hardware spec)",
        "Split the tensors over the off-chip memories rather than copying them (mapping runtime_args placement)",
        "Reduce the workload's tensor sizes (smaller batch, shorter sequence)",
    ),
)


def check_offchip_capacity(
    accelerator: Accelerator, workloads: Sequence[Workload], mappings: Sequence[Mapping]
) -> None:
    """Refuse workload inputs and outputs that overflow an off-chip memory they live in (:func:`offchip_holdings`),
    when there are several off-chip memories or the mapping places them."""
    if len(accelerator.offchip_memories) <= 1 and not any(mapping.edge_placements for mapping in mappings):
        return
    for memory, held in offchip_holdings(accelerator, workloads, mappings).items():
        bits_of = {
            tensor: float(share * tensor.size_bits(tuple(tensor.subview.source.type.get_shape())))
            for tensor, share in held.items()
        }
        bits = sum(bits_of.values())
        if bits > memory.capacity_bits:
            terms = " + ".join(_share_text(tensor.name, share) for tensor, share in held.items())
            summary = f"{memory.name} holds {terms} = {size_text(bits)} > {size_text(memory.capacity_bits)}"
            shares = {_share_text(tensor.name, held[tensor]): b / 8 for tensor, b in bits_of.items()}
            report = overflow_infeasibility(OFFCHIP_CAPACITY, memory, shares, memory.capacity_bits / 8, summary)
            raise InfeasibleAllocationError(report)


def _share_text(name: str, share: Fraction) -> str:
    if share == 1:
        return name
    return f"{name}/{share.denominator}" if share.numerator == 1 else f"{name}*{share}"
