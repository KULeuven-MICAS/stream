"""The layout of every tensor copy of a steady state, and the contiguous runs each transfer moves them in."""

from __future__ import annotations

from typing import TYPE_CHECKING

from stream.cost_model.layout import Layout, Rate, transfer_runs
from stream.hardware.ports import READ, WRITE
from stream.stages.estimation.core_cost_backends import contiguous_axes, select_backend
from stream.workload.workload import ComputationNode, InEdge, OutEdge, Tensor, TransferNode

if TYPE_CHECKING:
    from stream.cost_model.communication_manager import MulticastPathPlan
    from stream.hardware.architecture.core import Core
    from stream.opt.allocation.constraint_optimization.space import DecisionSpace


class CopyLayouts:
    """Each tensor copy's layout, chosen as an accelerator's compiler and DMAs would:

    - a model parameter is packed ahead of time in the layout its kernel reads, any other workload input and every
      output is row-major, as the host holds them;
    - a node writes its output with the axes its core produces together innermost;
    - a copy a node reads has the axes that node's core reads together innermost, reordered from the copy it is made
      from as little as that takes, so a transpose the kernel can read as it is moves nothing; where its readers
      differ, the first decides, and every copy one transfer makes has the layout of its first;
    - a copy in the memory it is copied from moves nothing, so it keeps its source's layout;
    - a copy on its way to another, in a memory between them, is laid out as its source or as the copy it feeds,
      whichever the two transfers move faster, so the DMA converts it on the hop that loses the least bandwidth, and
      before a hop that stays in one memory, which cannot.

    What a kernel reads and writes together comes from its core's cost backend (``contiguous_axes``); a backend that
    does not say leaves its copies as they come."""

    def __init__(self, space: DecisionSpace) -> None:
        self.space = space
        workload = space.workload
        self._producer: dict[Tensor, object] = {t: n for n in workload.nodes() for t in getattr(n, "outputs", ())}
        self._readers: dict[Tensor, list[object]] = {}
        for node in workload.nodes():
            for tensor in getattr(node, "inputs", ()):
                self._readers.setdefault(tensor, []).append(node)
        self._layouts: dict[Tensor, Layout] = {}
        self._needs: dict[tuple[ComputationNode, int], frozenset[int]] = {}
        self._runs: dict[tuple[TransferNode, MulticastPathPlan], tuple[float, float]] = {}

    def of(self, tensor: Tensor) -> Layout:
        """The layout of the copy ``tensor``."""
        if (layout := self._layouts.get(tensor)) is None:
            layout = self._layouts[tensor] = self._choose(tensor)
        return layout

    def runs(self, tr: TransferNode, choice: MulticastPathPlan) -> tuple[float, float]:
        """Contiguous bytes per run ``tr`` reads its tile in and writes it in on ``choice``."""
        if (found := self._runs.get((tr, choice))) is None:
            found = self._runs[(tr, choice)] = self._transfer_runs(
                tr, choice, self.of(tr.inputs[0]), self.of(tr.outputs[0])
            )
        return found

    def needs(self, node: ComputationNode, tensor: Tensor) -> frozenset[int]:
        """The axes of ``tensor`` the core running ``node`` reads or writes together."""
        index = node.tensors.index(tensor)
        if (found := self._needs.get((node, index))) is None:
            found = frozenset()
            for core in self.space.cost_lut.get_cores(node):
                axes = contiguous_axes(select_backend(core), self.space.cost_lut.get_cost(node, core))
                if axes:
                    found = axes[index]
                    break
            self._needs[(node, index)] = found
        return found

    def _choose(self, tensor: Tensor) -> Layout:
        row_major = Layout.row_major(len(tensor.shape))
        producer = self._producer.get(tensor)
        if isinstance(producer, InEdge):
            return self._wanted(tensor, row_major) if producer.parameter else row_major
        if isinstance(producer, ComputationNode):
            return row_major.with_innermost(self.needs(producer, tensor))
        if not isinstance(producer, TransferNode):
            return row_major
        return self._copy(producer, tensor)

    def _copy(self, producer: TransferNode, tensor: Tensor) -> Layout:
        """The layout of the copy ``tensor`` transfer ``producer`` makes."""
        if tensor is not producer.outputs[0]:
            return self.of(producer.outputs[0])  # one stream writes every copy of a multicast in one layout
        source = self.of(producer.inputs[0])
        if self.space.within_one_memory(producer):
            return source  # nothing moves, so nothing is reordered: the reader reads the copy as it is
        onward = [r for r in self._readers.get(tensor, []) if isinstance(r, TransferNode)]
        if not onward or len(onward) != len(self._readers[tensor]):
            return self._wanted(tensor, source)
        candidates = dict.fromkeys((source, self._wanted(tensor, source)))
        return min(candidates, key=lambda layout: self._cost(producer, source, layout, onward))

    def _wanted(self, tensor: Tensor, source: Layout) -> Layout:
        """The layout the first reader of ``tensor`` (through any further copies) wants a copy of a ``source`` copy
        in: one a node reads with its core's axes innermost, a workload output row-major."""
        readers = self._readers.get(tensor, [])
        for reader in readers:
            if isinstance(reader, ComputationNode) and (need := self.needs(reader, tensor)):
                return source.with_innermost(need)
        if any(isinstance(reader, OutEdge) for reader in readers):
            return Layout.row_major(len(tensor.shape))
        onward = next((r for r in readers if isinstance(r, TransferNode)), None)
        return self._wanted(onward.outputs[0], source) if onward is not None else source

    def _cost(self, tr: TransferNode, source: Layout, layout: Layout, onward: list[TransferNode]) -> float:
        """Cycles the transfer into a copy laid out ``layout`` and those out of it take."""
        cost = self._moved(tr, source, layout)
        for next_tr in onward:
            cost += self._moved(next_tr, layout, self._wanted(next_tr.outputs[0], layout))
        return cost

    def _moved(self, tr: TransferNode, source: Layout, target: Layout) -> float:
        """Cycles ``tr`` takes from a ``source`` layout to a ``target`` one at its slower side's rate: unbounded for one
        that stays in a memory, which moves nothing and so cannot reorder anything."""
        full = tuple(tr.inputs[0].subview.source.type.get_shape())
        if self.space.within_one_memory(tr):
            return 0.0 if source.same_data_order(target, full) else float("inf")
        choice = self.space.path_choices[tr][0] if self.space.path_choices.get(tr) else None
        if choice is None:
            return 0.0
        read, write = self._transfer_runs(tr, choice, source, target)
        rate = min(self._rate(tr, choice, READ)(read), self._rate(tr, choice, WRITE)(write))
        return self.space.moved_bits(tr, choice) / rate if rate > 0 else float("inf")

    def _transfer_runs(
        self, tr: TransferNode, choice: MulticastPathPlan, source: Layout, target: Layout
    ) -> tuple[float, float]:
        tensor = tr.inputs[0]
        full = tuple(tensor.subview.source.type.get_shape())
        bits = tensor.operand_type.bitwidth
        read, write = self._rate(tr, choice, READ), self._rate(tr, choice, WRITE)
        extents = (self._extent(tr.inputs[0], full), self._extent(tr.outputs[0], full))
        return transfer_runs(tuple(tensor.shape), full, bits, source, target, read, write, extents)

    def _extent(self, copy: Tensor, full: tuple[int, ...]) -> tuple[int, ...]:
        """The buffer ``copy`` lives in: the whole tensor off chip, where a workload input or output is, else the tile
        it holds on chip."""
        off_chip = isinstance(self._producer.get(copy), InEdge) or any(
            isinstance(reader, OutEdge) for reader in self._readers.get(copy, [])
        )
        return full if off_chip else tuple(copy.shape)

    def _rate(self, tr: TransferNode, choice: MulticastPathPlan, direction: str) -> Rate:
        """The rate of the slowest core on one side of ``choice``."""
        cores: tuple[Core, ...] = choice.sources if direction == READ else choice.targets
        sides = [self.space.side_rate(tr, core, direction) for core in cores]
        return lambda span: min((side(span) for side in sides), default=float("inf"))
