import logging
import os
from dataclasses import replace

from stream.datatypes import LayerDim
from stream.hardware.architecture.core import Core
from stream.mapping.chain_placement import (
    TILED_HANDOVER_FACTOR,
    bandwidth_bound,
    column_budget_options,
    layer_cost,
    row_counts,
    widest_columns,
)
from stream.mapping.mapping import Mapping
from stream.stages.context import StageContext
from stream.stages.stage import Stage, StageCallable
from stream.workload.workload import ComputationNode, Workload

logger = logging.getLogger(__name__)

ELEMENT_BYTES = 2
BUFFERS = 2


class PlacementGenerationStage(Stage):
    """Place the layers a mapping leaves unplaced, from the workload and the kernels.

    A fused group that is a linear chain short enough to stack gets a row tenancy: rows
    per layer from the bottleneck of measured kernel cycles (a layer wider than its
    consumer pays the tiled-handover factor), the shared dimension split over the widest
    column count it divides, cores ordered rows-outermost so every handover lands in the
    consumer's column. Any other multi-layer group gets disjoint column tenancies by the
    same bottleneck rule. A layer alone in its group takes the whole array -- all rows
    for a contracting kernel, one core per column beside the memory tile for a
    bandwidth-bound one. Declared placements are never touched.
    """

    REQUIRED_FIELDS = ("workload", "mapping", "accelerator")

    def __init__(self, list_of_callables: list[StageCallable], ctx: StageContext):
        super().__init__(list_of_callables, ctx)
        self.workload: Workload = self.ctx.get("workload")
        self.mapping: Mapping = self.ctx.get("mapping")
        self.accelerator = self.ctx.get("accelerator")

    def run(self):
        grid = {
            (c.col_id, c.row_id): c
            for c in self.accelerator.core_list
            if c.type == "compute" and c.col_id is not None and c.row_id is not None
        }
        self._pending_fallbacks: list = []
        self._pending_reserves: list = []
        if grid:
            for group in self.mapping.fused_groups:
                nodes = [n for name in group.layers if isinstance(n := self._node(name), ComputationNode)]
                if nodes and all(self._unplaced(n) for n in nodes):
                    self._place_group(nodes, grid)
            def materialize(pending):
                shapes: list[Mapping] = []
                for narrow in pending:
                    alt = (shapes[-1] if shapes else self.mapping).copy()
                    narrow(alt)
                    shapes.append(alt)
                return shapes

            self.ctx.set(placement_alternatives=materialize(self._pending_fallbacks))
            self.ctx.set(placement_reserves=materialize(self._pending_reserves))
        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], self.ctx)
        yield from sub_stage.run()

    def _node(self, name: str):
        try:
            return self.workload.get_node_by_name(name)
        except Exception:  # noqa: BLE001 -- a group may name a node this sub-workload lacks
            return None

    def _unplaced(self, node: ComputationNode) -> bool:
        nm = self.mapping.get(node)
        return not nm.resource_allocation and nm.kernel is not None

    def _place_group(self, nodes: list[ComputationNode], grid: dict[tuple[int, int], Core]) -> None:
        columns = sorted({col for col, _ in grid})
        rows = sorted({row for _, row in grid})
        if len(nodes) == 1:
            self._place_alone(nodes[0], grid, columns, rows)
        elif len(nodes) <= len(rows) and self._linear(nodes):
            self._place_stacked(nodes, grid, columns, rows)
        else:
            self._place_tenancies(nodes, grid, columns, rows)

    def _linear(self, nodes: list[ComputationNode]) -> bool:
        members = set(nodes)
        for i, node in enumerate(nodes[:-1]):
            downstream = {s for s in self.workload.successors(node) if s in members}
            if downstream != {nodes[i + 1]}:
                return False
        return True

    def _place_alone(self, node: ComputationNode, grid, columns, rows) -> None:
        kernel = self.mapping.get(node).kernel
        granule = dict(kernel.granule())
        if bandwidth_bound(kernel):
            self._place_one_row(node, kernel, grid, columns, rows, self.mapping)
            return
        if len(granule) == 2:
            # One core per column: the model ranks the wider and the rows-only shapes
            # above it and hardware refutes both (13.7% and 7.5% slower), so those are
            # reserves for an infeasible seed, never priced rivals.
            self._place_one_row(node, kernel, grid, columns, rows, self.mapping)
            depth = self._fitting_split(node, 0, len(rows), granule.get(0, 1))
            narrow = tuple(grid[(columns[0], row)] for row in rows[:depth])
            self._pending_reserves.append(
                lambda m, n=node, c=narrow, sp=depth: self._assign(n, c, ((0, sp),), m)
            )
            return
        # D2 splits across all columns or not at all: every measured design does one of
        # the two, and a partial-width scatter of a short output dimension is exactly the
        # shape that hung the generated k=3 attention scores on hardware.
        split = [(0, self._fitting_split(node, 0, len(rows), granule.get(0, 1)))]
        cols_used = 1
        if len(granule) > 2 and self._fitting_split(node, 2, len(columns), granule.get(2, 1)) == len(columns):
            cols_used = len(columns)
            split.append((2, cols_used))
        cores = tuple(grid[(col, row)] for col in columns[:cols_used] for row in rows[: split[0][1]])
        self._assign(node, cores, tuple(split))

    def _place_one_row(self, node, kernel, grid, columns, rows, mapping) -> None:
        row = rows[0]  # beside the memory tile that feeds it
        width = self._fitting_split(node, 0, len(columns))
        cores = tuple(grid[(col, row)] for col in columns[:width])
        self._assign(node, cores, ((0, width),), mapping)
        width = self._row_width(node, kernel, next(iter(grid.values())))
        self._retile(node, kernel, mapping, m=1, n=width, layout="contiguous")

    def _place_stacked(self, nodes: list[ComputationNode], grid, columns, rows) -> None:
        kernels = [self.mapping.get(n).kernel for n in nodes]
        # STREAM_WIDE_SOFTMAX: prototype flag to spend a spare row on the bottleneck stage.
        #   "memtile"/"1": model the handover as memory-tile-relaid (factor 1.0) and keep
        #                  the widened stage row major (relayout on the DMA).
        #   "tiled":       still widen (factor 1.0 so row_counts picks it) but keep the
        #                  core-to-core MAC-tiled handover, the model's own feasible design.
        mode = os.environ.get("STREAM_WIDE_SOFTMAX")
        wide = mode in ("1", "memtile", "tiled")
        counts = row_counts(
            [layer_cost(k) for k in kernels],
            len(rows),
            handover=1.0 if wide else TILED_HANDOVER_FACTOR,
        )
        granule = dict(kernels[0].granule()).get(0, 1)
        extent = self.workload.get_dimension_size(self.workload.get_dims(nodes[0])[0])
        width = widest_columns(extent, granule, max(counts), len(columns))
        # A stage split over two rows normally takes them contiguously, leaving its
        # single-row consumer beside only one half -- the other half's carried state then
        # crosses a non-neighbouring tile and spends a DMA channel. When the flag asks, seat
        # the consumer between the two producer rows so both handovers stay in shared memory.
        row_lists = self._flanking_rows(counts, rows) if wide else None
        offset = 0
        for i, (node, kernel, r) in enumerate(zip(nodes, kernels, counts)):
            layer_rows = row_lists[i] if row_lists is not None else rows[offset : offset + r]
            offset += r
            cores = tuple(grid[(col, row)] for row in layer_rows for col in columns[:width])
            self._assign(node, cores, ((0, width * r),))
        for node, kernel, r, r_next in zip(nodes, kernels, counts, (*counts[1:], counts[-1])):
            # "memtile" mode keeps the widened stage row major (relayout on the DMA);
            # "tiled" mode and the default keep the core-to-core MAC-tiled handover.
            if r > r_next and mode not in ("1", "memtile"):
                self._retile(node, kernel, tiled_out=True)

    def _flanking_rows(self, counts, rows):
        """Rows per stage that seat each single-row consumer between the two rows of a
        two-row producer, so the carried-state handover stays between neighbours. Only the
        one-wider-than-its-consumer case is reseated; anything else keeps contiguous rows."""
        assigned = [None] * len(counts)
        pool = list(rows)
        for i, r in enumerate(counts):
            nxt = counts[i + 1] if i + 1 < len(counts) else None
            if r == 2 and nxt == 1:
                # producer takes the outer two rows, its consumer the one between them
                trio = pool[:3]
                if len(trio) < 3:
                    return None  # not enough rows to flank; fall back to contiguous
                assigned[i] = [trio[0], trio[2]]
                assigned[i + 1] = [trio[1]]
                pool = pool[3:]
            elif assigned[i] is None:
                assigned[i] = pool[:r]
                pool = pool[r:]
        return assigned if all(a is not None for a in assigned) else None

    def _place_tenancies(self, nodes: list[ComputationNode], grid, columns, rows) -> None:
        kernels = [self.mapping.get(n).kernel for n in nodes]
        caps = [1 if len(k.granule()) == 2 else len(columns) for k in kernels]
        options = column_budget_options([layer_cost(k) for k in kernels], len(columns), caps)
        for budgets in options[1:]:
            self._pending_fallbacks.append(
                lambda m, b=budgets: self._assign_tenancy(nodes, kernels, b, grid, columns, rows, m)
            )
        self._assign_tenancy(nodes, kernels, options[0], grid, columns, rows, self.mapping)

    def _assign_tenancy(self, nodes, kernels, budgets, grid, columns, rows, mapping) -> None:
        first = 0
        for node, kernel, budget in zip(nodes, kernels, budgets):
            tenant = columns[first : first + budget]
            first += budget
            granule = dict(kernel.granule())
            width = 1
            split = [(0, self._fitting_split(node, 0, len(rows), granule.get(0, 1)))]
            if len(granule) > 2 and budget > 1:
                width = self._fitting_split(node, 2, budget, granule.get(2, 1))
                if width > 1:
                    split.append((2, width))
            cores = tuple(grid[(col, row)] for col in tenant[:width] for row in rows[: split[0][1]])
            self._assign(node, cores, tuple(split), mapping)

    def _fitting_split(self, node: ComputationNode, position: int, target: int, granule: int = 1) -> int:
        """The widest split that still hands every core whole granules."""
        extent = self.workload.get_dimension_size(self.workload.get_dims(node)[position])
        for split in range(target, 0, -1):
            if extent % (split * granule) == 0:
                return split
        return 1

    def _row_width(self, node: ComputationNode, kernel, core: Core) -> int:
        extent = self.workload.get_dimension_size(self.workload.get_dims(node)[1])
        operands = max(2, len(kernel.operand_layouts() or ()))
        budget = (core.get_memory_capacity() // 8) // (operands * BUFFERS * ELEMENT_BYTES)
        return max(w for w in range(1, min(extent, budget) + 1) if extent % w == 0)

    def _assign(
        self, node: ComputationNode, cores: tuple[Core, ...], split: tuple[tuple[int, int], ...], mapping=None
    ) -> None:
        nm = (mapping or self.mapping).get(node)
        nm.resource_allocation = (tuple(cores),)
        nm.inter_core_tiling = (tuple((LayerDim(position=p, prefix="d"), s) for p, s in split),)
        logger.info(
            "Placed %s on %s cores, split %s",
            node.name,
            len(cores),
            [(str(d), s) for d, s in nm.inter_core_tiling[0]],
        )

    def _retile(self, node: ComputationNode, kernel, mapping=None, **kwargs) -> None:
        try:
            (mapping or self.mapping).get(node).kernel = replace(kernel, **kwargs)
        except TypeError:
            logger.info("Kernel of %s takes no %s; placement keeps it as declared", node.name, sorted(kwargs))
