"""Give a node the state its kernel keeps between iterations.

Declared by the kernel and materialised here as an operand read at ``carry - 1``, which is
what :func:`~stream.workload.iterator_type.is_state_operand` recognises. An input and not an
output because the state never moves: it is resident on the cores its node runs on.
"""

from dataclasses import replace

from xdsl.ir.affine import AffineDimExpr, AffineExpr, AffineMap

from stream.mapping.mapping import Mapping
from stream.stages.stage import Stage, StageCallable, StageContext
from stream.workload.node import ComputationNode
from stream.workload.tensor import Tensor
from stream.workload.workload import Workload


def _extent(node: ComputationNode, position: int) -> int | None:
    """How far the node's iteration space runs along ``position``.

    Read off whichever operand indexes it directly, which is the same reading
    :meth:`Workload.get_dimension_sizes` takes.
    """
    for tensor, mapping in zip(node.tensors, node.operand_mapping, strict=True):
        for expr, size in zip(mapping.results, tensor.shape, strict=True):
            if isinstance(expr, AffineDimExpr) and expr.position == position:
                return size
    return None


def state_tensor(node: ComputationNode, state, positions: dict[str, int]) -> tuple[Tensor, AffineMap] | None:
    """The operand and map that make ``state`` a recurrence on ``node``: ``rows`` per step of the carried
    dimension, read at ``-1``, indexed by the dimension the node splits over. The carry costs ``rows`` however
    the carried dimension is tiled, since the extent reaching the tensor is clipped to its shape."""
    carried, index = positions[state.carried_over], positions[state.indexed_by]
    indexed = _extent(node, index)
    if indexed is None or _extent(node, carried) is None:
        return None
    rank = len(node.operand_mapping[0].results) if node.operand_mapping else 0
    dims = max(rank, carried + 1, index + 1)
    carry: AffineExpr = AffineDimExpr(carried) - 1
    mapping = AffineMap(dims, 0, (carry, AffineDimExpr(index)))
    tensor = Tensor.create(f"{state.name}_{node.name}", node.outputs[0].operand_type, (state.rows, indexed))
    return tensor, mapping


class KernelStateStage(Stage):
    """Add each node's kernel-carried state to it, before anything reads the iteration space."""

    REQUIRED_FIELDS = ("workload", "mapping")

    def __init__(self, list_of_callables: list[StageCallable], ctx: StageContext):
        super().__init__(list_of_callables, ctx)
        self.workload: Workload = self.ctx.get("workload")
        self.mapping: Mapping = self.ctx.get("mapping")

    def run(self):
        rebuilt, changed = {}, False
        for node in self.workload.nodes:
            kernel = self._kernel(node)
            declared = list(kernel.state_operands()) if kernel is not None else []
            if not declared:
                rebuilt[node.name] = node
                continue
            inputs, maps = list(node.inputs), list(node.operand_mapping)
            positions = kernel.dim_positions(node)
            for state in declared:
                built = state_tensor(node, state, positions)
                if built is None:
                    continue
                tensor, mapping = built
                # operand_mapping is inputs then outputs, so the state goes before the output.
                inputs.append(tensor)
                maps.insert(len(inputs) - 1, mapping)
                changed = True
            rebuilt[node.name] = replace(node, inputs=tuple(inputs), operand_mapping=tuple(maps))
        if changed:
            workload = Workload(rebuilt.values())
            self.ctx.set(workload=workload, mapping=self.mapping.with_updated_workload(workload, self.workload))
            for field in ("placement_alternatives", "placement_reserves"):
                if shapes := self.ctx.data.get(field):
                    self.ctx.set(**{field: [s.with_updated_workload(workload, self.workload) for s in shapes]})
        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], self.ctx)
        yield from sub_stage.run()

    def _kernel(self, node):
        if not isinstance(node, ComputationNode):
            return None
        return getattr(self.mapping.get(node), "kernel", None)
