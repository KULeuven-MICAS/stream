from typing import cast

from snaxc.dialects.tsl import TiledStridedLayoutAttr
from xdsl.context import Context
from xdsl.dialects import builtin, func
from xdsl.ir import Attribute, BlockArgument
from xdsl.passes import ModulePass
from xdsl_aie.dialects.aie import ObjectFIFO, ObjectFIFOSubview


class ClearMemorySpace(ModulePass):
    name = "clear-memory-space"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        # helper function to clear the memory space of a memref
        # also clears the layout information of the memref - not used anymore
        def clear_memory_space(t: Attribute) -> Attribute:
            if isinstance(t, builtin.MemRefType):
                if isinstance(t.layout, TiledStridedLayoutAttr):
                    return builtin.MemRefType(
                        t.element_type,
                        t.get_shape(),
                        builtin.NoneAttr(),
                        builtin.NoneAttr(),
                    )
                else:
                    return builtin.MemRefType(
                        t.element_type,
                        t.get_shape(),
                        t.layout,
                        builtin.NoneAttr(),
                    )

            if isinstance(t, ObjectFIFO):
                t = cast(ObjectFIFO[Attribute], t)
                return ObjectFIFO([clear_memory_space(t.buffer)])
            if isinstance(t, ObjectFIFOSubview):
                t = cast(ObjectFIFOSubview[Attribute], t)
                return ObjectFIFOSubview([clear_memory_space(t.buffer)])
            return t

        for op_in_module in op.walk():
            for operand in op_in_module.operands:
                operand.type = clear_memory_space(operand.type)
            for result in op_in_module.results:
                result.type = clear_memory_space(result.type)
            for key, property in op_in_module.properties.items():
                op_in_module.properties[key] = clear_memory_space(property)

            if isinstance(op_in_module, func.FuncOp):
                # special case for func ops because func ops do not have
                # operands, they have function_types which have ins & outs
                # Define new function type with updated inputs and outputs
                # mapped to a default memory space
                new_function_type = builtin.FunctionType.from_lists(
                    list(map(clear_memory_space, op_in_module.function_type.inputs)),
                    list(map(clear_memory_space, op_in_module.function_type.outputs)),
                )

                op_in_module.function_type = new_function_type

                # change block args ssa values
                if op_in_module.body.blocks:
                    old_args = [old_arg for old_arg in op_in_module.body.block._args]  # pyright: ignore
                    new_args = [
                        BlockArgument(
                            clear_memory_space(old_arg.type),
                            op_in_module.body.block,
                            index,
                        )
                        for index, old_arg in enumerate(old_args)
                    ]
                    for old_arg, new_arg in zip(old_args, new_args, strict=False):
                        old_arg.replace_by(new_arg)
                    op_in_module.body.block._args = tuple(new_args)  # pyright: ignore
