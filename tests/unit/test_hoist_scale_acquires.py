from xdsl.builder import ImplicitBuilder
from xdsl.context import Context
from xdsl.dialects.builtin import IntegerAttr, ModuleOp, bf16
from xdsl.ir import Block, Region
from xdsl_aie.dialects.aie import ObjectFifoAcquireOp, ObjectFifoPortEnum

from stream.compiler.transforms.hoist_scale_acquires import HoistFlashScaleAcquires


def acquire(name, port):
    return ObjectFifoAcquireOp(
        IntegerAttr.from_int_and_width(port.get_int(), 32),
        IntegerAttr.from_int_and_width(1, 32), name, (256,), bf16)


def names(module):
    return [o.objFifo_name.string_value() for o in module.walk() if isinstance(o, ObjectFifoAcquireOp)]


def build(order):
    block = Block()
    with ImplicitBuilder(block):
        for name, port in order:
            acquire(name, port)
    return ModuleOp(Region(block))


def test_scale_moves_before_the_first_consume():
    module = build([("of_out", ObjectFifoPortEnum.Produce), ("of_p", ObjectFifoPortEnum.Consume),
                    ("of_v", ObjectFifoPortEnum.Consume), ("flash_scale_a", ObjectFifoPortEnum.Consume)])
    HoistFlashScaleAcquires().apply(Context(), module)
    assert names(module) == ["of_out", "flash_scale_a", "of_p", "of_v"]


def test_already_first_stays_put():
    module = build([("flash_scale_a", ObjectFifoPortEnum.Consume), ("of_p", ObjectFifoPortEnum.Consume)])
    HoistFlashScaleAcquires().apply(Context(), module)
    assert names(module) == ["flash_scale_a", "of_p"]


def test_produce_side_is_untouched():
    module = build([("of_p", ObjectFifoPortEnum.Consume), ("flash_scale_a", ObjectFifoPortEnum.Produce)])
    HoistFlashScaleAcquires().apply(Context(), module)
    assert names(module) == ["of_p", "flash_scale_a"]
