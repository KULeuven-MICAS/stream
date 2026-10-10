"""Data layout: each tensor copy has an axis order, kernels say which axes they read together, and transfers convert
between layouts on the fly, at the bandwidth their contiguous runs get."""

import tempfile

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from stream.api import SolveOptions, evaluate_mapping
from stream.cost_model.bandwidth import BandwidthModel
from stream.cost_model.layout import Layout, shared_run_bytes, transfer_runs
from stream.opt.allocation.constraint_optimization import layouts as copy_layouts

TPU_V7 = "stream/inputs/examples/hardware/tpu_v7_ironwood.yaml"
BF16 = 16


def test_a_layout_moves_axes_innermost_keeping_the_others_in_order():
    layout = Layout.row_major(4)
    assert layout.with_innermost({1}) == Layout((0, 2, 3, 1))
    assert layout.with_innermost({1, 3}) == Layout((0, 2, 1, 3))
    assert layout.with_innermost({3, 2}) is layout  # already innermost, in any order
    assert layout.with_innermost(()) is layout


def test_a_block_is_contiguous_up_to_the_first_axis_it_does_not_span():
    full = (1024, 4096)
    assert Layout.row_major(2).contiguous_bytes((64, 512), full, BF16) == 512 * 2
    assert Layout.row_major(2).contiguous_bytes((8, 4096), full, BF16) == 8 * 4096 * 2
    assert Layout((1, 0)).contiguous_bytes((64, 512), full, BF16) == 64 * 2
    assert shared_run_bytes((64, 512), full, BF16, Layout((0, 1)), Layout((1, 0))) == 2
    three = (8, 16, 32)
    assert shared_run_bytes(three, three, BF16, Layout((1, 0, 2)), Layout((0, 1, 2))) == 32 * 2


def test_a_transfer_converts_on_the_side_that_loses_least():
    """Row-major to transposed: the DMA reads contiguously and scatters its writes, or gathers its reads and writes
    contiguously, whichever side tolerates short runs."""
    block = full = (256, 128)
    source, target = Layout((0, 1)), Layout((1, 0))
    burst = BandwidthModel(ceiling=1.0, contiguous=1.0, strided={"read": {}, "write": {}}, burst=64)

    def bursty(span):
        return burst.efficiency(span, "read")

    def flat(_):
        return 1.0

    assert transfer_runs(block, full, BF16, source, target, bursty, flat) == (256 * 128 * 2, 2)
    assert transfer_runs(block, full, BF16, source, target, flat, bursty) == (2, 256 * 128 * 2)
    assert transfer_runs(block, full, BF16, source, source, bursty, bursty) == (256 * 128 * 2,) * 2


def test_a_burst_memory_uses_only_the_part_of_each_access_a_short_run_fills():
    model = BandwidthModel.from_description({"ceiling": 100, "contiguous": 100, "burst": 32})
    assert model.efficiency(2, "read") == pytest.approx(2 / 32)
    assert model.efficiency(48, "write") == pytest.approx(48 / 64)
    assert model.efficiency(4096, "read") == 1.0
    assert BandwidthModel.flat(100).efficiency(2, "read") == 1.0


def _projection(path, weights_are_parameters: bool):
    """y = x @ w, with w a model parameter (an initializer) or an input the host provides."""
    x = helper.make_tensor_value_info("x", TensorProto.BFLOAT16, [512, 1024])
    y = helper.make_tensor_value_info("y", TensorProto.BFLOAT16, [512, 2048])
    w = numpy_helper.from_array(np.zeros((1024, 2048), np.float32), "w")
    w.data_type = TensorProto.BFLOAT16
    w.ClearField("raw_data")
    inputs, initializers = [x], []
    if weights_are_parameters:
        initializers.append(w)
    else:
        inputs.append(helper.make_tensor_value_info("w", TensorProto.BFLOAT16, [1024, 2048]))
    node = helper.make_node("MatMul", ["x", "w"], ["y"], name="projection")
    graph = helper.make_graph([node], "projection", inputs, [y], initializers)
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)]), path)
    return str(path)


def _layouts(path, tmp_path):
    spaces = []
    original = copy_layouts.CopyLayouts.__init__

    def record(self, space):
        original(self, space)
        spaces.append(space)

    copy_layouts.CopyLayouts.__init__ = record
    try:
        estimate = evaluate_mapping(TPU_V7, path, str(tmp_path), options=SolveOptions(artifacts=False))
    finally:
        copy_layouts.CopyLayouts.__init__ = original
    space = spaces[-1]
    return estimate, space, {tr.name: tr for tr in space.transfer_nodes}


def test_the_mxu_reads_a_matmul_operand_with_its_contraction_innermost(tmp_path):
    """The MXU unrolls the contraction over its rows, so the weights' contraction axis is what one memory word feeds;
    its output is written along the axis its spatial mapping unrolls widest, here the rows."""
    _, space, _ = _layouts(_projection(tmp_path / "p.onnx", True), tmp_path)
    node = next(n for n in space.ssc_nodes if n.name == "projection")
    _, w, y = node.tensors
    assert space.layouts.needs(node, w) == {0}
    assert space.layouts.needs(node, y) == {0}


def test_a_parameter_is_packed_for_its_kernel_and_crosses_no_conversion(tmp_path):
    estimate, space, transfers = _layouts(_projection(tmp_path / "p.onnx", True), tmp_path)
    weights = [tr for name, tr in transfers.items() if name.startswith("Transfer(w")]
    assert weights
    for tr in weights:
        assert space.layouts.of(tr.inputs[0]) == space.layouts.of(tr.outputs[0]) == Layout((1, 0))
    rows = estimate.context.get("allocation").get_ir()["performance"]["layouts"]
    assert [r["transfer"] for r in rows] == ["Transfer(y_2)"]


def test_a_host_input_is_converted_by_the_last_dma_that_can_and_in_the_memory_that_costs_nothing(tmp_path):
    """The host hands the weights row-major. VMEM feeds the MXU in place, so the HBM-to-VMEM DMA lays them out for
    the MXU: it reads HBM in long runs, whose 32-byte bursts short runs would waste, and scatters into VMEM."""
    estimate, space, transfers = _layouts(_projection(tmp_path / "h.onnx", False), tmp_path)
    into_vmem, into_mxu = transfers["Transfer(w)"], transfers["Transfer(w_1)"]
    assert space.layouts.of(into_vmem.inputs[0]) == Layout((0, 1))
    assert space.layouts.of(into_vmem.outputs[0]) == space.layouts.of(into_mxu.outputs[0]) == Layout((1, 0))
    read, write = space.layouts.runs(into_vmem, space.path_choices[into_vmem][0])
    assert read >= 32 and write == BF16 / 8
    rows = estimate.context.get("allocation").get_ir()["performance"]["layouts"]
    assert [(r["transfer"], r["source_order"], r["target_order"]) for r in rows] == [
        ("Transfer(w)", [0, 1], [1, 0]),
        ("Transfer(y_2)", [1, 0], [0, 1]),  # written with its rows innermost, handed back to the host row-major
    ]


def test_outputs_and_host_inputs_are_row_major(tmp_path):
    _, space, transfers = _layouts(_projection(tmp_path / "h.onnx", False), tmp_path)
    assert space.layouts.of(transfers["Transfer(x)"].inputs[0]) == Layout((0, 1))
    assert space.layouts.of(transfers["Transfer(y_2)"].outputs[0]) == Layout((0, 1))


def test_every_layout_is_a_permutation_of_its_tensors_axes(tmp_path):
    _, space, _ = _layouts(_projection(tmp_path / "h.onnx", False), tmp_path)
    for tr in space.transfer_nodes:
        for tensor in (*tr.inputs, *tr.outputs):
            assert sorted(space.layouts.of(tensor).order) == list(range(len(tensor.shape)))


def test_a_copy_staying_in_one_memory_keeps_its_layout():
    """VMEM feeds the MXU in place: nothing moves, so nothing can be reordered."""
    with tempfile.TemporaryDirectory() as tmp:
        _, space, _ = _layouts(_projection(f"{tmp}/h.onnx", False), tmp)
        for tr in space.transfer_nodes:
            if space.within_one_memory(tr):
                assert space.layouts.of(tr.inputs[0]) == space.layouts.of(tr.outputs[0])


def test_an_axis_of_one_element_moves_no_data():
    """[512, 1] laid out either way holds the same bytes in the same order."""
    full = (512, 1)
    assert Layout((1, 0)).same_data_order(Layout((0, 1)), full)
    assert Layout((1, 0)).contiguous_bytes(full, full, BF16) == 512 * 2

    def rate(_):
        return 1.0

    assert transfer_runs(full, full, BF16, Layout((1, 0)), Layout((0, 1)), rate, rate) == (1024, 1024)


def test_the_conversion_side_is_chosen_by_rate_not_by_efficiency_alone():
    """A narrow read port that needs long runs and a wide write port that tolerates short ones: scattering on the wide
    side is faster even though it loses more of that side's bandwidth."""
    block = full = (256, 128)
    narrow = BandwidthModel(ceiling=1, contiguous=1, strided={"read": {}, "write": {}}, burst=32)
    wide = BandwidthModel(ceiling=100, contiguous=100, strided={"read": {}, "write": {}}, burst=64)

    def read(span):
        return narrow.contiguous * narrow.efficiency(span, "read")

    def write(span):
        return wide.contiguous * wide.efficiency(span, "write")

    assert transfer_runs(block, full, BF16, Layout((0, 1)), Layout((1, 0)), read, write) == (256 * 128 * 2, 2)


def test_a_multicast_writes_all_its_copies_in_one_layout(tmp_path):
    """x feeds one matmul as it is and another through a transpose; the copies one transfer makes share a layout."""
    value = lambda name, shape: helper.make_tensor_value_info(name, TensorProto.BFLOAT16, shape)  # noqa: E731
    nodes = [
        helper.make_node("MatMul", ["x", "w1"], ["y1"], name="direct"),
        helper.make_node("Transpose", ["x"], ["xt"], perm=[1, 0], name="transpose"),
        helper.make_node("MatMul", ["xt", "w2"], ["y2"], name="transposed"),
        helper.make_node("Add", ["y1", "y2"], ["y"], name="add"),
    ]
    inputs = [value("x", [256, 256]), value("w1", [256, 256]), value("w2", [256, 256])]
    graph = helper.make_graph(nodes, "two", inputs, [value("y", [256, 256])])
    path = str(tmp_path / "two.onnx")
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)]), path)
    _, space, _ = _layouts(path, tmp_path)
    for tr in space.transfer_nodes:
        assert len({space.layouts.of(t) for t in tr.outputs}) == 1


@pytest.mark.parametrize("burst", [0, -32, 32.5, "32"])
def test_a_burst_is_a_positive_whole_number_of_bytes(burst):
    from zigzag.utils import open_yaml  # noqa: PLC0415

    from stream.parser.accelerator_validator import AcceleratorValidator  # noqa: PLC0415

    data = open_yaml(TPU_V7)
    data["bandwidth"]["16.hbm3e.rw_port_1"]["burst"] = burst
    validator = AcceleratorValidator(data, accelerator_path=TPU_V7)
    assert not validator.validate()
    assert any("burst" in error for error in validator.errors)
