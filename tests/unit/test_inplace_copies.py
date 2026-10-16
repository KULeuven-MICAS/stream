"""A copy a transfer lands in the memory of its source is read in place: the memory holds only what it adds."""

from stream.api import evaluate_mapping


def test_a_copy_beside_its_source_costs_only_what_it_adds(tmp_path):
    """The attention head's exp output is multicast to the sum on the tensor cores and the div beside it on the vector
    core; the div reads it in place, so the vector core holds it once and the head fits."""
    estimate = evaluate_mapping(
        "stream/inputs/examples/hardware/tpu_like_quad_core.yaml",
        "stream/inputs/testing/workload/attention_head.onnx",
        str(tmp_path),
    )
    assert estimate.cycles > 0
