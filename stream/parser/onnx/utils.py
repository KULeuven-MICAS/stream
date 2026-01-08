from onnx import TensorProto, ValueInfoProto
from zigzag.parser.onnx.utils import OnnxTensorType
from xdsl.dialects.builtin import FixedBitwidthType, i32, bf16, i8, i16, i32, f32

from stream.workload.workload import Tensor


def onnx_tensor_to_tensor(tensor: ValueInfoProto | OnnxTensorType | TensorProto) -> Tensor:
    onnx_type_to_xdsl_type: dict[int, FixedBitwidthType] = {
        TensorProto.FLOAT: f32,
        TensorProto.BFLOAT16: bf16,
        TensorProto.INT8: i8,
        TensorProto.INT16: i16,
        TensorProto.INT32: i32,
    }
    if isinstance(tensor, ValueInfoProto):
        return Tensor(
            onnx_type_to_xdsl_type[tensor.type.tensor_type.elem_type],
            tuple(d.dim_value for d in tensor.type.tensor_type.shape.dim),
        )
    elif isinstance(tensor, OnnxTensorType):
        return Tensor(onnx_type_to_xdsl_type[tensor.elem_type], tuple(tensor.shape))
    else:
        return Tensor(onnx_type_to_xdsl_type[tensor.data_type], tuple(tensor.dims))
