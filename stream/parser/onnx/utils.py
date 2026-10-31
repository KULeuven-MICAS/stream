from onnx import TensorProto, ValueInfoProto
from xdsl.dialects.builtin import FixedBitwidthType, IntegerType, Signedness, _FloatType, bf16, f16, f32, i8, i16, i32
from xdsl.ir import ParametrizedAttribute
from xdsl.irdl import irdl_attr_definition
from zigzag.parser.onnx.utils import OnnxTensorType

from stream.workload.workload import Tensor


# Built on xdsl's private float base: xdsl 0.29 has no 4- or 8-bit float types.
class _NarrowFloatType(ParametrizedAttribute, _FloatType):
    @property
    def format(self) -> str:
        raise NotImplementedError(f"{self.name} values have no struct format to be packed in.")


class _Float8Type(_NarrowFloatType):
    @property
    def bitwidth(self) -> int:
        return 8


@irdl_attr_definition
class Float8E4M3FNType(_Float8Type):
    name = "f8E4M3FN"


@irdl_attr_definition
class Float8E4M3FNUZType(_Float8Type):
    name = "f8E4M3FNUZ"


@irdl_attr_definition
class Float8E5M2Type(_Float8Type):
    name = "f8E5M2"


@irdl_attr_definition
class Float8E5M2FNUZType(_Float8Type):
    name = "f8E5M2FNUZ"


@irdl_attr_definition
class Float4E2M1FNType(_NarrowFloatType):
    name = "f4E2M1FN"

    @property
    def bitwidth(self) -> int:
        return 4


def onnx_tensor_to_tensor(tensor: ValueInfoProto | OnnxTensorType | TensorProto, name: str | None = None) -> Tensor:
    name = tensor.name if name is None else name
    onnx_type_to_xdsl_type: dict[int, FixedBitwidthType] = {
        TensorProto.FLOAT: f32,
        TensorProto.FLOAT16: f16,
        TensorProto.BFLOAT16: bf16,
        TensorProto.FLOAT8E4M3FN: Float8E4M3FNType(),
        TensorProto.FLOAT8E4M3FNUZ: Float8E4M3FNUZType(),
        TensorProto.FLOAT8E5M2: Float8E5M2Type(),
        TensorProto.FLOAT8E5M2FNUZ: Float8E5M2FNUZType(),
        TensorProto.FLOAT4E2M1: Float4E2M1FNType(),
        TensorProto.INT4: IntegerType(4),
        TensorProto.UINT4: IntegerType(4, Signedness.UNSIGNED),
        TensorProto.INT8: i8,
        TensorProto.UINT8: IntegerType(8, Signedness.UNSIGNED),
        TensorProto.INT16: i16,
        TensorProto.INT32: i32,
    }
    if isinstance(tensor, ValueInfoProto):
        elem_type = tensor.type.tensor_type.elem_type
        shape = tuple(d.dim_value for d in tensor.type.tensor_type.shape.dim)
    elif isinstance(tensor, OnnxTensorType):
        elem_type = tensor.elem_type
        shape = tuple(tensor.shape)
    else:
        elem_type = tensor.data_type
        shape = tuple(tensor.dims)
    if elem_type not in onnx_type_to_xdsl_type:
        raise NotImplementedError(f"Tensor {name} has unsupported element type {TensorProto.DataType.Name(elem_type)}.")
    operand_type = onnx_type_to_xdsl_type[elem_type]
    return Tensor.create(name=name, operand_type=operand_type, shape=shape)
