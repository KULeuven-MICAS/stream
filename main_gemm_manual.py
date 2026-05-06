import argparse
import logging as _logging
from collections.abc import Sequence

from xdsl.dialects.builtin import ModuleOp, bf16
from xdsl.parser import Parser
from xdsl.passes import ModulePass, PipelinePass
from xdsl.xdsl_opt_main import xDSLOptMain
from xdsl_aie.dialects.aie import AIE
from xdsl_aie.dialects.aiex import AIEX

from stream.compiler.context.aie_context import AIEContext
from stream.compiler.dialects.stream import Stream
from stream.compiler.kernels.gemm import GemmKernel
from stream.compiler.transforms.aie_convert_ofs import AIEConvertOfs
from stream.compiler.transforms.aie_dispatch import AIEDispatchPass
from stream.compiler.transforms.aie_move_tile_ops_up import AIEMoveTileOpsUp
from stream.compiler.transforms.clear_memory_space import ClearMemorySpace
from stream.compiler.transforms.convert_stream_to_aie import ConvertStreamToAIEPass
from stream.compiler.transforms.iteration_space_to_for import IterationSpaceToFor
from stream.compiler.transforms.unroll import SpatialUnrollPass

_logging_level = _logging.INFO
_logging_format = "%(asctime)s - %(name)s.%(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
_logging.basicConfig(level=_logging_level, format=_logging_format)


def generate_mlir(m_size, n_size, k_size):
    t0 = m_size // 32 // 4
    t1 = n_size // 32 // 8
    t2 = k_size // 32
    assert m_size % (32 * 4) == 0, "M must be divisible by 128"
    assert n_size % (32 * 8) == 0, "N must be divisible by 256"
    assert k_size % 32 == 0, "K must be divisible by 32"

    return f"""
builtin.module {{

  "stream.fusion_group"() ({{
    ^bb0(%0: !stream.strensor<{m_size}(c0)x{k_size}(c2)xbf16, ["tile_0_0"]>,
         %1: !stream.strensor<{k_size}(c2)x{n_size}(c1)xbf16, ["tile_0_0"]>,
         %2: !stream.strensor<{m_size}(c0)x{n_size}(c1)xbf16, ["tile_0_0"]>):

  // Inputs
  %4 = "stream.transfer"(%0) :
        (!stream.strensor<{m_size}(c0)x{k_size}(c2)xbf16, ["tile_0_0"]>)
     -> !stream.strensor<{t0}(t0)x{t1}(t1)x{t2}(t2)|8(a1)x4(s0)x32(k0)x32(k2)xbf16,
        ["tile_0_1", "tile_1_1", "tile_2_1", "tile_3_1"]>

  %8 = "stream.transfer"(%4) :
        (!stream.strensor<{t0}(t0)x{t1}(t1)x{t2}(t2)|8(a1)x4(s0)x32(k0)x32(k2)xbf16,
        ["tile_0_1", "tile_1_1", "tile_2_1", "tile_3_1"]>)
     -> !stream.strensor<{t0}(t0)x{t1}(t1)x{t2}(t2)|8(s1)x4(s0)x32(k0)x32(k2)xbf16,
        ["tile_0_2", "tile_0_3", "tile_0_4", "tile_0_5",
         "tile_1_2", "tile_1_3", "tile_1_4", "tile_1_5",
         "tile_2_2", "tile_2_3", "tile_2_4", "tile_2_5",
         "tile_3_2", "tile_3_3", "tile_3_4", "tile_3_5",
         "tile_4_2", "tile_4_3", "tile_4_4", "tile_4_5",
         "tile_5_2", "tile_5_3", "tile_5_4", "tile_5_5",
         "tile_6_2", "tile_6_3", "tile_6_4", "tile_6_5",
         "tile_7_2", "tile_7_3", "tile_7_4", "tile_7_5"]>

  // Weights
  %5 = "stream.transfer"(%1) :
        (!stream.strensor<{k_size}(c2)x{n_size}(c1)xbf16, ["tile_0_0"]>)
     -> !stream.strensor<{t0}(t0)x{t1}(t1)x{t2}(t2)|8(s1)x4(a0)x32(k2)x32(k1)xbf16,
        ["tile_0_1", "tile_1_1", "tile_2_1", "tile_3_1",
         "tile_4_1", "tile_5_1", "tile_6_1", "tile_7_1"]>

  %9 = "stream.transfer"(%5) :
        (!stream.strensor<{t0}(t0)x{t1}(t1)x{t2}(t2)|8(s1)x4(a0)x32(k2)x32(k1)xbf16,
        ["tile_0_1", "tile_1_1", "tile_2_1", "tile_3_1",
         "tile_4_1", "tile_5_1", "tile_6_1", "tile_7_1"]>)
     -> !stream.strensor<{t0}(t0)x{t1}(t1)x{t2}(t2)|8(s1)x4(s0)x32(k2)x32(k1)xbf16,
        ["tile_0_2", "tile_0_3", "tile_0_4", "tile_0_5",
         "tile_1_2", "tile_1_3", "tile_1_4", "tile_1_5",
         "tile_2_2", "tile_2_3", "tile_2_4", "tile_2_5",
         "tile_3_2", "tile_3_3", "tile_3_4", "tile_3_5",
         "tile_4_2", "tile_4_3", "tile_4_4", "tile_4_5",
         "tile_5_2", "tile_5_3", "tile_5_4", "tile_5_5",
         "tile_6_2", "tile_6_3", "tile_6_4", "tile_6_5",
         "tile_7_2", "tile_7_3", "tile_7_4", "tile_7_5"]>

  // Compute
  %12 = "stream.computation_node"(%8, %9)
        <{{kernel = "matmul_bf16_bf16_32_32_32"}}> :
        (!stream.strensor<{t0}(t0)x{t1}(t1)x{t2}(t2)|8(s1)x4(s0)x32(k0)x32(k2)xbf16,
         ["tile_0_2", "tile_0_3", "tile_0_4", "tile_0_5",
          "tile_1_2", "tile_1_3", "tile_1_4", "tile_1_5",
          "tile_2_2", "tile_2_3", "tile_2_4", "tile_2_5",
          "tile_3_2", "tile_3_3", "tile_3_4", "tile_3_5",
          "tile_4_2", "tile_4_3", "tile_4_4", "tile_4_5",
          "tile_5_2", "tile_5_3", "tile_5_4", "tile_5_5",
          "tile_6_2", "tile_6_3", "tile_6_4", "tile_6_5",
          "tile_7_2", "tile_7_3", "tile_7_4", "tile_7_5"]>,
         !stream.strensor<{t0}(t0)x{t1}(t1)x{t2}(t2)|8(s1)x4(s0)x32(k2)x32(k1)xbf16,
         ["tile_0_2", "tile_0_3", "tile_0_4", "tile_0_5",
          "tile_1_2", "tile_1_3", "tile_1_4", "tile_1_5",
          "tile_2_2", "tile_2_3", "tile_2_4", "tile_2_5",
          "tile_3_2", "tile_3_3", "tile_3_4", "tile_3_5",
          "tile_4_2", "tile_4_3", "tile_4_4", "tile_4_5",
          "tile_5_2", "tile_5_3", "tile_5_4", "tile_5_5",
          "tile_6_2", "tile_6_3", "tile_6_4", "tile_6_5",
          "tile_7_2", "tile_7_3", "tile_7_4", "tile_7_5"]>)
     -> !stream.strensor<{t0}(t0)x{t1}(t1)|{t2}(t2)x8(s1)x4(s0)x32(k0)x32(k1)xbf16,
        ["tile_0_2", "tile_0_3", "tile_0_4", "tile_0_5",
         "tile_1_2", "tile_1_3", "tile_1_4", "tile_1_5",
         "tile_2_2", "tile_2_3", "tile_2_4", "tile_2_5",
         "tile_3_2", "tile_3_3", "tile_3_4", "tile_3_5",
         "tile_4_2", "tile_4_3", "tile_4_4", "tile_4_5",
         "tile_5_2", "tile_5_3", "tile_5_4", "tile_5_5",
         "tile_6_2", "tile_6_3", "tile_6_4", "tile_6_5",
         "tile_7_2", "tile_7_3", "tile_7_4", "tile_7_5"]>

  // Output
  %21 = "stream.transfer"(%12) :
        (!stream.strensor<{t0}(t0)x{t1}(t1)|{t2}(t2)x8(s1)x4(s0)x32(k0)x32(k1)xbf16,
        ["tile_0_2", "tile_0_3", "tile_0_4", "tile_0_5",
         "tile_1_2", "tile_1_3", "tile_1_4", "tile_1_5",
         "tile_2_2", "tile_2_3", "tile_2_4", "tile_2_5",
         "tile_3_2", "tile_3_3", "tile_3_4", "tile_3_5",
         "tile_4_2", "tile_4_3", "tile_4_4", "tile_4_5",
         "tile_5_2", "tile_5_3", "tile_5_4", "tile_5_5",
         "tile_6_2", "tile_6_3", "tile_6_4", "tile_6_5",
         "tile_7_2", "tile_7_3", "tile_7_4", "tile_7_5"]>)
     -> !stream.strensor<{t0}(t0)x{t1}(t1)|{t2}(a2)x8(s1)x4(t0)x32(k0)x32(k1)xbf16,
        ["tile_0_1", "tile_1_1", "tile_2_1", "tile_3_1",
         "tile_4_1", "tile_5_1", "tile_6_1", "tile_7_1"]>

  %22 = "stream.transfer"(%21) :
        (!stream.strensor<{t0}(t0)x{t1}(t1)|{t2}(a2)x8(s1)x4(t0)x32(k0)x32(k1)xbf16,
        ["tile_0_1", "tile_1_1", "tile_2_1", "tile_3_1",
         "tile_4_1", "tile_5_1", "tile_6_1", "tile_7_1"]>)
     -> !stream.strensor<{m_size}(c0)x{n_size}(c1)xbf16, ["tile_0_0"]>

  "stream.yield"(%22) :
        (!stream.strensor<{m_size}(c0)x{n_size}(c1)xbf16, ["tile_0_0"]>) -> ()

  }}) : () -> ()
}}
"""


class StreamMain(xDSLOptMain):
    ctx: AIEContext

    def __init__(
        self,
        description: str = "SNAX modular optimizer driver",
        args: Sequence[str] | None = None,
    ):
        self.available_frontends = {}
        self.available_passes = {}
        self.available_targets = {}

        self.ctx = AIEContext()
        self.register_all_dialects()
        self.register_all_frontends()
        self.register_all_passes()
        self.register_all_targets()
        self.register_default_kernels()

        self.ctx.allow_unregistered = True

        self.setup_pipeline()

    def setup_pipeline(self):
        def callback(previous_pass: ModulePass, module: ModuleOp, next_pass: ModulePass) -> None:
            module.verify()

        pass_pipeline: list[ModulePass] = []

        # Transform passes:
        pass_pipeline.append(SpatialUnrollPass())
        pass_pipeline.append(AIEDispatchPass())
        pass_pipeline.append(IterationSpaceToFor())
        pass_pipeline.append(AIEConvertOfs())
        pass_pipeline.append(ConvertStreamToAIEPass())
        pass_pipeline.append(AIEMoveTileOpsUp())
        pass_pipeline.append(ClearMemorySpace())

        self.pipeline = PipelinePass(
            tuple(pass_pipeline),
            callback,
        )

    def register_default_kernels(self):
        for kernel in (GemmKernel(1, bf16, 32, 32, 32, "layout"),):
            self.ctx.registered_kernels[kernel.unique_name] = kernel

    def register_all_passes(self):
        self.register_pass("unroll", lambda: SpatialUnrollPass)
        self.register_pass("aie-dispatch", lambda: AIEDispatchPass)
        self.register_pass("iteration-space-to-for", lambda: IterationSpaceToFor)
        self.register_pass("aie-convert-ofs", lambda: AIEConvertOfs)
        self.register_pass("convert-stream-to-aie", lambda: ConvertStreamToAIEPass)
        self.register_pass("clear-memory-space", lambda: ClearMemorySpace)
        self.register_pass("aie-move-tile-ops-up", lambda: AIEMoveTileOpsUp)

    def register_all_dialects(self):
        super().register_all_dialects()
        self.ctx.register_dialect("aie", lambda: AIE)
        self.ctx.register_dialect("aiex", lambda: AIEX)
        if "stream" in self.ctx._registered_dialects:
            del self.ctx._registered_dialects["stream"]
        self.ctx.register_dialect("stream", lambda: Stream)

    def register_all_targets(self):
        super().register_all_targets()

    def register_all_frontends(self):
        super().register_all_frontends()


def run_main_aie_codegen_gemm(m_size, k_size, n_size):
    stream_main = StreamMain()
    module = Parser(
        stream_main.ctx,
        generate_mlir(m_size, n_size, k_size),
    ).parse_module()
    stream_main.pipeline.apply(stream_main.ctx, module)
    save_path = f"outputs/swiglu_module_{m_size}_{n_size}_{k_size}.mlir"
    with open(save_path, "w") as f:
        f.write(str(module))
    print(f"Saved generated module to {save_path}")

    return module


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AIE code generation for Gemm")
    parser.add_argument("--M", type=int, required=True, help="M parameter for the model")
    parser.add_argument("--N", type=int, required=True, help="N parameter for the model")
    parser.add_argument("--K", type=int, required=True, help="K parameter for the model")
    parser.add_argument("--m", type=int, default=32, help="m parameter for the model (default: 32)")
    parser.add_argument("--k", type=int, default=32, help="k parameter for the model (default: 32)")
    parser.add_argument("--n", type=int, default=32, help="n parameter for the model (default: 32)")
    parser.add_argument("--in_dtype", type=str, default="i16", help="Input data type (default: i16)")
    parser.add_argument("--out_dtype", type=str, default="i32", help="Output data type (default: i32)")
    parser.add_argument("--trace_size", type=int, default=1048576, help="Size of the trace buffer (default: 1048576)")
    parser.add_argument("--rows", type=int, default=2, help="Number of AIE rows to use (default: 2)")
    parser.add_argument("--cols", type=int, default=2, help="Number of AIE columns to use (default: 2)")
    parser.add_argument("--npu", type=str, default="npu2", help="NPU type to target (default: npu2)")
    args = parser.parse_args()

    stream_main = StreamMain()
    module = Parser(
        stream_main.ctx,
        generate_mlir(args.M, args.N, args.K),
    ).parse_module()
    stream_main.pipeline.apply(stream_main.ctx, module)
    save_path = f"outputs/swiglu_module_{args.M}_{args.N}_{args.K}.mlir"
    with open(save_path, "w") as f:
        f.write(str(module))
    print(f"Saved generated module to {save_path}")
