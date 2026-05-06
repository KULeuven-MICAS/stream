import argparse
from collections.abc import Sequence

from xdsl.dialects.builtin import bf16
from xdsl.xdsl_opt_main import xDSLOptMain
from xdsl_aie.dialects.aie import AIE
from xdsl_aie.dialects.aiex import AIEX

from stream.compiler.context.aie_context import AIEContext
from stream.compiler.dialects.stream import Stream
from stream.compiler.kernels.eltwise_mul import EltwiseMulKernel
from stream.compiler.kernels.gemm import GemmKernel
from stream.compiler.kernels.silu import SiluKernel
from stream.compiler.transforms.aie_convert_ofs import AIEConvertOfs
from stream.compiler.transforms.aie_dispatch import AIEDispatchPass
from stream.compiler.transforms.aie_move_tile_ops_up import AIEMoveTileOpsUp
from stream.compiler.transforms.clear_memory_space import ClearMemorySpace
from stream.compiler.transforms.convert_stream_to_aie import ConvertStreamToAIEPass
from stream.compiler.transforms.iteration_space_to_for import IterationSpaceToFor
from stream.compiler.transforms.unroll import SpatialUnrollPass


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

        # arg handling
        arg_parser = argparse.ArgumentParser(description=description)
        self.register_all_arguments(arg_parser)
        self.args = arg_parser.parse_args(args=args)

        self.ctx.allow_unregistered = self.args.allow_unregistered_dialect

        self.setup_pipeline()

    def register_default_kernels(self):
        for kernel in (
            GemmKernel(1, bf16, 32, 32, 64, "layout"),
            GemmKernel(1, bf16, 32, 64, 32, "layout"),
            GemmKernel(1, bf16, 32, 32, 32, "layout"),
            SiluKernel(1, bf16),
            EltwiseMulKernel(1, bf16),
        ):
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

        # def parse_onnx(io: IO[str]) -> ModuleOp:
        #     assert isinstance(io.name, str)
        #     return OnnxParser(io.name).parse()
        #
        # self.available_frontends["onnx"] = parse_onnx


def main():
    stream_main = StreamMain()
    stream_main.run()


if "__main__" == __name__:
    main()
