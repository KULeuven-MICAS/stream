import logging
import os

import yaml

from stream.mapping.generator import MappingGenerator
from stream.stages.context import StageContext
from stream.stages.stage import Stage, StageCallable

logger = logging.getLogger(__name__)


class MappingGenerationStage(Stage):
    REQUIRED_FIELDS = ("accelerator", "workload")

    def __init__(
        self,
        list_of_callables: list[StageCallable],
        ctx: StageContext,
    ):
        super().__init__(list_of_callables, ctx)
        self.accelerator = self.ctx.require_value("accelerator", self.__class__.__name__)
        self.workload = self.ctx.require_value("workload", self.__class__.__name__)
        self.output_path = self.ctx.require_value("output_path", self.__class__.__name__)
        seq_len_tile_size = self.ctx.get("seq_len_tile_size", 32)
        embedding_tile_size = self.ctx.get("embedding_tile_size", 128)
        hidden_tile_size = self.ctx.get("hidden_tile_size", 64)
        last_gemm_down = self.ctx.get("last_gemm_down", False)
        max_nb_mappings = self.ctx.get("max_nb_mappings", 200)
        self.mapping_generator = MappingGenerator(
            accelerator=self.accelerator,
            workload=self.workload,
            output_dir=self.output_path,
            seq_len_tile_size=seq_len_tile_size,
            embedding_tile_size=embedding_tile_size,
            hidden_tile_size=hidden_tile_size,
            last_gemm_down=last_gemm_down,
            max_variants=max_nb_mappings,
            layer_core_splits={
                "Gemm_Left": [4, 8, 16],
                "Gemm_Right": [4, 8, 16],
                "Silu": [1, 4, 8],
                "Elt_Mul": [1, 4, 8],
                "Gemm_Down": [4, 8, 16],  # only used if last_gemm_down=True
            },
        )

    def run(self):
        best_mapping_path = None
        best_context = None
        best_latency = float("inf")
        for i, variant, mapping in self.mapping_generator.run():
            output_path_i = os.path.join(self.output_path, f"{i}")
            os.makedirs(output_path_i, exist_ok=True)

            # Save mapping into the same folder passed onward to substages
            mapping_path = self.mapping_generator.save_mapping(
                mapping=mapping,
                variant=variant,
                idx=i,
                output_dir=output_path_i,
            )

            self.ctx.set(
                workload=self.workload,
                mapping_path=mapping_path,
                output_path=output_path_i,
            )
            logger.info(f"Evaluating mapping: {mapping_path}")
            sub_stage = self.list_of_callables[0](self.list_of_callables[1:], self.ctx)
            try:
                ctxs = list(sub_stage.run())
                ctx = ctxs[0]
                scheduler = ctx.get("scheduler", None)
                latency = scheduler.latency_total
            except RuntimeError as e:
                logger.error(f"Error evaluating mapping {mapping_path}: {e}")
                latency = float("inf")  # treat errors as infinite latency
            assert len(ctxs) == 1, f"Expected exactly one context, but got {len(ctxs)}"
            if best_latency is None or latency < best_latency:
                best_latency = latency
                best_mapping_path = mapping_path
                best_context = ctx
            # Save the latency to yaml for later analysis
            latency_yaml_path = os.path.join(output_path_i, "latency.yaml")
            with open(latency_yaml_path, "w") as f:
                yaml.dump({"latency": latency}, f)
            logger.info(f"Mapping {mapping_path} has latency {latency}")
        logger.info(f"Best mapping found with latency {best_latency}: {best_mapping_path}")
        yield best_context
