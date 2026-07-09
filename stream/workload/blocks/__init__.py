"""Registry of parameterized reference blocks, buildable by key via :func:`build_block`.

Out-of-tree blocks register through the ``stream.workload_blocks`` entry-point group.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import entry_points
from typing import Any

from stream.workload.blocks.library import (
    ChunkedSSMConfig,
    FlashAttentionConfig,
    MoEConfig,
    RMSNormConfig,
    SwiGLUConfig,
    build_chunked_ssm_block,
    build_flash_attention_block,
    build_moe_block,
    build_rmsnorm_block,
    build_swiglu_block,
)
from stream.workload.models import (
    MODEL_CATALOG,
    AttentionConfig,
    GQAConfig,
    KVCacheConfig,
    LinearAttentionConfig,
    MambaConfig,
)
from stream.workload.workload import Workload

logger = logging.getLogger(__name__)

__all__ = [
    "BlockSpec",
    "register_block",
    "get_block",
    "available_blocks",
    "build_block",
]

ENTRY_POINT_GROUP = "stream.workload_blocks"


@dataclass(frozen=True)
class BlockSpec:
    """A named, parameterized block; ``build(**params)`` returns a fresh ``Workload`` (unknown params raise)."""

    key: str
    label: str
    description: str
    build: Callable[..., Workload]


_REGISTRY: dict[str, BlockSpec] = {}
_LOAD_STATE: dict[str, bool] = {"plugins": False}


def register_block(spec: BlockSpec) -> BlockSpec:
    """Register ``spec`` under its ``key`` (idempotent; last registration wins)."""
    _REGISTRY[spec.key] = spec
    return spec


def _cfg_builder(build_fn: Callable[[Any], Workload], cfg_cls: type) -> Callable[..., Workload]:
    """Wrap a ``build_fn(config)`` into a uniform ``build(**params)`` that constructs the config."""
    return lambda **params: build_fn(cfg_cls(**params))


# built-ins: catalog configs + the modern library blocks
_CATALOG_CONFIGS: dict[str, type] = {
    "attention": AttentionConfig,
    "gqa": GQAConfig,
    "linear_attention": LinearAttentionConfig,
    "mamba": MambaConfig,
    "kv_cache": KVCacheConfig,
}

for _spec in MODEL_CATALOG:
    register_block(
        BlockSpec(_spec.key, _spec.label, _spec.description, _cfg_builder(_spec.build, _CATALOG_CONFIGS[_spec.key]))
    )

register_block(
    BlockSpec(
        "swiglu",
        "SwiGLU MLP",
        "Gated feed-forward: affine projections + Silu/Mul.",
        _cfg_builder(build_swiglu_block, SwiGLUConfig),
    )
)
register_block(
    BlockSpec(
        "rmsnorm",
        "RMSNorm",
        "Reduce-then-broadcast normalization + learned scale.",
        _cfg_builder(build_rmsnorm_block, RMSNormConfig),
    )
)
register_block(
    BlockSpec(
        "moe",
        "Mixture-of-Experts",
        "Dense per-expert GEMMs with data-dependent dispatch/combine; capacity C is a DSE lever.",
        _cfg_builder(build_moe_block, MoEConfig),
    )
)
register_block(
    BlockSpec(
        "chunked_ssm",
        "Chunked SSM",
        "SEQUENTIAL scan decomposed into a per-chunk reduction chain; chunk size is a DSE lever.",
        _cfg_builder(build_chunked_ssm_block, ChunkedSSMConfig),
    )
)
register_block(
    BlockSpec(
        "flash_attention",
        "Flash Attention (online softmax)",
        "Attention decomposed into an online-softmax scan over key blocks; block size is a DSE lever.",
        _cfg_builder(build_flash_attention_block, FlashAttentionConfig),
    )
)


def _load_plugins() -> None:
    """Discover out-of-tree blocks declared under the ``stream.workload_blocks`` entry-point group."""
    if _LOAD_STATE["plugins"]:
        return
    _LOAD_STATE["plugins"] = True
    try:
        eps = entry_points(group=ENTRY_POINT_GROUP)
    except Exception as exc:  # pragma: no cover - importlib.metadata edge cases
        logger.debug("block entry-point discovery failed: %s", exc)
        return
    for ep in eps:
        try:
            obj = ep.load()
            register_block(obj() if callable(obj) and not isinstance(obj, BlockSpec) else obj)
        except Exception as exc:  # pragma: no cover - a broken plugin must not break the registry
            logger.warning("skipping block plugin %r: %s", ep.name, exc)


def get_block(key: str) -> BlockSpec:
    """The registered block for ``key`` (loads plugins first). Raises ``KeyError`` if unknown."""
    _load_plugins()
    return _REGISTRY[key]


def available_blocks() -> tuple[BlockSpec, ...]:
    """Every registered block (built-in + discovered plugins)."""
    _load_plugins()
    return tuple(_REGISTRY.values())


def build_block(key: str, **params: Any) -> Workload:
    """Build the block ``key`` with per-block ``params`` (defaults when omitted)."""
    return get_block(key).build(**params)
