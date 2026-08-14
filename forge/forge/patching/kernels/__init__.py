"""Registry of kernel-specific patch adapters.

``core.py`` imports this registry and stays responsible only for traversal,
mutation, and restoration. Architecture- or kernel-specific extraction logic
lives in the adapter modules next to this file.
"""
from __future__ import annotations

from typing import Callable, Dict

from .basic import (
    make_embedding_forward,
    make_geglu_forward,
    make_rmsnorm_forward,
    make_swiglu_forward,
)
from .common import (
    MIN_FUSED_ELEMENTS,
    ForgeSkipPatch,
    enough_work,
    guard_stats,
    reset_guard_stats,
    rope_supported,
    with_guard,
)
from .fused_linear_ce import make_fused_linear_ce_forward
from .lora import make_lora_mlp_forward, make_lora_qkv_forward


FORWARD_MAKERS: Dict[str, Callable] = {
    "embedding": make_embedding_forward,
    "rmsnorm": make_rmsnorm_forward,
    "swiglu": make_swiglu_forward,
    "fused_linear_ce": make_fused_linear_ce_forward,
    "lora_mlp": make_lora_mlp_forward,
    "lora_qkv": make_lora_qkv_forward,
    "geglu": make_geglu_forward,
}

#: Kernels whose benefit depends on how much work is in the call, so they are
#: dispatched per call against MIN_FUSED_ELEMENTS. See common.py for the
#: measurements behind the threshold.
SHAPE_GUARDED = frozenset({"embedding", "rmsnorm", "swiglu", "geglu"})

#: Kernels with no role outside training: a loss kernel needs labels, and the
#: LoRA kernels exist to produce adapter gradients. Patching them for inference
#: adds a wrapper that can only cost time.
TRAINING_ONLY = frozenset({"fused_linear_ce", "lora_mlp", "lora_qkv"})

#: Predicate used to guard each module-level patch.
MODULE_LEVEL_PREDICATES: Dict[str, Callable] = {
    "rope": rope_supported,
}


def _embedding_width(module) -> int:
    weight = getattr(module, "weight", None)
    if weight is not None and weight.dim() == 2:
        return int(weight.shape[-1])
    return 1


#: How to recover the feature width for kernels whose input does not carry it.
#: Everything except embedding is called with the activation itself, so its
#: numel() is already the element count.
WORK_WIDTH: Dict[str, Callable] = {
    "embedding": _embedding_width,
}


__all__ = [
    "FORWARD_MAKERS",
    "MIN_FUSED_ELEMENTS",
    "MODULE_LEVEL_PREDICATES",
    "SHAPE_GUARDED",
    "TRAINING_ONLY",
    "WORK_WIDTH",
    "ForgeSkipPatch",
    "enough_work",
    "guard_stats",
    "reset_guard_stats",
    "rope_supported",
    "with_guard",
]
