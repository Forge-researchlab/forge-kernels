"""Forge patching — monkey-patch HF Causal LMs to use Forge kernels.

Two patch modalities, both reversible via unpatch():

1. Per-module-instance forward replacement (most kernels).
   Walks model.named_modules(), looks up each module's class name in the
   architecture mapping (QWEN3_MAPPING / GEMMA_MAPPING), and rebinds
   `module.forward` to a closure that calls the matching Forge kernel.
   This is the pattern locked in design_details.html.

2. Module-level function replacement (RoPE only, for now).
   `apply_rotary_pos_emb` is a module-level function inside
   `transformers.models.qwen2.modeling_qwen2` / `.gemma2.modeling_gemma2`.
   It is called inline from attention.forward — there is no module instance
   whose forward we can monkey-patch. So we swap the function at module level
   and remember the original for unpatch.

Shape-sensitive kernels are additionally guarded per call: a fused kernel only
beats eager once there is enough work in the launch to amortise it, and the row
count is a property of the batch rather than of the model. See
`forge.patching.kernels.common` for the threshold and the measurements behind it.
"""
from .core import patch, unpatch
from .kernels import MIN_FUSED_ELEMENTS, guard_stats, reset_guard_stats

__all__ = [
    "MIN_FUSED_ELEMENTS",
    "guard_stats",
    "patch",
    "reset_guard_stats",
    "unpatch",
]
