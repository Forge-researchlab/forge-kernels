"""Forge — custom Triton kernels for HuggingFace LLM fine-tuning.

Public API:
    forge.patch(model, kernels=None, mode="train")   # patch HF model in place
    forge.unpatch(model)                             # restore original forwards
    forge.guard_stats()                              # fused vs eager dispatch counts

`patch` is for fine-tuning. Pass mode="infer" when serving: the loss and LoRA
kernels cannot help without a backward pass, and are skipped. Either way the
shape-sensitive kernels fall back to eager per call when the batch is too small
for fusion to pay off — see forge.patching.kernels.common for the threshold.

The kernels referenced by the patching layer live under `forge.kernels.*`.
See `forge.patching.core` for the patching pattern (forward replacement +
closure factory) and the locked decisions behind it.
"""
import os as _os
import sys as _sys

# Hackathon shim: the POC kernels live at /workspace/kernel-POCs/kernels/*.
# Re-exporting them through forge.kernels.* requires the POC root on sys.path.
# Post-hackathon clean-up: copy each kernel file into forge/kernels/ and drop
# this block. Until then, this lets `pip install -e ./forge` work with the
# kernel POCs in their experimental locations.
_POC_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _POC_ROOT not in _sys.path:
    _sys.path.insert(0, _POC_ROOT)

from .patching import (  # noqa: E402
    MIN_FUSED_ELEMENTS,
    guard_stats,
    patch,
    reset_guard_stats,
    unpatch,
)

__version__ = "0.0.1.dev1"
__all__ = [
    "MIN_FUSED_ELEMENTS",
    "__version__",
    "guard_stats",
    "patch",
    "reset_guard_stats",
    "unpatch",
]
