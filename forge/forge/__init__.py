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
import importlib.util as _util
import os as _os
import sys as _sys

# forge.kernels.* re-exports the implementations in the sibling `kernels`
# package. An installed copy ships that package, so nothing is needed. A bare
# checkout does not put the repository root on sys.path, so add it there — and
# only there, since in an installed copy this path resolves to the parent of
# site-packages, which has no business on sys.path.
if _util.find_spec("kernels") is None:
    _REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
    if _REPO_ROOT not in _sys.path:
        _sys.path.insert(0, _REPO_ROOT)

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
