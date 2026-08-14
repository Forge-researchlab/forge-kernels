"""forge.patching.core — the patch/unpatch loop, closure factories, and dispatch.

Design (locked, per design_details.html#patching-design and #wiring):

  * Forward replacement, NOT module replacement.
        Module replacement breaks state_dict loading and confuses FSDP2's
        transformer-layer-cls wrap policies. We monkey-patch `module.forward`
        and remember the original on `model._forge_originals` so unpatch is
        bit-exact reversible.

  * Architecture detected via `model.config.model_type`, never type(model).__name__.
        Qwen2.5 / Qwen3 report "qwen2" (and sometimes "qwen3"); Gemma-2 reports
        "gemma2". The mapping registry keys on this string.

  * Closure factory is mandatory.
        Don't write `module.forward = lambda x: forge_module(x)` inside a loop.
        Python's late binding will bind `forge_module` to whatever the loop
        variable points to at the end — silently wrong. Always use a factory
        function that closes over the right reference.

  * Closures close over `module.weight` directly, not a copy.
        LoRA adapters that update the weight at training time must be visible
        through the closure on the next forward. Copying at patch time freezes it.

  * Double-patch raises RuntimeError, not silent no-op.
        Loud failure catches the bug where a test fixture forgets to unpatch.

  * Selective patching: `forge.patch(model, kernels=["embedding"])`.
        The #1 debugging tool when training diverges — enable kernels one at a
        time to bisect which one is broken.

  * Stubs raise NotImplementedError only when explicitly requested.
        forge.patch(model) silently skips kernels not yet built; the caller can
        force-fail by passing kernels=[<unbuilt name>] to confirm the gap.
"""
from __future__ import annotations

import importlib
from typing import Callable, Dict, List, Optional, Tuple

from .qwen3 import QWEN3_MAPPING, QWEN3_MODULE_LEVEL_PATCHES
from .gemma import GEMMA_MAPPING, GEMMA_MODULE_LEVEL_PATCHES
from .kernels import FORWARD_MAKERS as _FORWARD_MAKERS
from .kernels import (
    MIN_FUSED_ELEMENTS,
    MODULE_LEVEL_PREDICATES,
    SHAPE_GUARDED,
    TRAINING_ONLY,
    WORK_WIDTH,
    ForgeSkipPatch,
    enough_work,
    with_guard,
)


# ----------------------------------------------------------------------------
# Architecture detection
# ----------------------------------------------------------------------------

_ARCH_TO_MAPPING: Dict[str, Tuple[dict, dict]] = {
    # model_type -> (per_class_mapping, module_level_patches)
    "qwen2":  (QWEN3_MAPPING, QWEN3_MODULE_LEVEL_PATCHES),
    "qwen3":  (QWEN3_MAPPING, QWEN3_MODULE_LEVEL_PATCHES),
    "gemma":  (GEMMA_MAPPING, GEMMA_MODULE_LEVEL_PATCHES),
    "gemma2": (GEMMA_MAPPING, GEMMA_MODULE_LEVEL_PATCHES),
}


def _model_config(model):
    config = getattr(model, "config", None)
    if config is not None:
        return config
    base_model = getattr(model, "base_model", None)
    inner = getattr(base_model, "model", None)
    config = getattr(inner, "config", None)
    if config is not None:
        return config
    raise ValueError("forge.patch: model does not expose a Hugging Face config.")


def _detect_architecture(model) -> str:
    model_type = getattr(_model_config(model), "model_type", None)
    if model_type not in _ARCH_TO_MAPPING:
        raise ValueError(
            f"forge.patch: no Forge mapping for model_type={model_type!r}. "
            f"Supported: {sorted(_ARCH_TO_MAPPING)}."
        )
    return model_type


# ----------------------------------------------------------------------------
# Kernel-specific forward factories live in forge.patching.kernels.
# core.py owns only orchestration: validation, traversal, mutation, restoration.
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# Per-module-instance forward replacement
# ----------------------------------------------------------------------------

def _replace_forward(module, kernel_name: str, config: dict, originals: dict,
                     min_elements: int, collect_stats: bool) -> bool:
    """Replace `module.forward` via the matching closure factory.

    Returns True if the replacement happened, False if the kernel is a stub and
    we silently skipped it.

    Shape-sensitive kernels are wrapped so that each call falls back to the
    original forward when it is too small to be worth fusing. The threshold
    cannot be applied at patch time because the row count belongs to the batch,
    not to the model.
    """
    if kernel_name not in _FORWARD_MAKERS:
        raise KeyError(
            f"forge.patch: unknown kernel {kernel_name!r}. "
            f"Registered: {sorted(_FORWARD_MAKERS)}."
        )
    maker = _FORWARD_MAKERS[kernel_name]
    try:
        new_fwd = maker(module, config)
    except ForgeSkipPatch:
        return False

    original_fwd = module.forward
    if min_elements > 0 and kernel_name in SHAPE_GUARDED:
        width = WORK_WIDTH.get(kernel_name, lambda _module: 1)(module)
        new_fwd = with_guard(
            kernel_name, new_fwd, original_fwd,
            lambda args, kwargs, _w=width: enough_work(args, kwargs, min_elements, _w),
            collect_stats=collect_stats,
        )

    originals[id(module)] = (module, original_fwd)
    module.forward = new_fwd
    return True


# ----------------------------------------------------------------------------
# Module-level function replacement (RoPE)
# ----------------------------------------------------------------------------

_MODULE_LEVEL_KEY = "__forge_module_level__"


def _apply_module_level_patches(patches: List[Tuple[str, str, Callable, str]],
                                originals: dict, min_elements: int,
                                collect_stats: bool) -> None:
    """Replace module-level functions (e.g. transformers' apply_rotary_pos_emb).

    `patches` is a list of (module_path, attr_name, replacement, kernel_name)
    tuples. We import the module, save the original attribute, and overwrite.
    unpatch walks the saved list and restores.

    Where a predicate is registered for the kernel, the replacement is wrapped so
    each call can defer to the original. For RoPE that covers both calls too
    small to be worth fusing and head_dim values the kernel cannot handle.
    """
    saved = []
    for module_path, attr_name, replacement, kernel_name in patches:
        try:
            mod = importlib.import_module(module_path)
        except ImportError:
            # Quietly skip if the transformers submodule isn't installed for this arch
            continue
        original = getattr(mod, attr_name, None)
        if original is None:
            continue
        # Installed regardless of min_elements: these predicates also decide
        # whether the kernel *can* run on this shape at all, and turning off a
        # performance heuristic must not turn off a compatibility check. With
        # min_elements=0 the size half of the predicate is vacuously true.
        predicate = MODULE_LEVEL_PREDICATES.get(kernel_name)
        if predicate is not None:
            replacement = with_guard(
                kernel_name, replacement, original,
                lambda args, kwargs, _p=predicate: _p(args, kwargs, min_elements),
                collect_stats=collect_stats,
            )
        saved.append((mod, attr_name, original))
        setattr(mod, attr_name, replacement)
    if saved:
        originals[_MODULE_LEVEL_KEY] = saved


def _revert_module_level_patches(originals: dict) -> None:
    for mod, attr_name, original in originals.get(_MODULE_LEVEL_KEY, []):
        setattr(mod, attr_name, original)


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------

def patch(model, kernels: Optional[List[str]] = None, mode: str = "train",
          min_elements: int = MIN_FUSED_ELEMENTS, collect_stats: bool = False):
    """Patch an HF model in place. Returns the same model object.

    Args:
        model:   HF causal LM (currently Qwen2/Qwen3 or Gemma2).
        kernels: Optional whitelist of kernel names. None enables every kernel
                 with a real implementation (stubs are silently skipped).
                 Pass an explicit list to bisect during debugging:
                     forge.patch(model, kernels=["embedding"])
                     forge.patch(model, kernels=["rope"])
        mode:    "train" (default) patches everything applicable. "infer" also
                 skips the kernels that cannot help without a backward pass —
                 the loss kernel and both LoRA kernels — since there is no loss
                 and no adapter gradient to compute when serving.
        min_elements: per-call activation-element floor below which a
                 shape-sensitive kernel defers to the original forward. The
                 default is calibrated on an A100 in bf16; see
                 forge.patching.kernels.common for the measurements. Pass 0 to
                 disable the guard and patch unconditionally, which is what
                 benchmarking a kernel in isolation wants.
        collect_stats: count fused-vs-eager guard decisions, readable through
                 forge.guard_stats(). Off by default: the guard is on the hot
                 path and counting cost 2% on the small shapes it protects.

    Raises:
        RuntimeError if the model is already patched (call unpatch first).
        ValueError if model.config.model_type isn't supported, or mode is not
            "train" or "infer".
        NotImplementedError if `kernels` names a kernel that's still a stub.
    """
    if getattr(model, "_forge_patched", False):
        raise RuntimeError("Model already patched. Call forge.unpatch(model) first.")
    if mode not in ("train", "infer"):
        raise ValueError(f"forge.patch: mode must be 'train' or 'infer', got {mode!r}.")

    arch = _detect_architecture(model)
    class_mapping, module_level_patches = _ARCH_TO_MAPPING[arch]

    # In inference mode the training-only kernels are recorded as skipped rather
    # than dropped silently, so `model._forge_skipped` explains the difference
    # between what was asked for and what was applied.
    skipped: Dict[str, str] = {}
    if mode == "infer":
        for kernel_name in sorted(TRAINING_ONLY):
            if kernels is None or kernel_name in kernels:
                skipped[kernel_name] = "no effect without a backward pass (mode='infer')"

    # --- pre-validate: fail loudly BEFORE mutating any module, so a stub in the
    # `kernels` whitelist doesn't leave the model half-patched.
    if kernels is not None:
        for kernel_name in kernels:
            maker = _FORWARD_MAKERS.get(kernel_name)
            if maker is None:
                # Could still be valid as a module-level kernel (e.g., "rope")
                if kernel_name not in module_level_patches:
                    raise KeyError(
                        f"forge.patch: unknown kernel {kernel_name!r}. "
                        f"Known per-module: {sorted(_FORWARD_MAKERS)}. "
                        f"Known module-level: {sorted(module_level_patches)}."
                    )
                continue
            if getattr(maker, "__forge_stub__", False):
                raise NotImplementedError(
                    f"forge.patch: kernel {kernel_name!r} requested but not implemented yet "
                    f"({getattr(maker, '__forge_reason__', '')}). "
                    f"Drop {kernel_name!r} from the kernels=[...] list or build it first."
                )

    originals: dict = {}

    # --- per-module-instance forward replacement ---
    patched_count: Dict[str, int] = {}
    for _name, module in model.named_modules():
        cls_name = type(module).__name__
        if cls_name not in class_mapping:
            continue
        patch_specs = class_mapping[cls_name]
        if isinstance(patch_specs, tuple):
            patch_specs = [patch_specs]

        for kernel_name, config in patch_specs:
            # Filter by `kernels` whitelist
            if kernels is not None and kernel_name not in kernels:
                continue
            if kernel_name in skipped:
                continue

            maker = _FORWARD_MAKERS.get(kernel_name)
            if maker is None:
                continue
            # Stubs: silent skip when patching everything; pre-validation above already
            # caught the explicit-whitelist case.
            if getattr(maker, "__forge_stub__", False):
                continue

            if _replace_forward(module, kernel_name, config, originals, min_elements,
                                collect_stats):
                patched_count[kernel_name] = patched_count.get(kernel_name, 0) + 1
                break

    # --- module-level patches (RoPE et al.) ---
    selected_module_level = []
    for kernel_name, patch_spec in module_level_patches.items():
        if kernels is not None and kernel_name not in kernels:
            continue
        if kernel_name in skipped:
            continue
        # A kernel can patch more than one transformers module path (e.g. qwen2
        # and qwen3 have separate modeling modules in newer transformers). The
        # kernel name rides along so the guard predicate can be looked up.
        specs = [patch_spec] if isinstance(patch_spec, tuple) else list(patch_spec)
        selected_module_level.extend(
            (module_path, attr_name, replacement, kernel_name)
            for module_path, attr_name, replacement in specs
        )
    _apply_module_level_patches(selected_module_level, originals, min_elements,
                               collect_stats)

    model._forge_originals = originals
    model._forge_patched = True
    model._forge_patched_counts = patched_count
    model._forge_arch = arch
    model._forge_mode = mode
    model._forge_min_elements = min_elements
    model._forge_skipped = skipped
    return model


def unpatch(model):
    """Restore original forwards. Idempotent — calling on an unpatched model is a no-op."""
    if not getattr(model, "_forge_patched", False):
        return model

    originals = model._forge_originals

    # Restore module-instance forwards
    for key, payload in originals.items():
        if key == _MODULE_LEVEL_KEY:
            continue
        module, original_fwd = payload
        module.forward = original_fwd

    # Restore module-level functions
    _revert_module_level_patches(originals)

    del model._forge_originals
    del model._forge_patched
    for attr in ("_forge_patched_counts", "_forge_arch", "_forge_mode",
                 "_forge_min_elements", "_forge_skipped"):
        if hasattr(model, attr):
            delattr(model, attr)
    return model
