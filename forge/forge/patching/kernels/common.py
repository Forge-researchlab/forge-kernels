"""Shared helpers for Forge patch-adapter factories."""
from __future__ import annotations

import itertools
from collections import Counter
from typing import Callable

import torch


class ForgeSkipPatch(RuntimeError):
    """Internal sentinel: a real kernel is not applicable to this module."""


# ---------------------------------------------------------------------------
# Shape guard
# ---------------------------------------------------------------------------
# A fused Triton kernel replaces several eager ops with one launch, so it wins
# only when there is enough work in that launch to cover the launch and autotune
# overhead. Below that point it is strictly worse, and patching unconditionally
# turns a speedup into a regression without telling anyone.
#
# The deciding quantity is the number of activation elements per call — rows
# (batch x seq) times width — not the model, and not whether we are training.
# Measured on an idle A100-80GB in bf16 with forge.patch applied one kernel at a
# time, as ratio against eager:
#
#     hidden   rows   elements   rmsnorm   swiglu
#        896    128      0.11M     0.91x    0.90x
#        896   1024      0.92M     0.91x    0.91x
#        896   2048      1.84M     1.04x    1.01x
#        896   4096      3.67M     1.06x    1.02x
#       4096    128      0.52M     0.92x    0.92x
#       4096    512      2.10M     1.15x    1.15x
#
# The crossover tracks the element count across a 4.5x difference in width,
# landing between 0.92M and 1.84M. 1 << 20 sits inside that window.
#
# This is calibrated on an A100 in bf16. It is a heuristic, and the right way to
# change it is to re-measure with forge/benchmarks/bench_patch_bisect.py rather
# than to adjust it by feel.
MIN_FUSED_ELEMENTS = 1 << 20


#: Counts of guard decisions per kernel, as {"rmsnorm": {"fused": n, "eager": m}}.
#: Exposed through forge.guard_stats() so a caller can confirm which path ran.
_GUARD_STATS: dict[str, Counter] = {}


def guard_stats() -> dict[str, dict[str, int]]:
    """Return per-kernel counts of fused vs eager dispatch, and reset nothing."""
    return {name: dict(counter) for name, counter in _GUARD_STATS.items()}


def reset_guard_stats() -> None:
    _GUARD_STATS.clear()


def _first_activation(args, kwargs):
    """The first tensor argument with a feature dimension, or None.

    Patched forwards take the activation first in every case we handle, but
    kwargs-only calls happen, so both are scanned.
    """
    for value in itertools.chain(args, kwargs.values()):
        if isinstance(value, torch.Tensor) and value.dim() >= 2:
            return value
    return None


def enough_work(args, kwargs, min_elements: int = MIN_FUSED_ELEMENTS,
                width: int = 1) -> bool:
    """True when the call is large enough for a fused kernel to pay off.

    `width` scales the element count for kernels whose input does not already
    carry the feature dimension. An embedding is indexed by token ids, so its
    activation has shape (batch, seq) and numel() counts rows rather than
    elements; passing the embedding dimension recovers the real work.

    Returns True when no activation can be identified: declining to fuse on a
    shape we failed to read would silently disable the kernel, which is a worse
    failure than being slightly wrong about one call.
    """
    activation = _first_activation(args, kwargs)
    if activation is None:
        return True
    return activation.numel() * width >= min_elements


def rope_supported(args, kwargs, min_elements: int = MIN_FUSED_ELEMENTS) -> bool:
    """As enough_work, plus ForgeRoPEv3's power-of-two head_dim requirement.

    The kernel asserts on head_dim internally. Without this check, patching a
    model whose head_dim is not a power of two — 80 and 96 both occur in
    published models — raises a bare AssertionError from inside attention,
    several frames from anything the user wrote.
    """
    query = _first_activation(args, kwargs)
    if query is None:
        return True
    head_dim = query.shape[-1]
    if head_dim & (head_dim - 1) != 0:
        return False
    return query.numel() >= min_elements


def with_guard(kernel_name: str, fused: Callable, original: Callable,
               predicate: Callable, collect_stats: bool = False) -> Callable:
    """Wrap `fused` so it defers to `original` whenever `predicate` is False.

    The check is per call, not per patch, because the row count is a property of
    the batch rather than of the model — the same patched model is above the
    threshold during training and below it when serving one short prompt.

    `collect_stats` is off by default because the guard sits on the hot path. On
    a 24-layer model a forward pass crosses it ~1,370 times, and counting each
    decision cost a measurable 2% on the small shapes where every kernel
    declines — which is exactly the case the guard exists to protect. Turn it on
    when you need to confirm which path ran.
    """
    if collect_stats:
        stats = _GUARD_STATS.setdefault(kernel_name, Counter())

        def guarded(*args, **kwargs):
            if predicate(args, kwargs):
                stats["fused"] += 1
                return fused(*args, **kwargs)
            stats["eager"] += 1
            return original(*args, **kwargs)
    else:
        def guarded(*args, **kwargs):
            if predicate(args, kwargs):
                return fused(*args, **kwargs)
            return original(*args, **kwargs)

    guarded.__forge_kernel__ = kernel_name
    guarded.__forge_guarded__ = True
    return guarded
