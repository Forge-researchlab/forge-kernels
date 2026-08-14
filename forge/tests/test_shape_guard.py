"""Tests for the per-call shape guard in forge.patch.

The guard decides, per call, whether a fused kernel is worth using. Three things
have to hold or it is worse than not having it:

  1. Below the threshold it defers to eager, and the result is then bitwise
     identical to the unpatched model — the guard must be a pure dispatch
     decision, not a second numerical path.
  2. Above the threshold the fused kernel actually runs.
  3. A shape the kernel cannot handle is declined rather than asserted on.

Run directly, or under pytest. Needs one GPU.
"""
from __future__ import annotations

import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_FORGE_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
_REPO_ROOT = os.path.normpath(os.path.join(_FORGE_ROOT, ".."))
for _path in (_FORGE_ROOT, _REPO_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import forge  # noqa: E402


def _tiny_model(hidden=896, heads=14, kv_heads=2, layers=2, intermediate=4864,
                vocab=512, dtype=torch.bfloat16):
    """A structurally real Qwen2 with random weights, small enough to be fast.

    Widths matter (they set head_dim and drive the guard); depth and vocab do not,
    so both are kept minimal.
    """
    from transformers import AutoModelForCausalLM, Qwen2Config

    config = Qwen2Config(
        hidden_size=hidden, num_attention_heads=heads, num_key_value_heads=kv_heads,
        num_hidden_layers=layers, intermediate_size=intermediate, vocab_size=vocab,
    )
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(
        config, torch_dtype=dtype, attn_implementation="eager"
    )
    return model.to("cuda").eval()


def _forward(model, batch, seq, vocab=512):
    torch.manual_seed(1234)
    ids = torch.randint(0, vocab, (batch, seq), device="cuda")
    with torch.no_grad():
        return model(input_ids=ids).logits


def test_below_threshold_defers_to_eager():
    """Small call: every guarded kernel declines, and output is bit-identical."""
    model = _tiny_model()
    batch, seq = 1, 8
    assert batch * seq * 896 < forge.MIN_FUSED_ELEMENTS

    reference = _forward(model, batch, seq)

    forge.reset_guard_stats()
    forge.patch(model, collect_stats=True)
    guarded = _forward(model, batch, seq)
    stats = forge.guard_stats()
    forge.unpatch(model)

    assert stats, "guard recorded no decisions at all"
    for kernel, counts in stats.items():
        assert counts.get("fused", 0) == 0, f"{kernel} fused below the threshold"
        assert counts.get("eager", 0) > 0, f"{kernel} was never consulted"

    # Deferring must mean calling the original forward, so nothing may move.
    assert torch.equal(reference, guarded), "declined path changed the output"


def test_above_threshold_fuses():
    """Large call: the fused kernels run, and agree with eager to bf16 tolerance."""
    model = _tiny_model()
    batch, seq = 4, 512
    assert batch * seq * 896 >= forge.MIN_FUSED_ELEMENTS

    reference = _forward(model, batch, seq)

    forge.reset_guard_stats()
    forge.patch(model, collect_stats=True)
    fused = _forward(model, batch, seq)
    stats = forge.guard_stats()
    forge.unpatch(model)

    for kernel in ("rmsnorm", "swiglu", "embedding"):
        assert stats.get(kernel, {}).get("fused", 0) > 0, \
            f"{kernel} did not fuse above the threshold: {stats.get(kernel)}"

    cosine = torch.nn.functional.cosine_similarity(
        reference.float().flatten(), fused.float().flatten(), dim=0
    ).item()
    assert cosine > 0.999, f"fused output diverged from eager: cosine={cosine}"


def test_min_elements_zero_disables_the_size_check():
    """Benchmarking a kernel needs it to run at shapes the guard would refuse.

    min_elements=0 switches off the size heuristic only. Compatibility
    predicates stay in force, which is what
    test_rope_declines_unsupported_head_dim covers.
    """
    model = _tiny_model()

    forge.reset_guard_stats()
    forge.patch(model, min_elements=0, collect_stats=True)
    _forward(model, 1, 8)  # far below MIN_FUSED_ELEMENTS
    stats = forge.guard_stats()
    forge.unpatch(model)

    for kernel, counts in stats.items():
        assert counts.get("eager", 0) == 0, \
            f"{kernel} declined despite min_elements=0: {counts}"


def test_infer_mode_skips_training_only_kernels():
    """A loss kernel and LoRA kernels cannot help without a backward pass."""
    model = _tiny_model()
    forge.patch(model, mode="infer")
    skipped = dict(model._forge_skipped)
    patched = dict(model._forge_patched_counts)
    forge.unpatch(model)

    assert set(skipped) == {"fused_linear_ce", "lora_mlp", "lora_qkv"}, skipped
    for kernel in skipped:
        assert kernel not in patched, f"{kernel} was patched despite mode='infer'"


def test_train_mode_keeps_the_loss_kernel():
    model = _tiny_model()
    forge.patch(model, mode="train")
    patched = dict(model._forge_patched_counts)
    forge.unpatch(model)
    assert patched.get("fused_linear_ce", 0) == 1, patched


def test_rope_declines_unsupported_head_dim():
    """ForgeRoPEv3 requires a power-of-two head_dim.

    Regression test: without the guard this raised a bare AssertionError from
    inside attention. head_dim=80 (240/3) is not exotic — published models use
    80 and 96.
    """
    model = _tiny_model(hidden=240, heads=3, kv_heads=1, intermediate=512)
    assert model.config.hidden_size // model.config.num_attention_heads == 80

    reference = _forward(model, 1, 8)

    forge.reset_guard_stats()
    # min_elements=0 so the only thing that can decline RoPE is head_dim itself.
    forge.patch(model, kernels=["rope"], min_elements=0, collect_stats=True)
    try:
        result = _forward(model, 1, 8)
    finally:
        stats = forge.guard_stats()
        forge.unpatch(model)

    assert stats.get("rope", {}).get("fused", 0) == 0, "RoPE fused on head_dim=80"
    assert stats.get("rope", {}).get("eager", 0) > 0, "RoPE guard never ran"
    assert torch.equal(reference, result)


def test_unpatch_restores_everything():
    model = _tiny_model()
    reference = _forward(model, 1, 8)

    forge.patch(model)
    forge.unpatch(model)

    assert torch.equal(reference, _forward(model, 1, 8))
    for attr in ("_forge_patched", "_forge_originals", "_forge_mode",
                 "_forge_skipped", "_forge_min_elements"):
        assert not hasattr(model, attr), f"{attr} survived unpatch"


TESTS = [
    test_below_threshold_defers_to_eager,
    test_above_threshold_fuses,
    test_min_elements_zero_disables_the_size_check,
    test_infer_mode_skips_training_only_kernels,
    test_train_mode_keeps_the_loss_kernel,
    test_rope_declines_unsupported_head_dim,
    test_unpatch_restores_everything,
]


def main():
    if not torch.cuda.is_available():
        print("SKIP: needs a GPU")
        return 0

    results = []
    for test in TESTS:
        try:
            test()
        except AssertionError as exc:
            results.append((test.__name__, "FAIL", str(exc)))
        except Exception as exc:  # noqa: BLE001 - report, do not mask
            results.append((test.__name__, "ERROR", f"{type(exc).__name__}: {exc}"))
        else:
            results.append((test.__name__, "PASS", ""))

    print("=" * 72)
    print("VERDICT")
    print("=" * 72)
    for name, status, detail in results:
        print(f"  {status:<6} {name}")
        if detail:
            print(f"         {detail}")
    passed = sum(1 for _, status, _ in results if status == "PASS")
    print(f"\n  OVERALL ({passed}/{len(results)}): "
          f"{'PASS' if passed == len(results) else 'FAIL'}\n")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
