# The `forge` patching layer

`forge.patch(model)` swaps the kernels into a live Hugging Face model, with
architecture detection for `qwen2`, `qwen3`, `gemma`, and `gemma2`, and an
idempotent `unpatch`. See [`forge/forge/patching/core.py`](../../forge/forge/patching/core.py)
for the patching pattern (forward replacement + closure factory).

```python
import forge

forge.patch(model)                  # fine-tuning
forge.patch(model, mode="infer")    # serving: skips the loss and LoRA kernels
forge.patch(model, min_elements=0)  # patch unconditionally, for benchmarking
forge.unpatch(model)

model._forge_patched_counts   # {'rmsnorm': 49, 'swiglu': 24, ...}
model._forge_skipped          # what was declined at patch time, and why
```

Individual kernels fall back to the original forward on calls too small to be
worth fusing. See [what decides whether patching helps](#what-decides-whether-patching-helps-is-batch-size-not-training-vs-inference).
To see where the line falls for your own model and batch size, bisect it one
kernel at a time:

```bash
PYTHONPATH=forge:. python forge/benchmarks/bench_patch_bisect.py \
    --model Qwen/Qwen2.5-0.5B --mode train --batch 4 --seq-len 1024 --guard-stats
```

## Verified in a live model

Re-run 2026-08-14 on 2× A100-80GB, torch `2.4.1+cu121`, real `Qwen/Qwen2.5-0.5B` weights:

| Check | Result |
| ----- | ------ |
| `forge.patch` on Qwen2.5-0.5B | 24 LoRA-QKV + 24 LoRA-MLP + 1 fused-linear-CE replaced; loss delta 9.5e-07, min cosine 0.9994; `unpatch` restores |
| FSDP2 on 2 GPUs | forward bit-identical; grad cosine 0.99995; 5-step loss matches the single-GPU reference to 9.2e-05 relative |
| 200-step LoRA fine-tune under FSDP2 | loss 4.3686 → 0.0005, ~1.03 s/step, 1.75 GB peak per rank, 8.1M of 638M params trainable (1.27%) |
| Held-out generation | base "Good morning everyone!" → "Good morning, everyone!"; tuned → "Ahoy, ye good and true matey!" |

Artifacts land in [`artifacts/lora_demos/qwen/`](../../artifacts/lora_demos/qwen/) and
[`artifacts/fsdp2/`](../../artifacts/fsdp2/).

## What decides whether patching helps is batch size, not training vs inference

The variable that decides whether fusion pays off is the number of activation
elements in a call: rows (`batch × seq`) times width. Below roughly a million, a
fused kernel cannot cover its own launch overhead; above it, it wins. Training
usually sits above the line and single-stream serving below it, so measuring on
one shape can look like "helps training, hurts inference" when the real driver is
batch size.

Measured 2026-08-14 on an idle A100-80GB, real Qwen2.5-0.5B weights
(`hidden=896`), bf16, via
[`forge/benchmarks/bench_patch_bisect.py`](../../forge/benchmarks/bench_patch_bisect.py):

| Mode | batch × seq | elements/call | eager | patched | vs eager | Peak memory |
| ----- | ----------: | ------------: | ----: | ------: | -------: | ----------: |
| infer | 1 × 128     | 0.11 M | 31.9 ms | 37.4 ms | **0.85×** | 998 MB → 998 MB |
| infer | 1 × 1024    | 0.92 M | 33.3 ms | 38.4 ms | **0.87×** | 1269 MB → 1269 MB |
| infer | 4 × 1024    | 3.67 M | 71.9 ms | 64.0 ms | **1.12×** | 2201 MB → 2201 MB |
| train | 4 × 1024    | 3.67 M | 211.4 ms | 175.9 ms | **1.20×** | 21869 MB → **14071 MB** |

Two things follow. Training on this model is **1.20× faster and uses 36% less
memory**, driven by fused linear cross-entropy: Qwen2.5-0.5B pairs an 896-wide
hidden state with a 151936-token vocabulary, so the logits dominate and never
materialising them saves 6.2 GB by itself. Inference is not inherently slow
either: at batch 4 it is 1.12× faster. The same kernel that loses 9% at 128 rows
wins 6% at 4096, because below ~1024 rows this model is launch-bound rather than
compute-bound. Eager takes the same 32 ms for 128 tokens as for 1024, so eight
times the work is free while extra kernel launches can only cost.

Correctness is unaffected either way: greedy generation stays token-identical to
eager and logits agree to 1.8e-02 relative in bf16.

**`forge.patch` now decides per call.** Shape-sensitive kernels fall back to the
original forward when a call is below the threshold, so patching a small model no
longer produces a regression:

| Mode | batch × seq | Before the guard | With the guard |
| ----- | ----------: | ---------------: | -------------: |
| infer | 1 × 128     | 0.85× | **0.98×** |
| infer | 1 × 1024    | 0.87× | **1.00×** |
| infer | 4 × 1024    | 1.12× | **1.12×** |
| train | 4 × 1024    | 1.20× | **1.20×** |

The remaining 2% at the smallest shape is the guard's own per-call Python check,
paid ~1,370 times per forward on a 24-layer model. The threshold
(`forge.MIN_FUSED_ELEMENTS`) is a heuristic calibrated on an A100 in bf16, and it
tracked the crossover across a 4.5× change in width; re-measure with the bisect
script before changing it. Pass `min_elements=0` to patch unconditionally, which
is what benchmarking a kernel in isolation wants.

Pass `mode="infer"` when serving and the three kernels with no inference role,
fused linear cross-entropy and both LoRA kernels, are skipped rather than
wrapped, since there is no loss and no adapter gradient to compute.
`model._forge_skipped` records what was declined and why.

**"Multi-GPU" means the kernels survive FSDP2, not that they speed it up.** What
was verified is that patched forwards keep working when weights become sharded
DTensors, which is a real and non-trivial property, since a kernel that closes
over a raw `weight` tensor at patch time crashes under FSDP2, and an earlier
version of the LoRA MLP path did exactly that (see
[`phase3_fsdp2_design.md`](../design-notes/phase3_fsdp2_design.md) §7). There is no
custom collective, no communication/computation overlap, and no change to how
FSDP2 shards or all-gathers. The 2-GPU numbers above show *equivalence* to the
single-GPU reference, not a speedup from scaling.
