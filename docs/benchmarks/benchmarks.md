# Benchmarks

Full performance record behind the [README results table](../../README.md#results):
every measured speedup and every regression. All numbers are read from committed
result files, not estimates.

**Environment.** NVIDIA A100-SXM4-80GB, Triton `3.0.0`, `liger-kernel==0.8.0`.
RoPE / RMSNorm / SwiGLU / GeGLU / LoRA / LayerNorm rows: 2026-05-23/24, torch
`2.4.1+cu124`. Cross-Entropy and Embedding rows: 2026-08-14, torch `2.4.1+cu121`.

**Re-run 2026-08-14** on a second A100-80GB box: original figures hold. RoPE
7.12× vs eager (2.77× vs Unsloth fused-QK, 3.24× vs Liger), forward 30/30,
backward 8/8, fp64 gradcheck passing. RMSNorm 10.24× forward. `bench_all.py`
completes 11/11.


| Kernel                     | Best recorded speedup | Baseline               | Shape / config                      | Source                                                                               |
| -------------------------- | --------------------- | ---------------------- | ----------------------------------- | ------------------------------------------------------------------------------------ |
| **RoPE** (v3)              | **7.1×** fwd          | PyTorch eager          | 2×32/8×2048×128, GQA G=4, bf16      | [`v3_summary.md`](../../kernels/rope/benchmarks/results/v3_summary.md)                  |
|                            | 2.8× fwd              | Unsloth fused-QK       | same                                | same                                                                                 |
|                            | 3.2× fwd              | Liger                  | same                                | same                                                                                 |
| **RMSNorm** (v4)           | **10.2×** fwd         | PyTorch eager          | 2×2048×4096, bf16, offset=1.0       | [`v4_summary.md`](../../kernels/rmsnorm/benchmarks/results/v4_summary.md)               |
| **SwiGLU**                 | **6.5×** fwd          | PyTorch eager (packed) | 4×2048×11008, bf16                  | [`swiglu_a100_bf16.csv`](../../results/swiglu_a100_bf16.csv)                            |
|                            | 1.5× fwd              | Liger                  | 4×2048×11008, bf16, mult 0.7/1.3    | [`..._with_liger.csv`](../../results/swiglu_a100_bf16_with_liger.csv)                   |
| **GeGLU** (activation)     | **5.2×** fwd+bwd      | PyTorch eager (packed) | 4×2048×11008, bf16, tanh            | [`geglu_activation_a100_bf16.csv`](../../results/geglu_activation_a100_bf16.csv)        |
| **GeGLU** (gate+up fusion) | **1.40×** fwd         | separate PyTorch MLP   | 1×512×4096, i=11008, bf16           | [`geglu_gateup_a100_bf16.csv`](../../results/geglu_gateup_a100_bf16.csv)                |
| **LoRA QKV** (v4)          | **1.37×** fwd+bwd     | Unsloth                | 4×2048, h=4096, GQA 32/8, r=8, bf16 | [`CHANGELOG.md`](../../kernels/lora_qkv/CHANGELOG.md)                                   |
| **LoRA MLP** (v6)          | **1.18×** fwd         | Unsloth                | 4×2048, h=4096, i=14336, r=16, bf16 | [`v6_upgrade_1_latency_*.csv`](../../kernels/lora_mlp/benchmarks/results/)              |
| **LayerNorm**              | **1.57×** fwd+bwd     | PyTorch eager          | 8×2048×4096, fp32                   | [`layernorm_tests.executed.ipynb`](../../kernels/layernorm/layernorm_tests.executed.ipynb) |
| **Cross-Entropy**          | **2.77×** fwd+bwd     | PyTorch eager          | 8192×128256, fp32                   | [`cross_entropy_*.csv`](../../kernels/cross_entropy/benchmarks/results/)                |
|                            | 1.00× fwd+bwd         | Liger                  | same                                | same                                                                                 |
| **Fused Linear CE**        | **2.88× less memory** | PyTorch eager          | BT=8192, h=4096, v=128256, bf16     | [`fused_linear_cross_entropy_*.csv`](../../kernels/cross_entropy/benchmarks/results/)   |
|                            | 2.85× fwd+bwd         | Liger                  | same                                | same                                                                                 |
| **Embedding**              | **1.49×** fwd+bwd     | PyTorch eager          | v=32000, d=4096, s=8192, bf16       | [`bench_*.csv`](../../kernels/embedding/benchmarks/results/)                            |


## Where these kernels lose

Headline numbers are best-case. The same result files record these regressions:

- **SwiGLU is slower than Liger on 12 of the shape/mode combinations** benchmarked, worst case 0.84× (fwd+bwd, 2×2048×11008 bf16). It only beats Liger on large-hidden forward passes.
- **SwiGLU and GeGLU lose to PyTorch eager on small and irregular shapes**, down to 0.58× (SwiGLU, fp32, 2×7×11009) and 0.66× (GeGLU smoke suite). The win is concentrated at large hidden dimensions.
- **RMSNorm's 10× is forward-only.** With backward included it is roughly at parity with Liger (0.32×–2.68× across shapes, clustered near 1.0×) and generally slower than Unsloth, which does not compute `dW`.
- **GeGLU gate+up fusion narrows as the batch grows.** It reaches 2.84× only at a 14-token batch (2×7×4096); at the realistic 2×2048×4096 training shape it is 1.07× vs PyTorch and 0.97× vs Liger, slightly slower than Liger. It does hold a real memory edge there: 854 MiB vs PyTorch's 1112 MiB.
- **LayerNorm is 0.84× vs eager** at 4×2048×4096 bf16.
- **LoRA MLP memory matches Unsloth rather than beating it**, 737 MB vs 736 MB peak forward. The v6 in-place epilogue closed a 1185 MB → 737 MB gap against our own earlier version; it is not a win over the baseline.
- **Embedding loses to PyTorch on 94 of 108 benchmarked configs**, down to 0.31× at v=32000/d=768/s=512. Sorting indices to group duplicate rows only pays for itself at large embedding dimension and long sequences: every win is at `d=4096, s=8192`. The vendored Liger-style reference cannot be compared at all in bf16: `tl.atomic_add` rejects bf16 on Triton 3.0, whereas the sort-and-group backward accumulates in fp32 and is also deterministic.
- **Fused linear CE only saves memory once the logits dominate.** At BT=1024–2048 it is level with eager (1269 vs 1260 MB, 1536 vs 1520 MB) because the `(V, H)` weight gradient, not the logits chunk, sets the floor. The 2.88× saving appears at BT=8192 (2085 vs 6012 MB).
- **Plain CE's memory ratio is an artifact worth stating plainly.** It transforms the logits tensor in place, so it allocates ~0.1 MB against eager's 12 GB of intermediates at BT=8192. That is a real advantage but it is "no additional allocation", not "12000× less memory than a correct implementation needs".

## Fused linear cross-entropy: the first measurement and the fix

The first run showed the fused kernel 5× slower than eager and using more memory
than eager. Two bugs in the chunk loop, both fixed:

| Problem | Effect | Fix |
| --- | --- | --- |
| `grad_weight += torch.mm(dlogits.t(), input_chunk).float()` | Two `(V, H)` temporaries per chunk (~3 GB at `V=128256, H=4096`), not shrinking with `chunk_size`, so chunking saved no memory | `grad_weight.addmm_(...)` in place |
| `chunk_size = BT / (V/H)` gave 128-row chunks | `(V, H)` weight re-read from HBM per chunk for logits, `dX`, and `dW`; latency scaled with chunk count | Size chunks against a logits-memory budget |

Measured before and after on the same A100, bf16, `h=4096, v=128256`, full fwd+bwd:

| BT   | latency before | after    | vs eager | peak before | after       | eager   |
| ---- | -------------- | -------- | -------- | ----------- | ----------- | ------- |
| 1024 | 262.1 ms       | 15.2 ms  | 0.96×    | 4024 MB     | 1269 MB     | 1260 MB |
| 2048 | 269.9 ms       | 28.8 ms  | 1.00×    | 4040 MB     | 1536 MB     | 1520 MB |
| 4096 | 290.9 ms       | 56.9 ms  | 1.02×    | 4071 MB     | **2053 MB** | 3006 MB |
| 8192 | 321.7 ms       | 113.1 ms | 1.02×    | 4135 MB     | **2085 MB** | 6012 MB |

"Before" is [`fused_linear_cross_entropy_20260814_025924.csv`](../../kernels/cross_entropy/benchmarks/results/fused_linear_cross_entropy_20260814_025924.csv),
kept as the pre-fix baseline; "after" is the newest CSV in the same directory.

Result: eager latency parity with 2.88× less memory at BT=8192. Liger 0.8.0 uses
the same `V/H` chunk rule and is 2.8–17× slower here, which is a comparison
against its default configuration, not its ceiling.

## Correctness

All counts below are from a full re-run on 2026-08-14, each suite in its own
process, on an idle A100-80GB. 660 pass, 4 skip, 1 xfail, 0 fail.

| Kernel        | Recorded result                                                    | fp64 gradcheck                       |
| ------------- | ------------------------------------------------------------------ | ------------------------------------ |
| RoPE v3       | 4 pass (forward 30/30, backward 8/8 inside them)                   | PASS                                 |
| LayerNorm     | 98 pass, 4 skip, 1 xfail                                           | 3/3 PASS                             |
| LoRA MLP      | 152 pass                                                           | PASS (v2, v5, v6)                    |
| SwiGLU        | 75 pass                                                            | not run                              |
| LoRA QKV v4   | 90 pass                                                            | PASS (MHA, GQA, r=4/8/16)            |
| RMSNorm v4    | 32 pass (14 top-level + 18 under `kernels/rmsnorm/tests/`)         | —                                    |
| Cross-Entropy | 105 pass (incl. fused linear CE)                                   | uses `assert_close`, not `gradcheck` |
| Embedding     | 15 pass (fp32 + bf16, padding_idx, duplicate rows)                 | uses `assert_close`, not `gradcheck` |
| GeGLU         | 89 pass                                                            | explicitly deferred                  |

Two entries in this table were wrong before this run. GeGLU was recorded as
having no committed run although its 89 tests pass. The whole `tests/` directory
could not be collected: `tests/layernorm/` is a package, so pytest put `tests/`
on `sys.path` instead of the repository root and all nine LayerNorm modules
failed on `import kernels`. A root `conftest.py` fixes it.

Making them collectable surfaced four failures, all in the Liger dW/dB comparison
against eager in fp32 at larger shapes. They were tolerance, not gradient: the
two disagree by 1–6 fp32 epsilons relative to `|dW|max` because the kernel's
block-partial reduction accumulates in a different order. Against a float64
reference the kernel is closer to the truth than eager at every shape above one
row (ratios 0.57–0.74), so a flat `atol=1e-5` failed the more accurate result.
The bound now scales with gradient magnitude. The single xfail is deliberate:
Unsloth's backward returns `None` for dW/dB by design, so a full gradcheck on it
must fail.
