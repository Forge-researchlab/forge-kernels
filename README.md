# kernel-POCs

Triton kernels for LLM fine-tuning, benchmarked against PyTorch eager, [Liger Kernel](https://github.com/linkedin/Liger-Kernel), and [Unsloth](https://github.com/unslothai/unsloth), plus a `forge` package that patches them into Hugging Face Qwen and Gemma models.

Built during the [Forge Hackathon](https://xhitijc2.github.io/forge-hackathon-plan/index.html) (May 23–24, 2026) and extended afterwards. **All work is merged into `main`** — there are no feature branches to hunt through. The pre-cleanup hackathon tree, including the working notes under `context/`, is preserved on the `hackathon-archive` branch.

Contributions are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md). The one hard rule
is that a performance claim needs a committed result file behind it; [this
repository has already been bitten](#fused-linear-cross-entropy-what-the-first-measurement-found)
by a kernel whose harness had never been run. Benchmark results from GPUs other
than an A100-80GB are the most useful thing you can add right now, since every
number here comes from one card.

Licensed under [Apache-2.0](LICENSE).

## Results

All numbers below are read from committed result files, not estimates. Every run is on a **NVIDIA A100-SXM4-80GB**, Triton `3.0.0`, `liger-kernel==0.8.0`. The RoPE/RMSNorm/SwiGLU/GeGLU/LoRA/LayerNorm rows are from 2026-05-23/24 on torch `2.4.1+cu124`; the Cross-Entropy and Embedding rows are from 2026-08-14 on torch `2.4.1+cu121`.

Every benchmark was re-run from scratch on a second A100-80GB box on 2026-08-14 and
the original figures hold: RoPE **7.12×** vs eager (2.77× vs Unsloth fused-QK, 3.24×
vs Liger) with forward 30/30, backward 8/8 and fp64 gradcheck passing, and RMSNorm
**10.24×** forward. `bench_all.py` now completes 11/11.


| Kernel                     | Best recorded speedup | Baseline               | Shape / config                      | Source                                                                               |
| -------------------------- | --------------------- | ---------------------- | ----------------------------------- | ------------------------------------------------------------------------------------ |
| **RoPE** (v3)              | **7.1×** fwd          | PyTorch eager          | 2×32/8×2048×128, GQA G=4, bf16      | [`v3_summary.md`](kernels/rope/benchmarks/results/v3_summary.md)                     |
|                            | 2.8× fwd              | Unsloth fused-QK       | same                                | same                                                                                 |
|                            | 3.2× fwd              | Liger                  | same                                | same                                                                                 |
| **RMSNorm** (v4)           | **10.2×** fwd         | PyTorch eager          | 2×2048×4096, bf16, offset=1.0       | [`v4_summary.md`](kernels/rmsnorm/benchmarks/results/v4_summary.md)                  |
| **SwiGLU**                 | **6.5×** fwd          | PyTorch eager (packed) | 4×2048×11008, bf16                  | [`swiglu_a100_bf16.csv`](results/swiglu_a100_bf16.csv)                               |
|                            | 1.5× fwd              | Liger                  | 4×2048×11008, bf16, mult 0.7/1.3    | [`..._with_liger.csv`](results/swiglu_a100_bf16_with_liger.csv)                      |
| **GeGLU** (activation)     | **5.2×** fwd+bwd      | PyTorch eager (packed) | 4×2048×11008, bf16, tanh            | [`geglu_activation_a100_bf16.csv`](results/geglu_activation_a100_bf16.csv)           |
| **GeGLU** (gate+up fusion) | **1.40×** fwd         | separate PyTorch MLP   | 1×512×4096, i=11008, bf16           | [`geglu_gateup_a100_bf16.csv`](results/geglu_gateup_a100_bf16.csv)                   |
| **LoRA QKV** (v4)          | **1.37×** fwd+bwd     | Unsloth                | 4×2048, h=4096, GQA 32/8, r=8, bf16 | [`CHANGELOG.md`](kernels/lora_qkv/CHANGELOG.md)                                      |
| **LoRA MLP** (v6)          | **1.18×** fwd         | Unsloth                | 4×2048, h=4096, i=14336, r=16, bf16 | [`v6_upgrade_1_latency_*.csv`](kernels/lora_mlp/benchmarks/results/)                 |
| **LayerNorm**              | **1.57×** fwd+bwd     | PyTorch eager          | 8×2048×4096, fp32                   | [`layernorm_tests.executed.ipynb`](kernels/layernorm/layernorm_tests.executed.ipynb) |
| **Cross-Entropy**          | **2.77×** fwd+bwd     | PyTorch eager          | 8192×128256, fp32                   | [`cross_entropy_*.csv`](kernels/cross_entropy/benchmarks/results/)                   |
|                            | 1.00× fwd+bwd         | Liger                  | same                                | same                                                                                 |
| **Fused Linear CE**        | **2.88× less memory** | PyTorch eager          | BT=8192, h=4096, v=128256, bf16     | [`fused_linear_cross_entropy_*.csv`](kernels/cross_entropy/benchmarks/results/)      |
|                            | 2.85× fwd+bwd         | Liger                  | same                                | same                                                                                 |
| **Embedding**              | **1.49×** fwd+bwd     | PyTorch eager          | v=32000, d=4096, s=8192, bf16       | [`bench_*.csv`](kernels/embedding/benchmarks/results/)                               |


### Where these kernels lose

Headline numbers are best-case. The same result files record regressions, and they matter more than the wins:

- **SwiGLU is slower than Liger on 12 of the shape/mode combinations** benchmarked, worst case 0.84× (fwd+bwd, 2×2048×11008 bf16). It only beats Liger on large-hidden forward passes.
- **SwiGLU and GeGLU lose to PyTorch eager on small and irregular shapes** — down to 0.58× (SwiGLU, fp32, 2×7×11009) and 0.66× (GeGLU smoke suite). The win is concentrated at large hidden dimensions.
- **RMSNorm's 10× is forward-only.** With backward included it is roughly at parity with Liger (0.32×–2.68× across shapes, clustered near 1.0×) and generally slower than Unsloth, which does not compute `dW`.
- **GeGLU gate+up fusion narrows as the batch grows.** It reaches 2.84× only at a 14-token batch (2×7×4096); at the realistic 2×2048×4096 training shape it is 1.07× vs PyTorch and 0.97× vs Liger — slightly slower than Liger. It does hold a real memory edge there: 854 MiB vs PyTorch's 1112 MiB.
- **LayerNorm is 0.84× vs eager** at 4×2048×4096 bf16.
- **LoRA MLP memory matches Unsloth rather than beating it** — 737 MB vs 736 MB peak forward. The v6 in-place epilogue closed a 1185 MB → 737 MB gap against our own earlier version; it is not a win over the baseline.
- **Embedding loses to PyTorch on 94 of 108 benchmarked configs**, down to 0.31× at v=32000/d=768/s=512. Sorting indices to group duplicate rows only pays for itself at large embedding dimension and long sequences: every win is at `d=4096, s=8192`. The vendored Liger-style reference cannot be compared at all in bf16 — `tl.atomic_add` rejects bf16 on Triton 3.0, whereas the sort-and-group backward accumulates in fp32 and is also deterministic.
- **Fused linear CE only saves memory once the logits dominate.** At BT=1024–2048 it is level with eager (1269 vs 1260 MB, 1536 vs 1520 MB) because the `(V, H)` weight gradient, not the logits chunk, sets the floor. The 2.88× saving appears at BT=8192 (2085 vs 6012 MB).
- **Plain CE's memory ratio is an artifact worth stating plainly.** It transforms the logits tensor in place, so it allocates ~0.1 MB against eager's 12 GB of intermediates at BT=8192. That is a real advantage but it is "no additional allocation", not "12000× less memory than a correct implementation needs".

### Fused linear cross-entropy: what the first measurement found

The first time this harness was ever run it showed the fused kernel losing on both
axes it exists to win — 5× slower than eager *and* using more memory than eager.
Two causes, both in the chunk loop:

1. `grad_weight += torch.mm(dlogits.t(), input_chunk).float()` allocated two
   `(V, H)` temporaries per chunk. At `V=128256, H=4096` that is ~3 GB of transient
   allocation per chunk, and it does not shrink when `chunk_size` shrinks — so
   chunking bought no memory saving at all. Replaced with `grad_weight.addmm_(...)`.
2. `chunk_size` was derived as `BT / (V/H)`, giving 128-row chunks at these shapes.
   The `(V, H)` weight is re-read from HBM once per chunk for the logits matmul,
   again for `dX`, and read-modify-written again for `dW`, so latency scaled with
   the chunk count for no benefit. Now sized against a logits-memory budget.

Measured before and after on the same A100, bf16, `h=4096, v=128256`, full fwd+bwd:

| BT   | latency before | after    | vs eager | peak before | after       | eager   |
| ---- | -------------- | -------- | -------- | ----------- | ----------- | ------- |
| 1024 | 262.1 ms       | 15.2 ms  | 0.96×    | 4024 MB     | 1269 MB     | 1260 MB |
| 2048 | 269.9 ms       | 28.8 ms  | 1.00×    | 4040 MB     | 1536 MB     | 1520 MB |
| 4096 | 290.9 ms       | 56.9 ms  | 1.02×    | 4071 MB     | **2053 MB** | 3006 MB |
| 8192 | 321.7 ms       | 113.1 ms | 1.02×    | 4135 MB     | **2085 MB** | 6012 MB |

"Before" is [`fused_linear_cross_entropy_20260814_025924.csv`](kernels/cross_entropy/benchmarks/results/fused_linear_cross_entropy_20260814_025924.csv),
kept as the pre-fix baseline; "after" is the newest CSV in the same directory.

So the kernel now runs at eager latency parity while using 2.88× less memory at
BT=8192, where before it was 2.7× slower than eager and used less memory only
because eager's logits had grown past it. Liger 0.8.0 uses the same `V/H` chunk
rule and is unchanged, which is why it is now 2.8–17× slower here; that is a
comparison against Liger's default configuration, not a claim about its ceiling.

### Correctness


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

Two entries in this table were wrong before this run and are worth flagging:
GeGLU was recorded as having no committed run when its 89 tests pass, and the
whole `tests/` directory could not be collected at all — `tests/layernorm/` is a
package, so pytest put `tests/` on `sys.path` instead of the repository root and
all nine LayerNorm modules died on `import kernels`. A root `conftest.py` fixes
it.

Making them collectable surfaced four real failures, all in the Liger dW/dB
comparison against eager in fp32 at the larger shapes. They were tolerance, not
gradient: the two disagree by 1–6 fp32 epsilons relative to `|dW|max` because the
kernel's block-partial reduction accumulates in a different order. Checked
against a float64 reference, the *kernel* is closer to the truth than eager at
every shape above one row (ratios 0.57–0.74), so a flat `atol=1e-5` was failing
the more accurate of the two results. The bound now scales with the gradient
magnitude. The single xfail is unrelated and deliberate — it documents that
Unsloth's backward returns `None` for dW/dB by design, so a full gradcheck on it
must fail.


## Reproducing

Requires a CUDA GPU. Results above are from an A100-80GB; other hardware will differ.

```bash
uv sync --dev
python benchmarks/bench_all.py            # every benchmark
python benchmarks/bench_all.py --list     # what will run and what it writes
python benchmarks/bench_all.py --only rope swiglu
```

Each benchmark is also runnable on its own, for example:

```bash
python kernels/rope/benchmarks/bench_v3.py
python kernels/swiglu/benchmarks/benchmark_swiglu.py --suite a100 --dtype bf16 --save results/swiglu_a100_bf16.csv
```

Tests. **Each kernel suite must run in its own pytest process** — see Known gaps
for why combining them fails:

```bash
pytest tests/                        # SwiGLU, GeGLU, RMSNorm, LayerNorm — 276 pass, 4 skip, 1 xfail
pytest kernels/lora_mlp/tests/       # 152 pass
pytest kernels/lora_qkv/tests/       # 90 pass
pytest kernels/cross_entropy/tests/  # 105 pass, incl. fused linear CE
pytest kernels/embedding/tests/      # 15 pass
pytest kernels/rmsnorm/tests/        # 18 pass
pytest kernels/rope/tests/           # 4 pass
```

That is 660 passing tests, all re-run on 2026-08-14. The 4 skips are the
LayerNorm perf tests, which import a `benchmarks/harness.py` that was never
committed.

The `forge` integration checks need the package plus a real model, and the FSDP2
ones need two GPUs:

```bash
PYTHONPATH=forge:. python forge/tests/verify_lora_qwen_patch.py
PYTHONPATH=forge:. torchrun --nproc-per-node=2 --standalone forge/tests/verify_fsdp2_lora_qwen.py
PYTHONPATH=forge:. torchrun --nproc-per-node=2 --standalone forge/demos/train_lora_qwen_forge.py
PYTHONPATH=forge:. python forge/demos/plot_artifacts_qwen.py
PYTHONPATH=forge:. python forge/demos/run_inference_qwen.py
```

These need `transformers`, `peft`, and `matplotlib`, which are not in the root
`pyproject.toml` — install them alongside the pinned kernel deps.

## Layout

```
kernels/<name>/           # one directory per kernel
  experiments/v1..v6/     #   versioned implementations, kept for the perf history
  benchmarks/results/     #   committed CSV / JSON output
  docs/analysis/          #   why each version was faster or slower
  tests/
forge/                    # installable package: kernels + HF patching layer
  forge/kernels/          #   re-exports of the kernels above
  forge/patching/         #   forge.patch(model) for Qwen2/Qwen3/Gemma/Gemma2
  demos/                  #   Qwen2.5-0.5B and Gemma LoRA fine-tuning demos
  tests/                  #   real-model verification, FSDP2 checks
benchmarks/bench_all.py   # runs every benchmark above
docs/                     # per-kernel research notes
artifacts/                # LoRA demo curves, FSDP2 analysis dashboards
results/                  # SwiGLU and GeGLU CSVs
```

The `forge` package is the integration story: `forge.patch(model)` swaps kernels into a live Hugging Face model, with architecture detection for `qwen2`, `qwen3`, `gemma`, and `gemma2`, and an idempotent `unpatch`. See [`forge/forge/patching/core.py`](forge/forge/patching/core.py).

### Verified in a live model

Re-run 2026-08-14 on 2× A100-80GB, torch `2.4.1+cu121`, real `Qwen/Qwen2.5-0.5B` weights:

| Check | Result |
| ----- | ------ |
| `forge.patch` on Qwen2.5-0.5B | 24 LoRA-QKV + 24 LoRA-MLP + 1 fused-linear-CE replaced; loss delta 9.5e-07, min cosine 0.9994; `unpatch` restores |
| FSDP2 on 2 GPUs | forward bit-identical; grad cosine 0.99995; 5-step loss matches the single-GPU reference to 9.2e-05 relative |
| 200-step LoRA fine-tune under FSDP2 | loss 4.3686 → 0.0005, ~1.03 s/step, 1.75 GB peak per rank, 8.1M of 638M params trainable (1.27%) |
| Held-out generation | base "Good morning everyone!" → "Good morning, everyone!"; tuned → "Ahoy, ye good and true matey!" |

Artifacts land in [`artifacts/lora_demo_qwen/`](artifacts/lora_demo_qwen/) and
[`artifacts/fsdp2_analysis/`](artifacts/fsdp2_analysis/).

## Known gaps

Kept here so the repo does not overstate itself:

- `transformers`, `peft`, and `matplotlib` are needed by `forge/demos/` and `forge/tests/` but are declared only in [`forge/pyproject.toml`](forge/pyproject.toml), not in the root [`pyproject.toml`](pyproject.toml).
- **Every kernel suite must run in its own pytest process.** Each kernel puts its own `experiments/` package on `sys.path` under the same top-level name, so whichever is imported first wins and the others resolve to the wrong kernel. Collecting `lora_mlp`, `lora_qkv`, and `cross_entropy` together fails with `No module named 'experiments.v4.lora_qkv_kernel_v4'` and `cannot import name 'CrossEntropyOutput' from 'experiments.v2'`; embedding and cross-entropy collide the same way. Run separately, all 660 tests pass. The real fix is to make these proper subpackages (`kernels.<name>.experiments.v1`) instead of relying on `sys.path` insertion.
- **Benchmark latencies on a shared box are not trustworthy.** A pass taken while another tenant held the GPU reported every provider, PyTorch baseline included, at roughly half speed. Peak-memory figures are unaffected. Check `nvidia-smi` before believing a latency number.
- `tests/` at the top level only covers SwiGLU, GeGLU, RMSNorm, and LayerNorm. Every other kernel's suite lives under `kernels/<name>/tests/`, which is easy to miss.
- GeGLU fp64 gradcheck is deferred.
- **LayerNorm has no committed perf tests.** Four of them exist under `tests/layernorm/` but import a `benchmarks/harness.py` that was never committed, so they have never run and are now skipped explicitly. The LayerNorm figures come from [`layernorm_tests.executed.ipynb`](kernels/layernorm/layernorm_tests.executed.ipynb), which is the weakest form of evidence here.
- `ForgeRMSNorm` does not accept Gemma's `offset` parameter, so `forge.patch` skips RMSNorm on Gemma models.
- The LoRA QKV v4 CSV disagrees with the numbers in its own `CHANGELOG.md` and analysis doc; the table above uses the CHANGELOG figures, and the discrepancy is unresolved.

A fuller track-by-track assessment is in [`docs/hackathon_day1_audit.md`](docs/hackathon_day1_audit.md).

## Requirements

Python 3.11–3.12, CUDA 12.1+, `torch>=2.4,<2.5`, `triton>=3.0,<3.1`, `liger-kernel>=0.8.0`. Managed with [uv](https://docs.astral.sh/uv/).

## Contributors

- **Devansh Agarwal** — SwiGLU, RMSNorm, LoRA MLP
- **Sasank Tumpati** — fused linear cross-entropy, Qwen patch adapters
- **Shaurya Madukuri** — Embedding, Cross-Entropy, Qwen2.5-0.5B LoRA demo, FSDP2 analysis artifacts
- **Xhitij C** — LoRA QKV, GeGLU

Several commits were authored from a shared GPU box, so `git log` does not reflect ownership; the list above is the authoritative one.
