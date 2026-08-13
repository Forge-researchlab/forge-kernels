# kernel-POCs

Triton kernels for LLM fine-tuning, benchmarked against PyTorch eager, [Liger Kernel](https://github.com/linkedin/Liger-Kernel), and [Unsloth](https://github.com/unslothai/unsloth), plus a `forge` package that patches them into Hugging Face Qwen and Gemma models.

Built during the [Forge Hackathon](https://xhitijc2.github.io/forge-hackathon-plan/index.html) (May 23–24, 2026) and extended afterwards. **All work is merged into `main`** — there are no feature branches to hunt through.

## Results

All numbers below are read from committed result files, not estimates. Every run is on a **NVIDIA A100-SXM4-80GB**, torch `2.4.1+cu124`, Triton `3.0.0`, `liger-kernel==0.8.0`, dated 2026-05-23/24.


| Kernel                     | Best recorded speedup | Baseline               | Shape / config                      | Source                                                                               |
| -------------------------- | --------------------- | ---------------------- | ----------------------------------- | ------------------------------------------------------------------------------------ |
| **RoPE** (v3)              | **7.1×** fwd          | PyTorch eager          | 2×32/8×2048×128, GQA G=4, bf16      | `[v3_summary.md](kernels/rope/benchmarks/results/v3_summary.md)`                     |
|                            | 2.8× fwd              | Unsloth fused-QK       | same                                | same                                                                                 |
|                            | 3.2× fwd              | Liger                  | same                                | same                                                                                 |
| **RMSNorm** (v4)           | **10.2×** fwd         | PyTorch eager          | 2×2048×4096, bf16, offset=1.0       | `[v4_summary.md](kernels/rmsnorm/benchmarks/results/v4_summary.md)`                  |
| **SwiGLU**                 | **6.5×** fwd          | PyTorch eager (packed) | 4×2048×11008, bf16                  | `[swiglu_a100_bf16.csv](results/swiglu_a100_bf16.csv)`                               |
|                            | 1.5× fwd              | Liger                  | 4×2048×11008, bf16, mult 0.7/1.3    | `[..._with_liger.csv](results/swiglu_a100_bf16_with_liger.csv)`                      |
| **GeGLU** (activation)     | **5.2×** fwd+bwd      | PyTorch eager (packed) | 4×2048×11008, bf16, tanh            | `[geglu_activation_a100_bf16.csv](geglu_activation_a100_bf16.csv)`                   |
| **GeGLU** (gate+up fusion) | **1.40×** fwd         | separate PyTorch MLP   | 1×512×4096, i=11008, bf16           | `[geglu_gateup_a100_bf16.csv](geglu_gateup_a100_bf16.csv)`                           |
| **LoRA QKV** (v4)          | **1.37×** fwd+bwd     | Unsloth                | 4×2048, h=4096, GQA 32/8, r=8, bf16 | `[CHANGELOG.md](kernels/lora_qkv/CHANGELOG.md)`                                      |
| **LoRA MLP** (v6)          | **1.18×** fwd         | Unsloth                | 4×2048, h=4096, i=14336, r=16, bf16 | `[v6_upgrade_1_latency_*.csv](kernels/lora_mlp/benchmarks/results/)`                 |
| **LayerNorm**              | **1.57×** fwd+bwd     | PyTorch eager          | 8×2048×4096, fp32                   | `[layernorm_tests.executed.ipynb](kernels/layernorm/layernorm_tests.executed.ipynb)` |
| **Cross-Entropy**          | not yet measured      | —                      | harness ready, no run committed     | `[benchmarks.md](kernels/cross_entropy/docs/benchmarks.md)`                          |
| **Embedding**              | not yet measured      | —                      | harness ready, no run committed     | `[bench_embedding.py](kernels/embedding/benchmarks/bench_embedding.py)`              |


### Where these kernels lose

Headline numbers are best-case. The same result files record regressions, and they matter more than the wins:

- **SwiGLU is slower than Liger on 12 of the shape/mode combinations** benchmarked, worst case 0.84× (fwd+bwd, 2×2048×11008 bf16). It only beats Liger on large-hidden forward passes.
- **SwiGLU and GeGLU lose to PyTorch eager on small and irregular shapes** — down to 0.58× (SwiGLU, fp32, 2×7×11009) and 0.66× (GeGLU smoke suite). The win is concentrated at large hidden dimensions.
- **RMSNorm's 10× is forward-only.** With backward included it is roughly at parity with Liger (0.32×–2.68× across shapes, clustered near 1.0×) and generally slower than Unsloth, which does not compute `dW`.
- **GeGLU gate+up fusion narrows as the batch grows.** It reaches 2.84× only at a 14-token batch (2×7×4096); at the realistic 2×2048×4096 training shape it is 1.07× vs PyTorch and 0.97× vs Liger — slightly slower than Liger. It does hold a real memory edge there: 854 MiB vs PyTorch's 1112 MiB.
- **LayerNorm is 0.84× vs eager** at 4×2048×4096 bf16.
- **LoRA MLP memory matches Unsloth rather than beating it** — 737 MB vs 736 MB peak forward. The v6 in-place epilogue closed a 1185 MB → 737 MB gap against our own earlier version; it is not a win over the baseline.

### Correctness


| Kernel        | Recorded result                                                            | fp64 gradcheck                       |
| ------------- | -------------------------------------------------------------------------- | ------------------------------------ |
| RoPE v3       | forward 30/30, backward 8/8                                                | PASS                                 |
| LayerNorm     | 89 pass, 0 fail                                                            | 3/3 PASS                             |
| LoRA MLP      | 152 tests pass                                                             | PASS (v2, v5, v6)                    |
| SwiGLU        | 75 pass                                                                    | not run                              |
| LoRA QKV v4   | 18 tests pass                                                              | PASS (MHA, GQA, r=4/8/16)            |
| RMSNorm v4    | results committed under `[tests/results/](kernels/rmsnorm/tests/results/)` | —                                    |
| Cross-Entropy | 25 tests defined, no run committed                                         | uses `assert_close`, not `gradcheck` |
| GeGLU         | no run committed                                                           | explicitly deferred                  |


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

Tests:

```bash
pytest tests/                             # SwiGLU, GeGLU, RMSNorm, LayerNorm
pytest kernels/lora_mlp/tests/ kernels/lora_qkv/tests/ kernels/cross_entropy/tests/
```

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
results/                  # top-level SwiGLU CSVs
```

The `forge` package is the integration story: `forge.patch(model)` swaps kernels into a live Hugging Face model, with architecture detection for `qwen2`, `qwen3`, `gemma`, and `gemma2`, and an idempotent `unpatch`. See `[forge/forge/patching/core.py](forge/forge/patching/core.py)`.

## Known gaps

Kept here so the repo does not overstate itself:

- **Cross-Entropy and Embedding have no committed benchmark results.** Both harnesses are written and wired into `bench_all.py`, but `benchmarks/results/` for each contains only `.gitkeep`. No memory reduction has been measured for fused linear + cross-entropy at any vocabulary size.
- `requirements.txt` is empty — dependencies live in `[pyproject.toml](pyproject.toml)` and `uv.lock`.
- `tests/test_cross_entropy.py`, `test_lora_mlp.py`, `test_lora_qkv.py`, and `test_rope.py` at the top level are empty placeholders. The real suites are under `kernels/<name>/tests/`.
- GeGLU fp64 gradcheck is deferred.
- `ForgeRMSNorm` does not accept Gemma's `offset` parameter, so `forge.patch` skips RMSNorm on Gemma models.
- The LoRA QKV v4 CSV disagrees with the numbers in its own `CHANGELOG.md` and analysis doc; the table above uses the CHANGELOG figures, and the discrepancy is unresolved.

A fuller track-by-track assessment is in `[docs/hackathon_day1_audit.md](docs/hackathon_day1_audit.md)`.

## Requirements

Python 3.11–3.12, CUDA 12.1+, `torch>=2.4,<2.5`, `triton>=3.0,<3.1`, `liger-kernel>=0.8.0`. Managed with [uv](https://docs.astral.sh/uv/).

## Contributors

- **Devansh Agarwal** — SwiGLU,  RMSNorm, LoRA MLP
- **Sasank Tumpati** —  fused linear cross-entropy, Qwen patch adapters
- **Shaurya Madukuri** — Qwen2.5-0.5B LoRA demo,Cross-Entropy, FSDP2 analysis artifacts
- **Xhitij C** — LoRA QKV, GeGLU

Kernel attribution follows `git log`; several commits were authored from a shared GPU box, so treat the mapping as approximate.