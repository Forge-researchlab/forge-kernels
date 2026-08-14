# forge-kernels

Fused Triton kernels for LLM fine-tuning, plus a `forge` package that patches
them into Hugging Face models (Qwen2/Qwen3/Gemma/Gemma2) with a single call.
Benchmarked against PyTorch eager, [Liger Kernel](https://github.com/linkedin/Liger-Kernel),
and [Unsloth](https://github.com/unslothai/unsloth), with every performance claim
backed by a committed result file.

Licensed under [Apache-2.0](LICENSE). Contributions welcome — see
[CONTRIBUTING.md](CONTRIBUTING.md). Benchmark results from GPUs other than an
A100-80GB are the most useful thing you can add right now, since every number
here comes from one card.

## Install

```bash
pip install git+https://github.com/Forge-researchlab/forge-kernels.git
```

```python
import forge
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", torch_dtype="bfloat16")
forge.patch(model)          # fine-tuning; use mode="infer" when serving
```

`forge.patch` swaps the kernels into a live model with per-call shape guards (a
fused kernel falls back to the original forward when a call is too small to pay
off) and an idempotent `unpatch`. On Qwen2.5-0.5B this makes training **1.20×
faster and uses 36% less memory**, driven by fused linear cross-entropy. Full
API, the batch-size-vs-speedup analysis, and the FSDP2 multi-GPU verification are
in [`docs/patching.md`](docs/patching.md).

> `transformers` is pinned `<5.0` on purpose: v5 requires `torch>=2.5`, and
> against the `torch<2.5` pin here it disables its own torch backend and every
> `AutoModel` call then raises "requires the PyTorch library but it was not
> found". The install also ships a second top-level package named `kernels`, a
> layout wart to fix before any PyPI release ([#16](https://github.com/Forge-researchlab/forge-kernels/issues/16)).

## Results

Best recorded speedup per kernel. Every run is on an **A100-SXM4-80GB**, Triton
`3.0.0`, read from a committed result file — the [full benchmark record,
including where each kernel *loses*](docs/benchmarks.md) is the more honest read.

| Kernel                     | Best recorded speedup | Baseline               | Shape / config                      |
| -------------------------- | --------------------- | ---------------------- | ----------------------------------- |
| **RoPE** (v3)              | **7.1×** fwd          | PyTorch eager          | 2×32/8×2048×128, GQA G=4, bf16      |
| **RMSNorm** (v4)           | **10.2×** fwd         | PyTorch eager          | 2×2048×4096, bf16, offset=1.0       |
| **SwiGLU**                 | **6.5×** fwd          | PyTorch eager (packed) | 4×2048×11008, bf16                  |
| **GeGLU** (activation)     | **5.2×** fwd+bwd      | PyTorch eager (packed) | 4×2048×11008, bf16, tanh            |
| **LoRA QKV** (v4)          | **1.37×** fwd+bwd     | Unsloth                | 4×2048, h=4096, GQA 32/8, r=8, bf16 |
| **LoRA MLP** (v6)          | **1.18×** fwd         | Unsloth                | 4×2048, h=4096, i=14336, r=16, bf16 |
| **LayerNorm**              | **1.57×** fwd+bwd     | PyTorch eager          | 8×2048×4096, fp32                   |
| **Cross-Entropy**          | **2.77×** fwd+bwd     | PyTorch eager          | 8192×128256, fp32                   |
| **Fused Linear CE**        | **2.88× less memory** | PyTorch eager          | BT=8192, h=4096, v=128256, bf16     |
| **Embedding**              | **1.49×** fwd+bwd     | PyTorch eager          | v=32000, d=4096, s=8192, bf16       |

These are best-case numbers. Several kernels **lose** to eager or Liger on small,
irregular, or backward-inclusive shapes — the [benchmarks doc](docs/benchmarks.md)
lists every regression, the per-kernel correctness counts (660 pass, 4 skip, 1
xfail), and the story of the fused-linear-CE kernel that once ran 5× *slower* than
eager. That failure is why this repo's one rule is: a performance claim needs a
committed result file behind it.

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
docs/                     # benchmarks, patching internals, known gaps, research notes
artifacts/                # LoRA demo curves, FSDP2 analysis dashboards
results/                  # SwiGLU and GeGLU CSVs
```

## Reproducing

Requires a CUDA GPU; results above are from an A100-80GB and other hardware will differ.

```bash
uv sync --dev
python benchmarks/bench_all.py            # every benchmark
python benchmarks/bench_all.py --list     # what will run and what it writes
python benchmarks/bench_all.py --only rope swiglu
```

Tests. **Each kernel suite must run in its own pytest process** — each kernel
puts its own `experiments/` package on `sys.path` under the same top-level name,
so combining them resolves the wrong kernel (see [known gaps](docs/known_gaps.md)):

```bash
pytest tests/                        # SwiGLU, GeGLU, RMSNorm, LayerNorm — 276 pass, 4 skip, 1 xfail
pytest kernels/lora_mlp/tests/       # 152 pass
pytest kernels/lora_qkv/tests/       # 90 pass
pytest kernels/cross_entropy/tests/  # 105 pass, incl. fused linear CE
pytest kernels/embedding/tests/      # 15 pass
pytest kernels/rmsnorm/tests/        # 18 pass
pytest kernels/rope/tests/           # 4 pass
```

That is 660 passing tests, all re-run on 2026-08-14. The `forge` integration and
FSDP2 checks (the latter needing two GPUs) are documented in
[`docs/patching.md`](docs/patching.md); they need `transformers`, `peft`, and
`matplotlib`, installed via `uv sync --dev`.

## Docs

- [`docs/benchmarks.md`](docs/benchmarks.md) — full results, every regression, correctness counts, and the fused-linear-CE measurement story
- [`docs/patching.md`](docs/patching.md) — the `forge.patch` API, the batch-size-vs-speedup analysis, and FSDP2 multi-GPU verification
- [`docs/known_gaps.md`](docs/known_gaps.md) — what the repo does not yet do, kept so it does not overstate itself
- [`docs/hackathon_day1_audit.md`](docs/hackathon_day1_audit.md) — a fuller track-by-track assessment

## Requirements

Python 3.11–3.12, CUDA 12.1+, `torch>=2.4,<2.5`, `triton>=3.0,<3.1`, `liger-kernel>=0.8.0`. Managed with [uv](https://docs.astral.sh/uv/).

## Contributors

- **Devansh Agarwal** — SwiGLU, RMSNorm, LoRA MLP
- **Sasank Tumpati** — fused linear cross-entropy, Qwen patch adapters
- **Shaurya Madukuri** — Embedding, Cross-Entropy, Qwen2.5-0.5B LoRA demo, FSDP2 analysis artifacts
- **Xhitij C** — LoRA QKV, GeGLU

Several commits were authored from a shared GPU box, so `git log` does not reflect ownership; the list above is the authoritative one.
