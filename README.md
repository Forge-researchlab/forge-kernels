# forge-kernels

Fused Triton kernels for LLM fine-tuning, with a one-call patching layer for
Hugging Face models (Qwen2/Qwen2.5, Qwen3, Gemma, Gemma2). Benchmarked against
PyTorch eager, [Liger Kernel](https://github.com/linkedin/Liger-Kernel), and
[Unsloth](https://github.com/unslothai/unsloth). Every performance number is read
from a committed result file.

[Install](#install) · [Quickstart](#quickstart) · [Results](#results) · [Kernels](#kernels) · [Reproducing](#reproducing) · [Docs](#docs) · [Contributing](CONTRIBUTING.md)

## Key features

- **One-call patching.** `forge.patch(model)` swaps the kernels into a live model in place, with an idempotent `unpatch`. No model rewrites or trainer changes.
- **Per-call shape guard.** Each shape-sensitive kernel falls back to the original forward when a call is too small for fusion to pay off, so patching a small model does not regress it.
- **Exact, not approximate.** Forward and backward match eager within fp32/fp64 tolerances. 660 tests pass on an A100-80GB.
- **Evidence-backed.** Every speedup below maps to a committed CSV/JSON result file. The [benchmarks doc](docs/benchmarks/benchmarks.md) also lists where each kernel loses.
- **FSDP2-safe.** Patched forwards keep working when weights become sharded DTensors, verified for equivalence on 2 GPUs. This is compatibility, not a distributed speedup (see [Limitations](#limitations)).

## Install

```bash
pip install git+https://github.com/Forge-researchlab/forge-kernels.git
```

Requires a CUDA GPU. Python 3.11–3.12, CUDA 12.1+, `torch>=2.4,<2.5`,
`triton>=3.0,<3.1`, `liger-kernel>=0.8.0`. Managed with [uv](https://docs.astral.sh/uv/).

`transformers` is pinned `<5.0`: v5 requires `torch>=2.5`, which the pin above
excludes, and against it `transformers` silently disables its torch backend.

## Quickstart

```python
import forge
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", torch_dtype="bfloat16")

forge.patch(model)                   # fine-tuning
# forge.patch(model, mode="infer")   # serving: skips loss and LoRA kernels
# forge.unpatch(model)               # restore original forwards
```

On Qwen2.5-0.5B (batch 4 × seq 1024, bf16), training runs **1.20× faster and uses
36% less memory**, driven by fused linear cross-entropy. For a runnable version
that measures the delta live, see
[`examples/quickstart_qwen.ipynb`](examples/quickstart_qwen.ipynb). Full API, the
batch-size crossover analysis, and FSDP2 verification are in
[`docs/api-reference/patching.md`](docs/api-reference/patching.md).

## Supported models

| Architecture      | `model_type`      | Kernels applied                                             |
| ----------------- | ----------------- | ---------------------------------------------------------- |
| Qwen2 / Qwen2.5   | `qwen2`           | RMSNorm, RoPE, SwiGLU, Embedding, LoRA QKV/MLP, fused linear CE |
| Qwen3             | `qwen3`           | RMSNorm, RoPE, SwiGLU, Embedding, LoRA QKV/MLP, fused linear CE |
| Gemma / Gemma2    | `gemma`, `gemma2` | RoPE, GeGLU, Embedding, LoRA QKV/MLP                        |

LoRA QKV/MLP wire in only when PEFT adapters are present, otherwise the base
SwiGLU/GeGLU kernel is used. `ForgeRMSNorm` does not yet accept Gemma's `offset`,
so RMSNorm is skipped on Gemma. Adding an architecture means adding one mapping
file under `forge/forge/patching/`.

## Results

Best recorded speedup per kernel. Every run is on an **A100-SXM4-80GB**, Triton
`3.0.0`, read from a committed result file.

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

These are best-case shapes. Several kernels lose to eager or Liger on small,
irregular, or backward-inclusive shapes. [`docs/benchmarks/benchmarks.md`](docs/benchmarks/benchmarks.md)
lists every regression, the per-kernel correctness counts (660 pass, 4 skip, 1
xfail), and the fused-linear-CE fix that took it from 5× slower than eager to
memory parity.

## Kernels

```
kernels/<name>/
  experiments/v1..v6/     versioned implementations, kept for the perf history
  benchmarks/results/     committed CSV / JSON output
  docs/analysis/          why each version was faster or slower
  tests/
forge/
  forge/kernels/          re-exports of the kernels above
  forge/patching/         forge.patch(model) for Qwen2/Qwen3/Gemma/Gemma2
  demos/                  Qwen2.5-0.5B and Gemma LoRA fine-tuning demos
  tests/                  real-model verification, FSDP2 checks
benchmarks/bench_all.py   runs every benchmark
examples/                 runnable quickstart notebook
docs/
  api-reference/          forge.patch API
  benchmarks/             full results, comparison, known gaps
  design-notes/           per-kernel research and FSDP2 design
artifacts/
  lora_demos/             gemma and qwen LoRA fine-tune curves and adapters
  fsdp2/                   FSDP2 analysis dashboards
  patch_bisect/           per-shape patch guard sweeps
results/                  SwiGLU and GeGLU CSVs
```

## Reproducing

Results above are from an A100-80GB. Other hardware will differ.

```bash
uv sync --dev
python benchmarks/bench_all.py            # every benchmark
python benchmarks/bench_all.py --list     # what will run and what it writes
python benchmarks/bench_all.py --only rope swiglu
```

**Each kernel suite must run in its own pytest process.** Each kernel puts its
own `experiments/` package on `sys.path` under the same top-level name, so
combining suites resolves the wrong kernel (see [Limitations](#limitations)).

```bash
pytest tests/                        # SwiGLU, GeGLU, RMSNorm, LayerNorm: 276 pass, 4 skip, 1 xfail
pytest kernels/lora_mlp/tests/       # 152 pass
pytest kernels/lora_qkv/tests/       # 90 pass
pytest kernels/cross_entropy/tests/  # 105 pass, incl. fused linear CE
pytest kernels/embedding/tests/      # 15 pass
pytest kernels/rmsnorm/tests/        # 18 pass
pytest kernels/rope/tests/           # 4 pass
```

660 tests pass, all re-run on 2026-08-14. The `forge` integration and FSDP2
checks (the latter needing two GPUs) are documented in
[`docs/api-reference/patching.md`](docs/api-reference/patching.md) and need `transformers`, `peft`, and
`matplotlib` (`uv sync --dev`).

## Limitations

- **Multi-GPU means FSDP2 compatibility, not a distributed speedup.** There is no custom collective and no communication/computation overlap. The 2-GPU numbers show equivalence to the single-GPU reference.
- **Kernel suites cannot be collected together.** They share top-level `experiments` package names, so pytest resolves the wrong kernel. Run each suite in its own process.
- **The distribution ships a second top-level `kernels` package** and a vendored Unsloth baseline. Attribution and layout need sorting before any PyPI release ([#16](https://github.com/Forge-researchlab/forge-kernels/issues/16)).

Full list: [`docs/benchmarks/known_gaps.md`](docs/benchmarks/known_gaps.md).

## Docs

| Doc | Contents |
| --- | --- |
| [`docs/api-reference/patching.md`](docs/api-reference/patching.md) | `forge.patch` API, batch-size crossover, FSDP2 verification |
| [`docs/benchmarks/benchmarks.md`](docs/benchmarks/benchmarks.md) | Full results, regressions, correctness counts, fused-linear-CE fix |
| [`docs/benchmarks/comparison.md`](docs/benchmarks/comparison.md) | forge vs Liger, Unsloth, and torch.compile |
| [`docs/benchmarks/known_gaps.md`](docs/benchmarks/known_gaps.md) | What the repo does not do yet |
| [`docs/design-notes/`](docs/design-notes/) | Per-kernel research notes, FSDP2 design, hackathon audit |
| [`examples/quickstart_qwen.ipynb`](examples/quickstart_qwen.ipynb) | Patch a Qwen model and measure the speedup/memory delta |

## Contributing

Contributions welcome. See [CONTRIBUTING.md](CONTRIBUTING.md). Benchmark results
from GPUs other than an A100-80GB are the most useful addition right now, since
every number here comes from one card. Licensed under [Apache-2.0](LICENSE).

## Contributors

- **Devansh Agarwal**: SwiGLU, RMSNorm, LoRA MLP
- **Sasank Tumpati**: fused linear cross-entropy, Qwen patch adapters
- **Shaurya Madukuri**: Embedding, Cross-Entropy, Qwen2.5-0.5B LoRA demo, FSDP2 analysis artifacts
- **Xhitij C**: LoRA QKV, GeGLU
- **Gautam**
- **Srinivasan**
- **Jithamanyu**: benchmark harness

Some commits were authored from a shared GPU box, so `git log` does not reflect
ownership. The list above is authoritative.
