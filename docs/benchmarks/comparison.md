# forge vs Liger, Unsloth, and torch.compile

Where forge-kernels sits next to the tools it is benchmarked against. The forge
vs Liger and forge vs Unsloth numbers here come from committed result files (see
[`benchmarks.md`](benchmarks.md)). The torch.compile column is a capability
comparison, not a benchmark: this repo has no committed torch.compile run.

## What each one optimizes

| Tool | Approach | Scope | Multi-GPU | License |
| --- | --- | --- | --- | --- |
| **forge** | Hand-written Triton kernels, patched into HF models | 9 kernels, Qwen/Gemma | FSDP2-compatible (no distributed speedup) | Apache-2.0 |
| **Liger** | Hand-written Triton kernels, patched into HF models | Many kernels, 30+ architectures | FSDP, DeepSpeed, DDP | BSD-2 |
| **Unsloth** | Triton kernels plus a tuned training stack | Full fine-tuning path | Free tier is single-GPU; multi-GPU is paid | Apache-2.0 (OSS parts) |
| **torch.compile** | Graph capture and Inductor codegen | Any PyTorch model | Works with FSDP/DDP | PyTorch |

## Where forge wins and loses

Measured on an A100-80GB. Baselines are eager unless stated.

| Kernel | vs eager | vs Liger | vs Unsloth | Read |
| --- | --- | --- | --- | --- |
| RoPE | 7.1× fwd | 3.2× fwd | 2.8× fwd | forge wins against all three |
| RMSNorm | 10.2× fwd | ~parity with bwd | slower (Unsloth skips `dW`) | forward-only win |
| SwiGLU | 6.5× fwd | loses on 12 of the shapes tested | not benchmarked | wins only at large hidden, forward |
| Fused Linear CE | 2.88× less memory | 2.85× fwd+bwd | not benchmarked | forge wins on the memory-chunk config |
| Cross-Entropy | 2.77× fwd+bwd | 1.00× (parity) | not benchmarked | matches Liger |
| LoRA QKV | not benchmarked | not benchmarked | 1.37× fwd+bwd | small win over Unsloth |
| LoRA MLP | not benchmarked | not benchmarked | 1.18× fwd, memory matches (737 vs 736 MB) | parity |
| Embedding | 1.49× fwd+bwd | not benchmarked | not benchmarked | loses on 94 of 108 configs |

Full regression list: [`benchmarks.md#where-these-kernels-lose`](benchmarks.md#where-these-kernels-lose).

## torch.compile

torch.compile is the default baseline to beat because it needs no kernel code.
It fuses elementwise and reduction ops well and covers any model.

- It does not fuse a linear projection with the cross-entropy loss, so it still
  materializes the full `(BT, V)` logits tensor. That is the allocation forge's
  fused linear CE avoids, and it is the largest memory win here (6012 MB to 2085 MB
  at BT=8192). See [`benchmarks.md`](benchmarks.md).
- On the Qwen2.5-0.5B latency sweep, forge's `mode="reduce-overhead"` compile
  path is included as one of the compared implementations in
  `tests/layernorm/test_perf_time.py`.
- Where a compiled kernel already saturates bandwidth (elementwise norms on
  regular shapes), a hand-written Triton kernel has little room left.

## When to use forge

- You want kernels you can read, modify, and extend. Adding a model is one file
  under `forge/forge/patching/`.
- You want the fused-linear-CE memory saving on a large-vocabulary model and are
  fine-tuning, where the logits dominate.
- You are on multiple GPUs with FSDP2 and want the kernels to keep working when
  weights are sharded, without a paid tier.

## When not to use forge

- Production maturity and broad architecture coverage: Liger is further along.
- Fastest single-GPU fine-tuning out of the box: Unsloth's full stack is tuned
  for that.
- Zero-effort speedups across an arbitrary model: start with torch.compile and
  add forge only where a specific kernel beats it.
