# Contributing

This repository is a set of Triton kernels for LLM fine-tuning plus a `forge`
package that patches them into Hugging Face models. Contributions are welcome,
particularly benchmark results on hardware other than an A100-80GB.

By contributing you agree that your work is licensed under
[Apache-2.0](LICENSE).

## The one rule that matters here

**A performance claim without a committed result file does not land.**

This repository exists to measure kernels, and it has already been burned by the
alternative. Two kernels sat in `main` for months with harnesses that had never
been executed. When they were finally run, the fused linear cross-entropy kernel
turned out to be 5× *slower* than PyTorch eager while using *more* memory — the
exact opposite of the kernel's purpose — because of a per-chunk temporary that no
one had measured. See the [fused linear cross-entropy
section](docs/benchmarks.md#fused-linear-cross-entropy-what-the-first-measurement-found)
in the benchmarks doc.

So if your change touches performance:

1. Commit the CSV or JSON your harness produced, under
   `kernels/<name>/benchmarks/results/` or `results/`.
2. Quote the shape, dtype, and baseline with every number. "1.4× faster" means
   nothing without them; the same kernel here ranges from 0.31× to 1.49×
   depending only on shape.
3. Report the regressions too. The benchmarks doc has a
   [Where these kernels lose](docs/benchmarks.md#where-these-kernels-lose) section
   and it is load-bearing — reviewers trust the wins because the losses are listed.

## Branches and how work lands

```
your fork / feature branch  ──PR──▶  dev  ──PR──▶  main
```

- **`dev`** is the integration branch. Open your pull request against `dev`, not
  `main`. GitHub will offer `main` as the default base, so change it.
- **`main`** is release-only. Every commit on it should be a state we would tag.
- **`hackathon-archive`** is the frozen pre-cleanup tree from the May 2026
  hackathon, including the working notes that used to live in `context/`. Nothing
  is merged out of it; it exists so that history is not lost.

Both `main` and `dev` are protected: no direct pushes, no force pushes, no
deletion, CI must pass, and a maintainer review is required. So the only route in
is a pull request, including for the maintainer.

## Environment

Python 3.11–3.12, CUDA 12.1+, and a CUDA GPU. The kernel dependencies are
pinned deliberately (`torch>=2.4,<2.5`, `triton>=3.0,<3.1`) because the recorded
numbers are tied to them.

```bash
uv sync --dev   # includes transformers, peft, and matplotlib for forge/demos and forge/tests
```

## Before opening a pull request

```bash
uv run ruff check .
uv run pytest tests/                        # 276 pass, 4 skip, 1 xfail
uv run pytest kernels/lora_mlp/tests/       # 152
uv run pytest kernels/lora_qkv/tests/       # 90
uv run pytest kernels/cross_entropy/tests/  # 105
uv run pytest kernels/embedding/tests/      # 15
uv run pytest kernels/rmsnorm/tests/        # 18
uv run pytest kernels/rope/tests/           # 4
```

**Run each kernel suite as its own process.** Every kernel puts its own
`experiments/` package on `sys.path` under the same top-level name, so whichever
gets imported first wins and the rest resolve to the wrong kernel. Combining
them produces errors that look like missing code but are not — for example
`No module named 'experiments.v4.lora_qkv_kernel_v4'`. Separately, all 660 tests
pass.

If you changed a kernel, run its benchmark as well:

```bash
uv run python benchmarks/bench_all.py --list
uv run python benchmarks/bench_all.py --only <name>
```

## Benchmarking on a shared machine

Check `nvidia-smi` first and confirm the GPU is idle. A pass taken while another
process held the GPU reported every provider — including the PyTorch baseline —
at roughly half speed, which silently invalidates every ratio in the file. Peak
memory figures are not affected by contention.

Pin the device explicitly with `CUDA_VISIBLE_DEVICES=<n>` rather than trusting
device 0 to be free.

## Adding a kernel

Follow the existing layout so the perf history stays legible:

```
kernels/<name>/
  experiments/v1../vN/     # keep every version; the analysis docs reference them
  benchmarks/              # harness + results/
  docs/analysis/           # why each version was faster or slower
  tests/
```

Then wire the benchmark into [`benchmarks/bench_all.py`](benchmarks/bench_all.py)
so it runs with everything else, and re-export the kernel through
`forge/forge/kernels/` if it should be patchable into a live model.

## Issues

Use the templates: **bug** for something that is broken, **enhancement** for
improving something that already works, **feature** for something new. Please
include your GPU, torch version, and Triton version — several problems here were
version-specific, such as `tl.atomic_add` rejecting bf16 on Triton 3.0 and
`fully_shard` not being public before torch 2.6.
