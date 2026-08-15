# forge-kernels docs

Start with the [root README](../README.md) for install and quickstart. This
folder is organized by audience.

## api-reference

For users of the library.

| Doc | Contents |
| --- | --- |
| [`api-reference/patching.md`](api-reference/patching.md) | `forge.patch` API, mode/guard flags, batch-size crossover, FSDP2 verification |

## benchmarks

Performance record and positioning. Every number maps to a committed result file.

| Doc | Contents |
| --- | --- |
| [`benchmarks/benchmarks.md`](benchmarks/benchmarks.md) | Full results, regressions, correctness counts, fused-linear-CE fix |
| [`benchmarks/comparison.md`](benchmarks/comparison.md) | forge vs Liger, Unsloth, and torch.compile |
| [`benchmarks/known_gaps.md`](benchmarks/known_gaps.md) | What the repo does not do yet |

## design-notes

For contributors. Per-kernel research, the FSDP2 design, and the hackathon audit.
Not needed to use the library. See [`design-notes/`](design-notes/).

## examples

Runnable notebooks live in [`../examples/`](../examples/).
