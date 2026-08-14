## What this changes

<!-- One or two sentences. If it fixes an issue, link it. -->

## Numbers

<!--
Required for anything touching performance. Delete this section only if the
change cannot affect speed or memory.

Quote shape, dtype, and baseline, and link the committed result file.
Report regressions as well as wins — a PR that only shows its best shape will
be sent back for the rest.
-->

| Shape / dtype | Baseline | Before | After |
| ------------- | -------- | ------ | ----- |
|               |          |        |       |

Result file:

- [ ] GPU was idle when measured (`nvidia-smi`), and the device was pinned with `CUDA_VISIBLE_DEVICES`
- [ ] Result CSV/JSON is committed in this PR

## Correctness

- [ ] `uv run ruff check .`
- [ ] Relevant `pytest` suites pass (`kernels/embedding/tests/` in its own process)
- [ ] Numerics checked against a reference, not just "it runs"

## Anything a reviewer should know

<!-- Trade-offs, follow-up work, or things you deliberately left out. -->
