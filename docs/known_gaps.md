# Known gaps

Kept so the repo does not overstate itself.

- **The distribution declares its packages by hand.** `kernels/` mostly predates any packaging and has no `__init__.py`, so setuptools cannot auto-discover it without namespace discovery, which then makes the two-source-root mapping ambiguous. [`scripts/check_packages_declared.py`](../scripts/check_packages_declared.py) runs in CI to catch a new kernel that was never added to the list, but the list still has to be edited.
- **The vendored Unsloth baseline ships in the wheel.** `kernels/lora_mlp/reference/unsloth_baseline.py` is imported at runtime by LoRA MLP v5/v6, so it cannot be excluded. Attribution needs sorting out before any PyPI release ([#16](https://github.com/Forge-researchlab/forge-kernels/issues/16)).
- **Every kernel suite must run in its own pytest process.** Each kernel puts its own `experiments/` package on `sys.path` under the same top-level name, so whichever is imported first wins and the others resolve to the wrong kernel. Collecting `lora_mlp`, `lora_qkv`, and `cross_entropy` together fails with `No module named 'experiments.v4.lora_qkv_kernel_v4'` and `cannot import name 'CrossEntropyOutput' from 'experiments.v2'`; embedding and cross-entropy collide the same way. Run separately, all 660 tests pass. The real fix is to make these proper subpackages (`kernels.<name>.experiments.v1`) instead of relying on `sys.path` insertion.
- **Benchmark latencies on a shared box are not trustworthy.** A pass taken while another tenant held the GPU reported every provider, PyTorch baseline included, at roughly half speed. Peak-memory figures are unaffected. Check `nvidia-smi` before believing a latency number.
- `tests/` at the top level only covers SwiGLU, GeGLU, RMSNorm, and LayerNorm. Every other kernel's suite lives under `kernels/<name>/tests/`, which is easy to miss.
- GeGLU fp64 gradcheck is deferred.
- **LayerNorm has no committed perf tests.** Four of them exist under `tests/layernorm/` but import a `benchmarks/harness.py` that was never committed, so they have never run and are now skipped explicitly. The LayerNorm figures come from [`layernorm_tests.executed.ipynb`](../kernels/layernorm/layernorm_tests.executed.ipynb), which is the weakest form of evidence here.
- `ForgeRMSNorm` does not accept Gemma's `offset` parameter, so `forge.patch` skips RMSNorm on Gemma models.
- **`MIN_FUSED_ELEMENTS` is one threshold for every shape-sensitive kernel.** It was calibrated on RMSNorm and SwiGLU on an A100 in bf16; embedding and GeGLU almost certainly cross over somewhere else, and no threshold was measured on other hardware. A per-kernel table read off the committed CSVs would be better than one constant.

A fuller track-by-track assessment is in [`hackathon_day1_audit.md`](hackathon_day1_audit.md).
