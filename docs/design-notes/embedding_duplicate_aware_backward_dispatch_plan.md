# Embedding Duplicate-Aware Backward Dispatch Plan

This is the implementation plan for a clean v1 patch. The goal is to let the
data pipeline preselect the embedding backward path using duplicate statistics
from `input_ids`, while preserving current behavior when no dispatch config is
provided.

This doc is intentionally scoped to planning. It does not propose the full
precomputed-grouping or hot-token hybrid algorithm yet.


## 1. Goal

Current Forge embedding backward chooses the algorithm during backward by
inspecting `input_ids` on GPU:

```text
small input      -> index_add_
small groups     -> grouped Triton reduction
large groups     -> cooperative two-phase Triton reduction
```

The v1 change is:

```text
collator/training loop computes duplicate stats on CPU
        |
        v
EmbeddingBackwardDispatchConfig(path, n_tokens, max_count_bucket)
        |
        v
embedding forward saves config on autograd ctx
        |
        v
embedding backward uses preselected path
```

Primary v1 wins:

- remove the `counts.max().item()` sync from backward dispatch;
- make algorithm choice explicit and testable;
- enable future prewarm/precompile decisions;
- create a safe bridge for later CPU-precomputed grouping metadata.

Non-goals for v1:

- do not move gradient computation to CPU;
- do not add sparse optimizer semantics;
- do not change model forward signatures;
- do not require Hugging Face model code changes;
- do not implement hot-token hybrid routing yet;
- do not attempt CUDA graph or `torch.compile` guarantees beyond current Forge
  behavior.


## 2. Current Implementation

Current file:

```text
kernels/embedding/experiments/v1/embedding_kernel_v1_upgrade_1.py
```

Forward:

```python
ForgeEmbeddingFunction.forward(ctx, embeddings, indices)
```

- makes `embeddings` and `indices` contiguous;
- flattens `indices`;
- launches `forge_embedding_forward_kernel`;
- saves `indices_flat` and `embeddings` on `ctx`;
- returns output with original input shape plus embedding dim.

Backward:

```python
ForgeEmbeddingFunction.backward(ctx, grad_output)
```

Current dispatch:

```python
if n_elements >= SORT_BACKWARD_THRESHOLD:
    sorted_indices, sorted_order = torch.sort(indices_flat, stable=True)
    unique_tokens, counts = torch.unique_consecutive(
        sorted_indices,
        return_counts=True,
    )
    group_offsets = cumsum(counts)

    max_gs = int(counts.max().item())

    if max_gs > COOPERATIVE_GROUP_THRESHOLD:
        cooperative two-phase reduction
    else:
        grouped reduction
else:
    grad_weight.index_add_(0, indices_flat, grad_output)
```

Important current costs:

- GPU sort and unique are on the backward critical path.
- `counts.max().item()` forces CPU to wait for the GPU result.
- `max_gs` controls Triton constexprs:
  - grouped path uses `MAX_GROUP_SIZE`;
  - cooperative path uses `CHUNKS_PER_GROUP`.
- Runtime path choice is hidden inside backward, so it cannot be prewarmed or
  predicted by the data pipeline.


## 3. Clean Patch Principle

The v1 patch must be additive:

```text
no config supplied -> exact current behavior
valid config       -> use preselected path
invalid config     -> fail loudly in debug/tests, never silently corrupt grads
```

The config is advisory for performance, but correctness must not depend on
unchecked assumptions.

Key rule:

```text
Any config bucket used as a Triton loop bound must be an upper bound.
```

If the actual max duplicate group is 65, then a bucket of 64 is unsafe because
the grouped/cooperative loop could miss elements. A bucket of 128 is safe but
may do extra guarded iterations.


## 4. Proposed Data Structures

Add a small frozen dataclass, likely under a new runtime/config module:

```python
from dataclasses import dataclass
from typing import Literal

EmbeddingBackwardPath = Literal["auto", "index_add", "grouped", "cooperative"]

@dataclass(frozen=True)
class EmbeddingBackwardDispatchConfig:
    path: EmbeddingBackwardPath
    n_tokens: int
    max_count: int | None
    max_count_bucket: int | None
    n_unique: int | None = None
    duplicate_ratio: float | None = None
    source: str = "cpu_collator"
```

Field meanings:

| Field | Meaning | Required for v1 |
|---|---|---|
| `path` | Chosen backward path. | Yes |
| `n_tokens` | Number of flattened input IDs. | Yes, used for validation. |
| `max_count` | Exact maximum duplicate count if known. | Useful for logging/debug. |
| `max_count_bucket` | Safe power-of-two upper bound for launch constants. | Yes for grouped/cooperative. |
| `n_unique` | Number of unique token IDs. | Optional telemetry. |
| `duplicate_ratio` | `1 - n_unique / n_tokens`. | Optional telemetry. |
| `source` | Where config came from. | Optional debug. |

The public policy helper should mirror current thresholds:

```python
def choose_embedding_backward_path(n_tokens: int, max_count: int) -> str:
    if n_tokens < SORT_BACKWARD_THRESHOLD:
        return "index_add"
    if max_count > COOPERATIVE_GROUP_THRESHOLD:
        return "cooperative"
    return "grouped"
```

The helper that builds config from CPU `input_ids`:

```python
def build_embedding_backward_dispatch_config(input_ids: torch.Tensor) -> (
    EmbeddingBackwardDispatchConfig | None
):
    if input_ids.device.type != "cpu":
        return None

    flat = input_ids.reshape(-1)
    n_tokens = flat.numel()
    if n_tokens == 0:
        return EmbeddingBackwardDispatchConfig(
            path="index_add",
            n_tokens=0,
            max_count=0,
            max_count_bucket=0,
            n_unique=0,
            duplicate_ratio=0.0,
        )

    if n_tokens < SORT_BACKWARD_THRESHOLD:
        return EmbeddingBackwardDispatchConfig(
            path="index_add",
            n_tokens=n_tokens,
            max_count=None,
            max_count_bucket=None,
        )

    _, counts = torch.unique(flat, return_counts=True)
    max_count = int(counts.max().item())  # CPU item, no GPU sync
    n_unique = int(counts.numel())
    bucket = next_power_of_2(max_count)
    path = choose_embedding_backward_path(n_tokens, max_count)

    return EmbeddingBackwardDispatchConfig(
        path=path,
        n_tokens=n_tokens,
        max_count=max_count,
        max_count_bucket=bucket,
        n_unique=n_unique,
        duplicate_ratio=1.0 - n_unique / n_tokens,
    )
```

Implementation note: do not compute this helper on a CUDA tensor in v1. If
`input_ids` are already on GPU, return `None` and use current runtime behavior.


## 5. How Config Reaches the Embedding

Do not change model forward signatures. A Hugging Face model should still be
called as:

```python
model(input_ids=..., attention_mask=..., labels=...)
```

The cleanest v1 path is a context manager backed by `contextvars.ContextVar`:

```python
with forge.kernel_config(config):
    outputs = model(**batch)
```

The patched embedding forward reads the current config and passes only the
embedding part into the autograd function:

```python
def forward(input_ids):
    cfg = forge.current_kernel_config()
    emb_cfg = None if cfg is None else cfg.embedding_backward
    return ForgeEmbeddingFunction.apply(module.weight, input_ids, emb_cfg)
```

Why a context manager:

- no model signature changes;
- no extra keys accidentally passed into Hugging Face model forward;
- per-process isolation works naturally with DDP;
- nested forwards/microbatches can save their config on each autograd context;
- direct `ForgeEmbeddingFunction.apply(weight, indices)` remains valid.

The autograd function should accept an optional non-tensor argument:

```python
def forward(ctx, embeddings, indices, dispatch_config=None):
    ctx.dispatch_config = dispatch_config
    ...

def backward(ctx, grad_output):
    ...
    return grad_weight, None, None
```

Backward must not read a global context variable. It must use the config saved
on `ctx` during forward. This is required for gradient accumulation, delayed
backward, activation checkpointing, and multiple concurrent model forwards.


## 6. Backward Behavior With Config

The backward should be factored into small helpers:

```python
def _embedding_backward_auto(...):
    # Current behavior. Computes max_gs with counts.max().item().

def _embedding_backward_with_config(..., cfg):
    # New v1 behavior. Uses cfg.path and cfg.max_count_bucket.
```

Validation:

```python
if cfg is None or cfg.path == "auto":
    return _embedding_backward_auto(...)

if cfg.n_tokens != n_elements:
    raise RuntimeError("Embedding dispatch config does not match input_ids")
```

Path behavior:

### `index_add`

```python
grad_weight.index_add_(0, indices_flat, grad_output)
```

No sort, unique, or sync.

### `grouped`

Still does GPU sort/unique/cumsum in v1:

```python
sorted_indices, sorted_order = torch.sort(indices_flat, stable=True)
unique_tokens, counts = torch.unique_consecutive(sorted_indices, return_counts=True)
group_offsets = cumsum(counts)
```

But it does not call:

```python
counts.max().item()
```

Instead:

```python
MAX_GROUP_SIZE = cfg.max_count_bucket
```

Requirement:

```text
cfg.max_count_bucket >= actual max group size
```

### `cooperative`

Same sort/unique/cumsum as today, but launch constants come from config:

```python
CHUNK_SIZE = next_power_of_2(TARGET_CHUNK_SIZE)
CHUNKS_PER_GROUP = next_power_of_2(ceil(cfg.max_count_bucket / CHUNK_SIZE))
```

This may over-allocate scratch if the bucket is larger than the exact max, but
it remains correct.


## 7. Compatibility Rules

### PyTorch `nn.Embedding` Semantics

Current Forge embedding is only equivalent to default dense embedding behavior.
Before broad rollout, the patch layer must avoid unsupported semantics:

| Module option | Clean v1 behavior |
|---|---|
| `sparse=True` | Do not patch; use original PyTorch embedding. |
| `max_norm is not None` | Do not patch; PyTorch mutates/renorms selected rows in forward. |
| `scale_grad_by_freq=True` | Do not patch in v1, or implement count-based scaling first. |
| `padding_idx is not None` | Prefer fallback for v1 unless pad-row skip is implemented. |

This should be handled at patch time, not discovered during backward.

Proposed patching behavior:

```text
supported embedding module -> patch forward
unsupported embedding module -> leave original forward intact and record skip reason
```

Recording skip reasons helps debugging:

```python
model._forge_skipped = [
    ("model.embed_tokens", "embedding", "padding_idx not supported yet"),
]
```

### Hugging Face Trainer Compatibility

Do not put `forge_kernel_config` into a batch unless the training step removes
it before calling the model. Many Hugging Face models reject unknown keyword
arguments.

Safe manual loop:

```python
cfg = batch.pop("_forge_kernel_config", None)
with forge.kernel_config(cfg):
    outputs = model(**batch)
```

For Hugging Face `Trainer`, provide a small integration later:

- custom `Trainer.compute_loss` wrapper; or
- a Forge helper that pops config before model call.

Do not require this for default `forge.patch(model)`.

### DDP and Multi-GPU

DDP is safe if the config is per-process/per-rank:

- each rank receives its local `input_ids`;
- each rank computes local duplicate stats;
- each rank saves local config on local autograd ctx;
- DDP all-reduces the final dense `grad_weight` as usual.

No cross-rank synchronization is needed for v1.

Constraints:

- config must not contain CUDA tensor pointers;
- config must be picklable if produced in DataLoader workers;
- config must be immutable after creation;
- do not use one global mutable config object across ranks.

`torch.nn.DataParallel` is not a target for v1. It can split a batch inside one
process after config creation, which can make `n_tokens` and duplicate stats
wrong for each device shard. Use fallback/current behavior if DataParallel is
detected.

FSDP/tensor-parallel/vocab-parallel embeddings are not validated by the current
Forge embedding patch. Treat them as fallback until explicitly tested.

### Gradient Accumulation and Activation Checkpointing

Saving the config on `ctx` is required.

Bad pattern:

```python
backward reads current global config
```

Good pattern:

```python
forward reads current global config once
forward saves it on ctx
backward uses ctx.dispatch_config
```

This keeps multiple outstanding forwards safe.

### `torch.compile` and CUDA Graphs

Current Forge already uses Python patching, custom autograd, and Triton JIT
paths. v1 should not claim new graph-capture support.

Compatibility stance:

- if no config is supplied, behavior remains current;
- if `torch.compile` or CUDA graphs require stable branches, users can leave
  dispatch config disabled;
- future work can add fixed-path modes after profiling.


## 8. Rejected Designs

| Design | Why rejected for clean v1 |
|---|---|
| Add `kernel_config` as a model forward kwarg | Breaks Hugging Face model signatures and Trainer behavior. |
| Store config as a Python attribute on `input_ids` | Not reliable across tensor copies, H2D transfer, slicing, DDP, or serialization. |
| Read global config during backward | Incorrect with delayed backward, gradient accumulation, and activation checkpointing. |
| Compute stats from CUDA `input_ids` in the helper | Reintroduces GPU sync or GPU-side preprocessing outside the intended path. |
| Trust arbitrary user config silently | Can corrupt gradients if `max_count_bucket` is too small. |
| Implement CPU precomputed grouping in the first patch | Larger API surface, tensor metadata transfer, and more failure modes. |
| Implement hot-token hybrid immediately | Interesting, but needs a baseline config-only patch first. |


## 9. Staged Patch Plan

### Stage 0: Safety Gates

Add embedding patch compatibility checks:

- skip `sparse=True`;
- skip `max_norm is not None`;
- skip `scale_grad_by_freq=True`;
- skip or implement `padding_idx`;
- record skip reasons.

This prevents silent semantic drift.

### Stage 1: Config Dataclass and Policy Helper

Add:

- `EmbeddingBackwardDispatchConfig`;
- `choose_embedding_backward_path`;
- `build_embedding_backward_dispatch_config`;
- unit tests for small, unique-heavy, duplicate-heavy inputs.

The helper should only operate on CPU tensors in v1.

### Stage 2: Runtime Context

Add:

- `ForgeKernelConfig` container;
- `forge.kernel_config(config)` context manager;
- `forge.current_kernel_config()`;
- tests that nested contexts restore correctly.

Default behavior remains `None`.

### Stage 3: Autograd Function Accepts Optional Config

Change:

```python
ForgeEmbeddingFunction.apply(weight, indices)
```

to also support:

```python
ForgeEmbeddingFunction.apply(weight, indices, dispatch_config)
```

The old two-argument call must keep working.

Forward saves config on `ctx`.

Backward:

- `None` or `"auto"` uses current implementation;
- supplied config validates `n_tokens`;
- supplied config avoids `counts.max().item()` for grouped/cooperative;
- returns `None` for the non-tensor config arg.

### Stage 4: Patch Wrapper Integration

Update `_make_embedding_forward`:

```python
def forward(input_ids):
    cfg = current_kernel_config()
    emb_cfg = None if cfg is None else cfg.embedding_backward
    return ForgeEmbeddingFunction.apply(module.weight, input_ids, emb_cfg)
```

Keep direct model calls unchanged.

### Stage 5: Optional Collator Helper

Add an optional helper, not automatic behavior:

```python
batch["_forge_kernel_config"] = ForgeKernelConfig(
    embedding_backward=build_embedding_backward_dispatch_config(batch["input_ids"])
)
```

The training loop must pop the config before calling the model:

```python
cfg = batch.pop("_forge_kernel_config", None)
with forge.kernel_config(cfg):
    outputs = model(**batch)
```

This avoids breaking model signatures.

### Stage 6: Instrumentation and Benchmarks

Add debug counters/timers behind an opt-in flag:

- selected path;
- config source;
- `n_tokens`;
- `n_unique`;
- `max_count`;
- `max_count_bucket`;
- whether runtime fallback was used.

Benchmark:

- current auto dispatch vs config-only dispatch;
- all unique, medium duplicates, high duplicates;
- fp32 and bf16;
- `D` 768, 2048, 4096;
- `n_tokens` 512, 2048, 8192, 32768.


## 10. Test Plan

Correctness tests:

- no config equals current behavior;
- explicit `index_add` path matches PyTorch;
- explicit `grouped` path matches PyTorch;
- explicit `cooperative` path matches PyTorch;
- high duplicate stress case;
- all unique case;
- 2D input shape;
- mismatched `n_tokens` config raises;
- old two-argument API still works;
- patched model works without a config;
- patched model works inside `with forge.kernel_config(...)`.

Compatibility tests:

- unsupported `nn.Embedding` options are skipped or fall back;
- context manager restores previous config after exit;
- multiple forwards with different configs before backward each use their own
  saved config;
- CPU helper returns `None` or refuses CUDA tensors.

Performance tests:

- verify config-only path removes `counts.max().item()` from backward;
- measure whether removing sync changes kernel-local time;
- confirm no slowdown when config is absent.


## 11. Expected v1 Impact

This patch does not remove GPU sort/unique yet. Therefore v1 speedup may be
modest.

Expected improvements:

- eliminates one CPU/GPU synchronization point in grouped/cooperative dispatch;
- makes path choice explicit for logging and profiling;
- enables future prewarming using path/bucket statistics;
- creates a clean API for precomputed grouping metadata later.

Cases where v1 helps most:

- duplicate-heavy batches using cooperative path;
- unstable batches that trigger different `max_count` buckets;
- training loops where the `.item()` sync blocks useful CPU scheduling.

Cases where v1 may not help:

- tiny batches using `index_add`;
- all-unique batches where sort/unique dominates and `.item()` is small;
- CPU collator is already the bottleneck;
- no optional config is supplied.


## 12. Future v2 Extensions

Once v1 is validated, extend in this order:

1. Precomputed CPU grouping metadata:
   - `sorted_order`;
   - `unique_tokens`;
   - `group_offsets`;
   - `counts`.
2. Hot-token hybrid algorithm:
   - cooperative/grouped reduction for hot IDs;
   - simple scatter/index-add for cold IDs.
3. `scale_grad_by_freq` support using collator counts.
4. `padding_idx` support with skip/zero-grad behavior.
5. Sparse-gradient or fused sparse-optimizer path.

The v1 config path is the compatibility foundation for all of these.


## 13. Final Patch Shape

Clean v1 patch summary:

```text
Add optional CPU-built EmbeddingBackwardDispatchConfig.
Pass it through a context manager, not through model kwargs.
Save it on autograd ctx during forward.
Use it in backward to choose index_add/grouped/cooperative.
Keep current auto behavior when config is absent.
Add patch-time semantic guards for unsupported nn.Embedding options.
Do not add precomputed metadata or new algorithms in the first patch.
```

This keeps the change small, measurable, and compatible with existing Forge
usage while opening the path to the larger collator-aware optimization.
