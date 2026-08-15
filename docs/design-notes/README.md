# Design notes

Research and design records for contributors. These are working notes, not user
documentation. Use the [research template](research-template.md) when adding a
new kernel study.

## Per-kernel research

| Doc | Kernel |
| --- | --- |
| [`swiglu.md`](swiglu.md), [`swiglu_brief.md`](swiglu_brief.md) | SwiGLU |
| [`geglu.md`](geglu.md), [`geglu_brief.md`](geglu_brief.md) | GeGLU |
| [`layernorm.md`](layernorm.md) | LayerNorm |
| [`rmsnorm.md`](rmsnorm.md) | RMSNorm |
| [`embedding_collator_dispatch.md`](embedding_collator_dispatch.md) | Embedding: collator dispatch |
| [`embedding_duplicate_aware_backward_dispatch_plan.md`](embedding_duplicate_aware_backward_dispatch_plan.md) | Embedding: duplicate-aware backward |
| [`stateful_kernel_dispatch.md`](stateful_kernel_dispatch.md) | Shape-based dispatch design |

## System design

| Doc | Contents |
| --- | --- |
| [`phase3_fsdp2_design.md`](phase3_fsdp2_design.md) | FSDP2 patching design, locked decisions |
| [`hackathon_day1_audit.md`](hackathon_day1_audit.md) | Track-by-track assessment |
| [`research-template.md`](research-template.md) | Template for a new per-kernel study |
