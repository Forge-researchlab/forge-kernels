# ForgeRoPE V3 — Test + Benchmark Results

**Run:** 2026-08-14T04:34:06.791367+00:00
**Device:** NVIDIA A100-SXM4-80GB (compute [8, 0])
**Torch / Triton:** 2.4.1+cu121 / 3.0.0
**Kernel design:** V2 base + `@triton.autotune` over num_warps×num_stages, keyed on seq_len

## Correctness

| Suite | Passed | Total |
|---|---|---|
| Forward correctness | 30 | 30 |
| Backward correctness | 8 | 8 |
| Gradcheck (fp64) | PASS | 1 |

## Forward timing (median ms) — V3 vs V2 vs baselines

| Shape (G) | dtype | PyTorch | Liger | UnslQK | V1 | V2 | **V3** | V3/V2 | autotune (nw, ns) |
|---|---|---|---|---|---|---|---|---|---|
| qwen3_8b_short (G=4) | torch.bfloat16 | 0.2981 | 0.1143 | 0.1555 | 0.0997 | 0.0415 | **0.0383** | 1.08× | (2, 2) |
| qwen3_8b_short (G=4) | torch.float16 | 0.2544 | 0.1127 | 0.1668 | 0.0988 | 0.0412 | **0.0383** | 1.08× | (2, 3) |
| qwen3_8b_train (G=4) | torch.bfloat16 | 0.4716 | 0.2145 | 0.1833 | 0.1911 | 0.0752 | **0.0662** | 1.14× | (2, 3) |
| qwen3_8b_train (G=4) | torch.float16 | 0.4697 | 0.2137 | 0.1800 | 0.1908 | 0.0738 | **0.0654** | 1.13× | (2, 2) |
| mqa_extreme (G=8) | torch.bfloat16 | 0.0922 | 0.0648 | 0.1519 | 0.0289 | 0.0138 | **0.0256** | 0.54× | (2, 3) |
| mqa_extreme (G=8) | torch.float16 | 0.0927 | 0.0626 | 0.1501 | 0.0293 | 0.0138 | **0.0247** | 0.56× | (2, 2) |
| mha_no_gqa (G=1) | torch.bfloat16 | 0.1946 | 0.0902 | 0.1530 | 0.0678 | 0.0694 | **0.0431** | 1.61× | (2, 3) |
| mha_no_gqa (G=1) | torch.float16 | 0.1895 | 0.0902 | 0.1511 | 0.0674 | 0.0695 | **0.0431** | 1.61× | (2, 2) |

## Backward timing (median ms)

| Shape (G) | dtype | PyTorch | Liger | V1 | V2 | **V3** | V3/V2 |
|---|---|---|---|---|---|---|---|
| qwen3_8b_short (G=4) | torch.bfloat16 | 0.3271 | 0.1417 | 0.1156 | 0.1077 | **0.1419** | 0.76× |
| qwen3_8b_short (G=4) | torch.float16 | 0.3389 | 0.1473 | 0.1200 | 0.1071 | **0.1448** | 0.74× |
| qwen3_8b_train (G=4) | torch.bfloat16 | 0.5557 | 0.1662 | 0.1937 | 0.1142 | **0.1501** | 0.76× |
| qwen3_8b_train (G=4) | torch.float16 | 0.5544 | 0.1662 | 0.1918 | 0.1093 | **0.1421** | 0.77× |
| mqa_extreme (G=8) | torch.bfloat16 | 0.3194 | 0.1266 | 0.1060 | 0.1094 | **0.1404** | 0.78× |
| mqa_extreme (G=8) | torch.float16 | 0.3234 | 0.1245 | 0.1074 | 0.1121 | **0.1438** | 0.78× |
| mha_no_gqa (G=1) | torch.bfloat16 | 0.4539 | 0.2855 | 0.1073 | 0.1099 | **0.2023** | 0.54× |
| mha_no_gqa (G=1) | torch.float16 | 0.4579 | 0.2853 | 0.2603 | 0.2536 | **0.3009** | 0.84× |

## HBM bandwidth utilization (Forge V3)

| Shape | dtype | Traffic (MB) | V3 time (ms) | Achieved BW (GB/s) |
|---|---|---|---|---|
| qwen3_8b_short | torch.bfloat16 | 42.2 | 0.0383 | 1102 |
| qwen3_8b_short | torch.float16 | 42.2 | 0.0383 | 1102 |
| qwen3_8b_train | torch.bfloat16 | 84.9 | 0.0662 | 1284 |
| qwen3_8b_train | torch.float16 | 84.9 | 0.0654 | 1298 |
| mqa_extreme | torch.bfloat16 | 10.0 | 0.0256 | 389 |
| mqa_extreme | torch.float16 | 10.0 | 0.0247 | 404 |
| mha_no_gqa | torch.bfloat16 | 34.1 | 0.0431 | 790 |
| mha_no_gqa | torch.float16 | 34.1 | 0.0431 | 791 |