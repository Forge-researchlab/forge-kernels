"""Shared GPU benchmarking utilities for the forge-kernels test suite.

Provides timing and memory-measurement helpers used by the LayerNorm
performance tests (tests/layernorm/test_perf_time.py, test_perf_memory.py,
test_bandwidth.py, test_alignment_impact.py).

All functions assume CUDA is available and that torch.cuda has been initialized.
"""
from __future__ import annotations

import torch


def _sync_and_time(fn, warmup: int = 10, repeats: int = 50) -> float:
    """Time a zero-arg callable on the GPU and return the median in milliseconds.

    Steps:
      1. Run *fn* `warmup` times to populate caches and trigger JIT compilation.
      2. Collect `repeats` timed iterations using CUDA events.
      3. Return the **median** elapsed time across the timed iterations.

    Using CUDA events avoids CPU-side measurement noise and accounts for
    asynchronous kernel launches correctly.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    times.sort()
    mid = len(times) // 2
    if len(times) % 2 == 0:
        return (times[mid - 1] + times[mid]) / 2.0
    return times[mid]


def _measure_peak_memory(fn) -> float:
    """Run a zero-arg callable and return peak GPU memory allocated in MB.

    Resets the peak-memory stats before running, so the returned value
    reflects only the allocations caused by *fn* (plus any baseline that
    was already allocated and not freed).
    """
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()

    fn()

    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    return (peak - baseline) / (1024 * 1024)
