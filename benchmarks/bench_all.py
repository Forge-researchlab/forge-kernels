"""Run every per-kernel benchmark that exists in this repo.

Each entry below reproduces a result set that is already committed under
`results/` or `kernels/<kernel>/benchmarks/results/`. Kernels whose benchmark
harness exists but has no committed output yet (cross_entropy, embedding) are
included and will populate their empty results directories on first run.

    python benchmarks/bench_all.py --list
    python benchmarks/bench_all.py
    python benchmarks/bench_all.py --only swiglu rope
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Benchmark:
    name: str
    script: str
    args: list[str] = field(default_factory=list)
    writes: str = ""

    @property
    def path(self) -> Path:
        return REPO_ROOT / self.script


BENCHMARKS: tuple[Benchmark, ...] = (
    Benchmark(
        name="swiglu",
        script="kernels/swiglu/benchmarks/benchmark_swiglu.py",
        args=["--suite", "a100", "--dtype", "bf16", "--warmup", "20", "--rep", "50",
              "--save", "results/swiglu_a100_bf16.csv"],
        writes="results/swiglu_a100_bf16.csv",
    ),
    Benchmark(
        name="geglu-activation",
        script="kernels/geglu/benchmarks/benchmark_geglu.py",
        args=["--suite", "a100", "--dtype", "bf16", "--modes", "forward", "full",
              "--csv", "results/geglu_activation_a100_bf16.csv"],
        writes="results/geglu_activation_a100_bf16.csv",
    ),
    Benchmark(
        name="geglu-gateup",
        script="kernels/geglu/benchmarks/benchmark_geglu_gate_up_fusion.py",
        args=["--suite", "a100", "--dtype", "bf16", "--check",
              "--csv", "results/geglu_gateup_a100_bf16.csv"],
        writes="results/geglu_gateup_a100_bf16.csv",
    ),
    Benchmark(
        name="rope",
        script="kernels/rope/benchmarks/bench_v3.py",
        writes="kernels/rope/benchmarks/results/v3_{results.json,summary.md}",
    ),
    Benchmark(
        name="rmsnorm",
        script="kernels/rmsnorm/benchmarks/bench_v4.py",
        writes="kernels/rmsnorm/benchmarks/results/v4_{results.json,summary.md}",
    ),
    Benchmark(
        name="lora-mlp",
        script="kernels/lora_mlp/benchmarks/bench_lora_mlp.py",
        args=["--save", "kernels/lora_mlp/benchmarks/results/"],
        writes="kernels/lora_mlp/benchmarks/results/*.csv",
    ),
    Benchmark(
        name="lora-mlp-memory",
        script="kernels/lora_mlp/benchmarks/bench_memory.py",
        writes="kernels/lora_mlp/benchmarks/results/memory_*.csv",
    ),
    Benchmark(
        name="lora-qkv",
        script="kernels/lora_qkv/benchmarks/bench_lora_qkv.py",
        args=["--save", "kernels/lora_qkv/benchmarks/results/"],
        writes="kernels/lora_qkv/benchmarks/results/*.csv",
    ),
    Benchmark(
        name="cross-entropy",
        script="kernels/cross_entropy/benchmarks/bench_cross_entropy.py",
        args=["--bt", "1024", "2048", "4096", "8192", "--vocab", "128256", "--dtype", "fp32",
              "--save", "kernels/cross_entropy/benchmarks/results/"],
        writes="kernels/cross_entropy/benchmarks/results/*.csv",
    ),
    Benchmark(
        name="fused-linear-cross-entropy",
        script="kernels/cross_entropy/benchmarks/bench_fused_linear_cross_entropy.py",
        args=["--bt", "1024", "2048", "4096", "8192", "--hidden", "4096",
              "--vocab", "128256", "--dtype", "bf16",
              "--save", "kernels/cross_entropy/benchmarks/results/"],
        writes="kernels/cross_entropy/benchmarks/results/*.csv",
    ),
    Benchmark(
        name="embedding",
        script="kernels/embedding/benchmarks/bench_embedding.py",
        args=["--save", "kernels/embedding/benchmarks/results/"],
        writes="kernels/embedding/benchmarks/results/*.csv",
    ),
)


def describe_environment() -> str:
    try:
        import torch
    except ImportError:
        return "torch not importable — benchmarks will fail"
    if not torch.cuda.is_available():
        return f"torch {torch.__version__}, CUDA NOT available — benchmarks will abort"
    return f"torch {torch.__version__}, {torch.cuda.get_device_name(0)}"


def run(bench: Benchmark) -> tuple[int, float]:
    cmd = [sys.executable, str(bench.path), *bench.args]
    print(f"\n{'=' * 78}\n{bench.name}\n$ {' '.join(cmd)}\n{'=' * 78}", flush=True)
    start = time.perf_counter()
    completed = subprocess.run(cmd, cwd=REPO_ROOT, check=False)
    return completed.returncode, time.perf_counter() - start


def print_listing() -> None:
    width = max(len(b.name) for b in BENCHMARKS)
    for bench in BENCHMARKS:
        mark = " " if bench.path.exists() else "!"
        print(f"{mark} {bench.name:<{width}}  {bench.script}")
        print(f"{'':<{width + 4}}-> {bench.writes}")
    print("\n! = script missing from working tree")


def print_summary(outcomes: list[tuple[str, int, float]], total: int) -> None:
    width = max(len(name) for name, _, _ in outcomes)
    print(f"\n{'=' * 78}\nSummary\n{'=' * 78}")
    for name, code, elapsed in outcomes:
        status = "ok" if code == 0 else "FAIL"
        suffix = "" if code == 0 else f"  (exit {code})"
        print(f"{status:<5}{name:<{width}}  {elapsed:7.1f}s{suffix}")
    if total > len(outcomes):
        print(f"\n{total - len(outcomes)} benchmark(s) not run.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="+", metavar="NAME",
                        help="run only these benchmarks (see --list)")
    parser.add_argument("--list", action="store_true", help="list benchmarks and exit")
    parser.add_argument("--dry-run", action="store_true", help="print commands without running")
    parser.add_argument("--keep-going", action="store_true",
                        help="continue after a benchmark fails (default: stop)")
    return parser


def select(parser: argparse.ArgumentParser, only: list[str] | None) -> list[Benchmark]:
    if not only:
        return list(BENCHMARKS)
    unknown = sorted(set(only) - {b.name for b in BENCHMARKS})
    if unknown:
        parser.error(f"unknown benchmark(s): {', '.join(unknown)}. "
                     f"Use --list to see available names.")
    return [b for b in BENCHMARKS if b.name in set(only)]


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.list:
        print_listing()
        return 0

    selected = select(parser, args.only)

    missing = [b for b in selected if not b.path.exists()]
    if missing:
        for bench in missing:
            print(f"error: {bench.script} not found", file=sys.stderr)
        return 2

    if args.dry_run:
        for bench in selected:
            print(f"{bench.name}: {sys.executable} {bench.path} {' '.join(bench.args)}")
        return 0

    print(f"Environment: {describe_environment()}")
    print(f"Running {len(selected)} benchmark(s) from {REPO_ROOT}")

    outcomes: list[tuple[str, int, float]] = []
    for bench in selected:
        code, elapsed = run(bench)
        outcomes.append((bench.name, code, elapsed))
        if code != 0 and not args.keep_going:
            print(f"\n{bench.name} exited {code}; stopping. "
                  f"Use --keep-going to run the rest.", file=sys.stderr)
            break

    print_summary(outcomes, len(selected))
    return 1 if any(code != 0 for _, code, _ in outcomes) else 0


if __name__ == "__main__":
    sys.exit(main())
