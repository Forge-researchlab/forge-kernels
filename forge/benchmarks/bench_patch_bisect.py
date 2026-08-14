"""Per-kernel cost of forge.patch on a real model, one kernel at a time.

Answers the question `forge.patch(model)` alone cannot: which kernels help on
*this* model at *this* shape, and which ones cost you. A kernel that wins 6x at
hidden=4096 can lose at hidden=896, so the aggregate number hides the story.

Training and inference are separate regimes and are measured separately:
  --mode train  forward with labels + backward (what fine-tuning does)
  --mode infer  forward only under no_grad (what serving does)

Examples:
  # real model, training step
  PYTHONPATH=forge:. python forge/benchmarks/bench_patch_bisect.py \
      --model Qwen/Qwen2.5-0.5B --mode train --batch 4 --seq-len 1024

  # 7B-class per-layer shapes without downloading 15 GB: same widths, fewer
  # layers, random init. Kernel cost depends on per-op shape, not depth.
  PYTHONPATH=forge:. python forge/benchmarks/bench_patch_bisect.py \
      --model Qwen/Qwen2.5-0.5B --mode train --batch 4 --seq-len 1024 \
      --override hidden_size=4096,intermediate_size=11008,num_hidden_layers=8
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
import time
from pathlib import Path

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

# Kernels are bisected in this order. "rope" is a module-level patch and so does
# not appear in model._forge_patched_counts even when it is active.
KERNELS = ["embedding", "rmsnorm", "swiglu", "geglu", "rope", "fused_linear_ce",
           "lora_mlp", "lora_qkv"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--mode", choices=["train", "infer"], default="train")
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--attn-implementation", default="eager")
    p.add_argument("--lora", action="store_true",
                   help="wrap in PEFT LoRA so the lora_mlp/lora_qkv kernels are exercised")
    p.add_argument("--override", default="",
                   help="comma-separated config overrides, e.g. hidden_size=4096,"
                        "intermediate_size=11008,num_hidden_layers=8. Weights are "
                        "randomly initialised when this is used.")
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--warmup", type=int, default=4)
    p.add_argument("--csv", default="")
    p.add_argument("--guard", action="store_true",
                   help="enable forge's per-call shape guard. Off by default, so "
                        "that what is measured is the kernel itself rather than "
                        "the guard's decision to skip it.")
    p.add_argument("--guard-stats", action="store_true",
                   help="also count fused-vs-eager decisions, and report them. "
                        "Implies --guard. Kept separate because counting costs "
                        "about 2%% on small shapes, and the library default is "
                        "off, so the timings above should not include it.")
    args = p.parse_args()
    if args.guard_stats:
        args.guard = True
    return args


_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def build_model(args):
    from transformers import AutoConfig, AutoModelForCausalLM

    dtype = _DTYPES[args.dtype]
    config = AutoConfig.from_pretrained(args.model)

    if args.override:
        for pair in args.override.split(","):
            key, _, value = pair.partition("=")
            setattr(config, key.strip(), int(value))
        # Random init: we are measuring kernel shapes, not model quality.
        torch.manual_seed(0)
        model = AutoModelForCausalLM.from_config(
            config, torch_dtype=dtype, attn_implementation=args.attn_implementation
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=dtype,
            attn_implementation=args.attn_implementation,
        )

    model = model.to("cuda")
    if args.lora:
        from peft import LoraConfig, get_peft_model

        model = get_peft_model(model, LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
        ))
    model.train() if args.mode == "train" else model.eval()
    return model, config


def make_step(model, args, config):
    vocab = config.vocab_size
    ids = torch.randint(0, vocab, (args.batch, args.seq_len), device="cuda")

    if args.mode == "infer":
        @torch.no_grad()
        def step():
            model(input_ids=ids)
    else:
        def step():
            model.zero_grad(set_to_none=True)
            model(input_ids=ids, labels=ids).loss.backward()
    return step


def measure(step, iters, warmup):
    """Return (median ms, peak MB). Median because a stray outlier from an
    autotune retune should not decide a ratio."""
    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    samples = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        step()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(samples), torch.cuda.max_memory_allocated() / 2**20


def main():
    args = parse_args()
    import forge

    model, config = build_model(args)
    step = make_step(model, args, config)

    print(f"model            {args.model}"
          f"{' (random init, overridden)' if args.override else ''}")
    print(f"hidden           {config.hidden_size}")
    print(f"intermediate     {config.intermediate_size}")
    print(f"layers           {config.num_hidden_layers}")
    print(f"vocab            {config.vocab_size}")
    print(f"mode             {args.mode}   batch {args.batch}   seq {args.seq_len}"
          f"   {args.dtype}{'   +LoRA' if args.lora else ''}")
    guard_label = ("on (" + f"{forge.MIN_FUSED_ELEMENTS:,}" + " elements)"
                   if args.guard else "off")
    print(f"shape guard      {guard_label}"
          f"   activation elements/call = {args.batch * args.seq_len * config.hidden_size:,}")
    print(f"gpu              {torch.cuda.get_device_name(0)}\n")

    base_ms, base_mb = measure(step, args.iters, args.warmup)
    rows = [{"kernel": "(unpatched)", "ms": base_ms, "ratio": 1.0,
             "peak_mb": base_mb, "modules": ""}]
    print(f"{'kernel':<18} {'ms':>9} {'vs eager':>9} {'peak MB':>9}   modules patched")
    print(f"{'(unpatched)':<18} {base_ms:>9.2f} {'1.00x':>9} {base_mb:>9.0f}")

    patch_kwargs = {
        "mode": args.mode if args.mode == "train" else "infer",
        "min_elements": forge.MIN_FUSED_ELEMENTS if args.guard else 0,
        "collect_stats": args.guard_stats,
    }

    for kernel in KERNELS:
        try:
            forge.patch(model, kernels=[kernel], **patch_kwargs)
        except Exception as exc:
            print(f"{kernel:<18} {'-':>9} {'-':>9} {'-':>9}   "
                  f"not applied: {type(exc).__name__}")
            continue
        counts = dict(getattr(model, "_forge_patched_counts", {}))
        skipped = dict(getattr(model, "_forge_skipped", {}))
        ms, mb = measure(step, args.iters, args.warmup)
        forge.unpatch(model)
        if kernel in skipped:
            label = "(skipped: " + skipped[kernel].split("(")[0].strip() + ")"
        else:
            label = counts if counts else "(module-level)"
        print(f"{kernel:<18} {ms:>9.2f} {base_ms / ms:>8.2f}x {mb:>9.0f}   {label}")
        rows.append({"kernel": kernel, "ms": ms, "ratio": base_ms / ms,
                     "peak_mb": mb, "modules": str(label)})

    forge.reset_guard_stats()
    forge.patch(model, **patch_kwargs)
    counts = dict(getattr(model, "_forge_patched_counts", {}))
    ms, mb = measure(step, args.iters, args.warmup)
    print(f"{'ALL':<18} {ms:>9.2f} {base_ms / ms:>8.2f}x {mb:>9.0f}   {counts}")
    rows.append({"kernel": "ALL", "ms": ms, "ratio": base_ms / ms,
                 "peak_mb": mb, "modules": str(counts)})

    if args.guard_stats:
        stats = forge.guard_stats()
        if stats:
            print("\nguard dispatch (fused = kernel ran, eager = fell back):")
            for name in sorted(stats):
                counts_for = stats[name]
                print(f"  {name:<16} fused={counts_for.get('fused', 0):<8} "
                      f"eager={counts_for.get('eager', 0)}")
    if getattr(model, "_forge_skipped", None):
        print("\nskipped at patch time:")
        for name, reason in sorted(model._forge_skipped.items()):
            print(f"  {name:<16} {reason}")

    if args.csv:
        out = Path(args.csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=[
                "model", "mode", "batch", "seq_len", "dtype", "hidden",
                "intermediate", "layers", "vocab", "kernel", "ms", "ratio",
                "peak_mb", "modules"])
            writer.writeheader()
            for row in rows:
                writer.writerow({
                    "model": args.model, "mode": args.mode, "batch": args.batch,
                    "seq_len": args.seq_len, "dtype": args.dtype,
                    "hidden": config.hidden_size,
                    "intermediate": config.intermediate_size,
                    "layers": config.num_hidden_layers,
                    "vocab": config.vocab_size, **row})
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
