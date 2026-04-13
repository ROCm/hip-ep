#!/usr/bin/env python3
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""
Run a Qwen model through the HIP MLIR compiler backend.

Uses torch.compile with a custom backend that:
  - Partitions the model into supported/unsupported subgraphs
  - Supported ops → HIP compiled path (MLIR → GPU DLL)
  - Unsupported ops → PyTorch fallback

Shows: text generation output + backend coverage statistics.

Usage:
    python scripts/run_qwen_hip_backend.py
    python scripts/run_qwen_hip_backend.py --model Qwen/Qwen3-0.6B
    python scripts/run_qwen_hip_backend.py --model Qwen/Qwen3-0.6B --device cpu
"""

import argparse
import logging
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))
from hip_backend import hip_backend, get_stats, reset_stats, enable_compilation


def parse_args():
    p = argparse.ArgumentParser(
        description="Run Qwen through HIP MLIR compiler backend"
    )
    p.add_argument("--model", default="Qwen/Qwen3-0.6B", help="HuggingFace model name")
    p.add_argument("--prompt", default="The capital of France is", help="Input prompt")
    p.add_argument(
        "--max-new-tokens", type=int, default=30, help="Max tokens to generate"
    )
    p.add_argument("--device", default="auto", help="Device: auto, cpu, cuda")
    p.add_argument(
        "--compile",
        action="store_true",
        help="Actually compile supported subgraphs to GPU DLLs",
    )
    p.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show per-subgraph compilation details",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO, format="[HIP] %(message)s")
        # Suppress noisy loggers
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

    print("=" * 70)
    print("Qwen Model → HIP MLIR Compiler Backend")
    print("=" * 70)

    # ── Load model ──────────────────────────────────────────────────────
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"\nModel:  {args.model}")
    print(f"Device: {device}", end="")
    if device == "cuda" and torch.cuda.is_available():
        print(f" ({torch.cuda.get_device_name(0)})")
    else:
        print()

    print("\nLoading model...")
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device if device != "cpu" else "cpu",
    )
    model.eval()
    load_time = time.perf_counter() - t0
    print(f"Loaded in {load_time:.1f}s")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Apply torch.compile with HIP backend ────────────────────────────
    print("\nApplying torch.compile with HIP backend...")
    reset_stats()
    if args.compile:
        enable_compilation(True)
        print("  DLL compilation: ENABLED")

    # Compile the forward method directly — generate() will call this
    model.forward = torch.compile(model.forward, backend=hip_backend)
    compiled_model = model

    # ── Generate text ──────────────────────────────────────────────────
    inputs = tokenizer(args.prompt, return_tensors="pt")
    if device != "cpu":
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    print(f'Prompt: "{args.prompt}"')
    print(f"Generating {args.max_new_tokens} tokens...\n")

    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = compiled_model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )
    gen_time = time.perf_counter() - t0

    # Decode output
    new_tokens = outputs[0][inputs["input_ids"].shape[1] :]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    num_tokens = len(new_tokens)

    # ── Show results ──────────────────────────────────────────────────
    print("=" * 70)
    print(f"Generated: {text}")
    print("=" * 70)

    print(f"\nTokens:    {num_tokens}")
    print(f"Time:      {gen_time:.2f}s")
    if num_tokens > 0:
        print(f"Tok/s:     {num_tokens / gen_time:.1f}")

    if device == "cuda" and torch.cuda.is_available():
        print(f"Peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    # ── Backend statistics ──────────────────────────────────────────────
    stats = get_stats()
    total_sub = stats["total_subgraphs"]
    compiled_sub = stats["compiled_subgraphs"]
    fallback_sub = stats["fallback_subgraphs"]
    total_ops = stats["total_ops"]
    supported_ops = stats["supported_ops"]

    print(f"\n{'=' * 70}")
    print("HIP Backend Coverage Report")
    print(f"{'=' * 70}")
    print(f"Subgraphs:  {total_sub} total")
    print(
        f"  HIP compiled: {compiled_sub} ({compiled_sub * 100 // max(total_sub, 1)}%)"
    )
    print(
        f"  Fallback:     {fallback_sub} ({fallback_sub * 100 // max(total_sub, 1)}%)"
    )
    print(f"\nOps:        {total_ops} total")
    print(
        f"  Supported:    {supported_ops} ({supported_ops * 100 // max(total_ops, 1)}%)"
    )
    print(
        f"  Unsupported:  {stats['unsupported_ops']} "
        f"({stats['unsupported_ops'] * 100 // max(total_ops, 1)}%)"
    )

    if stats["unsupported_op_names"]:
        print("\nUnsupported ops (need TorchToHip patterns):")
        for op in sorted(stats["unsupported_op_names"]):
            print(f"  - {op}")

    print()


if __name__ == "__main__":
    main()
