#!/usr/bin/env python3
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""
Run any HuggingFace model with hybrid GPU DLL execution.

Usage:
    python -m hip_torch.run_model --model Qwen/Qwen3-0.6B
    python -m hip_torch.run_model --model Qwen/Qwen3.5-35B-A3B --device cpu
    python -m hip_torch.run_model --model gpt2
"""

import argparse
import logging
import time

import torch


def parse_args():
    p = argparse.ArgumentParser(description="Run HF model with HIP GPU offload")
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--prompt", default="The capital of France is")
    p.add_argument("--max-new-tokens", type=int, default=30)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--max-offload",
        action="store_true",
        help="Offload everything supported (MLP + attn projections)",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.INFO, format="[HIP] %(message)s")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .model_adapter import ModelAdapter

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    # Load model
    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device if device != "cpu" else "cpu",
    )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompt_len = len(tokenizer.encode(args.prompt))

    # Adapt model
    print(f"Adapting model (prompt_len={prompt_len})...")
    adapter = ModelAdapter(model, max_offload=args.max_offload)
    report = adapter.compile_and_replace(prompt_len=prompt_len)
    print(report.summary())

    if report.replaced_count == 0:
        print("No submodules offloaded — running fully on PyTorch")

    # Generate
    inputs = tokenizer(args.prompt, return_tensors="pt")
    if device != "cpu":
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    print(f"\nGenerating {args.max_new_tokens} tokens...")
    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=args.max_new_tokens, do_sample=False
        )
    gen_time = time.perf_counter() - t0

    tokens = outputs[0][inputs["input_ids"].shape[1] :]
    text = tokenizer.decode(tokens, skip_special_tokens=True)

    print(f"\nOutput: {text}")
    print(
        f"Tokens: {len(tokens)} | Time: {gen_time:.2f}s | Tok/s: {len(tokens) / gen_time:.1f}"
    )

    stats = adapter.get_execution_stats()
    print(
        f"DLL calls: {stats['dll_calls']} | Fallback: {stats['fallback_calls']} "
        f"| Offload: {stats['dll_ratio']:.0%}"
    )


if __name__ == "__main__":
    main()
