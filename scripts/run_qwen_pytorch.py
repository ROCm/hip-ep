#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""
Basic PyTorch validation script for Qwen models.
Confirms the model loads, runs inference, and produces coherent output.

Usage:
    python scripts/run_qwen_pytorch.py --model Qwen/Qwen3-0.6B
    python scripts/run_qwen_pytorch.py --model Qwen/Qwen3.5-35B-A3B
    python scripts/run_qwen_pytorch.py --model Qwen/Qwen3-0.6B --device cpu
"""

import argparse
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser(description="Run a Qwen model in PyTorch")
    p.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="HuggingFace model name or local path",
    )
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to run on (default: auto)",
    )
    p.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Model dtype (default: auto)",
    )
    p.add_argument(
        "--prompt",
        type=str,
        default="The capital of France is",
        help="Input prompt for generation",
    )
    p.add_argument(
        "--max-new-tokens", type=int, default=50, help="Maximum new tokens to generate"
    )
    p.add_argument(
        "--no-think",
        action="store_true",
        help="Disable thinking mode for Qwen3+ models",
    )
    return p.parse_args()


def get_torch_dtype(dtype_str):
    if dtype_str == "auto":
        return "auto"
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype_str]


def main():
    args = parse_args()

    # --- System info ---
    print(f"PyTorch {torch.__version__}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"VRAM: {props.total_memory / 1e9:.1f} GB")
    else:
        print("GPU: not available")
    print()

    # --- Resolve device ---
    if args.device == "auto":
        device_map = "auto" if torch.cuda.is_available() else "cpu"
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            print("ERROR: --device cuda but no GPU available", file=sys.stderr)
            sys.exit(1)
        device_map = "auto"
    else:
        device_map = "cpu"

    torch_dtype = get_torch_dtype(args.dtype)

    # --- Load tokenizer ---
    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Load model ---
    print(f"Loading model: {args.model} (dtype={args.dtype}, device_map={device_map})")
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )
    load_time = time.perf_counter() - t0
    print(f"Model loaded in {load_time:.1f}s")

    if torch.cuda.is_available():
        mem_gb = torch.cuda.max_memory_allocated() / 1e9
        print(f"GPU memory after load: {mem_gb:.2f} GB")
    print()

    # --- Prepare input ---
    prompt = args.prompt
    # For Qwen3+ models with thinking mode, optionally disable it
    if args.no_think and hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    else:
        prompt_text = prompt

    inputs = tokenizer(prompt_text, return_tensors="pt")
    if device_map != "cpu":
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    print(f"Prompt: {prompt}")
    print(f"Input tokens: {inputs['input_ids'].shape[1]}")
    print()

    # --- Generate ---
    print(f"Generating (max_new_tokens={args.max_new_tokens}, greedy)...")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )
    gen_time = time.perf_counter() - t0

    # --- Decode ---
    new_tokens = outputs[0][inputs["input_ids"].shape[1] :]
    generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    num_new = len(new_tokens)

    print(f"\n{'=' * 60}")
    print(f"Generated text: {generated_text}")
    print(f"{'=' * 60}\n")

    # --- Stats ---
    print(f"New tokens: {num_new}")
    print(f"Generation time: {gen_time:.2f}s")
    if num_new > 0:
        print(f"Tokens/sec: {num_new / gen_time:.1f}")
    if torch.cuda.is_available():
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        print(f"Peak GPU memory: {peak_gb:.2f} GB")

    print("\nSUCCESS: Model ran successfully")


if __name__ == "__main__":
    main()
