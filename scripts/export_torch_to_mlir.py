#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""
Export a HuggingFace model to a list of ATen ops via torch.compile.

This script captures the FX graph produced by torch.compile and reports
the unique ATen ops, their shapes, and the graph structure. This output
drives the TorchToHip conversion pattern work.

Usage:
    python scripts/export_torch_to_mlir.py --model Qwen/Qwen3-0.6B
    python scripts/export_torch_to_mlir.py --model Qwen/Qwen3-0.6B --max-layers 1
    python scripts/export_torch_to_mlir.py --model Qwen/Qwen3.5-35B-A3B --max-layers 1 --device cpu
"""

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

import torch


def parse_args():
    p = argparse.ArgumentParser(description="Export HF model ops via torch.compile")
    p.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B",
                    help="HuggingFace model name or local path")
    p.add_argument("--max-layers", type=int, default=0,
                    help="Limit number of decoder layers (0 = all)")
    p.add_argument("--seq-len", type=int, default=4,
                    help="Input sequence length for tracing")
    p.add_argument("--batch-size", type=int, default=1,
                    help="Batch size for tracing")
    p.add_argument("--dtype", type=str, default="float16",
                    choices=["float16", "bfloat16", "float32"],
                    help="Model dtype")
    p.add_argument("--device", type=str, default="cpu",
                    help="Device for tracing (cpu recommended)")
    p.add_argument("-o", "--output", type=str, default=None,
                    help="Output file for op report (default: stdout)")
    p.add_argument("--dump-graphs", action="store_true",
                    help="Dump full FX graph IR to output dir")
    return p.parse_args()


def get_torch_dtype(s):
    return {"float16": torch.float16, "bfloat16": torch.bfloat16,
            "float32": torch.float32}[s]


def format_op(target):
    """Extract a clean op name from a torch.compile target."""
    s = str(target)
    # Built-in functions: <built-in function add> -> add
    if s.startswith("<built-in function "):
        return "builtin." + s[len("<built-in function "):-1]
    # Built-in methods: <built-in method arange of type ...> -> torch.arange
    if s.startswith("<built-in method "):
        name = s[len("<built-in method "):].split(" of ")[0]
        return "torch." + name
    # Regular functions: <function silu at 0x...> -> silu
    if s.startswith("<function "):
        name = s[len("<function "):].split(" at ")[0]
        return "F." + name
    # aten ops: aten.linear.default -> aten.linear
    if hasattr(target, "__name__"):
        return target.__name__
    return s


def main():
    args = parse_args()
    dtype = get_torch_dtype(args.dtype)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {args.model}...", file=sys.stderr)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map=args.device,
    )
    model.eval()

    # Trim layers
    if args.max_layers > 0 and hasattr(model, "model") and hasattr(model.model, "layers"):
        original = len(model.model.layers)
        model.model.layers = model.model.layers[:args.max_layers]
        print(f"Trimmed {original} layers to {args.max_layers}", file=sys.stderr)

    # Prepare inputs
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    text = "Hello world" * max(1, args.seq_len // 2)
    inputs = tokenizer(text, return_tensors="pt", max_length=args.seq_len,
                        truncation=True, padding="max_length")
    input_ids = inputs["input_ids"].to(args.device)
    attention_mask = inputs["attention_mask"].to(args.device)
    print(f"input_ids: {input_ids.shape}, dtype: {input_ids.dtype}", file=sys.stderr)

    # Capture ops via torch.compile custom backend
    all_graphs = []
    op_counts = Counter()
    op_details = []  # (op_name, args_info)

    def capture_backend(gm, example_inputs):
        all_graphs.append(gm)
        for node in gm.graph.nodes:
            if node.op == "call_function":
                name = format_op(node.target)
                op_counts[name] += 1
                # Capture shape info from meta tensors if available
                shapes = []
                for arg in node.args:
                    if hasattr(arg, "meta") and "val" in arg.meta:
                        val = arg.meta["val"]
                        if isinstance(val, torch.Tensor):
                            shapes.append(f"{list(val.shape)}x{val.dtype}")
                op_details.append((name, shapes))
        return gm

    print("Compiling with custom backend...", file=sys.stderr)
    compiled = torch.compile(model, backend=capture_backend)

    with torch.no_grad():
        compiled(input_ids, attention_mask=attention_mask)

    print(f"Captured {len(all_graphs)} graph(s)", file=sys.stderr)

    # Format output
    out = sys.stdout
    if args.output:
        out = open(args.output, "w")

    print(f"# Op Report: {args.model}", file=out)
    print(f"# Layers: {args.max_layers or 'all'}, seq_len: {args.seq_len}, "
          f"dtype: {args.dtype}", file=out)
    print(f"# Graphs captured: {len(all_graphs)}", file=out)
    print(f"# Total op calls: {sum(op_counts.values())}", file=out)
    print(f"# Unique ops: {len(op_counts)}", file=out)
    print(file=out)

    print("## Op Frequency", file=out)
    for op, count in op_counts.most_common():
        print(f"  {count:4d}x  {op}", file=out)

    print(file=out)
    print("## Ops Sorted Alphabetically", file=out)
    for op in sorted(op_counts.keys()):
        print(f"  {op}", file=out)

    # Dump full graphs if requested
    if args.dump_graphs:
        out_dir = Path(args.output).parent if args.output else Path(".")
        for i, gm in enumerate(all_graphs):
            graph_file = out_dir / f"graph_{i}.txt"
            with open(graph_file, "w") as f:
                f.write(str(gm.graph))
            print(f"Wrote {graph_file}", file=sys.stderr)

    if args.output:
        out.close()
        print(f"Report written to {args.output}", file=sys.stderr)

    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
