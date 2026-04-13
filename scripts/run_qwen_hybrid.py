#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""
Run Qwen with hybrid execution: MLP blocks on HIP GPU DLLs, rest on PyTorch.

This script:
1. Loads a Qwen model
2. Extracts the MLP block, compiles it to a GPU DLL via hip-compiler
3. Replaces each layer's MLP with a DLL-backed wrapper
4. Runs text generation — MLP on our compiled DLL, everything else on PyTorch
"""
import argparse
import os
import subprocess
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))


def parse_args():
    p = argparse.ArgumentParser(description="Run Qwen with HIP MLP offload")
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--prompt", default="The capital of France is")
    p.add_argument("--max-new-tokens", type=int, default=30)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


class MlpProxy(nn.Module):
    """Standalone MLP that matches Qwen's structure for export."""
    def __init__(self, gate_proj, up_proj, down_proj, act_fn):
        super().__init__()
        self.gate_proj = gate_proj
        self.up_proj = up_proj
        self.down_proj = down_proj
        self.act_fn = act_fn

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class DllMlp(nn.Module):
    """MLP with two compiled DLLs: one for prefill, one for decode.

    Dispatches to the appropriate DLL based on input sequence length.
    Falls back to PyTorch only if neither DLL matches.
    """
    def __init__(self, decode_runner, prefill_runner, gate_w, up_w, down_w,
                 orig_mlp):
        super().__init__()
        self._gate_w = gate_w.cpu().contiguous()
        self._up_w = up_w.cpu().contiguous()
        self._down_w = down_w.cpu().contiguous()
        self._decode_runner = decode_runner
        self._prefill_runner = prefill_runner
        self._fallback = orig_mlp

        self._decode_shape = tuple(
            decode_runner.input_metas[-1]["shape"]) if decode_runner else None
        self._prefill_shape = tuple(
            prefill_runner.input_metas[-1]["shape"]) if prefill_runner else None

    # Class-level counters
    _decode_calls = 0
    _prefill_calls = 0
    _fallback_calls = 0

    def forward(self, x):
        shape = tuple(x.shape)

        if shape == self._decode_shape and self._decode_runner:
            DllMlp._decode_calls += 1
            runner = self._decode_runner
        elif shape == self._prefill_shape and self._prefill_runner:
            DllMlp._prefill_calls += 1
            runner = self._prefill_runner
        else:
            DllMlp._fallback_calls += 1
            return self._fallback(x)

        device = x.device
        dtype = x.dtype
        # Convert bf16→f16 for numpy compatibility; f32→f16 for DLL
        x_f16 = x.detach().cpu().to(torch.float16).contiguous()
        out = runner(
            self._gate_w, self._up_w, self._down_w, x_f16
        )[0]
        return out.to(dtype).to(device)


def compile_mlp_for_shape(mlp_module, hidden_size, seq_len, label,
                          use_2d=False):
    """Compile a Qwen MLP for a specific input shape.

    Args:
        use_2d: If True, use [seq_len, hidden] shape (for MoE shared expert).
                If False, use [1, seq_len, hidden] shape (for dense MLP).

    Returns HipDllRunner or None.
    """
    from fx_to_mlir import fx_graph_to_mlir
    from hip_dll_runner import HipDllRunner
    import copy

    proxy = MlpProxy(
        copy.deepcopy(mlp_module.gate_proj).cpu(),
        copy.deepcopy(mlp_module.up_proj).cpu(),
        copy.deepcopy(mlp_module.down_proj).cpu(),
        mlp_module.act_fn,
    ).eval().half()

    if use_2d:
        example = torch.randn(seq_len, hidden_size, dtype=torch.float16)
    else:
        example = torch.randn(1, seq_len, hidden_size, dtype=torch.float16)

    print(f"\n  [{label}] Compiling for shape [1, {seq_len}, {hidden_size}]")

    # Export
    print(f"    Step 1/4: torch.export")
    try:
        ep = torch.export.export(proxy, (example,))
        ops = set()
        for n in ep.graph_module.graph.nodes:
            if n.op == "call_function" and "aten." in str(n.target):
                ops.add(str(n.target).split(".")[-2])
        print(f"             ATen ops: {sorted(ops)}")
    except Exception as e:
        print(f"    Export failed: {e}")
        return None

    # MLIR
    print(f"    Step 2/4: FX → MLIR")
    try:
        mlir = fx_graph_to_mlir(ep, decompose=False)
        print(f"             {len(mlir)} chars, {mlir.count(chr(10))} lines")
    except Exception as e:
        print(f"    MLIR failed: {e}")
        return None

    # Compile
    therock = os.environ.get("THEROCK_DIST",
        "C:\\Users\\tsiddaga\\Documents\\code\\therock")
    compiler = os.path.normpath(os.path.join(
        os.path.dirname(__file__), "..", "..", "build",
        "onnx-hipdnn-ep", "bin", "hip-compiler.exe"))

    work = os.path.join(os.path.dirname(__file__), "..", "test",
                        "e2e_flow", f"qwen_mlp_{label}")
    os.makedirs(work, exist_ok=True)
    mlir_path = os.path.join(work, "mlp.mlir")
    dll_path = os.path.join(work, "mlp.dll")
    with open(mlir_path, "w") as f:
        f.write(mlir)

    cmd = os.path.join(work, "_c.cmd")
    with open(cmd, "w") as f:
        f.write("@echo off\n")
        f.write('call "C:\\Program Files\\Microsoft Visual Studio\\18\\'
                'Community\\VC\\Auxiliary\\Build\\vcvarsall.bat" x64 >nul 2>&1\n')
        f.write(f'set THEROCK_DIST={therock}\nset PATH={therock}\\bin;%PATH%\n')
        f.write(f'"{compiler}" "{mlir_path}" -o "{dll_path}"\n')

    print(f"    Step 3/4: hip-compiler → DLL")
    import time as _time
    t0 = _time.perf_counter()
    r = subprocess.run(["cmd", "/c", cmd], capture_output=True, text=True,
                       timeout=120)
    elapsed = _time.perf_counter() - t0
    if r.returncode != 0 or not os.path.exists(dll_path):
        print(f"    Compilation failed: {r.stderr[-200:]}")
        return None
    print(f"             {elapsed:.1f}s, {os.path.getsize(dll_path)/1024:.0f} KB")

    # Load
    print(f"    Step 4/4: ctypes load")
    os.environ["THEROCK_DIST"] = therock
    try:
        runner = HipDllRunner(dll_path, work_dir=work)
        print(f"             Inputs:  "
              f"{[m.get('shape') for m in runner.input_metas]}")
        print(f"             Outputs: "
              f"{[m.get('shape') for m in runner.output_metas]}")
        return runner
    except Exception as e:
        print(f"    Load failed: {e}")
        return None


def main():
    args = parse_args()

    print("=" * 70)
    print("Qwen Hybrid Execution: MLP on HIP GPU DLL + PyTorch Fallback")
    print("=" * 70)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    print(f"\nModel:  {args.model}")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device if device != "cpu" else "cpu",
    )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tc = model.config.get_text_config() if hasattr(model.config, 'get_text_config') else model.config
    hidden_size = tc.hidden_size
    num_layers = len(model.model.layers)
    prompt_len = len(tokenizer.encode(args.prompt))

    # Detect MoE vs dense architecture
    is_moe = hasattr(model.model.layers[0].mlp, 'shared_expert')
    if is_moe:
        print(f"Architecture: MoE (experts={tc.num_experts}, "
              f"shared_expert_inter={tc.shared_expert_intermediate_size})")
    else:
        print(f"Architecture: Dense")

    # ── Phase 1: PyTorch Eager Baseline ──────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"Phase 1: PyTorch Eager Baseline")
    print(f"{'=' * 70}")
    print(f"Prompt ({prompt_len} tokens): \"{args.prompt}\"")

    inputs = tokenizer(args.prompt, return_tensors="pt")
    if device != "cpu":
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Warmup
    with torch.no_grad():
        _ = model(inputs["input_ids"][:, :1])
    if device == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    with torch.no_grad():
        eager_out = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                    do_sample=False)
    if device == "cuda":
        torch.cuda.synchronize()
    t_eager = time.perf_counter() - t0

    eager_tokens = eager_out[0][inputs["input_ids"].shape[1]:]
    eager_text = tokenizer.decode(eager_tokens, skip_special_tokens=True)
    eager_tps = len(eager_tokens) / t_eager if t_eager > 0 else 0

    print(f"Output:  {eager_text}")
    print(f"Tokens:  {len(eager_tokens)}")
    print(f"Time:    {t_eager:.2f}s")
    print(f"Tok/s:   {eager_tps:.1f}")

    # ── Phase 2: HIP DLL Compilation ─────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"Phase 2: HIP DLL Compilation")
    print(f"{'=' * 70}")
    print(f"Hidden size: {hidden_size}, Layers: {num_layers}")

    # For MoE models, compile the shared_expert; for dense, compile the MLP
    if is_moe:
        target_mlp = model.model.layers[0].mlp.shared_expert
        target_label = "shared_expert"
    else:
        target_mlp = model.model.layers[0].mlp
        target_label = "MLP"

    print(f"Compiling {target_label} (hidden={hidden_size})")
    t_compile_start = time.perf_counter()

    decode_runner = compile_mlp_for_shape(
        target_mlp, hidden_size, seq_len=1, label="decode", use_2d=is_moe)
    prefill_runner = compile_mlp_for_shape(
        target_mlp, hidden_size, seq_len=prompt_len, label="prefill",
        use_2d=is_moe)

    t_compile = time.perf_counter() - t_compile_start
    success = decode_runner is not None

    print(f"\n  Compile time:  {t_compile:.1f}s")
    print(f"  Decode DLL:    [1, 1, {hidden_size}] → "
          f"{'OK' if decode_runner else 'FAILED'}")
    print(f"  Prefill DLL:   [1, {prompt_len}, {hidden_size}] → "
          f"{'OK' if prefill_runner else 'FAILED'}")

    if not success:
        print(f"\n  Compilation failed — cannot run hybrid mode")
        return

    # Replace target MLP in all layers
    replaced = 0
    for layer in model.model.layers:
        if is_moe:
            orig = layer.mlp.shared_expert
            gw = orig.gate_proj.weight.data.t().contiguous().cpu()
            uw = orig.up_proj.weight.data.t().contiguous().cpu()
            dw = orig.down_proj.weight.data.t().contiguous().cpu()
            layer.mlp.shared_expert = DllMlp(
                decode_runner, prefill_runner, gw, uw, dw, orig)
        else:
            orig = layer.mlp
            gw = orig.gate_proj.weight.data.t().contiguous().cpu()
            uw = orig.up_proj.weight.data.t().contiguous().cpu()
            dw = orig.down_proj.weight.data.t().contiguous().cpu()
            layer.mlp = DllMlp(
                decode_runner, prefill_runner, gw, uw, dw, orig)
        replaced += 1
    print(f"  Replaced {target_label} in {replaced}/{num_layers} layers")

    # ── Phase 3: Hybrid Execution ────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"Phase 3: Hybrid Execution (MLP → GPU DLL, rest → PyTorch)")
    print(f"{'=' * 70}")

    inputs = tokenizer(args.prompt, return_tensors="pt")
    if device != "cpu":
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    DllMlp._decode_calls = 0
    DllMlp._prefill_calls = 0
    DllMlp._fallback_calls = 0

    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        hybrid_out = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                     do_sample=False)
    if device == "cuda":
        torch.cuda.synchronize()
    t_hybrid = time.perf_counter() - t0

    hybrid_tokens = hybrid_out[0][inputs["input_ids"].shape[1]:]
    hybrid_text = tokenizer.decode(hybrid_tokens, skip_special_tokens=True)
    hybrid_tps = len(hybrid_tokens) / t_hybrid if t_hybrid > 0 else 0

    print(f"Output:  {hybrid_text}")
    print(f"Tokens:  {len(hybrid_tokens)}")
    print(f"Time:    {t_hybrid:.2f}s")
    print(f"Tok/s:   {hybrid_tps:.1f}")

    # ── Phase 4: Comparison ──────────────────────────────────────────
    dec = DllMlp._decode_calls
    pre = DllMlp._prefill_calls
    fb = DllMlp._fallback_calls
    total_calls = dec + pre + fb
    dll_calls = dec + pre
    match = eager_text.strip() == hybrid_text.strip()
    sample = (model.model.layers[0].mlp.shared_expert if is_moe
              else model.model.layers[0].mlp)

    print(f"\n{'=' * 70}")
    print(f"Results Comparison")
    print(f"{'=' * 70}")
    print(f"{'':30s} {'PyTorch Eager':>15s} {'Hybrid (DLL)':>15s}")
    print(f"{'-'*30} {'-'*15} {'-'*15}")
    print(f"{'Total time':30s} {t_eager:>14.2f}s {t_hybrid:>14.2f}s")
    print(f"{'Tokens generated':30s} {len(eager_tokens):>15d} {len(hybrid_tokens):>15d}")
    print(f"{'Throughput (tok/s)':30s} {eager_tps:>15.1f} {hybrid_tps:>15.1f}")
    print(f"{'Output match':30s} {'':>15s} {'YES' if match else 'NO':>15s}")
    print()
    print(f"DLL Compilation:")
    print(f"  Total compile time:    {t_compile:.1f}s")
    print(f"  Decode DLL shape:      [1, 1, {hidden_size}]")
    print(f"  Prefill DLL shape:     [1, {prompt_len}, {hidden_size}]")
    print(f"  Weight shapes:         gate [{sample._gate_w.shape[0]}x{sample._gate_w.shape[1]}]"
          f"  up [{sample._up_w.shape[0]}x{sample._up_w.shape[1]}]"
          f"  down [{sample._down_w.shape[0]}x{sample._down_w.shape[1]}]")
    print()
    print(f"MLP Offload Statistics:")
    print(f"  Prefill DLL calls:     {pre:5d}  (seq={prompt_len})")
    print(f"  Decode DLL calls:      {dec:5d}  (seq=1)")
    print(f"  PyTorch fallback:      {fb:5d}")
    print(f"  Total MLP calls:       {total_calls:5d}")
    print(f"  DLL offload rate:      {dll_calls*100//max(total_calls,1)}%")
    print()
    print(f"GPU Kernel Dispatch (per MLP DLL call):")
    print(f"  1. hip.matmul  gate_proj  [{hidden_size}]→[{sample._gate_w.shape[1]}]  hipBLASLt")
    print(f"  2. hip.silu    activation [{sample._gate_w.shape[1]}]              MIOpen")
    print(f"  3. hip.matmul  up_proj    [{hidden_size}]→[{sample._up_w.shape[1]}]  hipBLASLt")
    print(f"  4. hip.mul     gate*up    [{sample._gate_w.shape[1]}]              MIOpen")
    print(f"  5. hip.matmul  down_proj  [{sample._down_w.shape[1]}]→[{hidden_size}]  hipBLASLt")


if __name__ == "__main__":
    main()
