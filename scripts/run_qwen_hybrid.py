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
        out = runner(
            self._gate_w, self._up_w, self._down_w,
            x.detach().cpu().contiguous()
        )[0]
        return out.to(device)


def compile_mlp_for_shape(mlp_module, hidden_size, seq_len, label):
    """Compile a Qwen MLP for a specific input shape.

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

    # ── Load model ──────────────────────────────────────────────────────
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

    # ── Compile MLP to DLL ──────────────────────────────────────────────
    hidden_size = model.config.hidden_size
    print(f"\nCompiling MLP block (hidden_size={hidden_size})...")

    # Compile two DLLs: one for decode (seq=1), one for prefill (seq=prompt_len)
    layer0_mlp = model.model.layers[0].mlp
    prompt_len = len(tokenizer.encode(args.prompt))
    print(f"  Prompt length: {prompt_len} tokens")

    # Compile decode DLL (seq_len=1, used for each generated token)
    decode_runner = compile_mlp_for_shape(
        layer0_mlp, hidden_size, seq_len=1, label="decode")

    # Compile prefill DLL (seq_len=prompt_len, used for initial forward pass)
    prefill_runner = compile_mlp_for_shape(
        layer0_mlp, hidden_size, seq_len=prompt_len, label="prefill")

    success = decode_runner is not None
    num_layers = len(model.model.layers)

    if success:
        print(f"\n  Compilation summary:")
        print(f"    Decode DLL:  [1, 1, {hidden_size}] → "
              f"{'OK' if decode_runner else 'FAILED'}")
        print(f"    Prefill DLL: [1, {prompt_len}, {hidden_size}] → "
              f"{'OK' if prefill_runner else 'FAILED'}")

        # Replace MLP in ALL layers
        replaced = 0
        for i, layer in enumerate(model.model.layers):
            orig_mlp = layer.mlp
            gate_w = orig_mlp.gate_proj.weight.data.t().contiguous().cpu()
            up_w = orig_mlp.up_proj.weight.data.t().contiguous().cpu()
            down_w = orig_mlp.down_proj.weight.data.t().contiguous().cpu()
            layer.mlp = DllMlp(decode_runner, prefill_runner,
                                gate_w, up_w, down_w, orig_mlp)
            replaced += 1

        print(f"\n  Replaced MLP in {replaced}/{num_layers} layers")
        print(f"  Offloaded: linear(gate) + silu + linear(up) + mul + linear(down)")
        print(f"  PyTorch:   embedding, RMSNorm, attention, residual add")
    else:
        print(f"\n  Compilation failed — running fully on PyTorch")

    # ── Generate text ──────────────────────────────────────────────────
    inputs = tokenizer(args.prompt, return_tensors="pt")
    if device != "cpu":
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    print(f"\nPrompt: \"{args.prompt}\"")
    print(f"Generating {args.max_new_tokens} tokens...\n")

    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )
    gen_time = time.perf_counter() - t0

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    print("=" * 70)
    print(f"Generated: {text}")
    print("=" * 70)
    print(f"\nTokens: {len(new_tokens)}")
    print(f"Time:   {gen_time:.2f}s")
    if len(new_tokens) > 0:
        print(f"Tok/s:  {len(new_tokens) / gen_time:.1f}")

    if success:
        dec = DllMlp._decode_calls
        pre = DllMlp._prefill_calls
        fb = DllMlp._fallback_calls
        total = dec + pre + fb
        dll_total = dec + pre
        print(f"\n{'=' * 70}")
        print(f"Execution Summary")
        print(f"{'=' * 70}")
        print(f"Mode:   HYBRID (MLP on HIP GPU DLLs + PyTorch fallback)")
        print(f"Model:  {args.model} ({num_layers} layers)")
        print(f"Device: {device}")

        # Get a DllMlp instance for shape info
        sample = model.model.layers[0].mlp

        print(f"\nCompiled DLLs:")
        print(f"  Decode:   input {sample._decode_shape} → "
              f"{decode_runner.dll_path if decode_runner else 'N/A'}")
        print(f"  Prefill:  input {sample._prefill_shape} → "
              f"{prefill_runner.dll_path if prefill_runner else 'N/A'}")
        print(f"  Weights:  gate [{sample._gate_w.shape[0]}x{sample._gate_w.shape[1]}], "
              f"up [{sample._up_w.shape[0]}x{sample._up_w.shape[1]}], "
              f"down [{sample._down_w.shape[0]}x{sample._down_w.shape[1]}]")

        print(f"\nMLP Forward Calls:")
        print(f"  Decode DLL:   {dec:5d} calls  (seq=1)")
        print(f"  Prefill DLL:  {pre:5d} calls  (seq={prompt_len})")
        print(f"  PyTorch:      {fb:5d} calls  (shape mismatch)")
        print(f"  Total:        {total:5d} calls")
        if total > 0:
            print(f"  DLL offload:  {dll_total*100//total}%")

        print(f"\nOps on HIP GPU DLL (per MLP call):")
        print(f"  hip.matmul  (gate_proj):  hipBLASLt GEMM")
        print(f"  hip.silu    (activation): MIOpen")
        print(f"  hip.matmul  (up_proj):    hipBLASLt GEMM")
        print(f"  hip.mul     (gate*up):    MIOpen element-wise")
        print(f"  hip.matmul  (down_proj):  hipBLASLt GEMM")
        print(f"\nOps on PyTorch ({device}):")
        print(f"  Embedding, RMSNorm, Self-Attention (SDPA),")
        print(f"  Q/K/V/O projections, Residual Add, LM Head")
    else:
        print(f"\nExecution mode: PyTorch only ({device})")


if __name__ == "__main__":
    main()
