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
    """MLP that executes via a compiled HIP DLL.

    Falls back to PyTorch for shapes that don't match the compiled DLL.
    """
    def __init__(self, runner, gate_w, up_w, down_w, orig_mlp):
        super().__init__()
        self._gate_w = gate_w.cpu().contiguous()
        self._up_w = up_w.cpu().contiguous()
        self._down_w = down_w.cpu().contiguous()
        self.runner = runner
        # Keep original MLP for shape-mismatch fallback
        self._fallback = orig_mlp
        # Expected input shape from DLL metadata
        self._expected_shape = tuple(
            runner.input_metas[-1]["shape"])  # last input is activation

    def forward(self, x):
        # Check if input shape matches compiled DLL
        if tuple(x.shape) != self._expected_shape:
            # Shape mismatch (e.g. prefill with seq_len > 1) → PyTorch fallback
            return self._fallback(x)

        device = x.device
        out = self.runner(
            self._gate_w, self._up_w, self._down_w,
            x.detach().cpu().contiguous()
        )[0]
        return out.to(device)


def compile_mlp(mlp_module, hidden_size, device):
    """Compile a Qwen MLP module to a GPU DLL.

    Returns (DllMlp module, success boolean).
    """
    from fx_to_mlir import fx_graph_to_mlir
    from hip_dll_runner import HipDllRunner

    # Create proxy for export (extracts just the MLP compute)
    # Deep copy weights to CPU to avoid moving the model
    import copy
    proxy = MlpProxy(
        copy.deepcopy(mlp_module.gate_proj).cpu(),
        copy.deepcopy(mlp_module.up_proj).cpu(),
        copy.deepcopy(mlp_module.down_proj).cpu(),
        mlp_module.act_fn,
    ).eval().half()

    example = torch.randn(1, 1, hidden_size, dtype=torch.float16)

    # Export
    try:
        ep = torch.export.export(proxy, (example,))
    except Exception as e:
        print(f"    Export failed: {e}")
        return None, False

    # Generate MLIR
    try:
        mlir = fx_graph_to_mlir(ep, decompose=False)
    except Exception as e:
        print(f"    MLIR generation failed: {e}")
        return None, False

    # Compile
    work = os.path.join(os.path.dirname(__file__), "..", "test",
                        "e2e_flow", "qwen_mlp")
    os.makedirs(work, exist_ok=True)

    mlir_path = os.path.join(work, "mlp.mlir")
    dll_path = os.path.join(work, "mlp.dll")
    with open(mlir_path, "w") as f:
        f.write(mlir)

    therock = os.environ.get("THEROCK_DIST",
        "C:\\Users\\tsiddaga\\Documents\\code\\therock")
    compiler = os.path.normpath(os.path.join(
        os.path.dirname(__file__), "..", "..", "build",
        "onnx-hipdnn-ep", "bin", "hip-compiler.exe"))

    cmd = os.path.join(work, "_c.cmd")
    with open(cmd, "w") as f:
        f.write("@echo off\n")
        f.write('call "C:\\Program Files\\Microsoft Visual Studio\\18\\'
                'Community\\VC\\Auxiliary\\Build\\vcvarsall.bat" x64 >nul 2>&1\n')
        f.write(f'set THEROCK_DIST={therock}\nset PATH={therock}\\bin;%PATH%\n')
        f.write(f'"{compiler}" "{mlir_path}" -o "{dll_path}"\n')

    r = subprocess.run(["cmd", "/c", cmd], capture_output=True, text=True,
                       timeout=120)
    if r.returncode != 0 or not os.path.exists(dll_path):
        print(f"    Compilation failed: {r.stderr[-200:]}")
        return None, False

    # Load DLL
    os.environ["THEROCK_DIST"] = therock
    try:
        runner = HipDllRunner(dll_path, work_dir=work)
    except Exception as e:
        print(f"    DLL loading failed: {e}")
        return None, False

    # Prepare transposed weights for hip.matmul
    gate_w = proxy.gate_proj.weight.data.t().contiguous().cpu()
    up_w = proxy.up_proj.weight.data.t().contiguous().cpu()
    down_w = proxy.down_proj.weight.data.t().contiguous().cpu()

    return DllMlp(runner, gate_w, up_w, down_w, proxy), True


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

    # Compile using the first layer's MLP as template
    # Use seq_len=1 for decode-step execution. During prefill (first call),
    # the MLP will fall back to PyTorch since shape doesn't match.
    layer0_mlp = model.model.layers[0].mlp

    # Compute prompt length for prefill compilation
    prompt_len = len(tokenizer.encode(args.prompt))
    print(f"  Prompt length: {prompt_len} tokens")

    dll_mlp, success = compile_mlp(layer0_mlp, hidden_size, device)

    if success:
        print(f"  MLP compiled to GPU DLL successfully!")

        # Replace MLP in ALL layers with DLL-backed version
        num_layers = len(model.model.layers)
        replaced = 0
        for i, layer in enumerate(model.model.layers):
            orig_mlp = layer.mlp
            gate_w = orig_mlp.gate_proj.weight.data.t().contiguous().cpu()
            up_w = orig_mlp.up_proj.weight.data.t().contiguous().cpu()
            down_w = orig_mlp.down_proj.weight.data.t().contiguous().cpu()
            layer.mlp = DllMlp(dll_mlp.runner, gate_w, up_w, down_w, orig_mlp)
            replaced += 1

        print(f"  Replaced MLP in {replaced}/{num_layers} layers with DLL")
        print(f"  Offloaded ops: linear(gate) + silu + linear(up) + mul + linear(down)")
        print(f"  PyTorch ops:   embedding, RMSNorm, attention, residual add")
    else:
        print(f"  MLP compilation failed — running fully on PyTorch")

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
        print(f"\nExecution mode: HYBRID")
        print(f"  MLP (3 matmuls + SiLU + mul): HIP compiled GPU DLL")
        print(f"  Everything else:              PyTorch ({device})")
    else:
        print(f"\nExecution mode: PyTorch only ({device})")


if __name__ == "__main__":
    main()
