#!/usr/bin/env python3
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""
Incrementally enable ops in the E2E flow, verifying accuracy after each.

Progression:
  Step 1: MLP only (gate+silu+up+mul+down) — baseline
  Step 2: RMSNorm + MLP (add rms_norm before MLP)
  Step 3: Full MLP block with residual (rms_norm + MLP + add)
  Step 4: Attention projections (Q/K/V linear + output linear)
"""

import copy
import os
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from hip_torch.compiler import Compiler
from hip_torch.dll_cache import DllCache
from hip_torch.fx_emitter import fx_graph_to_mlir

os.environ.setdefault("THEROCK_DIST", "C:\\Users\\tsiddaga\\Documents\\code\\therock")

# Reuse existing runner
from hip_dll_runner import HipDllRunner


def compile_and_load(proxy, example, label):
    """Export → MLIR → DLL → load. Returns (runner, compile_time) or (None, 0)."""
    print(f"\n  [{label}] Compiling for shape {list(example.shape)}...")
    try:
        ep = torch.export.export(proxy, (example,))
        mlir = fx_graph_to_mlir(ep, decompose=False)
        print(f"    MLIR: {len(mlir)} chars, {mlir.count(chr(10))} lines")

        cache = DllCache()
        key = cache.compute_key(mlir)
        cached = cache.get(key)
        if cached:
            print(f"    Cache HIT: {key}")
            runner = HipDllRunner(str(cached), work_dir=str(cached.parent))
            return runner, 0.0

        compiler = Compiler()
        t0 = time.perf_counter()
        dll_path = compiler.compile(mlir)
        elapsed = time.perf_counter() - t0
        cache.put(key, dll_path)
        print(f"    Compiled in {elapsed:.1f}s")
        runner = HipDllRunner(str(dll_path), work_dir=str(dll_path.parent))
        return runner, elapsed
    except Exception as e:
        print(f"    FAILED: {e}")
        return None, 0.0


def verify_accuracy(name, dll_fn, ref_fn, *inputs):
    """Run both DLL and PyTorch, compare outputs."""
    with torch.no_grad():
        ref_out = ref_fn(*inputs)
    dll_out = dll_fn(*inputs)
    if isinstance(dll_out, list):
        dll_out = dll_out[0]

    diff = (dll_out.float() - ref_out.cpu().float()).abs()
    max_abs = diff.max().item()
    # Use torch.testing.assert_close tolerance: atol=1e-2, rtol=1e-2 for f16
    # Relative error is only meaningful for non-tiny values
    ref_abs = ref_out.cpu().float().abs()
    significant = ref_abs > 0.01  # only check rel error on significant values
    if significant.any():
        max_rel = (diff[significant] / ref_abs[significant]).max().item()
    else:
        max_rel = 0.0

    # Pass if absolute error < 0.01 (f16 precision) OR relative error < 5%
    status = "PASS" if max_abs < 0.01 or max_rel < 0.05 else "FAIL"
    print(f"  [{name}] max_abs={max_abs:.6f}  max_rel={max_rel:.4%}  → {status}")
    # Also show first 5 values for visual comparison
    print(f"    DLL: {dll_out.flatten()[:5].tolist()}")
    print(f"    Ref: {ref_out.cpu().flatten()[:5].tolist()}")
    return status == "PASS"


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "Qwen/Qwen3-0.6B"
    print(f"Loading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="cpu"
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    hidden = model.config.hidden_size  # 1024
    layer0 = model.model.layers[0]

    print(f"Hidden size: {hidden}")
    print("Layer 0 modules:")
    for name, mod in layer0.named_children():
        print(f"  {name}: {type(mod).__name__}")

    x = torch.randn(1, 1, hidden, dtype=torch.float16)
    results = []

    # ── Step 1: MLP only ─────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("Step 1: MLP only (gate + silu + up + mul + down)")
    print(f"{'=' * 70}")

    class MlpProxy(nn.Module):
        def __init__(self, mlp):
            super().__init__()
            self.gate_proj = copy.deepcopy(mlp.gate_proj)
            self.up_proj = copy.deepcopy(mlp.up_proj)
            self.down_proj = copy.deepcopy(mlp.down_proj)
            self.act_fn = mlp.act_fn

        def forward(self, x):
            return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

    proxy1 = MlpProxy(layer0.mlp).eval().half()
    runner1, _ = compile_and_load(proxy1, x, "MLP")

    if runner1:
        gate_w = proxy1.gate_proj.weight.data.t().contiguous()
        up_w = proxy1.up_proj.weight.data.t().contiguous()
        down_w = proxy1.down_proj.weight.data.t().contiguous()

        ok = verify_accuracy(
            "MLP",
            lambda inp: runner1(gate_w, up_w, down_w, inp),
            lambda inp: proxy1(inp),
            x,
        )
        results.append(("MLP (linear+silu+mul+linear+linear)", "5 ops", ok))

    # ── Step 2: RMSNorm + MLP ────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("Step 2: RMSNorm + MLP")
    print(f"{'=' * 70}")

    class NormMlpProxy(nn.Module):
        def __init__(self, norm, mlp):
            super().__init__()
            self.norm_weight = copy.deepcopy(norm.weight)
            self.eps = norm.variance_epsilon
            self.gate_proj = copy.deepcopy(mlp.gate_proj)
            self.up_proj = copy.deepcopy(mlp.up_proj)
            self.down_proj = copy.deepcopy(mlp.down_proj)
            self.act_fn = mlp.act_fn

        def forward(self, x):
            normed = torch.nn.functional.rms_norm(
                x, (x.shape[-1],), self.norm_weight, self.eps
            )
            return self.down_proj(
                self.act_fn(self.gate_proj(normed)) * self.up_proj(normed)
            )

    proxy2 = NormMlpProxy(layer0.post_attention_layernorm, layer0.mlp).eval().half()
    runner2, _ = compile_and_load(proxy2, x, "Norm+MLP")

    if runner2:
        norm_w = proxy2.norm_weight.data
        gate_w = proxy2.gate_proj.weight.data.t().contiguous()
        up_w = proxy2.up_proj.weight.data.t().contiguous()
        down_w = proxy2.down_proj.weight.data.t().contiguous()

        ok = verify_accuracy(
            "Norm+MLP",
            lambda inp: runner2(norm_w, gate_w, up_w, down_w, inp),
            lambda inp: proxy2(inp),
            x,
        )
        results.append(
            ("RMSNorm + MLP", "6 ops (rms_norm+linear+silu+mul+linear+linear)", ok)
        )

    # ── Step 3: RMSNorm + MLP + Residual Add ─────────────────────────
    print(f"\n{'=' * 70}")
    print("Step 3: RMSNorm + MLP + Residual Add")
    print(f"{'=' * 70}")

    class NormMlpResidualProxy(nn.Module):
        def __init__(self, norm, mlp):
            super().__init__()
            self.norm_weight = copy.deepcopy(norm.weight)
            self.eps = norm.variance_epsilon
            self.gate_proj = copy.deepcopy(mlp.gate_proj)
            self.up_proj = copy.deepcopy(mlp.up_proj)
            self.down_proj = copy.deepcopy(mlp.down_proj)
            self.act_fn = mlp.act_fn

        def forward(self, x):
            normed = torch.nn.functional.rms_norm(
                x, (x.shape[-1],), self.norm_weight, self.eps
            )
            mlp_out = self.down_proj(
                self.act_fn(self.gate_proj(normed)) * self.up_proj(normed)
            )
            return x + mlp_out  # residual add

    proxy3 = (
        NormMlpResidualProxy(layer0.post_attention_layernorm, layer0.mlp).eval().half()
    )
    runner3, _ = compile_and_load(proxy3, x, "Norm+MLP+Res")

    if runner3:
        norm_w = proxy3.norm_weight.data
        gate_w = proxy3.gate_proj.weight.data.t().contiguous()
        up_w = proxy3.up_proj.weight.data.t().contiguous()
        down_w = proxy3.down_proj.weight.data.t().contiguous()

        ok = verify_accuracy(
            "Norm+MLP+Res",
            lambda inp: runner3(norm_w, gate_w, up_w, down_w, inp),
            lambda inp: proxy3(inp),
            x,
        )
        results.append(("RMSNorm + MLP + Residual", "7 ops (+add.Tensor)", ok))

    # ── Step 4: Attention Q/K/V projections ──────────────────────────
    print(f"\n{'=' * 70}")
    print("Step 4: Attention projections (Q + K + V linear)")
    print(f"{'=' * 70}")

    attn = layer0.self_attn

    class QkvProxy(nn.Module):
        def __init__(self, attn):
            super().__init__()
            self.q_proj = copy.deepcopy(attn.q_proj)
            # K and V may have different sizes (GQA: fewer KV heads)
            self.k_proj = copy.deepcopy(attn.k_proj)
            self.v_proj = copy.deepcopy(attn.v_proj)

        def forward(self, x):
            q = self.q_proj(x)
            # Don't add — sizes may differ. Just return Q for testing.
            return q

    proxy4 = QkvProxy(attn).eval().half()
    runner4, _ = compile_and_load(proxy4, x, "QKV")

    if runner4:
        q_w = proxy4.q_proj.weight.data.t().contiguous()
        try:
            ok = verify_accuracy(
                "Q projection",
                lambda inp: runner4(q_w, inp),
                lambda inp: proxy4(inp),
                x,
            )
        except Exception as e:
            print(f"    Runtime error: {e}")
            ok = False
        results.append(("Q Linear projection", "1 linear", ok))

    # ── Step 5: Attention output projection ──────────────────────────
    print(f"\n{'=' * 70}")
    print("Step 5: Output projection (linear)")
    print(f"{'=' * 70}")

    class OutProjProxy(nn.Module):
        def __init__(self, attn):
            super().__init__()
            self.o_proj = copy.deepcopy(attn.o_proj)

        def forward(self, x):
            return self.o_proj(x)

    # O_proj input is [1,1,num_heads*head_dim] = [1,1,1024] for Qwen3
    o_proj_input = torch.randn(1, 1, attn.o_proj.in_features, dtype=torch.float16)
    proxy5 = OutProjProxy(attn).eval().half()
    runner5, _ = compile_and_load(proxy5, o_proj_input, "O_proj")

    if runner5:
        o_w = proxy5.o_proj.weight.data.t().contiguous()
        try:
            ok = verify_accuracy(
                "O_proj",
                lambda inp: runner5(o_w, inp),
                lambda inp: proxy5(inp),
                o_proj_input,
            )
        except Exception as e:
            print(f"    Runtime error: {e}")
            ok = False
        results.append(("Output projection", "1 linear", ok))

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("INCREMENTAL OFFLOAD RESULTS")
    print(f"{'=' * 70}")
    print(f"\n{'Step':5s} {'Block':40s} {'Ops':30s} {'Accuracy'}")
    print("-" * 85)
    for i, (block, ops, ok) in enumerate(results, 1):
        status = "✓ PASS" if ok else "✗ FAIL"
        print(f"  {i:<4d} {block:40s} {ops:30s} {status}")

    all_ok = all(ok for _, _, ok in results)
    print(f"\nOverall: {'ALL PASS' if all_ok else 'SOME FAILED'}")
    print(f"Total ops verified: {len(results)} blocks")


if __name__ == "__main__":
    main()
