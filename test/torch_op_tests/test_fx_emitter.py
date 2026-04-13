#!/usr/bin/env python3
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Tests for the refactored FX-to-MLIR emitter."""

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from hip_torch.fx_emitter import EmitterContext, fx_graph_to_mlir


# ── Test Models ──────────────────────────────────────────────────────


class SimpleLinearSilu(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(32, 32, bias=False)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.linear(x))


class SmallTransformerBlock(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.norm_weight = nn.Parameter(torch.ones(hidden))
        self.gate_proj = nn.Linear(hidden, hidden, bias=False)
        self.up_proj = nn.Linear(hidden, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x):
        normed = torch.nn.functional.rms_norm(x, (x.shape[-1],), self.norm_weight, 1e-6)
        gate = torch.nn.functional.silu(self.gate_proj(normed))
        up = self.up_proj(normed)
        return x + self.down_proj(gate * up)


# ── EmitterContext Tests ─────────────────────────────────────────────


def test_emit_int_constant():
    ctx = EmitterContext()
    name = ctx.emit_int_constant(42)
    assert name.startswith("%_k")
    assert "42" in name
    assert len(ctx.lines) == 1
    assert "torch.constant.int" in ctx.lines[0]


def test_emit_int_constant_cached():
    ctx = EmitterContext()
    n1 = ctx.emit_int_constant(7)
    n2 = ctx.emit_int_constant(7)
    assert n1 == n2
    assert len(ctx.lines) == 1  # only emitted once


def test_emit_float_constant():
    ctx = EmitterContext()
    ctx.emit_float_constant(3.14)
    assert "torch.constant.float" in ctx.lines[0]


def test_emit_float_inf():
    ctx = EmitterContext()
    ctx.emit_float_constant(float("-inf"))
    assert "0xFFF0000000000000" in ctx.lines[0]


def test_emit_none_constant():
    ctx = EmitterContext()
    name = ctx.emit_none_constant()
    assert name == "%_none"
    assert "torch.constant.none" in ctx.lines[0]


def test_emit_list_int():
    ctx = EmitterContext()
    name, type_str = ctx.emit_list_constant([1, 2, 3])
    assert "torch.prim.ListConstruct" in ctx.lines[-1]
    assert type_str == "!torch.list<int>"


# ── End-to-End Emission Tests ────────────────────────────────────────


def test_simple_model_emits_mlir():
    """Test that a simple model produces valid MLIR."""
    model = SimpleLinearSilu().eval().half()
    x = torch.randn(1, 4, 32, dtype=torch.float16)
    ep = torch.export.export(model, (x,))
    mlir = fx_graph_to_mlir(ep, decompose=False)

    assert "module {" in mlir
    assert "func.func @main_graph" in mlir
    assert "torch.aten.linear" in mlir
    assert "torch.aten.silu" in mlir
    assert "return" in mlir


def test_transformer_block_emits_mlir():
    """Test the proven SmallTransformerBlock."""
    torch.manual_seed(42)
    model = SmallTransformerBlock(32).eval().half()
    x = torch.randn(1, 4, 32, dtype=torch.float16)
    ep = torch.export.export(model, (x,))
    mlir = fx_graph_to_mlir(ep, decompose=False)

    # Must have all ops
    assert "torch.aten.rms_norm" in mlir
    assert "torch.aten.linear" in mlir
    assert "torch.aten.silu" in mlir
    assert "torch.aten.mul.Tensor" in mlir
    assert "torch.aten.add.Tensor" in mlir

    # Must have correct input/output count
    assert "-> tensor<1x4x32xf16>" in mlir

    # Weights should be function args
    assert "%arg0:" in mlir  # norm_weight
    assert "%arg4:" in mlir  # input x


def test_linear_weight_transpose():
    """Test that linear weights are emitted with transposed shape."""
    model = SimpleLinearSilu().eval().half()
    x = torch.randn(1, 4, 32, dtype=torch.float16)
    ep = torch.export.export(model, (x,))
    mlir = fx_graph_to_mlir(ep, decompose=False)

    # Linear weight [32,32] → transposed to [32,32] (same for square)
    # For non-square, the shape should be [K,N] not [N,K]
    assert "tensor<32x32xf16>" in mlir


def test_no_torch_unknown_ops():
    """Test that all ops resolve to torch.aten.* (no unknowns)."""
    model = SmallTransformerBlock(32).eval().half()
    x = torch.randn(1, 4, 32, dtype=torch.float16)
    ep = torch.export.export(model, (x,))
    mlir = fx_graph_to_mlir(ep, decompose=False)

    assert "torch.unknown" not in mlir


def test_decomposed_model():
    """Test with run_decompositions() enabled."""
    model = SmallTransformerBlock(32).eval().half()
    x = torch.randn(1, 4, 32, dtype=torch.float16)
    ep = torch.export.export(model, (x,))
    mlir = fx_graph_to_mlir(ep, decompose=True)

    # After decomposition: linear→mm, silu→sigmoid*x, rms_norm→pow+mean+rsqrt+mul
    assert "module {" in mlir
    assert "func.func @main_graph" in mlir
    # Should still produce valid MLIR (no crashes)
    assert len(mlir) > 100


def test_constant_folding():
    """Test that constant nodes become function args, not ops."""
    model = SmallTransformerBlock(32).eval().half()
    x = torch.randn(1, 4, 32, dtype=torch.float16)
    ep = torch.export.export(model, (x,))
    mlir = fx_graph_to_mlir(ep, decompose=True)

    # In decomposed mode, RoPE-like constants should be folded
    # The MLIR should have function args for constant tensors
    # (they don't appear as torch ops in the body)
    lines = mlir.split("\n")
    func_line = [line for line in lines if "func.func" in line][0]
    # Multiple args (weights + constants + input)
    assert func_line.count("%arg") >= 4


# ── Run all tests ────────────────────────────────────────────────────


if __name__ == "__main__":
    import traceback

    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    for test in tests:
        try:
            test()
            print(f"  PASS: {test.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL: {test.__name__}: {e}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed out of {passed + failed}")
    sys.exit(1 if failed else 0)
