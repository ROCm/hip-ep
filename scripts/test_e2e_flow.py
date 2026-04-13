#!/usr/bin/env python3
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""
End-to-end validation: PyTorch model -> FX export -> MLIR -> hip-compiler -> GPU DLL

This script:
1. Defines a simple PyTorch model (RMSNorm + Linear + SiLU)
2. Exports via torch.export to FX graph
3. Emits Torch dialect MLIR via fx_to_mlir
4. Saves the MLIR to disk for hip-compiler
5. Reports the generated MLIR and instructions to compile/run
"""

import os
import sys
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from fx_to_mlir import fx_graph_to_mlir

# ── Step 1: Define a simple model ──────────────────────────────────────────


class SmallTransformerBlock(nn.Module):
    """Simplified transformer block using ops we support in TorchToHip."""

    def __init__(self, hidden=128):
        super().__init__()
        self.norm_weight = nn.Parameter(torch.ones(hidden))
        self.gate_proj = nn.Linear(hidden, hidden, bias=False)
        self.up_proj = nn.Linear(hidden, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x):
        # RMSNorm
        normed = torch.nn.functional.rms_norm(x, (x.shape[-1],), self.norm_weight, 1e-6)
        # SwiGLU-like MLP: silu(gate(x)) * up(x) -> down
        gate = torch.nn.functional.silu(self.gate_proj(normed))
        up = self.up_proj(normed)
        hidden = gate * up
        out = self.down_proj(hidden)
        # Residual
        return x + out


# ── Step 2: Export ─────────────────────────────────────────────────────────

print("Step 1: Creating model (hidden=128, f16)...")
model = SmallTransformerBlock(hidden=128).eval().half()

batch, seq, hidden = 1, 4, 128
x = torch.randn(batch, seq, hidden, dtype=torch.float16)

print("Step 2: Exporting via torch.export...")
ep = torch.export.export(model, (x,))

# Show ops
ops = set()
for node in ep.graph_module.graph.nodes:
    if node.op == "call_function":
        target = str(node.target)
        if "aten." in target:
            parts = target.split(".")
            idx = parts.index("aten")
            ops.add(parts[idx + 1])
print(f"  ATen ops used: {sorted(ops)}")

# ── Step 3: Emit MLIR ─────────────────────────────────────────────────────

print("Step 3: Emitting Torch dialect MLIR...")
mlir_text = fx_graph_to_mlir(ep)
print(f"  Generated {len(mlir_text)} chars, {mlir_text.count(chr(10))} lines")

# Save
out_dir = os.path.join(os.path.dirname(__file__), "..", "test", "e2e_flow")
os.makedirs(out_dir, exist_ok=True)
mlir_path = os.path.join(out_dir, "small_transformer.mlir")
with open(mlir_path, "w") as f:
    f.write(mlir_text)
print(f"  Saved to: {mlir_path}")

# ── Step 4: Print MLIR for inspection ──────────────────────────────────────

print("\n" + "=" * 70)
print("Generated MLIR:")
print("=" * 70)
print(mlir_text)
print("=" * 70)

# ── Step 5: Print compile/run instructions ─────────────────────────────────

dll_path = os.path.join(out_dir, "small_transformer.dll")
print(f"""
Next steps (run from project root in a VS Developer Command Prompt):

  1. Compile MLIR to GPU DLL:
     ..\\build\\onnx-hipdnn-ep\\bin\\hip-compiler.exe {mlir_path} -o {dll_path}

  2. Execute on GPU:
     set PATH=C:\\Users\\tsiddaga\\Documents\\code\\therock\\bin;%PATH%
     ..\\build\\onnx-hipdnn-ep\\bin\\hip-test-dll.exe {dll_path} --verbose --validate

  3. Or run the automated test (requires _build.cmd):
     See test_e2e_compile_run.cmd
""")
