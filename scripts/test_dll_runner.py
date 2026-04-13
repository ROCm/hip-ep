#!/usr/bin/env python3
"""Test the DLL runner with our proven simple model."""
import os, subprocess, sys, torch
sys.path.insert(0, os.path.dirname(__file__))
from fx_to_mlir import fx_graph_to_mlir
from hip_dll_runner import HipDllRunner

class SmallBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.norm_weight = torch.nn.Parameter(torch.ones(32))
        self.gate_proj = torch.nn.Linear(32, 32, bias=False)
        self.up_proj = torch.nn.Linear(32, 32, bias=False)
        self.down_proj = torch.nn.Linear(32, 32, bias=False)
    def forward(self, x):
        normed = torch.nn.functional.rms_norm(x, (32,), self.norm_weight, 1e-6)
        gate = torch.nn.functional.silu(self.gate_proj(normed))
        up = self.up_proj(normed)
        return x + self.down_proj(gate * up)

torch.manual_seed(42)
model = SmallBlock().eval().half()
x = torch.randn(1, 4, 32, dtype=torch.float16)

# 1. Export
print("1. Exporting to MLIR...")
ep = torch.export.export(model, (x,))
mlir = fx_graph_to_mlir(ep, decompose=False)

work = os.path.join(os.path.dirname(__file__), "..", "test", "e2e_flow", "dll_test")
os.makedirs(work, exist_ok=True)
mlir_path = os.path.join(work, "model.mlir")
dll_path = os.path.join(work, "model.dll")
with open(mlir_path, "w") as f:
    f.write(mlir)

# 2. Compile
print("2. Compiling to DLL...")
therock = "C:\\Users\\tsiddaga\\Documents\\code\\therock"
compiler = os.path.normpath(os.path.join(
    os.path.dirname(__file__), "..", "..", "build",
    "onnx-hipdnn-ep", "bin", "hip-compiler.exe"))
cmd = os.path.join(work, "_compile.cmd")
with open(cmd, "w") as f:
    f.write("@echo off\n")
    f.write('call "C:\\Program Files\\Microsoft Visual Studio\\18\\'
            'Community\\VC\\Auxiliary\\Build\\vcvarsall.bat" x64 >nul 2>&1\n')
    f.write(f'set THEROCK_DIST={therock}\nset PATH={therock}\\bin;%PATH%\n')
    f.write(f'"{compiler}" "{mlir_path}" -o "{dll_path}"\n')
r = subprocess.run(["cmd", "/c", cmd], capture_output=True, text=True, timeout=60)
if r.returncode != 0:
    print(f"   FAILED: {r.stderr[-200:]}"); sys.exit(1)
print(f"   OK: {dll_path}")

# 3. Run via hip-test-dll
print("3. Running on GPU via hip-test-dll...")
os.environ["THEROCK_DIST"] = therock
runner = HipDllRunner(dll_path, work_dir=work)
result = runner.run()
print(f"   Success: {result['success']}")
print(f"   GPU output: {result['output_values']}")

if result['success']:
    print("\n   FULL PIPELINE: PyTorch -> MLIR -> hip-compiler -> GPU DLL -> PASS")
else:
    print("\n   FULL PIPELINE: FAILED")
