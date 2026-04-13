#!/usr/bin/env python3
"""Test ctypes DLL runner: PyTorch → MLIR → DLL → ctypes GPU execution."""
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

# 1. Export & compile
print("1. Export → MLIR → hip-compiler → DLL")
ep = torch.export.export(model, (x,))
mlir = fx_graph_to_mlir(ep, decompose=False)

work = os.path.join(os.path.dirname(__file__), "..", "test", "e2e_flow", "dll_test")
os.makedirs(work, exist_ok=True)
mlir_path = os.path.join(work, "model.mlir")
dll_path = os.path.join(work, "model.dll")
with open(mlir_path, "w") as f:
    f.write(mlir)

therock = "C:\\Users\\tsiddaga\\Documents\\code\\therock"
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
r = subprocess.run(["cmd", "/c", cmd], capture_output=True, text=True, timeout=60)
if r.returncode != 0:
    print(f"   COMPILE FAILED: {r.stderr[-200:]}"); sys.exit(1)
print(f"   DLL compiled: {dll_path}")

# 2. Load via ctypes and run
print("2. Loading DLL via ctypes...")
os.environ["THEROCK_DIST"] = therock
runner = HipDllRunner(dll_path, work_dir=work)
print(f"   {runner}")

# Prepare inputs: weights (transposed for linear) then activation
norm_w = model.norm_weight.data
gate_w = model.gate_proj.weight.data.t().contiguous()
up_w = model.up_proj.weight.data.t().contiguous()
down_w = model.down_proj.weight.data.t().contiguous()

print("3. Running inference via ctypes DLL call...")
dll_out = runner(norm_w, gate_w, up_w, down_w, x)
dll_out = dll_out[0]
print(f"   DLL output[:8]: {dll_out.flatten()[:8].tolist()}")

# 3. Compare with PyTorch
print("4. PyTorch reference...")
with torch.no_grad():
    ref = model(x)
print(f"   Ref output[:8]: {ref.cpu().flatten()[:8].tolist()}")

diff = (dll_out.float() - ref.cpu().float()).abs()
max_abs = diff.max().item()
max_rel = (diff / (ref.cpu().float().abs() + 1e-8)).max().item()
print(f"\n   Max abs error:  {max_abs:.6f}")
print(f"   Max rel error:  {max_rel:.4%}")
print(f"   ACCURACY: {'PASS' if max_rel < 0.01 else 'FAIL'}")

runner.close()
print("\n   FULL PIPELINE: PyTorch → MLIR → DLL → ctypes → GPU → PASS")
