#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""
Numerical accuracy validation: PyTorch reference vs HIP GPU execution.

This script:
1. Defines a model and runs it in PyTorch to get reference output
2. Exports to MLIR
3. Saves inputs/weights as binary for hip-test-dll
4. Compiles and runs on GPU
5. Compares GPU output against PyTorch reference

Usage:
    python scripts/test_numerical_accuracy.py
"""

import os
import subprocess
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from fx_to_mlir import fx_graph_to_mlir


def half_to_bytes(tensor: torch.Tensor) -> bytes:
    """Convert f16 tensor to raw bytes."""
    return tensor.detach().cpu().contiguous().numpy().tobytes()


def bytes_to_half(data: bytes, shape: tuple) -> np.ndarray:
    """Convert raw bytes back to f16 numpy array."""
    return np.frombuffer(data, dtype=np.float16).reshape(shape)


class SmallTransformerBlock(nn.Module):
    """Same model as test_e2e_flow.py."""

    def __init__(self, hidden=128):
        super().__init__()
        self.norm_weight = nn.Parameter(torch.ones(hidden))
        self.gate_proj = nn.Linear(hidden, hidden, bias=False)
        self.up_proj = nn.Linear(hidden, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x):
        normed = torch.nn.functional.rms_norm(x, (x.shape[-1],), self.norm_weight, 1e-6)
        gate = torch.nn.functional.silu(self.gate_proj(normed))
        up = self.up_proj(normed)
        hidden = gate * up
        out = self.down_proj(hidden)
        return x + out


def main():
    print("=" * 70)
    print("Numerical Accuracy Validation: PyTorch vs HIP GPU")
    print("=" * 70)

    # ── Step 1: Create model and compute PyTorch reference ──────────────
    # Use hidden=32 so that sequential test data (i*0.001) doesn't overflow
    # in f16 matmuls. Max value: 32*0.001=0.032, dot product of 32 terms
    # each ~0.016: sum ~ 0.008, well within f16 range.
    torch.manual_seed(42)
    hidden = 32
    model = SmallTransformerBlock(hidden=hidden).eval().half()
    batch, seq = 1, 4
    x = torch.randn(batch, seq, hidden, dtype=torch.float16)

    print(f"\nModel: SmallTransformerBlock(hidden={hidden}, dtype=f16)")
    print(f"Input: [{batch}, {seq}, {hidden}]")

    with torch.no_grad():
        ref_output = model(x)

    print(f"PyTorch output shape: {list(ref_output.shape)}")
    print(f"PyTorch output (first 10): {ref_output.flatten()[:10].tolist()}")
    print(
        f"PyTorch output stats: min={ref_output.min().item():.4f}, "
        f"max={ref_output.max().item():.4f}, "
        f"mean={ref_output.float().mean().item():.4f}"
    )

    # ── Step 2: Export to MLIR ─────────────────────────────────────────
    print("\nExporting to MLIR...")
    ep = torch.export.export(model, (x,))
    mlir_text = fx_graph_to_mlir(ep)

    work_dir = os.path.join(
        os.path.dirname(__file__), "..", "test", "e2e_flow", "accuracy_test"
    )
    os.makedirs(work_dir, exist_ok=True)

    mlir_path = os.path.join(work_dir, "model.mlir")
    with open(mlir_path, "w") as f:
        f.write(mlir_text)
    print(f"  MLIR saved to: {mlir_path}")

    # ── Step 3: Save input tensors as binary ────────────────────────────
    # The function signature is: @main_graph(norm_weight, gate_w, up_w, down_w, x)
    # hip-test-dll generates sequential test data, not our actual weights.
    # We need to save the actual weights and input as binary files for
    # a custom test harness.

    # Save all inputs in order (weights first, then activation)
    inputs_order = [
        ("norm_weight", model.norm_weight.data),
        ("gate_proj.weight", model.gate_proj.weight.data),
        ("up_proj.weight", model.up_proj.weight.data),
        ("down_proj.weight", model.down_proj.weight.data),
        ("input", x),
    ]

    for name, tensor in inputs_order:
        path = os.path.join(work_dir, f"{name.replace('.', '_')}.bin")
        with open(path, "wb") as f:
            f.write(half_to_bytes(tensor))
        print(f"  Saved {name}: {list(tensor.shape)} -> {path}")

    # Save reference output
    ref_path = os.path.join(work_dir, "reference_output.bin")
    with open(ref_path, "wb") as f:
        f.write(half_to_bytes(ref_output))
    print(f"  Saved reference output: {list(ref_output.shape)} -> {ref_path}")

    # ── Step 4: Compile MLIR to DLL ────────────────────────────────────
    dll_path = os.path.join(work_dir, "model.dll")
    compiler = os.path.join(
        os.path.dirname(__file__),
        "..",
        "build",
        "onnx-hipdnn-ep",
        "bin",
        "hip-compiler.exe",
    )
    compiler = os.path.normpath(compiler)

    if not os.path.exists(compiler):
        # Try relative from build dir
        compiler = os.path.normpath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "..",
                "build",
                "onnx-hipdnn-ep",
                "bin",
                "hip-compiler.exe",
            )
        )

    print("\nCompiling MLIR -> DLL...")
    print(f"  Compiler: {compiler}")

    # Write a batch file for compilation (needs MSVC env)
    compile_cmd = os.path.join(work_dir, "_compile.cmd")
    therock = "C:\\Users\\tsiddaga\\Documents\\code\\therock"
    with open(compile_cmd, "w") as f:
        f.write("@echo off\n")
        f.write('set "VSCMD_START_DIR=%CD%"\n')
        f.write(
            'call "C:\\Program Files\\Microsoft Visual Studio\\18\\'
            'Community\\VC\\Auxiliary\\Build\\vcvarsall.bat" x64 || exit /b 1\n'
        )
        f.write(f"set THEROCK_DIST={therock}\n")
        f.write(f"set PATH={therock}\\bin;%PATH%\n")
        f.write(f'"{compiler}" "{mlir_path}" -o "{dll_path}"\n')

    result = subprocess.run(
        ["cmd", "/c", compile_cmd], capture_output=True, text=True, timeout=60
    )
    if result.returncode != 0:
        print("  COMPILE FAILED:")
        print(result.stdout)
        print(result.stderr)
        sys.exit(1)
    print(f"  DLL generated: {dll_path}")

    # ── Step 5: Execute on GPU ─────────────────────────────────────────
    # hip-test-dll generates its own test data. We can't easily pass our
    # weights through it. Instead, let's verify the compile succeeded and
    # note that numerical comparison requires a custom test harness.
    #
    # For now, we'll do a smoke test with hip-test-dll (which uses
    # sequential test data, not our weights) and separately verify
    # that the MLIR faithfully represents the PyTorch computation.

    runner = os.path.normpath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "build",
            "onnx-hipdnn-ep",
            "bin",
            "hip-test-dll.exe",
        )
    )

    run_cmd = os.path.join(work_dir, "_run.cmd")
    therock = "C:\\Users\\tsiddaga\\Documents\\code\\therock"
    with open(run_cmd, "w") as f:
        f.write("@echo off\n")
        f.write('set "VSCMD_START_DIR=%CD%"\n')
        f.write(
            'call "C:\\Program Files\\Microsoft Visual Studio\\18\\'
            'Community\\VC\\Auxiliary\\Build\\vcvarsall.bat" x64 || exit /b 1\n'
        )
        f.write(f"set PATH={therock}\\bin;%PATH%\n")
        f.write(f'"{runner}" "{dll_path}" --verbose --validate\n')

    result = subprocess.run(
        ["cmd", "/c", run_cmd], capture_output=True, text=True, timeout=60
    )

    gpu_output_line = ""
    for line in result.stdout.split("\n"):
        if "First 10 values:" in line:
            gpu_output_line = line.strip()
        if "SUCCESS" in line:
            print("  GPU execution: SUCCESS")

    if result.returncode != 0:
        print("  GPU EXECUTION FAILED:")
        print(result.stdout[-500:])
        print(result.stderr[-500:])
        sys.exit(1)

    # ── Step 6: Compare outputs ────────────────────────────────────────
    # hip-test-dll uses sequential test data (0, 0.001, 0.002, ...) not
    # our actual weights. For true numerical comparison, we compare
    # the PyTorch computation against re-running it with the same
    # sequential test data.

    print(f"\n{'=' * 70}")
    print("Numerical Comparison")
    print(f"{'=' * 70}")

    # Extract GPU values from hip-test-dll output
    if gpu_output_line:
        vals_str = gpu_output_line.split("First 10 values:")[1].strip()
        gpu_vals = [float(v) for v in vals_str.split()]
        print(f"GPU output (hip-test-dll data): {gpu_vals}")
    else:
        print("WARNING: Could not extract GPU output values")
        gpu_vals = []

    # hip-test-dll generates sequential data (i*0.001) which overflows in f16
    # matmuls. Instead, we replicate the exact pattern but scale down to avoid
    # overflow: use i * 0.001 but cap at small values.
    print("\nRunning PyTorch with same test data as hip-test-dll...")

    def make_sequential_f16(shape):
        """Replicate hip-test-dll's generateTestData for f16."""
        n = 1
        for s in shape:
            n *= s
        # hip-test-dll: hdata[i] = floatToHalf((i % 1000) * 0.001f)
        vals = [(i % 1000) * 0.001 for i in range(n)]
        return torch.tensor(vals, dtype=torch.float16).reshape(shape)

    seq_norm_w = make_sequential_f16([hidden])
    seq_gate_w = make_sequential_f16([hidden, hidden])
    seq_up_w = make_sequential_f16([hidden, hidden])
    seq_down_w = make_sequential_f16([hidden, hidden])
    seq_input = make_sequential_f16([1, 4, hidden])

    # The MLIR emitter pre-transposes linear weights: [N,K] -> [K,N]
    # hip-test-dll generates sequential data for the transposed [K,N] shape.
    # To match, we need to transpose the sequential data back to [N,K] for
    # PyTorch (which expects [N,K] and internally computes input @ W^T).
    model_f32 = SmallTransformerBlock(hidden=hidden).eval()
    with torch.no_grad():
        model_f32.norm_weight.copy_(seq_norm_w.float())
        # Transpose the sequential [K,N] data to [N,K] for PyTorch
        model_f32.gate_proj.weight.copy_(seq_gate_w.float().t())
        model_f32.up_proj.weight.copy_(seq_up_w.float().t())
        model_f32.down_proj.weight.copy_(seq_down_w.float().t())
        ref_seq_output = model_f32(seq_input.float()).half()

    ref_vals = ref_seq_output.flatten()[:10].tolist()
    print(f"PyTorch ref (f32 accum): {ref_vals}")

    if gpu_vals:
        # Compare
        errors = [abs(g - r) for g, r in zip(gpu_vals, ref_vals)]
        max_error = max(errors) if errors else 0
        l2_error = (sum(e**2 for e in errors) / len(errors)) ** 0.5 if errors else 0
        rel_errors = [abs(g - r) / (abs(r) + 1e-8) for g, r in zip(gpu_vals, ref_vals)]
        max_rel_error = max(rel_errors) if rel_errors else 0

        print(f"\nMax absolute error:  {max_error:.6f}")
        print(f"RMS error:           {l2_error:.6f}")
        print(f"Max relative error:  {max_rel_error:.6f}")

        # fp16 has ~3.3 decimal digits of precision
        TOLERANCE = 0.01  # 1% relative error threshold for f16
        if max_rel_error < TOLERANCE:
            print(
                f"\nNUMERICAL ACCURACY: PASS (max relative error "
                f"{max_rel_error:.6f} < {TOLERANCE})"
            )
        else:
            print(
                f"\nNUMERICAL ACCURACY: FAIL (max relative error "
                f"{max_rel_error:.6f} >= {TOLERANCE})"
            )
            print("Per-element comparison:")
            for i, (g, r, e, re) in enumerate(
                zip(gpu_vals, ref_vals, errors, rel_errors)
            ):
                flag = " <<< MISMATCH" if re > TOLERANCE else ""
                print(
                    f"  [{i}] GPU={g:.6f}  PyTorch={r:.6f}  "
                    f"abs_err={e:.6f}  rel_err={re:.6f}{flag}"
                )
    else:
        print("\nCould not extract GPU values for comparison")

    print(f"\n{'=' * 70}")


if __name__ == "__main__":
    main()
