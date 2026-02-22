#!/usr/bin/env python3
#
# Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""
GPU vs CPU Comparison Test

Compares outputs from GPU execution (via compiled DLL) vs CPU execution
(via ONNX Runtime CPU EP) to verify correctness.

Usage:
    python test_gpu_vs_cpu.py <model.onnx> [--tolerance 1e-5]

Requirements:
    - onnxruntime (CPU execution provider)
    - numpy
    - Compiled DLL from the same model

Prerequisites:
    1. Compile ONNX model to DLL:
       mlir-hip-compiler model.mlir --from-onnx-mlir -o model_gpu.dll

    2. Ensure TheRock DLLs are in PATH (Windows) or LD_LIBRARY_PATH (Linux)

Example:
    python test_gpu_vs_cpu.py demo_two_layer_conv.onnx --tolerance 1e-4
"""

import sys
import os
import argparse
import numpy as np
import ctypes
import platform

# Fix Windows console encoding for Unicode characters
if platform.system() == "Windows":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

try:
    import onnxruntime as ort
except ImportError:
    print("ERROR: onnxruntime not installed. Install with: pip install onnxruntime")
    sys.exit(1)


class GPUModel:
    """Wrapper for compiled DLL model"""

    def __init__(self, dll_path):
        """Load DLL and resolve interface functions"""
        self.dll_path = dll_path

        # Load DLL
        if platform.system() == "Windows":
            self.lib = ctypes.CDLL(dll_path)
        else:
            self.lib = ctypes.CDLL(dll_path, mode=ctypes.RTLD_NOW)

        # Define C structures (must match hipdnn_ep_runtime.h)
        class TensorT(ctypes.Structure):
            _fields_ = [
                ("data", ctypes.c_void_p),
                ("shape", ctypes.POINTER(ctypes.c_int64)),
                ("rank", ctypes.c_size_t),
            ]

        class SpanT(ctypes.Structure):
            _fields_ = [
                ("data", ctypes.POINTER(TensorT)),
                ("count", ctypes.c_size_t),
            ]

        self.TensorT = TensorT
        self.SpanT = SpanT

        # Resolve functions
        self.init = self.lib.inference_init
        self.init.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self.init.restype = ctypes.c_int

        self.compute = self.lib.inference_compute
        self.compute.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(SpanT),
            ctypes.POINTER(SpanT),
        ]
        self.compute.restype = ctypes.c_int

        self.cleanup = self.lib.inference_cleanup
        self.cleanup.argtypes = [ctypes.c_void_p]
        self.cleanup.restype = ctypes.c_int

        # Initialize model
        self.state = ctypes.c_void_p()
        ret = self.init(ctypes.byref(self.state))
        if ret != 0:
            raise RuntimeError(f"GPU model init failed with code {ret}")

    def run(self, inputs):
        """
        Run inference on GPU

        Args:
            inputs: List of numpy arrays

        Returns:
            List of output numpy arrays
        """
        # Prepare input span
        input_tensors = []
        for inp in inputs:
            tensor = self.TensorT()
            tensor.data = inp.ctypes.data_as(ctypes.c_void_p)
            shape_arr = (ctypes.c_int64 * len(inp.shape))(*inp.shape)
            tensor.shape = shape_arr
            tensor.rank = len(inp.shape)
            input_tensors.append(tensor)

        input_span = self.SpanT()
        input_span.data = (self.TensorT * len(input_tensors))(*input_tensors)
        input_span.count = len(input_tensors)

        # Prepare output span (assuming 1 output for demo)
        # TODO: Get output shape from model metadata
        output_shape = (1, 64, 112, 112)  # Demo model output shape
        output_data = np.zeros(output_shape, dtype=np.float32)

        output_tensor = self.TensorT()
        output_tensor.data = output_data.ctypes.data_as(ctypes.c_void_p)
        shape_arr = (ctypes.c_int64 * len(output_shape))(*output_shape)
        output_tensor.shape = shape_arr
        output_tensor.rank = len(output_shape)

        output_span = self.SpanT()
        output_span.data = ctypes.pointer(output_tensor)
        output_span.count = 1

        # Run inference
        ret = self.compute(self.state, ctypes.byref(input_span), ctypes.byref(output_span))
        if ret != 0:
            raise RuntimeError(f"GPU inference failed with code {ret}")

        return [output_data]

    def __del__(self):
        """Cleanup GPU resources"""
        if hasattr(self, "state") and self.state:
            self.cleanup(self.state)


def run_cpu(model_path, inputs):
    """Run model on CPU using ONNX Runtime"""
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    # Get input names
    input_names = [inp.name for inp in session.get_inputs()]

    # Create input dict
    input_dict = {name: inp for name, inp in zip(input_names, inputs)}

    # Run inference
    outputs = session.run(None, input_dict)

    return outputs


def compare_outputs(gpu_outputs, cpu_outputs, tolerance):
    """Compare GPU and CPU outputs"""
    if len(gpu_outputs) != len(cpu_outputs):
        print(f"ERROR: Output count mismatch: GPU={len(gpu_outputs)}, CPU={len(cpu_outputs)}")
        return False

    all_match = True
    for i, (gpu_out, cpu_out) in enumerate(zip(gpu_outputs, cpu_outputs)):
        # Check shape
        if gpu_out.shape != cpu_out.shape:
            print(f"✗ Output {i}: Shape mismatch - GPU={gpu_out.shape}, CPU={cpu_out.shape}")
            all_match = False
            continue

        # Check values
        try:
            np.testing.assert_allclose(gpu_out, cpu_out, rtol=tolerance, atol=tolerance, err_msg=f"Output {i}")
            max_diff = np.max(np.abs(gpu_out - cpu_out))
            mean_diff = np.mean(np.abs(gpu_out - cpu_out))
            print(f"✓ Output {i}: PASS")
            print(f"    Shape: {gpu_out.shape}")
            print(f"    Max diff: {max_diff:.2e}")
            print(f"    Mean diff: {mean_diff:.2e}")
        except AssertionError as e:
            print(f"✗ Output {i}: FAIL")
            print(f"    {e}")
            all_match = False

    return all_match


def main():
    parser = argparse.ArgumentParser(description="Compare GPU vs CPU execution")
    parser.add_argument("onnx_model", help="Path to ONNX model")
    parser.add_argument("--dll", help="Path to compiled GPU DLL (default: <model>_gpu.dll)")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-5,
        help="Numerical tolerance for comparison (default: 1e-5)",
    )
    args = parser.parse_args()

    # Determine DLL path
    dll_path = args.dll
    if not dll_path:
        base = os.path.splitext(args.onnx_model)[0]
        dll_path = f"{base}_gpu.dll"

    # Convert to absolute path for DLL loading
    dll_path = os.path.abspath(dll_path)
    args.onnx_model = os.path.abspath(args.onnx_model)

    if not os.path.exists(args.onnx_model):
        print(f"ERROR: ONNX model not found: {args.onnx_model}")
        return 1

    if not os.path.exists(dll_path):
        print(f"ERROR: GPU DLL not found: {dll_path}")
        print(f"Compile with: mlir-hip-compiler {args.onnx_model} -o {dll_path}")
        return 1

    print("=" * 60)
    print("GPU vs CPU Comparison Test")
    print("=" * 60)
    print(f"ONNX Model: {args.onnx_model}")
    print(f"GPU DLL: {dll_path}")
    print(f"Tolerance: {args.tolerance}")
    print()

    # Generate test input
    # TODO: Get input shape from ONNX model metadata
    input_shape = (1, 3, 224, 224)  # Demo model input
    test_input = np.random.randn(*input_shape).astype(np.float32)
    print(f"Test input shape: {input_shape}")
    print()

    # Run on CPU
    print("--- Running on CPU (ONNX Runtime) ---")
    try:
        cpu_outputs = run_cpu(args.onnx_model, [test_input])
        print("✓ CPU execution succeeded")
        print(f"  Outputs: {len(cpu_outputs)}")
        for i, out in enumerate(cpu_outputs):
            print(f"  Output {i}: shape={out.shape}, dtype={out.dtype}")
        print()
    except Exception as e:
        print(f"✗ CPU execution failed: {e}")
        return 1

    # Run on GPU
    print("--- Running on GPU (Compiled DLL) ---")
    try:
        gpu_model = GPUModel(dll_path)
        gpu_outputs = gpu_model.run([test_input])
        print("✓ GPU execution succeeded")
        print(f"  Outputs: {len(gpu_outputs)}")
        for i, out in enumerate(gpu_outputs):
            print(f"  Output {i}: shape={out.shape}, dtype={out.dtype}")
        print()
    except Exception as e:
        print(f"✗ GPU execution failed: {e}")
        import traceback

        traceback.print_exc()
        return 1

    # Compare outputs
    print("--- Comparing Outputs ---")
    if compare_outputs(gpu_outputs, cpu_outputs, args.tolerance):
        print()
        print("=" * 60)
        print("SUCCESS: GPU and CPU outputs match!")
        print("=" * 60)
        return 0
    else:
        print()
        print("=" * 60)
        print("FAILURE: GPU and CPU outputs differ")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
