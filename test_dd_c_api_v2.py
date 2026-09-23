#!/usr/bin/env python3
"""
Test DD C API integration - Weight Quantization Issue Demonstration

STATUS: DD C API works (no access violations), but NPU is NOT being used.

ISSUE: Dynamic Dispatch's combined_gemm expects quantized uint8 weights,
       but hip-ep is passing float16 weights. DD initialization fails with:
       "Unsupported const spec for combined_gemm (Const param dim == 2)"

RESULT: Test runs successfully but falls back to CPU/GPU instead of NPU.
        Task Manager shows NO NPU activity during 5-second test loop.

See docs/DynamicDispatch-Weight-Quantization-Issue.md for details.
"""
import os
import sys
import pathlib
import numpy as np
import time

# Enable Dynamic Dispatch
os.environ['HIPEP_USE_DYNAMIC_DISPATCH'] = '1'

REPO_ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "test" / "python"))

import onnxruntime as ort
from conftest import register_morphizen_ep
import onnx
from onnx import helper, TensorProto

print("=" * 70)
print("DD C API Test - Weight Quantization Issue")
print("=" * 70)
print("This test demonstrates that DD C API is invoked but NPU is NOT used")
print("due to weight format mismatch (float16 vs. expected quantized uint8).")
print("=" * 70)

devices = register_morphizen_ep(REPO_ROOT)

# Create simple MatMul with float16 constant weight
# This will trigger DD but fail initialization
M, K, N = 11, 271, 173

A = helper.make_tensor_value_info('A', TensorProto.FLOAT16, [M, K])
C = helper.make_tensor_value_info('C', TensorProto.FLOAT16, [M, N])

# Weight as float16 constant (this is the problem - DD expects uint8)
B_data = np.random.rand(K, N).astype(np.float16)
B_initializer = helper.make_tensor(
    name='B',
    data_type=TensorProto.FLOAT16,
    dims=[K, N],
    vals=B_data.tobytes(),
    raw=True
)

matmul = helper.make_node('MatMul', inputs=['A', 'B'], outputs=['C'])
graph = helper.make_graph([matmul], 'dd_weight_issue_test', [A], [C], [B_initializer])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])

print(f"\nModel: {M}x{K} @ {K}x{N} MatMul with float16 weights")
print(f"  Input A: dynamic float16 [{M}x{K}]")
print(f"  Weight B: constant float16 [{K}x{N}]  <-- PROBLEM: DD expects uint8")
print(f"  Output C: float16 [{M}x{N}]")

sess = ort.InferenceSession(
    model.SerializeToString(),
    providers=[('AMDGPUExecutionProvider', {})]
)

A_data = np.random.rand(M, K).astype(np.float16)

print("\n" + "=" * 70)
print("Running 5-second inference loop...")
print("CHECK TASK MANAGER: NPU usage should be 0% (confirming fallback)")
print("=" * 70)

start_time = time.time()
iteration = 0

try:
    while time.time() - start_time < 5.0:
        result = sess.run(None, {'A': A_data})
        iteration += 1
        if iteration == 1 or iteration % 10 == 0:
            elapsed = time.time() - start_time
            print(f"  [{elapsed:5.2f}s] Iteration {iteration:3d} - shape: {result[0].shape}")

    elapsed = time.time() - start_time

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  Iterations: {iteration}")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Avg: {elapsed/iteration*1000:.2f}ms per iteration")
    print(f"  Output: shape={result[0].shape}, dtype={result[0].dtype}")
    print("\n" + "=" * 70)
    print("ANALYSIS (check stderr)")
    print("=" * 70)
    print("  [DD] Creating combined_gemm        -> DD is invoked")
    print("  initialize_weights failed: -2       -> Initialization fails")
    print("  Unsupported const spec             -> Root cause: weight format")
    print("  NO '[DD] executing combined_gemm'  -> NPU NOT used")
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print("  C API: WORKING (no access violations)")
    print("  NPU Usage: NOT WORKING (falls back to CPU/GPU)")
    print("  Root Cause: DD expects quantized uint8 weights, not float16")
    print(f"\n  See: docs/DynamicDispatch-Weight-Quantization-Issue.md")
    print("=" * 70)

except Exception as e:
    print(f"\n{'='*70}")
    print(f"EXCEPTION: {e}")
    print(f"{'='*70}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
