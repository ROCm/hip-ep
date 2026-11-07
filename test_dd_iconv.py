#!/usr/bin/env python3
"""
Test DD C API integration - iconv (integer convolution) operator

This tests whether DD's iconv operator can activate the NPU.
Like combined_gemm, iconv likely expects quantized uint8 weights.
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
print("DD iconv Test - Testing NPU activation with Conv2D")
print("=" * 70)

devices = register_morphizen_ep(REPO_ROOT)

# Create a simple Conv2D model
# Input: [N=1, C=3, H=224, W=224]
# Kernel: [K=64, C=3, KH=3, KW=3]
# Output: [N=1, K=64, H=222, W=222] (no padding, stride=1)

N, C, H, W = 1, 3, 224, 224
K, KH, KW = 64, 3, 3
out_H, out_W = H - KH + 1, W - KW + 1  # No padding, stride=1

# Input tensor
X = helper.make_tensor_value_info('input', TensorProto.FLOAT16, [N, C, H, W])

# Output tensor
Y = helper.make_tensor_value_info('output', TensorProto.FLOAT16, [N, K, out_H, out_W])

# Weight tensor (constant) - float16
W_data = np.random.randn(K, C, KH, KW).astype(np.float16)
W_init = helper.make_tensor(
    name='weight',
    data_type=TensorProto.FLOAT16,
    dims=[K, C, KH, KW],
    vals=W_data.tobytes(),
    raw=True
)

# Conv node (no bias for simplicity)
conv = helper.make_node(
    'Conv',
    inputs=['input', 'weight'],
    outputs=['output'],
    kernel_shape=[KH, KW],
    strides=[1, 1],
    pads=[0, 0, 0, 0],  # No padding
    dilations=[1, 1],
    group=1
)

graph = helper.make_graph(
    [conv],
    'dd_iconv_test',
    [X],
    [Y],
    [W_init]
)

model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])

print(f"\nModel: Conv2D")
print(f"  Input:  [{N}, {C}, {H}, {W}] (NCHW) - float16")
print(f"  Weight: [{K}, {C}, {KH}, {KW}] - float16 constant")
print(f"  Output: [{N}, {K}, {out_H}, {out_W}] - float16")
print(f"  Params: kernel={KH}x{KW}, stride=1, padding=0")
print(f"\n  NOTE: DD iconv expects uint8 weights, but we're passing float16")

sess = ort.InferenceSession(
    model.SerializeToString(),
    providers=[('AMDGPUExecutionProvider', {})]
)

X_data = np.random.randn(N, C, H, W).astype(np.float16)

print("\n" + "=" * 70)
print("Running 5-second inference loop...")
print("CHECK TASK MANAGER: Monitor NPU usage!")
print("=" * 70)

start_time = time.time()
iteration = 0

try:
    while time.time() - start_time < 5.0:
        result = sess.run(None, {'input': X_data})
        iteration += 1
        if iteration == 1 or iteration % 5 == 0:
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
    print("  Look for:")
    print("    '[DD] Creating iconv'               -> iconv wrapper invoked")
    print("    'dd_iconv_initialize_weights failed' -> Weight init failure")
    print("    '[DD] iconv executing'              -> Success! NPU is used")
    print("    'Unsupported const spec'            -> Same issue as combined_gemm")
    print("\n" + "=" * 70)
    print("EXPECTED OUTCOME")
    print("=" * 70)
    print("  If iconv has the same weight format requirement as combined_gemm,")
    print("  we'll see initialization failure and 0% NPU usage.")
    print("  If iconv accepts float16 weights, we should see NPU activity!")
    print("=" * 70)

except Exception as e:
    print(f"\n{'='*70}")
    print(f"EXCEPTION: {e}")
    print(f"{'='*70}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
