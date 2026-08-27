#!/usr/bin/env python3
"""
Test script to verify DynamicDispatch compilation.

Usage:
    set HIPEP_USE_DYNAMIC_DISPATCH=1
    set MORPHIZEN_DEBUG_MLIR_BACKEND=1
    python test_dynamic_dispatch.py
"""
import os
import sys
import pathlib
import tempfile
import numpy as np

# Set debug flags to see compilation output
os.environ['MORPHIZEN_DEBUG_MLIR_BACKEND'] = '1'
os.environ['HIPDNN_EP_DEBUG'] = '1'  # Enable COMPILER_DEBUG_LOG
os.environ['HIPEP_USE_DYNAMIC_DISPATCH'] = '1'
# Point to the actual install location (one level up from repo)
os.environ['HIPEP_EP_BIN'] = str(pathlib.Path(__file__).resolve().parent.parent / "install" / "bin")

# Add DynamicDispatch and XRT DLL directories to PATH (if DYNAMICDISPATCH_ROOT is set)
dd_root = os.environ.get('DYNAMICDISPATCH_ROOT')
if dd_root:
    dd_bin_dir = os.path.join(dd_root, "bin")
    xrt_bin_dir = os.path.join(dd_root, "xrt", "bin")
    path_additions = dd_bin_dir + os.pathsep + xrt_bin_dir
    if 'PATH' in os.environ:
        os.environ['PATH'] = path_additions + os.pathsep + os.environ['PATH']
    else:
        os.environ['PATH'] = path_additions
    print(f"Added DynamicDispatch paths to PATH:")
    print(f"  DD bin: {dd_bin_dir}")
    print(f"  XRT bin: {xrt_bin_dir}")
else:
    print("DYNAMICDISPATCH_ROOT not set - skipping DD/XRT DLL path additions")

# Add test path for conftest
REPO_ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "test" / "python"))

import onnxruntime as ort
from conftest import register_morphizen_ep, EP_PROVIDER_OPTIONS, EP_REGISTRATION_NAME

print("=" * 70)
print("DynamicDispatch Test Script")
print("=" * 70)
print(f"HIPEP_USE_DYNAMIC_DISPATCH = {os.environ.get('HIPEP_USE_DYNAMIC_DISPATCH', '<not set>')}")
print(f"MORPHIZEN_DEBUG_MLIR_BACKEND = {os.environ.get('MORPHIZEN_DEBUG_MLIR_BACKEND', '<not set>')}")
print()

# Register the EP
print("Registering AMDGPU EP...")
print(f"HIPEP_EP_BIN = {os.environ.get('HIPEP_EP_BIN', '<not set>')}")
print(f"REPO_ROOT = {REPO_ROOT}")
try:
    devices = register_morphizen_ep(REPO_ROOT)
    print(f"Devices returned: {devices}")
    if not devices:
        print("ERROR: Failed to register EP (returned None or empty)")
        sys.exit(1)
except Exception as e:
    print(f"ERROR: Exception during EP registration: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
print(f"EP devices: {devices}")
print()

# Create a simple ONNX model (MatMul) in memory
print("Creating test model (MatMul)...")
import onnx
from onnx import helper, TensorProto

# Create a simple MatMul: C = A @ B
# A: [128, 256] fp16
# B: [256, 512] fp16
# C: [128, 512] fp16

A = helper.make_tensor_value_info('A', TensorProto.FLOAT16, [128, 256])
B = helper.make_tensor_value_info('B', TensorProto.FLOAT16, [256, 512])
C = helper.make_tensor_value_info('C', TensorProto.FLOAT16, [128, 512])

matmul_node = helper.make_node(
    'MatMul',
    inputs=['A', 'B'],
    outputs=['C']
)

graph = helper.make_graph(
    [matmul_node],
    'test_matmul',
    [A, B],
    [C]
)

model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])

# Save to temp file
with tempfile.NamedTemporaryFile(mode='wb', suffix='.onnx', delete=False) as f:
    model_path = f.name
    onnx.save(model, f)

print(f"Model saved to: {model_path}")
print()

try:
    # Create session with EP
    print("Creating InferenceSession with AMDGPU EP...")
    print("This should trigger compilation and show debug output...")
    print()

    so = ort.SessionOptions()
    so.log_severity_level = 0  # Verbose logging
    so.add_provider_for_devices(devices, EP_PROVIDER_OPTIONS)

    session = ort.InferenceSession(model_path, sess_options=so)

    print()
    print("=" * 70)
    print("Session created successfully!")
    print()

    # Check providers
    providers = session.get_providers()
    print(f"Active providers: {providers}")

    if EP_REGISTRATION_NAME in providers:
        print(f"[OK] {EP_REGISTRATION_NAME} is active")
    else:
        print(f"[FAIL] {EP_REGISTRATION_NAME} is NOT active")

    print()
    print("Look for compilation debug output above.")
    print("You should see lines containing:")
    print("  - 'useDynamicDispatch=true'")
    print("  - 'HIPEP_USE_DYNAMIC_DISPATCH env var: 1'")
    print("  - 'Compilation options (JSON):' with 'use_dynamic_dispatch': true")
    print()

    # Run a simple inference
    print("Running inference...")
    A_data = np.random.randn(128, 256).astype(np.float16)
    B_data = np.random.randn(256, 512).astype(np.float16)

    outputs = session.run(None, {'A': A_data, 'B': B_data})

    print(f"Output shape: {outputs[0].shape}")
    print("[OK] Inference completed successfully")

finally:
    # Cleanup
    try:
        os.unlink(model_path)
    except:
        pass

print()
print("=" * 70)
print("Test completed!")
print("=" * 70)
