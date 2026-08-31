#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Run an ONNX model or OGA benchmark on the MorphiZen EP.

Sets up the JIT linker's LIB and the umbrella EP path from the installed wheels,
then either:
  - Runs a plain ONNX model with random inputs (default), or
  - Delegates to benchmark_e2e.py for OGA benchmarks (--benchmark).

Prerequisites (see docs/quick_start.md):
  - pip install the onnxruntime + onnxruntime_ep_amdgpu + onnxruntime_ep_hip wheels

Usage:
  python run_onnx.py path/to/model.onnx
  python run_onnx.py --benchmark benchmark_e2e.py -i model_dir [args...]
"""

import os
import subprocess
import sys

import numpy as np
import onnxruntime as ort

# Importing this runs os.add_dll_directory on the package directory, where every
# DLL in the chain lives, so it has to precede ORT loading any of them.
import onnxruntime_ep_amdgpu

EP_NAME = "MorphiZenEP"

_NP_DTYPE = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(double)": np.float64,
    "tensor(int64)": np.int64,
    "tensor(int32)": np.int32,
    "tensor(int8)": np.int8,
    "tensor(uint8)": np.uint8,
    "tensor(bool)": np.bool_,
}


def _setup_env():
    pkg = os.path.dirname(onnxruntime_ep_amdgpu.__file__)

    # The JIT linker resolves the CRT and ROCm import libs off %LIB%.
    os.environ["LIB"] = os.pathsep.join(filter(None, [pkg, os.environ.get("LIB", "")]))
    # OGA looks for the umbrella next to its own DLL / onnxruntime.dll / the exe,
    # none of which is this directory. An env var also reaches the --benchmark
    # child, unlike an in-process registration.
    os.environ["AMDGPU_EP_PATH"] = onnxruntime_ep_amdgpu.get_library_path()
    return pkg


def _random_input(spec):
    dtype = _NP_DTYPE.get(spec.type, np.float32)
    shape = [d if isinstance(d, int) and d > 0 else 1 for d in spec.shape]
    if np.issubdtype(dtype, np.floating):
        return np.random.rand(*shape).astype(dtype)
    if dtype == np.bool_:
        return np.random.randint(0, 2, size=shape).astype(np.bool_)
    return np.random.randint(0, 2, size=shape).astype(dtype)


def _run_onnx(model_path, pkg):
    ort.register_execution_provider_library(EP_NAME, os.path.join(pkg, "hipgpu.dll"))

    devices = [d for d in ort.get_ep_devices() if d.ep_name == EP_NAME]
    if not devices:
        print(f"ERROR: no {EP_NAME} device found", file=sys.stderr)
        return 1
    print(f"{EP_NAME} device(s): {len(devices)} ({[d.ep_vendor for d in devices]})")

    so = ort.SessionOptions()
    so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    so.add_provider_for_devices(devices, {})

    sess = ort.InferenceSession(model_path, sess_options=so)
    print(f"providers: {sess.get_providers()}")

    feeds = {i.name: _random_input(i) for i in sess.get_inputs()}
    outputs = sess.run(None, feeds)

    for spec, value in zip(sess.get_outputs(), outputs):
        print(f"output {spec.name}: shape={value.shape} dtype={value.dtype}")
    return 0


def main():
    if len(sys.argv) >= 3 and sys.argv[1] == "--benchmark":
        _setup_env()
        return subprocess.call([sys.executable, *sys.argv[2:]])

    if len(sys.argv) != 2:
        print(f"usage: python {os.path.basename(__file__)} model.onnx", file=sys.stderr)
        print(
            f"       python {os.path.basename(__file__)} --benchmark benchmark_e2e.py [args...]",
            file=sys.stderr,
        )
        return 2

    pkg = _setup_env()
    return _run_onnx(sys.argv[1], pkg)


if __name__ == "__main__":
    sys.exit(main())
