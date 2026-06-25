#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Run an ONNX model or OGA benchmark on the MorphiZen EP.

Sets up the JIT-link / runtime-DLL environment (THEROCK_DIST / LIB / DLL dirs)
from the installed wheels, then either:
  - Runs a plain ONNX model with random inputs (default), or
  - Delegates to benchmark_e2e.py for OGA benchmarks (--benchmark).

Prerequisites (see docs/python_package_guide.md):
  - pip install the onnxruntime + onnxruntime_ep_hip wheels
  - rocm-sdk init        (expands rocm[devel] import libs)

Usage:
  python run_onnx.py path/to/model.onnx
  python run_onnx.py --benchmark benchmark_e2e.py -i model_dir [args...]
"""

import glob
import os
import subprocess
import sys

import numpy as np
import onnxruntime as ort

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
    sp = os.path.dirname(os.path.dirname(ort.__file__))
    capi = os.path.join(sp, "onnxruntime", "capi")

    os.environ["THEROCK_DIST"] = os.path.join(sp, "_rocm_sdk_devel")
    os.environ["LIB"] = os.pathsep.join(filter(None, [capi, os.environ.get("LIB", "")]))

    dll_dirs = [
        d
        for d in [capi, *glob.glob(os.path.join(sp, "_rocm_sdk_*", "bin"))]
        if os.path.isdir(d)
    ]
    os.environ["PATH"] = os.pathsep.join([*dll_dirs, os.environ.get("PATH", "")])
    for d in dll_dirs:
        os.add_dll_directory(d)
    return capi


def _random_input(spec):
    dtype = _NP_DTYPE.get(spec.type, np.float32)
    shape = [d if isinstance(d, int) and d > 0 else 1 for d in spec.shape]
    if np.issubdtype(dtype, np.floating):
        return np.random.rand(*shape).astype(dtype)
    if dtype == np.bool_:
        return np.random.randint(0, 2, size=shape).astype(np.bool_)
    return np.random.randint(0, 2, size=shape).astype(dtype)


def _run_onnx(model_path, capi):
    ort.register_execution_provider_library(EP_NAME, os.path.join(capi, "hipgpu.dll"))

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

    capi = _setup_env()
    return _run_onnx(sys.argv[1], capi)


if __name__ == "__main__":
    sys.exit(main())
