#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Minimal standalone example: run an ONNX model on the MorphiZen EP.

Registers the wheel-colocated MorphiZen EP plugin, builds a session over the
discovered AMD device, feeds random inputs, and prints the output shapes. All
the JIT-link / runtime-DLL environment (THEROCK_DIST / LIB / DLL dirs) is wired
up from the installed wheels by this script -- no manual env setup needed.

Prerequisites (see docs/python_package_guide.md):
  - pip install the onnxruntime + onnxruntime_morphizen_ep wheels
  - rocm-sdk init        (expands rocm[devel] import libs)

Usage:
  python run_onnx.py path/to/model.onnx
"""

import glob
import os
import sys

import numpy as np
import onnxruntime as ort

EP_NAME = "MorphiZenEP"

# ORT tensor type string -> numpy dtype (common cases).
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
    # Everything is derived from the installed onnxruntime location; no external
    # env vars required. site-packages = .../onnxruntime/.. ; the EP native files
    # live in onnxruntime/capi and pip rocm in _rocm_sdk_* siblings.
    sp = os.path.dirname(os.path.dirname(ort.__file__))
    capi = os.path.join(sp, "onnxruntime", "capi")

    # JIT link inputs: ROCm import libs (rocm[devel]) + CRT/custom kernels (capi).
    os.environ["THEROCK_DIST"] = os.path.join(sp, "_rocm_sdk_devel")
    os.environ["LIB"] = os.pathsep.join(filter(None, [capi, os.environ.get("LIB", "")]))

    # Runtime DLLs: EP + hip-compiler (capi) + ROCm runtime (rocm wheel bins).
    # The EP loads hip-compiler.dll by name via PATH, so prepend these dirs to
    # PATH (and register them as DLL dirs for ORT's own loader).
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
    # Replace dynamic / symbolic dims (None or str) with 1.
    shape = [d if isinstance(d, int) and d > 0 else 1 for d in spec.shape]
    if np.issubdtype(dtype, np.floating):
        return np.random.rand(*shape).astype(dtype)
    if dtype == np.bool_:
        return np.random.randint(0, 2, size=shape).astype(np.bool_)
    return np.random.randint(0, 2, size=shape).astype(dtype)


def main():
    if len(sys.argv) != 2:
        print(f"usage: python {os.path.basename(__file__)} model.onnx", file=sys.stderr)
        return 2
    model_path = sys.argv[1]

    capi = _setup_env()

    ort.register_execution_provider_library(
        EP_NAME, os.path.join(capi, "onnxruntime_morphizen_ep.dll")
    )

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


if __name__ == "__main__":
    sys.exit(main())
