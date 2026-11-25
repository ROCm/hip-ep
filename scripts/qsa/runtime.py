#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Strict hip-ep sessions and zero-copy Torch/ORT GPU tensor bindings."""
# DLL search paths must be configured before importing Torch and ORT.
# ruff: noqa: E402

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "_local/qsa"
INSTALL = Path(
    os.environ.get("QSA_HIP_EP_INSTALL", ROOT / "_local/ring-1259/install/existing")
)
SDK = Path(sys.base_prefix) / "Lib/site-packages/_rocm_sdk_devel"
CORE = Path(sys.base_prefix) / "Lib/site-packages/_rocm_sdk_core"

# Keep handles alive: Windows removes a DLL directory when its handle closes.
DLL_HANDLES = []
for directory in [INSTALL / "bin", SDK / "bin", CORE / "bin", SDK / "lib/llvm/bin"]:
    if directory.exists():
        DLL_HANDLES.append(os.add_dll_directory(str(directory)))
        os.environ["PATH"] = str(directory) + os.pathsep + os.environ["PATH"]
os.environ["LIB"] = os.pathsep.join(
    [str(INSTALL / "lib"), str(SDK / "lib"), os.environ.get("LIB", "")]
)
os.environ["THEROCK_DIST"] = str(SDK)
os.environ["HIPDNN_EP_STRICT"] = "1"
os.environ["HIPDNN_EP_DEBUG"] = "0"
os.environ["HIPDNN_EP_PERF"] = "0"
os.environ["HIPDNN_EP_MATMUL_DP4A"] = "0"

import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.capi import _pybind_state as capi

EP_NAME = "OLMoHipEP"
_DEVICE = None


def register():
    global _DEVICE
    if _DEVICE is None:
        ort.register_execution_provider_library(
            EP_NAME, str(INSTALL / "bin/hipgpu.dll")
        )
        devices = [d for d in ort.get_ep_devices() if d.ep_name == EP_NAME]
        if len(devices) != 1:
            raise RuntimeError(f"Expected one hip-ep GPU, got {len(devices)}")
        _DEVICE = devices[0]
    return _DEVICE


def session(path):
    options = ort.SessionOptions()
    if os.environ.get("QSA_ORT_PROFILE_DIR"):
        directory = Path(os.environ["QSA_ORT_PROFILE_DIR"])
        directory.mkdir(parents=True, exist_ok=True)
        options.enable_profiling = True
        options.profile_file_prefix = str(
            directory / (path.parent.name + "-" + path.stem)
        )
    options.log_severity_level = 3
    options.intra_op_num_threads = 2
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    options.add_provider_for_devices(
        [register()],
        {
            "config_file": str(INSTALL / "bin/morphizen_config.json"),
            "profile": "hip",
            "cacheDir": os.environ.get("QSA_MODEL_CACHE", str(WORK / "model-cache")),
        },
    )
    return ort.InferenceSession(str(path), sess_options=options)


DTYPES = {
    torch.float16: np.float16,
    torch.float32: np.float32,
    torch.int64: np.int64,
    torch.int32: np.int32,
}


def binding(sess, inputs, outputs):
    io = sess.io_binding()
    # The four-argument constructor carries the AMD PCI vendor ID. Using a
    # generic CUDA memory label can make ORT insert copies or reject the buffer.
    gpu = capi.OrtDevice(
        capi.OrtDevice.gpu(), capi.OrtDevice.default_memory(), 0x1002, 0
    )
    cpu = capi.OrtDevice(capi.OrtDevice.cpu(), capi.OrtDevice.default_memory(), 0)
    for direction, tensors in [("input", inputs), ("output", outputs)]:
        for name, tensor in tensors.items():
            if not tensor.is_contiguous():
                raise ValueError(f"Noncontiguous bound tensor: {name}")
            device = gpu if tensor.is_cuda else cpu
            getattr(io._iobinding, "bind_" + direction)(
                name,
                device,
                DTYPES[tensor.dtype],
                list(tensor.shape),
                tensor.data_ptr(),
            )
    return io


def run(sess, io):
    # Runtime-owned streams are independent of Torch's stream. The initial
    # correctness path uses explicit device synchronization at the handoff.
    torch.cuda.synchronize()
    sess.run_with_iobinding(io)
    torch.cuda.synchronize()
