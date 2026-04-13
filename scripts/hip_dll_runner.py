#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""
Load and run compiled HIP model DLLs from Python via ctypes.

Provides a callable wrapper around the C ABI:
  inference_init / inference_compute / inference_cleanup

Usage:
    runner = HipDllRunner("model.dll")
    outputs = runner(input_tensor1, input_tensor2, ...)
    runner.close()
"""

import ctypes
import json
import os
from typing import List, Optional

import numpy as np
import torch


# ── C struct definitions matching hip-test-dll ──────────────────────────

class TensorT(ctypes.Structure):
    """Mirrors tensor_t in hipdnn_ep_runtime.h"""
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("rank", ctypes.c_size_t),
        ("element_size", ctypes.c_size_t),
    ]


class SpanT(ctypes.Structure):
    """Mirrors span_t in hipdnn_ep_runtime.h"""
    _fields_ = [
        ("data", ctypes.POINTER(TensorT)),
        ("count", ctypes.c_size_t),
    ]


class HipDllRunner:
    """Loads a compiled HIP model DLL and provides a callable interface."""

    def __init__(self, dll_path: str, work_dir: Optional[str] = None):
        """Load DLL and initialize inference context.

        Args:
            dll_path: Path to the compiled model DLL
            work_dir: Working directory for constants.bin (default: dll's dir)
        """
        self.dll_path = os.path.abspath(dll_path)
        self.work_dir = work_dir or os.path.dirname(self.dll_path)

        # Load DLL
        # Add TheRock bin to DLL search path for amdhip64.dll
        therock = os.environ.get("THEROCK_DIST", "")
        if therock:
            os.add_dll_directory(os.path.join(therock, "bin"))

        self.dll = ctypes.CDLL(self.dll_path)

        # Bind functions
        self._init = self.dll.inference_init
        self._init.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
        self._init.restype = ctypes.c_int

        self._compute = self.dll.inference_compute
        self._compute.argtypes = [ctypes.c_void_p,
                                   ctypes.POINTER(SpanT),
                                   ctypes.POINTER(SpanT)]
        self._compute.restype = ctypes.c_int

        self._cleanup = self.dll.inference_cleanup
        self._cleanup.argtypes = [ctypes.c_void_p]
        self._cleanup.restype = ctypes.c_int

        self._get_metadata = self.dll.inference_get_metadata_json
        self._get_metadata.argtypes = []
        self._get_metadata.restype = ctypes.c_char_p

        # Parse metadata
        meta_str = self._get_metadata()
        self.metadata = json.loads(meta_str)
        self.input_metas = self.metadata.get("inputs", [])
        self.output_metas = self.metadata.get("outputs", [])

        # Initialize state
        self.state = ctypes.c_void_p()
        old_cwd = os.getcwd()
        os.chdir(self.work_dir)
        ret = self._init(ctypes.byref(self.state), None)
        os.chdir(old_cwd)
        if ret != 0:
            raise RuntimeError(f"inference_init failed with code {ret}")

    def __call__(self, *inputs: torch.Tensor) -> List[torch.Tensor]:
        """Run inference with PyTorch tensors."""
        if len(inputs) != len(self.input_metas):
            raise ValueError(
                f"Expected {len(self.input_metas)} inputs, got {len(inputs)}")

        # Prepare input tensors
        input_arrays = []
        input_shapes = []
        input_tensors = (TensorT * len(inputs))()

        for i, (tensor, meta) in enumerate(zip(inputs, self.input_metas)):
            arr = tensor.detach().cpu().contiguous().numpy()
            input_arrays.append(arr)  # keep alive

            shape = (ctypes.c_int64 * len(arr.shape))(*arr.shape)
            input_shapes.append(shape)  # keep alive

            input_tensors[i].data = arr.ctypes.data
            input_tensors[i].shape = shape
            input_tensors[i].rank = len(arr.shape)
            input_tensors[i].element_size = arr.itemsize

        input_span = SpanT()
        input_span.data = input_tensors
        input_span.count = len(inputs)

        # Prepare output tensors
        output_arrays = []
        output_shapes = []
        output_tensors = (TensorT * len(self.output_metas))()

        for i, meta in enumerate(self.output_metas):
            shape = meta.get("shape", [1])
            elem_size = meta.get("element_size", 2)

            dtype = {2: np.float16, 4: np.float32, 8: np.int64, 1: np.uint8}
            arr = np.zeros(shape, dtype=dtype.get(elem_size, np.float16))
            output_arrays.append(arr)

            shape_arr = (ctypes.c_int64 * len(shape))(*shape)
            output_shapes.append(shape_arr)

            output_tensors[i].data = arr.ctypes.data
            output_tensors[i].shape = shape_arr
            output_tensors[i].rank = len(shape)
            output_tensors[i].element_size = elem_size

        output_span = SpanT()
        output_span.data = output_tensors
        output_span.count = len(self.output_metas)

        # Run inference
        ret = self._compute(self.state,
                            ctypes.byref(input_span),
                            ctypes.byref(output_span))
        if ret != 0:
            raise RuntimeError(f"inference_compute failed with code {ret}")

        # Convert outputs to PyTorch tensors
        results = []
        for arr in output_arrays:
            results.append(torch.from_numpy(arr.copy()))

        return results

    def close(self):
        """Clean up inference state."""
        if self.state:
            self._cleanup(self.state)
            self.state = None

    def __del__(self):
        self.close()

    def __repr__(self):
        return (f"HipDllRunner('{os.path.basename(self.dll_path)}', "
                f"inputs={len(self.input_metas)}, "
                f"outputs={len(self.output_metas)})")
