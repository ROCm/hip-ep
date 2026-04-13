#!/usr/bin/env python3
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""
Load and run compiled HIP model DLLs from Python via ctypes.

Provides a callable wrapper around the compiled DLL's C ABI:
  inference_init / inference_compute / inference_cleanup
"""

import ctypes
import json
import os
from typing import List, Optional

import numpy as np
import torch


class TensorT(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("rank", ctypes.c_size_t),
        ("element_size", ctypes.c_size_t),
    ]


class SpanT(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.POINTER(TensorT)),
        ("count", ctypes.c_size_t),
    ]


class HipDllRunner:
    """Loads a compiled HIP model DLL and runs inference via ctypes."""

    def __init__(self, dll_path: str, work_dir: Optional[str] = None):
        self.dll_path = os.path.abspath(dll_path)
        self.work_dir = work_dir or os.path.dirname(self.dll_path)

        # Add TheRock to DLL search path for amdhip64.dll
        therock = os.environ.get("THEROCK_DIST", "")
        if therock:
            try:
                os.add_dll_directory(os.path.join(therock, "bin"))
            except OSError:
                pass

        # Load DLL
        self.dll = ctypes.CDLL(self.dll_path)

        # Bind C ABI functions
        self._init = self.dll.inference_init
        self._init.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
        self._init.restype = ctypes.c_int

        self._compute = self.dll.inference_compute
        self._compute.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(SpanT),
            ctypes.POINTER(SpanT),
        ]
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

        # Initialize: pass NULL filesystem (OK for models without constants)
        self.state = ctypes.c_void_p()
        old_cwd = os.getcwd()
        os.chdir(self.work_dir)
        ret = self._init(ctypes.byref(self.state), None)
        os.chdir(old_cwd)
        if ret != 0:
            raise RuntimeError(f"inference_init failed with code {ret}")

    def __call__(self, *inputs: torch.Tensor) -> List[torch.Tensor]:
        """Run inference. Inputs/outputs are PyTorch tensors (host memory)."""
        # Prepare inputs as contiguous numpy arrays
        np_inputs = []
        shape_bufs = []
        input_tensors = (TensorT * len(inputs))()

        for i, tensor in enumerate(inputs):
            t = tensor.detach().cpu()
            # Convert bf16→f16 since numpy doesn't support bfloat16
            if t.dtype == torch.bfloat16:
                t = t.to(torch.float16)
            arr = t.contiguous().numpy()
            np_inputs.append(arr)
            shape = (ctypes.c_int64 * len(arr.shape))(*arr.shape)
            shape_bufs.append(shape)

            input_tensors[i].data = arr.ctypes.data
            input_tensors[i].shape = shape
            input_tensors[i].rank = len(arr.shape)
            input_tensors[i].element_size = arr.itemsize

        input_span = SpanT(data=input_tensors, count=len(inputs))

        # Prepare outputs
        np_outputs = []
        out_shape_bufs = []
        output_tensors = (TensorT * len(self.output_metas))()

        _DTYPE_MAP = {2: np.float16, 4: np.float32, 8: np.int64, 1: np.uint8}
        for i, meta in enumerate(self.output_metas):
            shape = meta.get("shape", [1])
            elem_size = meta.get("element_size", 2)
            arr = np.zeros(shape, dtype=_DTYPE_MAP.get(elem_size, np.float16))
            np_outputs.append(arr)

            shape_arr = (ctypes.c_int64 * len(shape))(*shape)
            out_shape_bufs.append(shape_arr)

            output_tensors[i].data = arr.ctypes.data
            output_tensors[i].shape = shape_arr
            output_tensors[i].rank = len(shape)
            output_tensors[i].element_size = elem_size

        output_span = SpanT(data=output_tensors, count=len(self.output_metas))

        # Execute
        ret = self._compute(
            self.state, ctypes.byref(input_span), ctypes.byref(output_span)
        )
        if ret != 0:
            raise RuntimeError(f"inference_compute failed with code {ret}")

        return [torch.from_numpy(arr.copy()) for arr in np_outputs]

    def close(self):
        if self.state:
            self._cleanup(self.state)
            self.state = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self):
        return (
            f"HipDllRunner(inputs={len(self.input_metas)}, "
            f"outputs={len(self.output_metas)})"
        )
