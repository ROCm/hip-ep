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
    """Runs a compiled HIP model DLL via hip-test-dll subprocess.

    Uses hip-test-dll.exe as the execution engine since it handles
    the DiskFileSystem, GPU memory management, and tensor marshaling.
    This avoids needing to replicate the C++ FileSystem class in Python.

    For production, this would be replaced with direct ctypes calls
    using a Python-accessible filesystem binding.
    """

    def __init__(self, dll_path: str, work_dir: Optional[str] = None):
        self.dll_path = os.path.abspath(dll_path)
        self.work_dir = work_dir or os.path.dirname(self.dll_path)
        self._therock = os.environ.get(
            "THEROCK_DIST",
            "C:\\Users\\tsiddaga\\Documents\\code\\therock")

        # Find hip-test-dll.exe
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        self._runner = os.path.join(
            os.path.dirname(project_root), "build",
            "onnx-hipdnn-ep", "bin", "hip-test-dll.exe")

        # Parse metadata from DLL
        self.metadata = self._get_metadata()
        self.input_metas = self.metadata.get("inputs", [])
        self.output_metas = self.metadata.get("outputs", [])

    def _get_metadata(self) -> dict:
        """Extract metadata JSON from the DLL via hip-test-dll."""
        import subprocess
        cmd = (f'set PATH={self._therock}\\bin;%PATH% && '
               f'"{self._runner}" "{self.dll_path}" --verbose 2>&1')

        # Run just to get metadata (it will also execute, which is fine)
        result = subprocess.run(
            ["cmd", "/c", cmd],
            capture_output=True, text=True, timeout=30,
            cwd=self.work_dir)

        # Parse JSON from output
        for line in result.stdout.split("\n"):
            if line.strip().startswith("{"):
                try:
                    return json.loads(line.strip() +
                                    result.stdout[result.stdout.index(line) +
                                                  len(line):].split("}")[0] + "}")
                except (json.JSONDecodeError, ValueError):
                    pass

        # Try to find multi-line JSON
        in_json = False
        json_lines = []
        for line in result.stdout.split("\n"):
            stripped = line.strip()
            # Remove test output prefix like "1: " or "49: "
            if ":" in stripped and stripped.split(":")[0].strip().isdigit():
                stripped = ":".join(stripped.split(":")[1:]).strip()
            if stripped == "{":
                in_json = True
                json_lines = ["{"]
            elif in_json:
                json_lines.append(stripped)
                if stripped == "}":
                    try:
                        return json.loads("\n".join(json_lines))
                    except json.JSONDecodeError:
                        in_json = False

        return {"inputs": [], "outputs": []}

    def run(self) -> dict:
        """Execute the model with default test data via hip-test-dll.

        Returns dict with output values.
        """
        import subprocess
        cmd_path = os.path.join(self.work_dir, "_run.cmd")
        with open(cmd_path, "w") as f:
            f.write("@echo off\n")
            f.write(f'set PATH={self._therock}\\bin;%PATH%\n')
            f.write(f'"{self._runner}" "{self.dll_path}" '
                    f'--verbose --validate\n')

        result = subprocess.run(
            ["cmd", "/c", cmd_path],
            capture_output=True, text=True, timeout=30,
            cwd=self.work_dir)

        success = "SUCCESS" in result.stdout
        output_vals = []
        for line in result.stdout.split("\n"):
            if "First 10 values:" in line:
                vals_str = line.split("First 10 values:")[1].strip()
                output_vals = [float(v) for v in vals_str.split() if v]

        return {
            "success": success,
            "output_values": output_vals,
            "returncode": result.returncode,
        }

    def close(self):
        pass

    def __repr__(self):
        return (f"HipDllRunner('{os.path.basename(self.dll_path)}', "
                f"inputs={len(self.input_metas)}, "
                f"outputs={len(self.output_metas)})")
