#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""hipep Execution Provider for ONNX Runtime -- packaging marker.

This distribution carries no Python API and no payload in this package. Its
native artifacts (the EP plugin ``hipep.dll``, its hip-compiler JIT plugin, the
per-arch custom-kernels, and the AMD GPU umbrella chain ``amdgpu-ep.dll`` +
``hipep-backend.dll``) are installed directly into ``onnxruntime/capi/`` -- next
to ``onnxruntime.dll`` -- by the wheel. The ROCm import libraries come from
``rocm[devel]``.

Install this wheel AFTER ``onnxruntime`` so ``onnxruntime/capi/`` exists. Then:

- ONNX Runtime GenAI (OGA) discovers and registers the EP automatically (it
  searches next to ``onnxruntime.dll``).
- ONNX Runtime: register the colocated plugin, e.g.
  ``ort.register_execution_provider_library("hipep",
  <onnxruntime capi>/hipep.dll)``.

ROCm runtime DLLs must be on ``PATH`` (e.g. from ``rocm[libraries]``).
"""
