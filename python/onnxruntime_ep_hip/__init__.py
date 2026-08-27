#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""hipgpu Execution Provider for ONNX Runtime -- packaging marker.

This distribution carries no Python API and no payload in this package. Its
native artifacts (the EP plugin ``hipgpu.dll``, which carries the hip-compiler
JIT inside it, the per-arch custom-kernels, and the AMD GPU umbrella chain
``amdgpu-ep.dll`` + ``hip-backend.dll``) are installed directly into
``onnxruntime/capi/`` -- next to ``onnxruntime.dll`` -- by the wheel. The ROCm
import libraries come from ``rocm[devel]``.

Install this wheel AFTER ``onnxruntime`` so ``onnxruntime/capi/`` exists. Then:

- ONNX Runtime GenAI (OGA) discovers and registers the EP automatically (it
  searches next to ``onnxruntime.dll``).
- ONNX Runtime: register the colocated plugin, e.g.
  ``ort.register_execution_provider_library("hipgpu",
  <onnxruntime capi>/hipgpu.dll)``.

ROCm runtime DLLs must be on ``PATH`` (e.g. from ``rocm[libraries]``).
"""
