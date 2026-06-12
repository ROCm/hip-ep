#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""MorphiZen Execution Provider for ONNX Runtime -- packaging marker.

This distribution carries no Python API and no payload in this package. Its
native artifacts (the EP plugin ``onnxruntime_morphizen_ep.dll``, its
hip-compiler JIT plugin, and the custom-kernels + CRT import libraries the JIT
linker resolves) are installed directly into ``onnxruntime/capi/`` -- next to
``onnxruntime.dll`` -- by the wheel. The ROCm import libraries come from
``rocm[devel]``.

Install this wheel AFTER ``onnxruntime`` so ``onnxruntime/capi/`` exists. Then:

- ONNX Runtime GenAI (OGA) discovers and registers the EP automatically (it
  searches next to ``onnxruntime.dll``); reference it by the provider name
  ``MorphiZenEP`` in genai_config.
- ONNX Runtime: register the colocated plugin, e.g.
  ``ort.register_execution_provider_library("MorphiZenEP",
  <onnxruntime capi>/onnxruntime_morphizen_ep.dll)``.

At inference the JIT linker needs the ROCm import libs (from ``rocm[devel]`` via
``THEROCK_DIST``) and ``onnxruntime/capi`` on ``LIB``; ROCm runtime DLLs must be
on ``PATH`` (e.g. from ``rocm[libraries]``).
"""
