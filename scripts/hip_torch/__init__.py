#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""
hip_torch: PyTorch frontend for the HIP MLIR compiler.

Provides tools to compile PyTorch models to GPU DLLs via MLIR:
  - op_registry: Declarative op support registry
  - compiler: hip-compiler.exe orchestration
  - dll_cache: Hash-based persistent DLL cache
  - dll_runner: ctypes-based in-process DLL execution
  - fx_emitter: FX graph → Torch dialect MLIR conversion
  - model_adapter: Generic HuggingFace model adaptation
  - backend: torch.compile custom backend
"""

__version__ = "0.1.0"
