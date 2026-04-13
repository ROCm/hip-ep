#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""
torch.compile custom backend for the HIP MLIR compiler.

Partitions the FX graph into subgraphs:
  - Supported ops → compiled through hip-compiler (MLIR → GPU DLL)
  - Unsupported ops → fallback to PyTorch eager execution

Usage:
    import torch
    from hip_backend import hip_backend

    model = ...
    compiled = torch.compile(model, backend=hip_backend)
    output = compiled(input)
"""

import logging
import os
import subprocess
import tempfile
from typing import Callable, List, Optional

import torch

log = logging.getLogger("hip_backend")

# Whether to actually compile subgraphs to DLLs (vs just classify them)
_compile_enabled = False

# Paths
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
_BUILD_DIR = os.path.join(os.path.dirname(_PROJECT_ROOT), "build", "onnx-hipdnn-ep")
_HIP_COMPILER = os.path.join(_BUILD_DIR, "bin", "hip-compiler.exe")
_THEROCK_DIST = os.environ.get(
    "THEROCK_DIST", os.path.join(os.path.dirname(_PROJECT_ROOT), "therock")
)


def enable_compilation(enabled: bool = True):
    """Enable/disable actual DLL compilation for supported subgraphs."""
    global _compile_enabled
    _compile_enabled = enabled


# ──────────────────────────────────────────────────────────────────────
# Op support registry
# ──────────────────────────────────────────────────────────────────────

# Ops we have TorchToHip conversion patterns for (after decomposition)
_SUPPORTED_OPS = {
    # Elementwise (tensor-tensor only, not tensor-scalar)
    "aten.add.Tensor",
    "aten.mul.Tensor",
    "aten.sub.Tensor",
    # Matmul
    "aten.mm.default",
    "aten.bmm.default",
    "aten.matmul.default",
    "aten.linear.default",
    # Activation
    "aten.sigmoid.default",
    "aten.silu.default",
    "aten._softmax.default",
    "aten.softmax.int",
    # Norm (decomposed RMSNorm components)
    "aten.pow.Tensor_Scalar",
    "aten.mean.dim",
    "aten.rsqrt.default",
    "aten.rms_norm.default",
    # Shape (zero-cost metadata ops)
    "aten.view.default",
    "aten.reshape.default",
    "aten.permute.default",
    "aten.transpose.int",
    "aten.unsqueeze.default",
    "aten.squeeze.dim",
    "aten.expand.default",
    "aten.slice.Tensor",
    "aten.cat.default",
    "aten.split.Tensor",
    # Type conversion
    "aten._to_copy.default",
    "aten.to.dtype",
    # Elementwise unary
    "aten.neg.default",
    "aten.cos.default",
    "aten.sin.default",
    # Other
    "aten.clone.default",
    "aten.alias.default",
    # Reduction
    "aten.sum.dim_IntList",
    # Attention
    "aten.scaled_dot_product_attention.default",
    # Boolean/mask ops (compile-time constants, passed through)
    "aten.le.Tensor",
    "aten.le.Scalar",
    "aten.bitwise_and.Tensor",
    "aten.logical_not.default",
    "aten.eq.Scalar",
    "aten.ne.Scalar",
    "aten.where.self",
    "aten.any.default",
    "aten.full.default",
    "aten.full_like.default",
    "aten.scalar_tensor.default",
    # Index/creation (handled at compile time)
    "aten.arange.default",
    "aten.arange.start",
    "aten.arange.start_step",
    "aten.embedding.default",
    # Misc
    "aten.index.Tensor",
}

# Ops that are "free" — they don't need GPU compute but should be allowed
# in supported subgraphs (they get lowered to tensor metadata ops)
_FREE_OPS = {
    "aten.view.default",
    "aten.reshape.default",
    "aten.permute.default",
    "aten.transpose.int",
    "aten.unsqueeze.default",
    "aten.squeeze.dim",
    "aten.expand.default",
    "aten.slice.Tensor",
    "aten.clone.default",
    "aten.alias.default",
    "aten._to_copy.default",
}


def _get_op_name(node) -> str:
    """Extract the aten op name from an FX node target."""
    target = node.target
    if hasattr(target, "__name__"):
        return target.__name__
    s = str(target)
    # Handle: aten.mm.default -> aten.mm.default
    if "aten." in s:
        return s.split("torch.ops.")[-1] if "torch.ops." in s else s
    return s


# Built-in Python ops that torch.compile injects (always allowed)
_BUILTIN_OPS = {
    "<built-in function getitem>",
    "getitem",
    "operator.getitem",
    "_operator.getitem",
}

# Framework ops that can be ignored (no compute semantics)
_SKIP_OPS = {
    "_log_api_usage_once",
    "torch._C._log_api_usage_once",
    "_assert_tensor_metadata",
}


def _is_supported(node) -> bool:
    """Check if a single FX node's op is supported by our HIP backend."""
    if node.op != "call_function":
        return True  # placeholders, output, get_attr are always fine

    name = _get_op_name(node)
    target_str = str(node.target)

    # Built-in ops (getitem, etc.) are always supported
    if target_str in _BUILTIN_OPS or name in _BUILTIN_OPS:
        return True
    if "getitem" in target_str or "and_" in name:
        return True

    # Skip framework ops
    if name in _SKIP_OPS or any(s in target_str for s in _SKIP_OPS):
        return True

    # Check direct support
    if name in _SUPPORTED_OPS:
        return True

    # Check if it's a variant we support (e.g. aten.mul.Tensor matches mul.Tensor)
    short = name.split("aten.")[-1] if "aten." in name else name
    for supported in _SUPPORTED_OPS:
        if short in supported:
            return True

    # Check with "aten." prefix
    if f"aten.{name}" in _SUPPORTED_OPS:
        return True

    return False


def _check_subgraph_support(gm: torch.fx.GraphModule) -> tuple:
    """Check what percentage of a subgraph's ops we support.

    Returns (supported_ops, unsupported_ops, total_compute_ops).
    """
    supported = []
    unsupported = []

    for node in gm.graph.nodes:
        if node.op != "call_function":
            continue
        name = _get_op_name(node)
        if _is_supported(node):
            supported.append(name)
        else:
            unsupported.append(name)

    return supported, unsupported, len(supported) + len(unsupported)


# ──────────────────────────────────────────────────────────────────────
# Backend implementation
# ──────────────────────────────────────────────────────────────────────

# Global counters for reporting
_stats = {
    "total_subgraphs": 0,
    "compiled_subgraphs": 0,
    "fallback_subgraphs": 0,
    "total_ops": 0,
    "supported_ops": 0,
    "unsupported_ops": 0,
    "unsupported_op_names": set(),
}


def get_stats() -> dict:
    """Get compilation statistics."""
    return dict(_stats)


def reset_stats():
    """Reset compilation statistics."""
    _stats.update(
        {
            "total_subgraphs": 0,
            "compiled_subgraphs": 0,
            "fallback_subgraphs": 0,
            "total_ops": 0,
            "supported_ops": 0,
            "unsupported_ops": 0,
            "unsupported_op_names": set(),
        }
    )


def _hip_compiler_backend(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    """The core backend: receives a subgraph, decides compile or fallback."""
    _stats["total_subgraphs"] += 1

    supported, unsupported, total = _check_subgraph_support(gm)
    _stats["total_ops"] += total
    _stats["supported_ops"] += len(supported)
    _stats["unsupported_ops"] += len(unsupported)
    _stats["unsupported_op_names"].update(unsupported)

    if unsupported:
        _stats["fallback_subgraphs"] += 1
        log.info(
            f"Subgraph {_stats['total_subgraphs']}: FALLBACK "
            f"({len(supported)}/{total} supported, "
            f"unsupported: {set(unsupported)})"
        )
        return gm.forward

    _stats["compiled_subgraphs"] += 1
    log.info(f"Subgraph {_stats['total_subgraphs']}: HIP COMPILED ({total} ops)")

    # Try to compile to DLL and return a wrapper
    if _compile_enabled:
        try:
            compiled_fn = _compile_subgraph(
                gm, example_inputs, _stats["total_subgraphs"]
            )
            if compiled_fn is not None:
                return compiled_fn
        except Exception as e:
            log.warning(
                f"Subgraph {_stats['total_subgraphs']}: "
                f"compilation failed ({e}), falling back to eager"
            )

    return gm.forward


def _compile_subgraph(
    gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor], subgraph_id: int
) -> Optional[Callable]:
    """Compile a supported subgraph to a GPU DLL and return a wrapper.

    Steps:
    1. Export FX graph to torch.export ExportedProgram
    2. Convert to Torch dialect MLIR via fx_to_mlir
    3. Compile MLIR to DLL via hip-compiler
    4. Load DLL and return a callable wrapper
    """
    from fx_to_mlir import fx_graph_to_mlir
    from hip_dll_runner import HipDllRunner

    # Create work directory for this subgraph
    work_dir = os.path.join(
        tempfile.gettempdir(), "hip_backend", f"subgraph_{subgraph_id}"
    )
    os.makedirs(work_dir, exist_ok=True)

    # Step 1: Export the subgraph
    # The gm is already an FX GraphModule. We need to wrap it as an
    # ExportedProgram for fx_to_mlir. For now, use torch.export.export.
    try:
        ep = torch.export.export(gm, tuple(example_inputs), strict=False)
    except Exception as e:
        log.warning(f"torch.export failed for subgraph {subgraph_id}: {e}")
        return None

    # Step 2: Convert to MLIR
    try:
        mlir_text = fx_graph_to_mlir(ep, decompose=False)  # already decomposed
    except Exception as e:
        log.warning(f"MLIR generation failed for subgraph {subgraph_id}: {e}")
        return None

    mlir_path = os.path.join(work_dir, "model.mlir")
    with open(mlir_path, "w") as f:
        f.write(mlir_text)

    # Step 3: Compile to DLL
    dll_path = os.path.join(work_dir, "model.dll")
    compile_cmd = os.path.join(work_dir, "_compile.cmd")
    with open(compile_cmd, "w") as f:
        f.write("@echo off\n")
        f.write(
            'call "C:\\Program Files\\Microsoft Visual Studio\\18\\'
            'Community\\VC\\Auxiliary\\Build\\vcvarsall.bat" x64 >nul 2>&1\n'
        )
        f.write(f"set THEROCK_DIST={_THEROCK_DIST}\n")
        f.write(f"set PATH={_THEROCK_DIST}\\bin;%PATH%\n")
        f.write(f'"{_HIP_COMPILER}" "{mlir_path}" -o "{dll_path}"\n')


    result = subprocess.run(
        ["cmd", "/c", compile_cmd], capture_output=True, text=True, timeout=120
    )
    if result.returncode != 0 or not os.path.exists(dll_path):
        log.warning(
            f"hip-compiler failed for subgraph {subgraph_id}: {result.stderr[-200:]}"
        )
        return None

    log.info(f"Subgraph {subgraph_id}: compiled to {dll_path}")

    # Step 4: Load DLL and create wrapper
    os.environ["THEROCK_DIST"] = _THEROCK_DIST
    try:
        runner = HipDllRunner(dll_path, work_dir=work_dir)
    except Exception as e:
        log.warning(f"DLL loading failed for subgraph {subgraph_id}: {e}")
        return None

    log.info(f"Subgraph {subgraph_id}: DLL loaded, {runner}")

    # Return a callable that matches the FX graph's signature
    def compiled_forward(*args):
        outputs = runner(*args)
        return tuple(outputs) if len(outputs) > 1 else outputs[0]

    return compiled_forward


def hip_backend(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    """torch.compile backend entry point."""
    return _hip_compiler_backend(gm, example_inputs)
