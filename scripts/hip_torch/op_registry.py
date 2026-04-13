#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""
Op registry: single source of truth for supported torch ops.

Each op has metadata describing its category, argument structure,
and test shapes. This drives:
  - torch.compile backend (which ops to compile vs fallback)
  - FX-to-MLIR emitter (per-op emission rules)
  - Test generation (auto-generated LIT and accuracy tests)

In the future, this file will be auto-generated from C++ REGISTER_TORCH_OP
macros via tools/hip-op-registry-gen.py. For now, it's hand-maintained.
"""

from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Tuple
import hashlib


@dataclass(frozen=True)
class OpInfo:
    """Metadata for a supported torch op."""

    category: str  # e.g. "elementwise", "matmul", "activation", "shape"
    required_args: int  # minimum arg count
    optional_args: int = 0  # trailing optional args
    default_values: Tuple = ()  # defaults for optional args, e.g. ((1,),) for alpha
    free: bool = False  # True = metadata-only op (no GPU compute)
    test_shapes: str = ""  # e.g. "2x3xf16,2x3xf16->2x3xf16"
    hip_op: str = ""  # expected HIP dialect op, e.g. "hip.add"


# ──────────────────────────────────────────────────────────────────────
# Op Registry
# ──────────────────────────────────────────────────────────────────────

OP_REGISTRY: Dict[str, OpInfo] = {
    # ── Elementwise (tensor-tensor) ──────────────────────────────────
    "torch.aten.add.Tensor": OpInfo(
        "elementwise", 3, 0, (), False, "2x3xf16,2x3xf16->2x3xf16", "hip.add"
    ),
    "torch.aten.mul.Tensor": OpInfo(
        "elementwise", 2, 0, (), False, "2x3xf16,2x3xf16->2x3xf16", "hip.mul"
    ),
    "torch.aten.sub.Tensor": OpInfo(
        "elementwise", 3, 0, (), False, "2x3xf16,2x3xf16->2x3xf16", "hip.sub"
    ),
    # ── Matmul ───────────────────────────────────────────────────────
    "torch.aten.mm": OpInfo(
        "matmul", 2, 0, (), False, "4x8xf16,8x16xf16->4x16xf16", "hip.matmul"
    ),
    "torch.aten.bmm": OpInfo(
        "matmul", 2, 0, (), False, "2x4x8xf16,2x8x16xf16->2x4x16xf16", "hip.matmul"
    ),
    "torch.aten.matmul": OpInfo(
        "matmul", 2, 0, (), False, "4x8xf16,8x16xf16->4x16xf16", "hip.matmul"
    ),
    "torch.aten.linear": OpInfo(
        "matmul",
        2,
        1,
        ((None,),),
        False,
        "1x4x32xf16,32x32xf16->1x4x32xf16",
        "hip.matmul",
    ),
    # ── Activation ───────────────────────────────────────────────────
    "torch.aten.sigmoid": OpInfo(
        "activation", 1, 0, (), False, "2x3xf16->2x3xf16", "hip.sigmoid"
    ),
    "torch.aten.silu": OpInfo(
        "activation", 1, 0, (), False, "2x3xf16->2x3xf16", "hip.silu"
    ),
    "torch.aten.softmax.int": OpInfo(
        "activation", 3, 0, (), False, "2x3xf16->2x3xf16", "hip.miopen.softmax"
    ),
    # ── Normalization ────────────────────────────────────────────────
    "torch.aten.rms_norm": OpInfo(
        "norm", 4, 0, (), False, "", "hip.rms_norm"  # test_shapes empty: complex args, hand-written test exists
    ),
    # ── Attention ────────────────────────────────────────────────────
    "torch.aten.scaled_dot_product_attention": OpInfo(
        "attention",
        3,
        0,
        (),
        False,
        "1x4x4x32xf16,1x4x4x32xf16,1x4x4x32xf16->1x4x4x32xf16",
        "hip.gqa",
    ),
    # ── Quantized ────────────────────────────────────────────────────
    "torch.aten.matmul_nbits": OpInfo(
        "quantized", 3, 2, (), False, "", "hip.matmul_nbits"
    ),
    "torch.aten.qmoe": OpInfo("quantized", 7, 7, (), False, "", "hip.qmoe"),
    # ── Type conversion ──────────────────────────────────────────────
    "torch.aten.to.dtype": OpInfo("cast", 2, 0, (), False, "", "hip.cast"),
    "torch.aten._to_copy": OpInfo("cast", 1, 0, (), True, "", ""),
    # ── Shape ops (free — no GPU compute) ────────────────────────────
    "torch.aten.view": OpInfo("shape", 2, 0, (), True, "", ""),
    "torch.aten.reshape": OpInfo("shape", 2, 0, (), True, "", ""),
    "torch.aten.permute": OpInfo("shape", 2, 0, (), True, "", ""),
    "torch.aten.transpose.int": OpInfo("shape", 3, 0, (), True, "", "hip.transpose"),
    "torch.aten.unsqueeze": OpInfo("shape", 2, 0, (), True, "", ""),
    "torch.aten.squeeze.dim": OpInfo("shape", 2, 0, (), True, "", ""),
    "torch.aten.expand": OpInfo("shape", 2, 0, (), True, "", ""),
    "torch.aten.slice.Tensor": OpInfo("shape", 5, 0, (), True, "", ""),
    "torch.aten.cat": OpInfo("shape", 2, 0, (), False, "", ""),
    "torch.aten.split.Tensor": OpInfo("shape", 3, 0, (), True, "", ""),
    "torch.aten.clone": OpInfo("shape", 1, 0, (), True, "", ""),
    "torch.aten.alias": OpInfo("shape", 1, 0, (), True, "", ""),
    "torch.aten.contiguous": OpInfo("shape", 1, 0, (), True, "", ""),
    # ── Reduction ────────────────────────────────────────────────────
    "torch.aten.sum.dim_IntList": OpInfo(
        "reduction", 3, 0, (), False, "", "hip.reduce_sum"
    ),
    # ── Index ops ────────────────────────────────────────────────────
    "torch.aten.index_select": OpInfo("index", 3, 0, (), False, "", "hip.gather"),
    # ── Creation/constant ops (allowed in subgraphs) ─────────────────
    "torch.aten.arange": OpInfo("creation", 1, 0, (), True, "", ""),
    "torch.aten.arange.start": OpInfo("creation", 2, 0, (), True, "", ""),
    "torch.aten.arange.start_step": OpInfo("creation", 3, 0, (), True, "", ""),
    "torch.aten.full": OpInfo("creation", 2, 0, (), True, "", ""),
    "torch.aten.full_like": OpInfo("creation", 2, 0, (), True, "", ""),
    "torch.aten.zeros": OpInfo("creation", 1, 0, (), True, "", ""),
    "torch.aten.ones": OpInfo("creation", 1, 0, (), True, "", ""),
    "torch.aten.scalar_tensor": OpInfo("creation", 1, 0, (), True, "", ""),
    "torch.aten.embedding": OpInfo("index", 2, 0, (), False, "", ""),
    # ── Boolean/mask ops ─────────────────────────────────────────────
    "torch.aten.le.Tensor": OpInfo("comparison", 2, 0, (), True, "", ""),
    "torch.aten.le.Scalar": OpInfo("comparison", 2, 0, (), True, "", ""),
    "torch.aten.eq.Scalar": OpInfo("comparison", 2, 0, (), True, "", ""),
    "torch.aten.ne.Scalar": OpInfo("comparison", 2, 0, (), True, "", ""),
    "torch.aten.bitwise_and.Tensor": OpInfo("comparison", 2, 0, (), True, "", ""),
    "torch.aten.logical_not": OpInfo("comparison", 1, 0, (), True, "", ""),
    "torch.aten.where.self": OpInfo("comparison", 3, 0, (), False, "", ""),
    "torch.aten.any": OpInfo("comparison", 1, 0, (), True, "", ""),
    # ── Unary elementwise ────────────────────────────────────────────
    "torch.aten.neg": OpInfo("elementwise", 1, 0, (), False, "2x3xf16->2x3xf16", ""),
    "torch.aten.cos": OpInfo("elementwise", 1, 0, (), False, "2x3xf32->2x3xf32", ""),
    "torch.aten.sin": OpInfo("elementwise", 1, 0, (), False, "2x3xf32->2x3xf32", ""),
    # ── Decomposed RMSNorm components ────────────────────────────────
    "torch.aten.pow.Tensor_Scalar": OpInfo("elementwise", 2, 0, (), False, "", ""),
    "torch.aten.mean.dim": OpInfo("reduction", 3, 0, (), False, "", ""),
    "torch.aten.rsqrt": OpInfo("elementwise", 1, 0, (), False, "2x3xf32->2x3xf32", ""),
    # ── Index/gather ─────────────────────────────────────────────────
    "torch.aten.index.Tensor": OpInfo("index", 2, 0, (), False, "", ""),
}

# ──────────────────────────────────────────────────────────────────────
# Derived Sets (used by backend and emitter)
# ──────────────────────────────────────────────────────────────────────

SUPPORTED_OPS: FrozenSet[str] = frozenset(OP_REGISTRY.keys())

FREE_OPS: FrozenSet[str] = frozenset(k for k, v in OP_REGISTRY.items() if v.free)

COMPUTE_OPS: FrozenSet[str] = SUPPORTED_OPS - FREE_OPS

# Built-in ops always allowed (Python operators, not aten)
BUILTIN_OPS: FrozenSet[str] = frozenset(["getitem", "operator.getitem", "and_", "or_"])

# Framework ops to skip silently (no compute, no fallback needed)
SKIP_OPS: FrozenSet[str] = frozenset(["_log_api_usage_once", "_assert_tensor_metadata"])

# ──────────────────────────────────────────────────────────────────────
# Registry Version (for cache invalidation)
# ──────────────────────────────────────────────────────────────────────

_REGISTRY_CONTENT = str(sorted(OP_REGISTRY.items()))
REGISTRY_VERSION: str = hashlib.sha256(_REGISTRY_CONTENT.encode()).hexdigest()[:12]


# ──────────────────────────────────────────────────────────────────────
# Query API
# ──────────────────────────────────────────────────────────────────────


def is_supported(op_name: str) -> bool:
    """Check if an op is supported (directly or via fuzzy match)."""
    if op_name in SUPPORTED_OPS:
        return True
    # Check builtins
    if any(b in op_name for b in BUILTIN_OPS):
        return True
    # Check skip ops
    if any(s in op_name for s in SKIP_OPS):
        return True
    # Fuzzy: try with/without overload suffix
    short = op_name.split(".")[-1] if "aten." in op_name else op_name
    for supported in SUPPORTED_OPS:
        if short in supported:
            return True
    return False


def get_info(op_name: str) -> Optional[OpInfo]:
    """Get metadata for a supported op."""
    return OP_REGISTRY.get(op_name)


def get_ops_by_category(category: str) -> List[str]:
    """Get all ops in a category."""
    return [k for k, v in OP_REGISTRY.items() if v.category == category]


def get_default_values(op_name: str) -> Tuple:
    """Get default trailing arg values for an op (for emitter padding)."""
    info = OP_REGISTRY.get(op_name)
    return info.default_values if info else ()
