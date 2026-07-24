#!/usr/bin/env python3
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Identify missing / unsupported ONNX ops for hip-ep model runs.

Runs ``onnxruntime_perf_test`` (or parses existing logs / MLIR IR dumps) and
extracts ONNX ops that block hip-ep compilation or runtime dispatch.

Typical failure signatures:
  - ``error: op was not bufferized`` — unconverted ``onnx.*`` survived to bufferize
  - surviving ``"onnx.OpName"`` in MLIR IR dumps (``HIPDNN_EP_IR_DUMP_TREE=1``)
  - ``Compilation failed`` / ``MLIR pass pipeline failed``
  - ``[REAL] wrap_*: unsupported ...`` — runtime kernel variant gaps
  - ORT session errors when ``session.disable_cpu_ep_fallback|1`` is set

USAGE
-----
  # Smoke-run one model (short, CPU fallback disabled)
  python .cursor/skills/identify-missing-ops/scripts/identify_missing_ops.py model.onnx

  # Batch over a model tree
  python .cursor/skills/identify-missing-ops/scripts/identify_missing_ops.py "<models-dir>" --glob "*.onnx"
  # Batch output includes a BATCH ROLLUP section (by model + by op) at the top.

  # Parse an existing perf_test log
  python .cursor/skills/identify-missing-ops/scripts/identify_missing_ops.py --parse-log run.log

  # Parse MLIR dump tree (HIPDNN_EP_IR_DUMP_TREE output)
  python .cursor/skills/identify-missing-ops/scripts/identify_missing_ops.py --parse-mlir-dir output_test.1

  # Static ONNX scan vs hip-ep converters (no GPU run)
  python .cursor/skills/identify-missing-ops/scripts/identify_missing_ops.py --scan-onnx model.onnx

Windows (gpu-test-package layout):
  cd gpu-test-package\\bin
  python ..\\.cursor\\skills\\identify-missing-ops\\scripts\\identify_missing_ops.py model.onnx
  # Monorepo workspace (hip-ep is a subfolder): use ..\\..\\.cursor\\skills\\identify-missing-ops\\scripts\\...

Default stdout is a structured compatibility summary. Use --quiet for op names only.
Use --show-next-step to include recommended work items in gap analysis tables.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
_ONNX_TO_HIP = Path("lib") / "Conversion" / "OnnxToHip"


def _resolve_hip_ep_root(script_dir: Path) -> Path:
    """Locate hip-ep root whether workspace is hip-ep or a parent monorepo."""
    for parent in script_dir.parents:
        if (parent / _ONNX_TO_HIP).is_dir():
            return parent
        sibling = parent / "hip-ep"
        if (sibling / _ONNX_TO_HIP).is_dir():
            return sibling
    sys.exit(
        "Could not locate hip-ep root (expected lib/Conversion/OnnxToHip). "
        "Open hip-ep or its parent monorepo as the workspace."
    )


HIP_EP_ROOT = _resolve_hip_ep_root(SCRIPT_DIR)
ONNX_TO_HIP_DIR = HIP_EP_ROOT / "lib" / "Conversion" / "OnnxToHip"

# --- log / IR parsing patterns ------------------------------------------------

RE_SURVIVING_ONNX_OP = re.compile(r'"onnx\.(\w+)"\s*\(')
RE_ONNX_NODE_NAME = re.compile(r'onnx_node_name\s*=\s*"([^"]+)"')
RE_REAL_UNSUPPORTED = re.compile(
    r"\[REAL\]\s+(wrap_\w+):\s*unsupported\s+(.+)", re.IGNORECASE
)
RE_COMPILATION_FAILED = re.compile(
    r"(?:MLIR compilation failed|Compilation failed|MLIR pass pipeline failed)",
    re.IGNORECASE,
)
RE_BUFFERIZE_FAILURE = re.compile(r"error:\s*op was not bufferized", re.IGNORECASE)
RE_CPU_FALLBACK = re.compile(
    r"(nodes assigned to CPU|Session creation failed|Inference failed|fallback disabled)",
    re.IGNORECASE,
)
RE_ONNX_REWRITE_PATTERN = re.compile(r'RewritePattern\(\s*"onnx\.(\w+)"')
# com.microsoft ops lower via onnx.Custom + function_name= in Conversion/*.cpp.
RE_MS_CUSTOM_FUNC = re.compile(
    r'funcNameAttr(?:\.getValue\(\))?\s*(?:!=|==)\s*"([A-Za-z0-9_]+)"'
)
RE_MLIR_CUSTOM_FUNC = re.compile(r'function_name\s*=\s*"([A-Za-z0-9_]+)"')

SUCCESS_QPS = re.compile(
    r"Number of inferences per second:\s+"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
)

# Ops handled by non-RewritePattern passes (verified in lib/Conversion/OnnxToHip).
_STRUCTURAL_KNOWN_OPS = frozenset({"Loop"})

# Pre-lowering fusion on primitive chains (FastGeluFusion.cpp / ErfGeluFusion.cpp).
# Does NOT cover com.microsoft FastGelu/QuickGelu exported as onnx.Custom.
_FUSION_HANDLING: dict[str, str] = {
    "Gelu": "onnx.Gelu -> hip.gelu (ActivationConversion)",
    "Erf": "ErfGeluFusion -> onnx.Gelu (inlined export chain only)",
}

# Partial fusion: applies only to specific export forms, not com.microsoft Custom ops.
_PARTIAL_FUSION_OPS: dict[str, str] = {
    "FastGelu": "no (Custom op); partial fusion for inlined Pow/Mul/Tanh chain only",
    "GemmFastGelu": "no (Custom op); partial fusion for inlined chain only",
}

# Actionable next-step hints for common gap patterns (OnnxToHip / Runtime / model).
_GAP_NEXT_STEP: dict[str, str] = {
    "Greater": (
        "OnnxToHip: add GreaterConversion (decompose like GreaterOrEqual -> "
        "hip.not+hip.less) | Runtime: reuse hip_elementwise_less + hip.not"
    ),
    "IsInf": (
        "OnnxToHip: add IsInfConversion.cpp | Runtime: add wrap_isinf + "
        "hip_elementwise_isinf kernel"
    ),
    "Abs": (
        "OnnxToHip: add AbsConversion.cpp | Runtime: add wrap_abs + kernel "
        "(or float decompose via Sqrt+Mul; model uses i64)"
    ),
    "Log": (
        "OnnxToHip: add LogConversion.cpp | Runtime: add wrap_log + "
        "hip_elementwise_log kernel"
    ),
    "FastGelu": (
        "OnnxToHip: add FastGelu Custom converter -> hip.gelu | "
        "Alt: re-export with inlined Gelu chain for FastGeluFusion | "
        "Runtime: wrap_gelu exists"
    ),
    "QuickGelu": (
        "OnnxToHip: add QuickGelu Custom converter (decompose to Mul+Sigmoid "
        "or map to hip.gelu) | Runtime: wrap_gelu exists; no QuickGelu kernel"
    ),
    "BiasGelu": (
        "OnnxToHip: add BiasGelu Custom converter | Runtime: likely wrap_gelu + add"
    ),
    "GemmFastGelu": (
        "OnnxToHip: add GemmFastGelu Custom converter or fusion | "
        "Runtime: wrap_gelu + wrap_gemm exist separately"
    ),
}

# Converter exists but pattern match fails on dynamic spatial shapes (H/W or 1D static).
_SHAPE_BLOCKER_NEXT_STEP: dict[str, str] = {
    "Conv": (
        "Extend ConvConversion.cpp for dynamic spatial output dims (H/W), or "
        "re-export ONNX with fixed latent/image resolution (static H/W)"
    ),
    "Resize": (
        "Extend ResizeConversion.cpp for dynamic output spatial dims, or "
        "re-export ONNX with fixed upsampler output sizes"
    ),
}

RE_MLIR_ONNX_OP_TYPES = re.compile(
    r'"onnx\.(\w+)"\([^)]*\)[^{]*\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}\s*:\s*\(([^)]*)\)\s*->\s*(.+)$'
)
RE_TENSOR_TYPE = re.compile(r"tensor<([^>]+)>")

MORPHIZEN_SCHEMA_FILE = (
    HIP_EP_ROOT
    / "morphizen"
    / "morphizen-core"
    / "src"
    / "binary"
    / "onnx_schema_json_binary.hpp"
)
RUNTIME_HEADER = HIP_EP_ROOT / "lib" / "Runtime" / "hipdnn_ep_runtime.h"
KERNELS_HEADER = (
    HIP_EP_ROOT / "lib" / "Runtime" / "Kernels" / "include" / "hip_custom_kernels.h"
)
RUNTIME_REAL_DIR = HIP_EP_ROOT / "lib" / "Runtime" / "real"

RE_MORPHIZEN_SCHEMA = re.compile(r'RegisterSchema\(OpSchema\("([A-Za-z0-9_]+)"')
RE_WRAP_FN = re.compile(r"\bint wrap_(\w+)\(")
RE_ELEM_KERNEL = re.compile(r"\bhip_elementwise_(\w+)\(")

# wrap_* / kernel suffix -> ONNX op name
_WRAP_SUFFIX_TO_OP: dict[str, str] = {
    "gelu": "Gelu",
    "less": "Less",
    "equal": "Equal",
    "not": "Not",
    "neg": "Neg",
    "sign": "Sign",
    "sqrt": "Sqrt",
    "exp": "Exp",
    "cos": "Cos",
    "sin": "Sin",
    "and": "And",
    "mod": "Mod",
    "matmul_nbits": "MatMulNBits",
    "gather_block_quantized": "GatherBlockQuantized",
    "group_query_attention": "GroupQueryAttention",
    "multi_head_attention": "MultiHeadAttention",
    "rotary_embedding": "RotaryEmbedding",
    "linear_attention": "LinearAttention",
    "qmoe": "QMoE",
    "causal_conv_with_state": "CausalConvWithState",
    "layer_normalization": "LayerNormalization",
    "skip_simplified_layer_norm": "SkipSimplifiedLayerNormalization",
}

_RUNTIME_PARTIAL: dict[str, str] = {
    "Greater": "partial (hip.less + hip.not; no wrap_greater)",
    "FastGelu": "partial (wrap_gelu for Gelu path; no FastGelu Custom wrap)",
    "QuickGelu": "partial (wrap_gelu; no QuickGelu-specific kernel)",
    "GemmFastGelu": "partial (wrap_gelu + wrap_gemm separately)",
    "BiasGelu": "partial (wrap_gelu + elementwise add likely)",
}

INVENTORY_STATUS_LABELS = {
    "compile_blocker": "COMPILE BLOCKER (survived lowering)",
    "shape_blocker": "SHAPE BLOCKER (converter exists; dynamic spatial dims)",
    "variant_blocker": "VARIANT BLOCKER (converter exists; unsupported attributes/shape)",
    "dtype_blocker": "DTYPE BLOCKER (converter exists; unsupported element type)",
    "runtime_blocker": "RUNTIME BLOCKER (compiled; kernel rejected variant)",
    "partial": "PARTIAL (no converter; some runtime pieces exist)",
    "lowered": "LOWERED by hip-ep",
    "fused": "FUSED by hip-ep",
    "lowered_incomplete": "LOWERED (converter in repo; still in MLIR at failure)",
    "unsupported": "UNSUPPORTED (no converter)",
    "lowered_unverified": "LOWERED (repo converter; no MLIR dump to confirm)",
}

BLOCKING_STATUSES = frozenset(
    {
        "compile_blocker",
        "shape_blocker",
        "variant_blocker",
        "dtype_blocker",
        "runtime_blocker",
        "partial",
        "unsupported",
        "lowered_incomplete",
    }
)

# ONNX element types that commonly block converters despite op support existing.
_DTYPE_BLOCKER_RULES: dict[str, frozenset[str]] = {
    "Abs": frozenset({"int64", "int32", "uint64", "uint32"}),
    "Sign": frozenset({"int64", "int32"}),
    "Neg": frozenset({"int64", "int32"}),
    "Cast": frozenset({"double", "float64", "bfloat16"}),
}

RE_MLIR_ATTR = re.compile(r'(\w+)\s*=\s*(?:"([^"]*)"|\[([^\]]*)\]|([^,}\s]+))')

# Infrastructure ops lowered before bufferize — not tracked in inventory.
_IGNORE_ONNX_OPS = frozenset({"Constant", "CastLike"})

FAILURE_STAGE_LABELS = {
    "none": "none (model ran on hip-ep)",
    "native_crash": "native crash (before EP output)",
    "mlir_bufferize": "MLIR one-shot-bufferize",
    "mlir_compile": "MLIR compile pipeline (OnnxToHip / passes)",
    "ort_session": "ORT session (graph not fully on hip-ep)",
    "runtime_kernel": "runtime kernel dispatch",
    "static_analysis": "static ONNX analysis (no hip-ep converter)",
    "unknown": "unknown",
}

SOURCE_LABELS = {
    "onnx_static": "ONNX graph (static scan)",
    "mlir_dump": "MLIR IR (survived lowering)",
    "perf_test_log": "compile / run log",
    "log": "log file",
}


@dataclass
class OpFinding:
    op: str
    source: str
    detail: str = ""
    count: int = 1

    def key(self) -> tuple[str, str, str]:
        return (self.op, self.source, self.detail)


@dataclass
class OpGapLayers:
    """Per-op hip-ep stack coverage for gap / roadmap reporting."""

    schema: str
    onnx_converter: str
    fusion: str
    runtime: str
    next_step: str


@dataclass
class ShapeBlockerDetail:
    """Converter exists but lowering blocked (shape / variant / dtype / runtime)."""

    reason: str
    count: int = 0
    example: str = ""
    source: str = "mlir"  # mlir | onnx | log
    kind: str = "shape"  # shape | variant | dtype | runtime


@dataclass
class OpInventoryItem:
    op: str
    count: int
    status: str
    hip_ep_handling: str
    gap: OpGapLayers | None = None


@dataclass
class ModelReport:
    model: str
    status: str  # ok | compile_failed | runtime_failed | unknown
    exit_code: int | None = None
    missing_ops: list[OpFinding] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    qps: float | None = None
    log_path: str | None = None
    mlir_dump_path: str | None = None
    hip_ep_result: str = "UNKNOWN"
    failure_level: str = "unknown"
    failure_detail: str = ""
    op_inventory: list[OpInventoryItem] = field(default_factory=list)

    @property
    def unique_missing_ops(self) -> list[str]:
        if self.op_inventory:
            return sorted(
                item.op
                for item in self.op_inventory
                if item.status in BLOCKING_STATUSES
            )
        seen: set[str] = set()
        out: list[str] = []
        for finding in self.missing_ops:
            if finding.op not in seen:
                seen.add(finding.op)
                out.append(finding.op)
        return sorted(out)

    def to_dict(self) -> dict:
        return {
            "model": _model_display_path(self.model),
            "status": self.status,
            "hip_ep_result": self.hip_ep_result,
            "failure_level": self.failure_level,
            "failure_stage": FAILURE_STAGE_LABELS.get(
                self.failure_level, self.failure_level
            ),
            "failure_detail": self.failure_detail,
            "exit_code": self.exit_code,
            "qps": self.qps,
            "log_path": self.log_path,
            "mlir_dump_path": self.mlir_dump_path,
            "missing_ops": [
                {
                    "op": f.op,
                    "source": f.source,
                    "detail": f.detail,
                    "count": f.count,
                }
                for f in self.missing_ops
            ],
            "unique_missing_ops": self.unique_missing_ops,
            "op_inventory": [
                {
                    "op": item.op,
                    "count": item.count,
                    "status": item.status,
                    "hip_ep_handling": item.hip_ep_handling,
                    "gap": (
                        {
                            "schema": item.gap.schema,
                            "onnx_converter": item.gap.onnx_converter,
                            "fusion": item.gap.fusion,
                            "runtime": item.gap.runtime,
                            "next_step": item.gap.next_step,
                        }
                        if item.gap
                        else None
                    ),
                }
                for item in self.op_inventory
            ],
            "errors": self.errors,
        }


@dataclass
class RepoOpSupport:
    """Ops with real converters discovered from lib/Conversion/OnnxToHip."""

    onnx_ops: frozenset[str]
    ms_custom_ops: frozenset[str]
    fusion_ops: frozenset[str]


_REPO_OP_SUPPORT: RepoOpSupport | None = None


def _discover_repo_op_support() -> RepoOpSupport:
    """Scan OnnxToHip sources for RewritePattern + com.microsoft Custom matchers."""
    onnx_ops: set[str] = set(_STRUCTURAL_KNOWN_OPS)
    ms_custom: set[str] = set()

    if ONNX_TO_HIP_DIR.is_dir():
        for path in ONNX_TO_HIP_DIR.rglob("*.cpp"):
            text = path.read_text(encoding="utf-8", errors="ignore")
            for match in RE_ONNX_REWRITE_PATTERN.finditer(text):
                op = match.group(1)
                if op != "Custom":
                    onnx_ops.add(op)
            for match in RE_MS_CUSTOM_FUNC.finditer(text):
                ms_custom.add(match.group(1))

    return RepoOpSupport(
        onnx_ops=frozenset(onnx_ops),
        ms_custom_ops=frozenset(ms_custom),
        fusion_ops=frozenset(_FUSION_HANDLING),
    )


def _repo_op_support() -> RepoOpSupport:
    global _REPO_OP_SUPPORT
    if _REPO_OP_SUPPORT is None:
        _REPO_OP_SUPPORT = _discover_repo_op_support()
    return _REPO_OP_SUPPORT


def _repo_known_onnx_ops() -> set[str]:
    """All ONNX + com.microsoft op types with a verified hip-ep converter."""
    support = _repo_op_support()
    return set(support.onnx_ops) | set(support.ms_custom_ops) | set(support.fusion_ops)


_MORPHIZEN_MS_SCHEMAS: frozenset[str] | None = None
_RUNTIME_BY_OP: dict[str, dict[str, str]] | None = None


def _discover_morphizen_ms_schemas() -> frozenset[str]:
    global _MORPHIZEN_MS_SCHEMAS
    if _MORPHIZEN_MS_SCHEMAS is not None:
        return _MORPHIZEN_MS_SCHEMAS
    schemas: set[str] = set()
    if MORPHIZEN_SCHEMA_FILE.is_file():
        text = MORPHIZEN_SCHEMA_FILE.read_text(encoding="utf-8", errors="ignore")
        for match in RE_MORPHIZEN_SCHEMA.finditer(text):
            schemas.add(match.group(1))
    _MORPHIZEN_MS_SCHEMAS = frozenset(schemas)
    return _MORPHIZEN_MS_SCHEMAS


def _snake_to_op_name(snake: str) -> str:
    if snake in _WRAP_SUFFIX_TO_OP:
        return _WRAP_SUFFIX_TO_OP[snake]
    parts = snake.split("_")
    return "".join(p[:1].upper() + p[1:] for p in parts if p)


def _discover_runtime_by_op() -> dict[str, dict[str, str]]:
    """Map ONNX op name -> {wrap, kernel} discovered from Runtime sources."""
    global _RUNTIME_BY_OP
    if _RUNTIME_BY_OP is not None:
        return _RUNTIME_BY_OP

    by_op: dict[str, dict[str, str]] = {}

    if RUNTIME_HEADER.is_file():
        text = RUNTIME_HEADER.read_text(encoding="utf-8", errors="ignore")
        for match in RE_WRAP_FN.finditer(text):
            wrap = f"wrap_{match.group(1)}"
            op = _snake_to_op_name(match.group(1))
            bucket = by_op.setdefault(op, {"wrap": "no", "kernel": "no"})
            bucket["wrap"] = wrap

    if KERNELS_HEADER.is_file():
        text = KERNELS_HEADER.read_text(encoding="utf-8", errors="ignore")
        for match in RE_ELEM_KERNEL.finditer(text):
            kernel = f"hip_elementwise_{match.group(1)}"
            op = _snake_to_op_name(match.group(1))
            bucket = by_op.setdefault(op, {"wrap": "no", "kernel": "no"})
            bucket["kernel"] = kernel

    # Real dispatch files (e.g. matmul_nbits.cpp) reinforce mapping.
    if RUNTIME_REAL_DIR.is_dir():
        skip = {"miopen", "hip", "hipblas", "memory", "test_hip_from_dll"}
        for path in RUNTIME_REAL_DIR.glob("*.cpp"):
            stem = path.stem
            if stem in skip:
                continue
            op = _snake_to_op_name(stem)
            by_op.setdefault(op, {"wrap": f"wrap_{stem}", "kernel": "no"})

    _RUNTIME_BY_OP = by_op
    return _RUNTIME_BY_OP


def _default_gap_next_step(
    op: str,
    *,
    has_converter: bool,
    runtime_label: str,
) -> str:
    if has_converter:
        return "Supported — no action needed"
    if op in _GAP_NEXT_STEP:
        return _GAP_NEXT_STEP[op]
    if runtime_label.startswith("partial"):
        return f"Add OnnxToHip converter for {op} (some runtime pieces already exist)"
    if runtime_label.startswith("yes"):
        return f"Add OnnxToHip converter for {op} (runtime wrap/kernel exists)"
    return f"Add OnnxToHip converter + runtime kernel for {op}"


def _analyze_op_gap(op: str) -> OpGapLayers:
    """Report schema / converter / fusion / runtime coverage and next step."""
    support = _repo_op_support()
    ms_schemas = _discover_morphizen_ms_schemas()
    runtime_map = _discover_runtime_by_op()

    if op in ms_schemas:
        schema = "yes (com.microsoft / morphizen)"
    else:
        schema = "yes (ONNX standard)"

    if op in support.ms_custom_ops:
        onnx_converter = "yes (Custom -> hip.*)"
    elif op in support.onnx_ops:
        onnx_converter = "yes (RewritePattern -> hip.*)"
    elif op in _PARTIAL_FUSION_OPS:
        onnx_converter = _PARTIAL_FUSION_OPS[op]
    else:
        onnx_converter = "no"

    if op in support.fusion_ops:
        fusion = f"yes ({_FUSION_HANDLING.get(op, 'fusion pass')})"
    elif op in _PARTIAL_FUSION_OPS:
        fusion = "partial (inlined primitive chain only)"
    else:
        fusion = "no"

    rt = runtime_map.get(op)
    if rt and rt.get("wrap", "no") != "no":
        parts = [rt["wrap"]]
        if rt.get("kernel", "no") != "no":
            parts.append(rt["kernel"])
        runtime = "yes (" + ", ".join(parts) + ")"
    elif op in _RUNTIME_PARTIAL:
        runtime = _RUNTIME_PARTIAL[op]
    else:
        runtime = "no"

    has_converter = onnx_converter.startswith("yes")
    next_step = _default_gap_next_step(
        op, has_converter=has_converter, runtime_label=runtime
    )

    return OpGapLayers(
        schema=schema,
        onnx_converter=onnx_converter,
        fusion=fusion,
        runtime=runtime,
        next_step=next_step,
    )


def _extract_surviving_ops_from_line(
    line: str,
    known: set[str],
) -> list[tuple[str, str]]:
    """Return [(op_name, detail), ...] for unlowered ops on one MLIR/log line."""
    found: list[tuple[str, str]] = []

    for match in RE_SURVIVING_ONNX_OP.finditer(line):
        op = match.group(1)
        if op in _IGNORE_ONNX_OPS or op in known:
            continue
        if op == "Custom":
            continue
        name_match = RE_ONNX_NODE_NAME.search(line)
        detail = name_match.group(1) if name_match else "surviving onnx op in IR/log"
        found.append((op, detail))

    if '"onnx.Custom"' in line:
        func_match = RE_MLIR_CUSTOM_FUNC.search(line)
        if func_match:
            custom_op = func_match.group(1)
            if custom_op not in known and custom_op not in _IGNORE_ONNX_OPS:
                name_match = RE_ONNX_NODE_NAME.search(line)
                detail = (
                    name_match.group(1) if name_match else f"com.microsoft.{custom_op}"
                )
                found.append((custom_op, detail))

    return found


def _extract_all_surviving_ops_from_line(line: str) -> list[tuple[str, str]]:
    """Like _extract_surviving_ops_from_line but without filtering known converters."""
    found: list[tuple[str, str]] = []

    for match in RE_SURVIVING_ONNX_OP.finditer(line):
        op = match.group(1)
        if op in _IGNORE_ONNX_OPS or op == "Custom":
            continue
        name_match = RE_ONNX_NODE_NAME.search(line)
        detail = name_match.group(1) if name_match else "survived in MLIR IR"
        found.append((op, detail))

    if '"onnx.Custom"' in line:
        func_match = RE_MLIR_CUSTOM_FUNC.search(line)
        if func_match:
            custom_op = func_match.group(1)
            if custom_op not in _IGNORE_ONNX_OPS:
                name_match = RE_ONNX_NODE_NAME.search(line)
                detail = (
                    name_match.group(1) if name_match else f"com.microsoft.{custom_op}"
                )
                found.append((custom_op, detail))

    return found


def _hip_ep_handled(op: str, known_ops: set[str]) -> bool:
    return op in known_ops


def _analyze_blocker_gap(op: str, detail: ShapeBlockerDetail) -> OpGapLayers:
    base = _analyze_op_gap(op)
    if detail.kind == "shape":
        next_step = _SHAPE_BLOCKER_NEXT_STEP.get(
            op,
            f"Extend {op} converter for dynamic spatial shapes or re-export with fixed H/W",
        )
    elif detail.kind == "dtype":
        next_step = (
            f"Extend {op} converter/runtime for this dtype, or insert Cast nodes "
            f"to a supported type in the ONNX export"
        )
    elif detail.kind == "runtime":
        next_step = f"Extend wrap_/kernel for {op} to handle: {detail.reason[:80]}"
    elif detail.kind == "variant":
        next_step = (
            f"Extend {op} OnnxToHip converter for this attribute/shape variant, "
            f"or re-export ONNX avoiding: {detail.reason[:60]}"
        )
    else:
        next_step = _default_gap_next_step(
            op,
            has_converter=base.onnx_converter.startswith("yes"),
            runtime_label=base.runtime,
        )
    return OpGapLayers(
        schema=base.schema,
        onnx_converter=base.onnx_converter,
        fusion=base.fusion,
        runtime=base.runtime,
        next_step=f"{next_step} ({detail.reason})",
    )


def _hip_ep_handling(
    op: str,
    known: set[str],
    *,
    in_survivors: bool = False,
    shape_detail: ShapeBlockerDetail | None = None,
) -> str:
    gap = _analyze_op_gap(op)
    if gap.onnx_converter.startswith("yes"):
        if op in _repo_op_support().ms_custom_ops:
            snake = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", op).lower()
            handling = f"com.microsoft Custom -> hip.{snake} (OnnxToHip converter)"
        else:
            handling = f"onnx.{op} -> hip.* (OnnxToHip converter)"
    elif gap.fusion.startswith("partial") or op in _PARTIAL_FUSION_OPS:
        handling = gap.onnx_converter
    else:
        handling = "no converter in hip-ep repo"

    if in_survivors and _hip_ep_handled(op, known):
        handling += " (still onnx.* / Custom at bufferize dump)"
    if shape_detail is not None:
        label = {
            "shape": "shape blocker",
            "variant": "variant blocker",
            "dtype": "dtype blocker",
            "runtime": "runtime blocker",
        }.get(shape_detail.kind, "blocker")
        handling += f" [{label}: {shape_detail.reason}]"
        if shape_detail.example:
            handling += f" (e.g. {shape_detail.example})"
    return handling


def enumerate_onnx_ops(model_path: Path) -> dict[str, int]:
    """Return {op_type: node_count} for all nodes in the ONNX graph."""
    try:
        import onnx
    except ImportError as exc:
        raise RuntimeError(
            "ONNX graph scan requires the 'onnx' package (pip install onnx)"
        ) from exc

    model = onnx.load(str(model_path), load_external_data=False)
    op_counts: dict[str, int] = {}
    for node in model.graph.node:
        op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1
    for fn in model.functions:
        for node in fn.node:
            op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1
    return op_counts


def _tensor_dims_from_type(type_fragment: str) -> list[str]:
    """Return dimension tokens from ``tensor<?x512x?x?xf32>`` (excludes element type)."""
    match = RE_TENSOR_TYPE.search(type_fragment)
    if not match:
        return []
    parts = match.group(1).split("x")
    if not parts:
        return []
    if not re.fullmatch(r"[?0-9]+", parts[-1]):
        parts = parts[:-1]
    return parts


def _conv_shape_issue(output_type: str, input_types: list[str]) -> str | None:
    """Mirror ConvConversion.cpp limits for dynamic / 1D shapes."""
    out_dims = _tensor_dims_from_type(output_type)
    if not out_dims:
        return None
    rank = len(out_dims)
    if rank == 3:
        for dims in (out_dims, *(_tensor_dims_from_type(t) for t in input_types)):
            if dims and any(d == "?" for d in dims):
                return "1D Conv requires fully static input/output shapes"
    elif rank >= 4:
        for i in range(2, rank):
            if out_dims[i] == "?":
                return (
                    "dynamic output spatial dims (H/W) — ConvToHip only "
                    "supports dynamic batch (N)"
                )
    return None


def _resize_shape_issue(output_type: str) -> str | None:
    """Mirror ResizeConversion.cpp: dynamic output spatial dims unsupported."""
    out_dims = _tensor_dims_from_type(output_type)
    if len(out_dims) < 3:
        return None
    for i in range(2, len(out_dims)):
        if out_dims[i] == "?":
            return "dynamic output spatial dims (H/W) unsupported by hip.resize"
    return None


def _shape_issue_for_op(
    op: str, output_type: str, input_types: list[str]
) -> str | None:
    if op in {"Conv", "ConvTranspose"}:
        return _conv_shape_issue(output_type, input_types)
    if op == "Resize":
        return _resize_shape_issue(output_type)
    if op in {"MaxPool", "AveragePool", "GlobalMaxPool", "GlobalAveragePool"}:
        out_dims = _tensor_dims_from_type(output_type)
        if len(out_dims) >= 3:
            for i in range(2, len(out_dims)):
                if out_dims[i] == "?":
                    return "dynamic spatial output dims unsupported for pool ops"
    return None


def _mlir_attr_dict(line: str) -> dict[str, str]:
    attrs: dict[str, str] = {}
    brace = line.find("{")
    if brace < 0:
        return attrs
    segment = line[brace:]
    for match in RE_MLIR_ATTR.finditer(segment):
        key = match.group(1)
        value = match.group(2) or match.group(3) or match.group(4) or ""
        attrs[key] = value.strip()
    return attrs


def _mlir_attr_int(attrs: dict[str, str], key: str, default: int = 0) -> int:
    raw = attrs.get(key, str(default))
    match = re.search(r"-?\d+", raw)
    return int(match.group(0)) if match else default


def _variant_issue_for_op(
    op: str,
    *,
    line: str = "",
    input_types: list[str] | None = None,
    output_type: str = "",
    onnx_node=None,
    values_by_name: dict | None = None,
) -> str | None:
    """Attribute / rank / mode limits that block lowering despite a converter."""
    input_types = input_types or []
    attrs = _mlir_attr_dict(line) if line else {}

    if op in {"Conv", "ConvTranspose"}:
        in_dims = _tensor_dims_from_type(input_types[0]) if input_types else []
        if len(in_dims) == 5:
            return "3D Conv (rank-5) not supported — only 1D/2D Conv"
        if op == "Conv" and len(in_dims) == 3:
            group = _mlir_attr_int(attrs, "group", 1)
            dilations = attrs.get("dilations", "[1]")
            if group != 1:
                return "1D Conv with group != 1 is not supported"
            if dilations not in {"[1]", "[1, 1]"} and "1" not in dilations.replace(
                " ", ""
            ):
                if not re.fullmatch(r"\[1(?:,\s*1)*\]", dilations.replace(" ", "")):
                    return "1D Conv with dilation != 1 is not supported"

    if op == "Resize":
        mode = attrs.get("mode", "nearest").strip('"')
        if mode == "cubic":
            return "Resize mode=cubic not supported (nearest/linear only)"
        if _mlir_attr_int(attrs, "antialias", 0) != 0:
            return "Resize antialias not supported"
        if _mlir_attr_int(attrs, "exclude_outside", 0) != 0:
            return "Resize exclude_outside not supported"
        kar = attrs.get("keep_aspect_ratio_policy", "stretch").strip('"')
        if kar and kar != "stretch":
            return "Resize keep_aspect_ratio_policy must be 'stretch'"
        if (
            input_types
            and len(input_types) >= 2
            and "none" not in input_types[1].lower()
        ):
            return "Resize with roi (tf_crop_and_resize) not supported"

    if op in {"MaxPool", "AveragePool"}:
        in_type = input_types[0] if input_types else ""
        if in_type and not re.search(r"f16|f32|f64", in_type):
            return "Pool runtime supports float element types only"
        in_dims = _tensor_dims_from_type(in_type)
        if len(in_dims) > 5:
            return "Pool spatial rank > 3 not supported"

    if op == "Slice" and onnx_node is not None and values_by_name is not None:
        return _onnx_slice_variant_issue(onnx_node, values_by_name)

    if onnx_node is not None and values_by_name is not None:
        if op in {"Conv", "ConvTranspose"}:
            return _onnx_conv_variant_issue(onnx_node, values_by_name)
        if op == "Resize":
            return _onnx_resize_variant_issue(onnx_node, values_by_name)
        if op in {"MaxPool", "AveragePool"}:
            return _onnx_pool_variant_issue(onnx_node, values_by_name)

    return None


def _onnx_elem_type_name(value) -> str | None:
    if hasattr(value, "type"):
        tensor_type = value.type.tensor_type
        elem_type = tensor_type.elem_type
    elif hasattr(value, "data_type"):
        elem_type = value.data_type
    else:
        return None
    if not elem_type:
        return None
    try:
        import onnx

        return onnx.TensorProto.DataType.Name(tensor_type.elem_type).lower()
    except Exception:
        return None


def _onnx_get_int_attr(node, name: str, default: int = 0) -> int:
    for attr in node.attribute:
        if attr.name == name and attr.type == 2:  # INT
            return int(attr.i)
    return default


def _onnx_get_str_attr(node, name: str, default: str = "") -> str:
    for attr in node.attribute:
        if attr.name == name and attr.type == 3:  # STRING
            return attr.s.decode() if isinstance(attr.s, bytes) else str(attr.s)
    return default


def _onnx_conv_variant_issue(node, values_by_name: dict) -> str | None:
    if not node.input:
        return None
    input_value = values_by_name.get(node.input[0])
    if input_value is None:
        return None
    rank = _onnx_value_rank(input_value)
    if rank == 5:
        return "3D Conv (rank-5) not supported — only 1D/2D Conv"
    if rank == 3 and node.op_type == "Conv":
        group = _onnx_get_int_attr(node, "group", 1)
        if group != 1:
            return "1D Conv with group != 1 is not supported"
        dilations = [
            int(a.i) for a in node.attribute if a.name == "dilations" and a.type == 7
        ]
        if dilations and any(d != 1 for d in dilations):
            return "1D Conv with dilation != 1 is not supported"
        for name in node.input[:2]:
            value = values_by_name.get(name)
            if value is None or not hasattr(value, "type"):
                continue
            tensor_type = value.type.tensor_type
            if not tensor_type.HasField("shape"):
                return "1D Conv requires fully static input/output shapes"
            if any(_onnx_dims_dynamic(tensor_type.shape.dim)):
                return "1D Conv requires fully static input/output shapes"
    return None


def _onnx_resize_variant_issue(node, values_by_name: dict) -> str | None:
    mode = _onnx_get_str_attr(node, "mode", "nearest")
    if mode == "cubic":
        return "Resize mode=cubic not supported (nearest/linear only)"
    if _onnx_get_int_attr(node, "antialias", 0) != 0:
        return "Resize antialias not supported"
    if _onnx_get_int_attr(node, "exclude_outside", 0) != 0:
        return "Resize exclude_outside not supported"
    kar = _onnx_get_str_attr(node, "keep_aspect_ratio_policy", "stretch")
    if kar and kar != "stretch":
        return "Resize keep_aspect_ratio_policy must be 'stretch'"
    if len(node.input) >= 2 and node.input[1]:
        return "Resize with roi (tf_crop_and_resize) not supported"
    return None


def _onnx_pool_variant_issue(node, values_by_name: dict) -> str | None:
    if not node.input:
        return None
    input_value = values_by_name.get(node.input[0])
    if input_value is None:
        return None
    rank = _onnx_value_rank(input_value)
    if rank > 5:
        return "Pool spatial rank > 3 not supported"
    elem = _onnx_elem_type_name(input_value)
    if elem and elem not in {"float", "float16", "double"}:
        return "Pool runtime supports float element types only"
    return None


def _onnx_slice_variant_issue(node, values_by_name: dict) -> str | None:
    if len(node.input) < 3:
        return None
    for idx in (1, 2):
        if idx >= len(node.input):
            continue
        value = values_by_name.get(node.input[idx])
        if value is None:
            return "Slice with dynamic starts/ends not supported"
        if not hasattr(value, "type"):
            continue
        tensor_type = value.type.tensor_type
        if not tensor_type.HasField("shape"):
            return "Slice with dynamic starts/ends not supported"
        if any(_onnx_dims_dynamic(tensor_type.shape.dim)):
            return "Slice with dynamic starts/ends not supported"
    return None


def _dtype_issue_for_op(
    op: str,
    *,
    onnx_node=None,
    values_by_name: dict | None = None,
    input_types: list[str] | None = None,
) -> str | None:
    blocked_types = _DTYPE_BLOCKER_RULES.get(op)
    if not blocked_types:
        return None
    if onnx_node is not None and values_by_name is not None and onnx_node.input:
        value = values_by_name.get(onnx_node.input[0])
        if value is not None and hasattr(value, "type"):
            elem = _onnx_elem_type_name(value)
            if elem and elem in blocked_types:
                return f"{op} on {elem} — converter targets float paths only"
    if input_types:
        for type_str in input_types:
            match = re.search(r"([a-z0-9]+)$", type_str.strip().lower())
            if match and match.group(1) in blocked_types:
                return f"{op} on {match.group(1)} — converter targets float paths only"
    return None


def _record_blocker(
    blockers: dict[str, ShapeBlockerDetail],
    op: str,
    reason: str,
    *,
    source: str,
    kind: str,
    example: str = "",
) -> None:
    key = f"{op}:{kind}"
    detail = blockers.setdefault(
        key,
        ShapeBlockerDetail(reason=reason, source=source, kind=kind),
    )
    detail.count += 1
    if example and not detail.example:
        detail.example = example


def _blockers_for_op(
    blockers: dict[str, ShapeBlockerDetail], op: str
) -> dict[str, ShapeBlockerDetail]:
    return {
        detail.kind: detail
        for key, detail in blockers.items()
        if key.startswith(f"{op}:")
    }


def _pick_blocker_for_op(
    blockers: dict[str, ShapeBlockerDetail], op: str
) -> ShapeBlockerDetail | None:
    """Choose highest-priority blocker detail for inventory classification."""
    by_kind = _blockers_for_op(blockers, op)
    for kind in ("runtime", "dtype", "shape", "variant"):
        detail = by_kind.get(kind)
        if detail is not None:
            return detail
    return None


def _blocker_status_for_kind(kind: str) -> str:
    return {
        "shape": "shape_blocker",
        "variant": "variant_blocker",
        "dtype": "dtype_blocker",
        "runtime": "runtime_blocker",
    }.get(kind, "variant_blocker")


def _parse_mlir_onnx_op_types(line: str) -> tuple[str, list[str], str] | None:
    match = RE_MLIR_ONNX_OP_TYPES.search(line.strip())
    if not match:
        return None
    op = match.group(1)
    input_types = [part.strip() for part in match.group(2).split(",") if part.strip()]
    output_type = match.group(3).strip().rstrip(")")
    return op, input_types, output_type


def _onnx_dims_dynamic(dims: Iterable) -> list[bool]:
    flags: list[bool] = []
    for dim in dims:
        if getattr(dim, "dim_param", None):
            flags.append(True)
        elif hasattr(dim, "HasField") and dim.HasField("dim_value"):
            flags.append(False)
        else:
            flags.append(True)
    return flags


def _onnx_value_rank(value) -> int:
    if hasattr(value, "type"):
        tensor_type = value.type.tensor_type
        if not tensor_type.HasField("shape"):
            return 0
        return len(tensor_type.shape.dim)
    if hasattr(value, "dims"):
        return len(value.dims)
    return 0


def _onnx_spatial_dynamic(value) -> bool:
    if hasattr(value, "type"):
        tensor_type = value.type.tensor_type
        if not tensor_type.HasField("shape"):
            return False
        dynamic = _onnx_dims_dynamic(tensor_type.shape.dim)
    elif hasattr(value, "dims"):
        dynamic = [not dim for dim in value.dims]
    else:
        return False
    rank = len(dynamic)
    if rank < 3:
        return False
    spatial_start = 2 if rank >= 4 else 2
    return any(dynamic[i] for i in range(spatial_start, rank))


def analyze_blockers_from_mlir(mlir_base: Path | None) -> dict[str, ShapeBlockerDetail]:
    """Detect shape/variant/dtype failures from surviving MLIR ``onnx.*`` lines."""
    if mlir_base is None:
        return {}

    roots = discover_mlir_dump_dirs(mlir_base)
    if not roots:
        return {}

    blockers: dict[str, ShapeBlockerDetail] = {}
    for root in roots:
        mlir_files = [p for p in root.rglob("*.mlir") if "bufferize" in p.name.lower()]
        if not mlir_files:
            convert_files = sorted(
                p
                for p in root.rglob("*.mlir")
                if "convert-onnx-to-hip" in p.name.lower()
            )
            mlir_files = (
                convert_files[-1:]
                if convert_files
                else sorted(root.rglob("*.mlir"))[-1:]
            )
        for mlir_path in mlir_files:
            text = mlir_path.read_text(encoding="utf-8", errors="ignore")
            for line in text.splitlines():
                parsed = _parse_mlir_onnx_op_types(line)
                if parsed is None:
                    continue
                op, input_types, output_type = parsed
                name_match = RE_ONNX_NODE_NAME.search(line)
                example = name_match.group(1) if name_match else output_type.strip()
                for kind, reason in (
                    ("shape", _shape_issue_for_op(op, output_type, input_types)),
                    (
                        "variant",
                        _variant_issue_for_op(
                            op,
                            line=line,
                            input_types=input_types,
                            output_type=output_type,
                        ),
                    ),
                    (
                        "dtype",
                        _dtype_issue_for_op(op, input_types=input_types),
                    ),
                ):
                    if reason:
                        _record_blocker(
                            blockers,
                            op,
                            reason,
                            source="mlir",
                            kind=kind,
                            example=example,
                        )
    return blockers


def _load_onnx_graph(model_path: Path):
    import onnx
    from onnx import shape_inference

    model = onnx.load(str(model_path), load_external_data=False)
    try:
        model = shape_inference.infer_shapes(model)
    except Exception:
        pass
    values_by_name: dict[str, object] = {}
    for src in (
        list(model.graph.input),
        list(model.graph.output),
        list(model.graph.value_info),
        list(getattr(model.graph, "initializer", [])),
    ):
        for value in src:
            if value.name:
                values_by_name[value.name] = value
    nodes = list(model.graph.node)
    for fn in model.functions:
        nodes.extend(fn.node)
    return model, values_by_name, nodes


def analyze_blockers_from_onnx(model_path: Path) -> dict[str, ShapeBlockerDetail]:
    """Predict shape/variant/dtype blockers from ONNX graph metadata."""
    try:
        _model, values_by_name, nodes = _load_onnx_graph(model_path)
    except ImportError:
        return {}
    except Exception:
        return {}

    blockers: dict[str, ShapeBlockerDetail] = {}
    for node in nodes:
        op = node.op_type
        if not node.output:
            continue
        output_value = values_by_name.get(node.output[0])
        input_types: list[str] = []
        if node.input:
            input_value = values_by_name.get(node.input[0])
            if input_value is not None and hasattr(input_value, "type"):
                tt = input_value.type.tensor_type
                if tt.HasField("shape"):
                    dims = []
                    for dim in tt.shape.dim:
                        if dim.dim_param:
                            dims.append("?")
                        elif dim.HasField("dim_value"):
                            dims.append(str(dim.dim_value))
                        else:
                            dims.append("?")
                    elem = _onnx_elem_type_name(input_value) or "f32"
                    input_types.append(f"tensor<{'x'.join(dims)}x{elem}>")
        output_type = ""
        if output_value is not None and hasattr(output_value, "type"):
            tt = output_value.type.tensor_type
            if tt.HasField("shape"):
                dims = []
                for dim in tt.shape.dim:
                    if dim.dim_param:
                        dims.append("?")
                    elif dim.HasField("dim_value"):
                        dims.append(str(dim.dim_value))
                    else:
                        dims.append("?")
                elem = _onnx_elem_type_name(output_value) or "f32"
                output_type = f"tensor<{'x'.join(dims)}x{elem}>"
        example = node.name or (node.output[0] if node.output else op)
        for kind, reason in (
            ("shape", _shape_issue_for_op(op, output_type, input_types)),
            (
                "variant",
                _variant_issue_for_op(
                    op,
                    onnx_node=node,
                    values_by_name=values_by_name,
                    input_types=input_types,
                    output_type=output_type,
                ),
            ),
            (
                "dtype",
                _dtype_issue_for_op(op, onnx_node=node, values_by_name=values_by_name),
            ),
        ):
            if reason:
                _record_blocker(
                    blockers,
                    op,
                    reason,
                    source="onnx",
                    kind=kind,
                    example=example,
                )
    return blockers


def analyze_runtime_blockers(
    runtime_findings: list[OpFinding] | None,
) -> dict[str, ShapeBlockerDetail]:
    blockers: dict[str, ShapeBlockerDetail] = {}
    for finding in runtime_findings or []:
        if "runtime unsupported" not in finding.detail:
            continue
        op = finding.op
        if op.startswith("wrap_"):
            op = _snake_to_op_name(op[5:])
        reason = finding.detail.split("runtime unsupported:", 1)[-1].strip()
        _record_blocker(
            blockers,
            op,
            reason or "runtime kernel rejected this variant",
            source="log",
            kind="runtime",
            example=finding.detail[:120],
        )
    return blockers


def collect_shape_blockers(
    model_path: Path,
    *,
    mlir_base: Path | None = None,
    runtime_findings: list[OpFinding] | None = None,
) -> dict[str, ShapeBlockerDetail]:
    """Merge MLIR, ONNX, and runtime-detected blockers keyed as ``Op:kind``."""
    merged: dict[str, ShapeBlockerDetail] = {}
    for src in (
        analyze_blockers_from_onnx(model_path),
        analyze_blockers_from_mlir(mlir_base),
        analyze_runtime_blockers(runtime_findings),
    ):
        for key, detail in src.items():
            bucket = merged.get(key)
            if bucket is None:
                merged[key] = ShapeBlockerDetail(
                    reason=detail.reason,
                    count=detail.count,
                    example=detail.example,
                    source=detail.source,
                    kind=detail.kind,
                )
                continue
            bucket.count = max(bucket.count, detail.count)
            if detail.source == "mlir":
                bucket.source = "mlir"
                bucket.reason = detail.reason
            elif detail.source == "log" and bucket.source != "mlir":
                bucket.source = "log"
                bucket.reason = detail.reason
            if detail.example and not bucket.example:
                bucket.example = detail.example
    return merged


def collect_mlir_bufferize_survivors(mlir_base: Path | None) -> dict[str, int]:
    """Ops still present as onnx.* / Custom at the bufferize-failure MLIR dump."""
    if mlir_base is None:
        return {}

    roots = discover_mlir_dump_dirs(mlir_base)
    if not roots:
        return {}

    op_counts: dict[str, int] = {}
    for root in roots:
        mlir_files = [p for p in root.rglob("*.mlir") if "bufferize" in p.name.lower()]
        if not mlir_files:
            mlir_files = sorted(root.rglob("*.mlir"))[-1:]
        for mlir_path in mlir_files:
            text = mlir_path.read_text(encoding="utf-8", errors="ignore")
            for line in text.splitlines():
                for op, _detail in _extract_all_surviving_ops_from_line(line):
                    op_counts[op] = op_counts.get(op, 0) + 1
    return op_counts


def build_op_inventory(
    model_path: Path,
    *,
    known_ops: set[str],
    mlir_base: Path | None = None,
    compile_failed: bool = False,
    report_shape_blockers: bool = False,
    runtime_findings: list[OpFinding] | None = None,
) -> list[OpInventoryItem]:
    """Classify every ONNX op in the model for hip-ep support."""
    graph_ops = enumerate_onnx_ops(model_path)
    survivors = collect_mlir_bufferize_survivors(mlir_base)
    all_blockers = collect_shape_blockers(
        model_path,
        mlir_base=mlir_base,
        runtime_findings=runtime_findings,
    )
    has_mlir = bool(survivors) or (
        mlir_base is not None and bool(discover_mlir_dump_dirs(mlir_base))
    )

    support = _repo_op_support()
    inventory: list[OpInventoryItem] = []
    seen_ops: set[str] = set()

    def _append_item(
        op: str,
        count: int,
        status: str,
        handling: str,
        gap: OpGapLayers | None,
    ) -> None:
        inventory.append(
            OpInventoryItem(
                op=op,
                count=count,
                status=status,
                hip_ep_handling=handling,
                gap=gap,
            )
        )
        seen_ops.add(op)

    for op, count in sorted(graph_ops.items()):
        if op in _IGNORE_ONNX_OPS:
            continue

        in_survivors = op in survivors
        handled = _hip_ep_handled(op, known_ops)
        blocker_detail = _pick_blocker_for_op(all_blockers, op)
        handling = _hip_ep_handling(
            op,
            known_ops,
            in_survivors=in_survivors,
            shape_detail=blocker_detail,
        )

        status: str
        gap: OpGapLayers | None = None

        if op in support.fusion_ops:
            status = "fused"
        elif blocker_detail and blocker_detail.kind == "runtime":
            status = "runtime_blocker"
            gap = _analyze_blocker_gap(op, blocker_detail)
        elif (
            blocker_detail
            and blocker_detail.kind == "dtype"
            and (in_survivors or report_shape_blockers or compile_failed)
        ):
            status = "dtype_blocker"
            gap = _analyze_blocker_gap(op, blocker_detail)
        elif (
            blocker_detail
            and blocker_detail.kind == "shape"
            and (in_survivors or report_shape_blockers or compile_failed)
        ):
            status = "shape_blocker"
            gap = _analyze_blocker_gap(op, blocker_detail)
        elif (
            blocker_detail
            and blocker_detail.kind == "variant"
            and (in_survivors or report_shape_blockers or compile_failed)
        ):
            status = "variant_blocker"
            gap = _analyze_blocker_gap(op, blocker_detail)
        elif op in support.ms_custom_ops:
            status = "lowered"
        elif handled:
            if in_survivors and compile_failed:
                status = "lowered_incomplete"
            elif in_survivors:
                status = "lowered_incomplete"
            elif compile_failed and not has_mlir:
                status = "lowered_unverified"
            else:
                status = "lowered"
        elif op in _RUNTIME_PARTIAL:
            status = "partial"
            gap = _analyze_op_gap(op)
        elif in_survivors:
            status = "compile_blocker"
            gap = _analyze_op_gap(op)
        else:
            status = "unsupported"
            gap = _analyze_op_gap(op)

        _append_item(op, count, status, handling, gap)

    for key, detail in all_blockers.items():
        if detail.kind != "runtime":
            continue
        op = key.split(":", 1)[0]
        if op in seen_ops:
            continue
        handling = _hip_ep_handling(op, known_ops, shape_detail=detail)
        _append_item(
            op,
            detail.count,
            "runtime_blocker",
            handling,
            _analyze_blocker_gap(op, detail),
        )

    return inventory


def _merge_findings(findings: list[OpFinding]) -> list[OpFinding]:
    merged: dict[tuple[str, str, str], OpFinding] = {}
    for item in findings:
        key = item.key()
        if key in merged:
            merged[key].count += item.count
        else:
            merged[key] = OpFinding(
                op=item.op,
                source=item.source,
                detail=item.detail,
                count=item.count,
            )
    return sorted(merged.values(), key=lambda f: (f.op, f.source, f.detail))


def _group_findings_by_op(findings: list[OpFinding]) -> dict[str, dict]:
    grouped: dict[str, dict] = {}
    for item in findings:
        bucket = grouped.setdefault(
            item.op,
            {"count": 0, "sources": set(), "details": []},
        )
        bucket["count"] += item.count
        bucket["sources"].add(item.source)
        if item.detail and item.detail not in bucket["details"]:
            bucket["details"].append(item.detail)
    return grouped


def analyze_failure(
    output: str,
    errors: list[str],
    exit_code: int | None,
    findings: list[OpFinding],
    qps: float | None,
) -> tuple[str, str, str]:
    """Return (hip_ep_result, failure_level, failure_detail)."""
    text = f"{output}\n" + "\n".join(errors)
    compile_blockers = [
        f
        for f in findings
        if f.source in {"mlir_dump", "perf_test_log", "log"}
        and "runtime unsupported" not in f.detail
    ]
    runtime_findings = [f for f in findings if "runtime unsupported" in f.detail]

    if qps is not None and not findings:
        return (
            "OK",
            "none",
            "Model compiled and inference ran on hip-ep.",
        )

    if exit_code is not None and (exit_code & 0xFFFFFFFF) == 0xC0000005:
        return (
            "NATIVE CRASH",
            "native_crash",
            "Process crashed (ACCESS_VIOLATION) before hip-ep produced output. "
            "Check EP DLL (use hipgpu|hipgpu.dll on Windows).",
        )

    if RE_BUFFERIZE_FAILURE.search(text):
        return (
            "COMPILE FAILED",
            "mlir_bufferize",
            "Unconverted onnx.* ops survived convert-onnx-to-hip and failed "
            "MLIR one-shot-bufferize (error: op was not bufferized).",
        )

    if RE_COMPILATION_FAILED.search(text):
        return (
            "COMPILE FAILED",
            "mlir_compile",
            "hip-ep MLIR compile pipeline failed (OnnxToHip lowering or a later pass).",
        )

    if compile_blockers:
        return (
            "COMPILE FAILED",
            "mlir_bufferize",
            "Unconverted onnx.* ops were found in MLIR IR after lowering.",
        )

    if runtime_findings:
        return (
            "RUNTIME FAILED",
            "runtime_kernel",
            "Graph compiled but a runtime kernel rejected an unsupported variant.",
        )

    if RE_CPU_FALLBACK.search(text):
        return (
            "SESSION FAILED",
            "ort_session",
            "ORT could not run the full graph on hip-ep with CPU fallback disabled. "
            "Some nodes were left on CPU EP.",
        )

    if findings:
        return (
            "UNSUPPORTED OPS",
            "static_analysis",
            "ONNX graph contains op types with no matching hip-ep converter in this repo.",
        )

    if exit_code is not None and exit_code not in (0, 1):
        unsigned = exit_code & 0xFFFFFFFF
        return (
            "RUNTIME FAILED",
            "unknown",
            f"onnxruntime_perf_test exited with code {exit_code} (0x{unsigned:08X}).",
        )

    if exit_code == 1 and errors:
        return (
            "FAILED",
            "unknown",
            "Run failed; see errors in --verbose or use --save-log.",
        )

    return ("OK", "none", "No unsupported ops identified.")


def finalize_model_report(report: ModelReport, output: str = "") -> ModelReport:
    result, level, detail = analyze_failure(
        output,
        report.errors,
        report.exit_code,
        report.missing_ops,
        report.qps,
    )
    report.hip_ep_result = result
    report.failure_level = level
    report.failure_detail = detail
    return report


def _model_dump_tag(model_path: Path) -> str:
    """Build a filesystem-safe tag from a model path (includes parent folder when generic)."""
    resolved = model_path.resolve()
    stem = resolved.stem
    parent = resolved.parent.name

    generic_stems = {"model", "graph", "network", "onnx"}
    if stem.lower() in generic_stems and parent:
        raw = f"{parent}_{stem}"
    else:
        raw = stem

    safe = re.sub(r"[^\w\-.]+", "_", raw)
    safe = re.sub(r"_+", "_", safe).strip("._")
    return safe[:120] or "model"


def parse_text_for_missing_ops(
    text: str,
    *,
    source_label: str = "log",
    known_ops: set[str] | None = None,
) -> tuple[list[OpFinding], list[str], float | None]:
    """Extract missing-op findings and error lines from arbitrary EP output."""
    findings: list[OpFinding] = []
    errors: list[str] = []
    known = known_ops or _repo_known_onnx_ops()

    for line in text.splitlines():
        if RE_BUFFERIZE_FAILURE.search(line):
            errors.append(line.strip())

        if RE_COMPILATION_FAILED.search(line):
            errors.append(line.strip())

        if RE_CPU_FALLBACK.search(line):
            errors.append(line.strip())

        for op, detail in _extract_surviving_ops_from_line(line, known):
            findings.append(
                OpFinding(
                    op=op,
                    source=source_label,
                    detail=detail,
                )
            )

        for match in RE_REAL_UNSUPPORTED.finditer(line):
            findings.append(
                OpFinding(
                    op=match.group(1),
                    source=source_label,
                    detail=f"runtime unsupported: {match.group(2).strip()}",
                )
            )

    qps_match = SUCCESS_QPS.search(text)
    qps = float(qps_match.group(1)) if qps_match else None
    return _merge_findings(findings), errors, qps


def discover_mlir_dump_dirs(requested_path: Path) -> list[Path]:
    """Locate MLIR tree dumps; EP appends compile counter (.0, .1, ...) to the path."""
    candidates: list[Path] = []
    if requested_path.is_dir():
        candidates.append(requested_path)

    parent = requested_path.parent
    stem = requested_path.name
    if parent.is_dir():
        for sibling in sorted(parent.glob(f"{stem}.*")):
            if sibling.is_dir():
                candidates.append(sibling)

    roots: list[Path] = []
    seen: set[Path] = set()
    for cand in candidates:
        try:
            resolved = cand.resolve()
        except OSError:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        if any(cand.rglob("*.mlir")):
            roots.append(resolved)
    return roots


def parse_mlir_tree(
    dump_dir: Path,
    *,
    known_ops: set[str] | None = None,
) -> tuple[list[OpFinding], list[str]]:
    """Scan MLIR dump directory for surviving ``onnx.*`` ops."""
    roots = discover_mlir_dump_dirs(dump_dir)
    if not roots:
        hint = (
            f"No MLIR dumps found near {dump_dir}. "
            "hip-ep writes tree dumps to <HIPDNN_EP_IR_DUMP_PATH>.0/, .1/, ..."
        )
        return [], [hint]

    findings: list[OpFinding] = []
    errors: list[str] = []
    known = known_ops or _repo_known_onnx_ops()

    mlir_files: list[Path] = []
    for root in roots:
        mlir_files.extend(sorted(root.rglob("*.mlir")))
    mlir_files = sorted(set(mlir_files))

    if not mlir_files:
        return [], [f"No .mlir files under {dump_dir}"]

    errors.append("MLIR dump dirs: " + ", ".join(str(r) for r in roots))

    # Prefer the latest bufferize-failure dump when present.
    bufferize_fail = [p for p in mlir_files if "bufferize" in p.name.lower()]
    scan_files = bufferize_fail if bufferize_fail else mlir_files[-1:]

    op_counts: dict[str, int] = {}
    op_examples: dict[str, str] = {}

    for mlir_path in scan_files:
        text = mlir_path.read_text(encoding="utf-8", errors="ignore")
        if RE_BUFFERIZE_FAILURE.search(text):
            errors.append(f"bufferize failure in {mlir_path.name}")

        for line in text.splitlines():
            for op, detail in _extract_surviving_ops_from_line(line, known):
                op_counts[op] = op_counts.get(op, 0) + 1
                if op not in op_examples:
                    op_examples[op] = detail

    for op, count in sorted(op_counts.items()):
        detail = op_examples.get(op, "")
        findings.append(
            OpFinding(
                op=op,
                source="mlir_dump",
                detail=detail,
                count=count,
            )
        )

    return _merge_findings(findings), errors


def scan_onnx_model(
    model_path: Path,
    *,
    known_ops: set[str] | None = None,
) -> tuple[list[OpFinding], list[str]]:
    """Static scan: ONNX graph op_types not covered by hip-ep converters."""
    try:
        import onnx
    except ImportError as exc:
        raise RuntimeError(
            "Static ONNX scan requires the 'onnx' package (pip install onnx)"
        ) from exc

    known = known_ops or _repo_known_onnx_ops()
    model = onnx.load(str(model_path), load_external_data=False)

    op_counts: dict[str, int] = {}
    for node in model.graph.node:
        op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1

    # Walk function bodies (ORT may inline Gelu etc. unless disabled).
    for fn in model.functions:
        for node in fn.node:
            op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1

    findings: list[OpFinding] = []
    for op, count in sorted(op_counts.items()):
        if op not in known:
            findings.append(
                OpFinding(
                    op=op,
                    source="onnx_static",
                    detail="no matching hip-ep converter found in repo scan",
                    count=count,
                )
            )

    return findings, []


def _exe_name(stem: str) -> str:
    return f"{stem}.exe" if os.name == "nt" else stem


def _perf_test_name() -> str:
    return _exe_name("onnxruntime_perf_test")


def _package_has_perf_test(package_dir: Path) -> bool:
    return (package_dir / "bin" / _perf_test_name()).is_file()


def resolve_package_dir(raw: Path) -> Path:
    """Resolve gpu-test-package root, even when cwd is inside package/bin."""
    cwd = Path.cwd()

    if raw.is_absolute() and _package_has_perf_test(raw):
        return raw

    candidates: list[Path] = []

    # Explicit relative path from cwd (e.g. ../gpu-test-package).
    candidates.append(cwd / raw)

    # cwd is already gpu-test-package/bin — use the parent package root.
    if cwd.name == "bin":
        candidates.append(cwd.parent)

    # Walk up from cwd for a gpu-test-package tree.
    for parent in (cwd, *cwd.parents):
        candidates.append(parent / raw.name)
        candidates.append(parent / "gpu-test-package")

    # Common layouts: package next to hip-ep, inside hip-ep workspace, or monorepo sibling.
    candidates.append(HIP_EP_ROOT / raw.name)
    candidates.append(HIP_EP_ROOT / "gpu-test-package")
    candidates.append(HIP_EP_ROOT.parent / raw.name)
    candidates.append(HIP_EP_ROOT.parent / "gpu-test-package")

    seen: set[Path] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        if _package_has_perf_test(resolved):
            return resolved

    return (cwd / raw).resolve() if not raw.is_absolute() else raw


def resolve_perf_exe(package_dir: Path, perf_exe: Path | None) -> Path:
    """Locate onnxruntime_perf_test under the package or on PATH."""
    if perf_exe is not None:
        path = perf_exe.expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        if path.is_file():
            return path
        raise FileNotFoundError(f"onnxruntime_perf_test not found: {path}")

    default = package_dir / "bin" / _perf_test_name()
    if default.is_file():
        return default

    # cwd may be package/bin even if package_dir resolution was wrong.
    cwd_candidate = Path.cwd() / _perf_test_name()
    if cwd_candidate.is_file():
        return cwd_candidate.resolve()

    import shutil

    on_path = shutil.which(_perf_test_name())
    if on_path:
        return Path(on_path).resolve()

    raise FileNotFoundError(
        "onnxruntime_perf_test not found. Tried:\n"
        f"  - {default}\n"
        f"  - {cwd_candidate}\n"
        "Hint: run from gpu-test-package\\bin or pass --package-dir <gpu-test-package>."
    )


def _describe_exit_code(code: int | None) -> str:
    if code is None:
        return "Process did not return an exit code."
    unsigned = code & 0xFFFFFFFF
    if unsigned == 0xC0000005:
        return (
            f"Exit code {code} (0xC0000005 ACCESS_VIOLATION): native crash before "
            "any EP output. On Windows gpu-test-package use hipgpu, not amdgpu-ep:\n"
            '  --plugin-eps hipgpu --plugin-ep-libs "hipgpu|hipgpu.dll" '
            '--plugin-ep-options "config_file|..\\morphizen_config.json"'
        )
    if unsigned == 0xC0000135:
        return f"Exit code {code} (0xC0000135): DLL not found - check PATH/LIB."
    if unsigned == 0xC0000139:
        return f"Exit code {code} (0xC0000139): entry point not found in a DLL."
    return f"Exit code {code} (0x{unsigned:08X})."


def _resolve_ep_settings(package_dir: Path) -> tuple[str, str, str | None]:
    """Pick EP plugin name/libs/options from files present in *package_dir*."""
    pkg_bin = package_dir / "bin"
    ep_opts: str | None = None
    morphizen_cfg = package_dir / "morphizen_config.json"
    if morphizen_cfg.is_file():
        rel = (
            "..\\morphizen_config.json"
            if os.name == "nt"
            else "../morphizen_config.json"
        )
        ep_opts = f"config_file|{rel}"

    if os.name == "nt":
        # hipgpu is the stable Windows CI path; amdgpu-ep.dll can AV on some setups.
        for plugin_name, lib in (
            ("hipgpu", "hipgpu.dll"),
            ("AMDGPUExecutionProvider", "amdgpu-ep.dll"),
            ("hipep", "libhipep.dll"),
        ):
            if (pkg_bin / lib).is_file():
                return plugin_name, f"{plugin_name}|{lib}", ep_opts
        return "hipgpu", "hipgpu|hipgpu.dll", ep_opts

    for lib_name, plugin_name in (
        ("libhipep.so", "hipep"),
        ("libamdgpu-ep.so", "AMDGPUExecutionProvider"),
        ("libonnxruntime_morphizen_ep.so", "MorphiZenEP"),
    ):
        lib = package_dir / "lib" / lib_name
        if lib.is_file():
            return plugin_name, f"{plugin_name}|{lib.resolve()}", ep_opts
    lib = package_dir / "lib" / "libhipep.so"
    return "hipep", f"hipep|{lib.resolve()}", ep_opts


def _run_and_capture_log(
    cmd: list[str],
    *,
    log_path: Path | None,
    cwd: str,
    env: dict[str, str],
    timeout_sec: int,
) -> tuple[int | None, str]:
    """Run *cmd*, capturing merged stdout/stderr (optionally writing *log_path*)."""
    header = (
        f"[CMD] {subprocess.list2cmdline(cmd)}\n[cwd] {cwd}\n[timeout] {timeout_sec}s\n"
    )
    if log_path is not None:
        log_path.write_text(header, encoding="utf-8")

    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    lines: list[str] = []
    try:
        assert proc.stdout is not None
        log_file = (
            open(log_path, "a", encoding="utf-8") if log_path is not None else None
        )
        try:
            for line in proc.stdout:
                lines.append(line)
                if log_file is not None:
                    log_file.write(line)
                    log_file.flush()
        finally:
            if log_file is not None:
                log_file.close()
        proc.wait(timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        note = f"\n[identify_missing_ops] timeout after {timeout_sec}s\n"
        if log_path is not None:
            with open(log_path, "a", encoding="utf-8") as log_file:
                log_file.write(note)
        return None, header + "".join(lines) + note

    output = header + "".join(lines)
    if not "".join(lines).strip():
        note = (
            "\n[identify_missing_ops] Process produced no output.\n"
            f"{_describe_exit_code(proc.returncode)}\n"
        )
        if log_path is not None:
            with open(log_path, "a", encoding="utf-8") as log_file:
                log_file.write(note)
        output += note

    return proc.returncode, output


def _build_package_env(package_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    bin_dir = package_dir / "bin"
    lib_dir = package_dir / "lib"
    therock = env.get("THEROCK_DIST", "").strip()

    path_parts = [str(bin_dir)]
    if therock:
        path_parts.insert(0, str(Path(therock) / "bin"))
    if os.name == "nt":
        env["PATH"] = ";".join(path_parts + [env.get("PATH", "")])
        if therock:
            env["LIB"] = f"{lib_dir};{Path(therock) / 'lib'}"
        else:
            env["LIB"] = str(lib_dir)
    else:
        env["PATH"] = ":".join(path_parts + [env.get("PATH", "")])
        ld = ":".join(
            p
            for p in (
                str(lib_dir),
                str(bin_dir),
                str(Path(therock) / "lib") if therock else "",
                str(Path(therock) / "bin") if therock else "",
            )
            if p
        )
        env["LD_LIBRARY_PATH"] = f"{ld}:{env.get('LD_LIBRARY_PATH', '')}"
        env["LIBRARY_PATH"] = f"{ld}:{env.get('LIBRARY_PATH', '')}"
    return env


def build_perf_test_cmd(
    perf_exe: Path,
    model_path: Path,
    *,
    plugin_eps: str,
    plugin_ep_libs: str,
    plugin_ep_options: str | None,
    repeats: int,
    disable_cpu_fallback: bool,
    extra_args: list[str],
) -> list[str]:
    cmd = [
        str(perf_exe),
        "--plugin_ep_libs",
        plugin_ep_libs,
        "--plugin_eps",
        plugin_eps,
        "-r",
        str(repeats),
        "-c",
        "1",
        "-s",
        "-I",
        str(model_path),
    ]
    if plugin_ep_options:
        cmd += ["--plugin_ep_options", plugin_ep_options]
    if disable_cpu_fallback:
        cmd += ["-C", "session.disable_cpu_ep_fallback|1"]
    cmd += extra_args
    return cmd


def run_perf_test(
    model_path: Path,
    *,
    package_dir: Path,
    perf_exe: Path | None,
    plugin_eps: str | None,
    plugin_ep_libs: str | None,
    plugin_ep_options: str | None,
    repeats: int,
    disable_cpu_fallback: bool,
    dump_mlir: bool,
    save_log: bool,
    log_dir: Path,
    timeout_sec: int,
    extra_args: list[str],
    known_ops: set[str] | None = None,
) -> ModelReport:
    package_dir = resolve_package_dir(package_dir)
    pkg_bin = package_dir / "bin"
    perf = resolve_perf_exe(package_dir, perf_exe)

    default_eps, default_libs, default_ep_opts = _resolve_ep_settings(package_dir)
    eps = plugin_eps or os.environ.get("HIP_EP_PLUGIN_EPS", default_eps)
    libs = plugin_ep_libs or os.environ.get("HIP_EP_PLUGIN_LIBS", default_libs)
    ep_opts = (
        plugin_ep_options
        or os.environ.get("HIP_EP_PLUGIN_OPTIONS", "").strip()
        or default_ep_opts
    )

    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    model_tag = _model_dump_tag(model_path)
    log_file = log_dir / f"{model_tag}_missing_ops_{stamp}.log" if save_log else None

    env = _build_package_env(package_dir)
    mlir_base: Path | None = None
    if dump_mlir:
        mlir_base = log_dir / f"{model_tag}_mlir_{stamp}"
        # EP creates <path>.0/, <path>.1/, ... — do not mkdir the base path.
        env["HIPDNN_EP_IR_DUMP_TREE"] = "1"
        env["HIPDNN_EP_IR_DUMP_PATH"] = str(mlir_base)

    cmd = build_perf_test_cmd(
        perf,
        model_path,
        plugin_eps=eps,
        plugin_ep_libs=libs,
        plugin_ep_options=ep_opts,
        repeats=repeats,
        disable_cpu_fallback=disable_cpu_fallback,
        extra_args=extra_args,
    )

    try:
        return_code, output = _run_and_capture_log(
            cmd,
            log_path=log_file,
            cwd=str(pkg_bin),
            env=env,
            timeout_sec=timeout_sec,
        )
    except OSError as exc:
        note = f"[identify_missing_ops] failed to launch perf_test: {exc}\n"
        if log_file is not None:
            log_file.write_text(note, encoding="utf-8")
        return ModelReport(
            model=_model_display_path(model_path),
            status="runtime_failed",
            exit_code=None,
            errors=[note.strip()],
            log_path=str(log_file) if log_file is not None else None,
        )

    if return_code is None:
        return ModelReport(
            model=_model_display_path(model_path),
            status="runtime_failed",
            exit_code=None,
            errors=[f"timeout after {timeout_sec}s"],
            log_path=str(log_file) if log_file is not None else None,
        )

    proc_returncode = return_code

    findings, errors, qps = parse_text_for_missing_ops(
        output, source_label="perf_test_log", known_ops=known_ops
    )

    if dump_mlir and mlir_base is not None:
        mlir_findings, mlir_errors = parse_mlir_tree(mlir_base, known_ops=known_ops)
        if mlir_findings:
            findings = _merge_findings(findings + mlir_findings)
        errors.extend(mlir_errors)
        mlir_roots = discover_mlir_dump_dirs(mlir_base)
        mlir_dump_path = str(mlir_roots[0]) if mlir_roots else None
    else:
        mlir_dump_path = None

    # Always scan the ONNX graph: perf_test logs often lack specific op names.
    try:
        static_findings, _ = scan_onnx_model(model_path, known_ops=known_ops)
        if static_findings:
            findings = _merge_findings(findings + static_findings)
    except RuntimeError:
        if not findings:
            errors.append("static ONNX scan unavailable (pip install onnx)")
    except Exception as exc:
        if not findings:
            errors.append(f"static ONNX scan failed: {exc}")

    status = "ok"
    if findings or errors or proc_returncode != 0:
        if (
            findings
            or RE_COMPILATION_FAILED.search(output)
            or RE_BUFFERIZE_FAILURE.search(output)
        ):
            status = "compile_failed"
        elif proc_returncode != 0:
            status = "runtime_failed"
        elif qps is None:
            status = "unknown"
    if status == "ok" and qps is None and (findings or errors):
        status = "compile_failed"

    if proc_returncode != 0 and not errors and not findings:
        errors.append(_describe_exit_code(proc_returncode))

    try:
        op_inventory = build_op_inventory(
            model_path,
            known_ops=known_ops or _repo_known_onnx_ops(),
            mlir_base=mlir_base if dump_mlir else None,
            compile_failed=status != "ok",
            runtime_findings=[f for f in findings if "runtime unsupported" in f.detail],
        )
    except RuntimeError:
        op_inventory = []
    except Exception as exc:
        errors.append(f"op inventory failed: {exc}")
        op_inventory = []

    return finalize_model_report(
        ModelReport(
            model=_model_display_path(model_path),
            status=status,
            exit_code=proc_returncode,
            missing_ops=findings,
            errors=sorted(set(errors)),
            qps=qps,
            log_path=str(log_file) if log_file is not None else None,
            mlir_dump_path=mlir_dump_path,
            op_inventory=op_inventory,
        ),
        output=output,
    )


def collect_models(paths: Iterable[Path], glob_pattern: str) -> list[Path]:
    models: list[Path] = []
    for raw in paths:
        path = raw.resolve()
        if path.is_file() and path.suffix.lower() == ".onnx":
            models.append(path)
        elif path.is_dir():
            models.extend(sorted(p.resolve() for p in path.rglob(glob_pattern)))
        else:
            raise FileNotFoundError(f"Not a file or directory: {path}")
    # De-dupe while preserving order.
    seen: set[Path] = set()
    unique: list[Path] = []
    for model in models:
        if model not in seen:
            seen.add(model)
            unique.append(model)
    return unique


def _model_display_path(model: str | Path) -> str:
    """Absolute path string for report output."""
    path = Path(model)
    try:
        return str(path.resolve())
    except OSError:
        return str(path)


def _model_short_label(model: str | Path) -> str:
    """Compact label for batch tables: ``parent/model.onnx`` or ``model.onnx``."""
    path = Path(model)
    try:
        path = path.resolve()
    except OSError:
        pass
    if path.parent.name and path.parent.name not in {".", ""}:
        return f"{path.parent.name}/{path.name}"
    return path.name


@dataclass
class ModelIssueSummary:
    """Blocking ops for one model (batch rollup)."""

    model: str
    model_short: str
    status: str
    hip_ep_result: str
    compile_blockers: list[str] = field(default_factory=list)
    shape_blockers: list[str] = field(default_factory=list)
    variant_blockers: list[str] = field(default_factory=list)
    dtype_blockers: list[str] = field(default_factory=list)
    runtime_blockers: list[str] = field(default_factory=list)
    partial: list[str] = field(default_factory=list)
    incomplete: list[str] = field(default_factory=list)
    unsupported: list[str] = field(default_factory=list)

    @property
    def all_blocking_ops(self) -> list[str]:
        return sorted(
            set(
                self.compile_blockers
                + self.shape_blockers
                + self.variant_blockers
                + self.dtype_blockers
                + self.runtime_blockers
                + self.partial
                + self.incomplete
                + self.unsupported
            )
        )


def _issue_ops_from_report(report: ModelReport) -> dict[str, list[str]]:
    """Return blocking ops grouped by issue kind."""
    buckets: dict[str, list[str]] = {
        "compile_blocker": [],
        "shape_blocker": [],
        "variant_blocker": [],
        "dtype_blocker": [],
        "runtime_blocker": [],
        "partial": [],
        "incomplete": [],
        "unsupported": [],
    }
    if report.op_inventory:
        for item in report.op_inventory:
            key = item.status
            if key == "lowered_incomplete":
                key = "incomplete"
            if key in buckets:
                buckets[key].append(item.op)
    elif report.missing_ops:
        for finding in report.missing_ops:
            detail = finding.detail.lower()
            if "runtime unsupported" in detail:
                buckets["runtime_blocker"].append(finding.op)
            elif "shape blocker" in detail:
                buckets["shape_blocker"].append(finding.op)
            elif "variant blocker" in detail:
                buckets["variant_blocker"].append(finding.op)
            elif "dtype blocker" in detail:
                buckets["dtype_blocker"].append(finding.op)
            elif "no converter" in detail:
                buckets["unsupported"].append(finding.op)
            else:
                buckets["compile_blocker"].append(finding.op)
    else:
        buckets["compile_blocker"] = list(report.unique_missing_ops)
    return {key: sorted(set(values)) for key, values in buckets.items() if values}


def collect_batch_issue_summaries(
    reports: list[ModelReport],
) -> list[ModelIssueSummary]:
    summaries: list[ModelIssueSummary] = []
    for report in reports:
        issues = _issue_ops_from_report(report)
        summaries.append(
            ModelIssueSummary(
                model=_model_display_path(report.model),
                model_short=_model_short_label(report.model),
                status=report.status,
                hip_ep_result=report.hip_ep_result,
                compile_blockers=issues.get("compile_blocker", []),
                shape_blockers=issues.get("shape_blocker", []),
                variant_blockers=issues.get("variant_blocker", []),
                dtype_blockers=issues.get("dtype_blocker", []),
                runtime_blockers=issues.get("runtime_blocker", []),
                partial=issues.get("partial", []),
                incomplete=issues.get("incomplete", []),
                unsupported=issues.get("unsupported", []),
            )
        )
    return summaries


def build_batch_op_index(
    summaries: Iterable[ModelIssueSummary],
) -> dict[str, list[str]]:
    """Map each blocking op to the model short labels that need it."""
    index: dict[str, list[str]] = {}
    for summary in summaries:
        if not summary.all_blocking_ops:
            continue
        for op in summary.all_blocking_ops:
            index.setdefault(op, []).append(summary.model_short)
    return {op: sorted(set(models)) for op, models in sorted(index.items())}


def batch_summary_to_dict(reports: list[ModelReport]) -> dict:
    summaries = collect_batch_issue_summaries(reports)
    with_issues = [s for s in summaries if s.all_blocking_ops or s.status != "ok"]
    ok_count = len(reports) - len(with_issues)
    return {
        "models_tested": len(reports),
        "models_ok": ok_count,
        "models_with_issues": len(with_issues),
        "by_model": [
            {
                "model": s.model,
                "model_short": s.model_short,
                "status": s.status,
                "hip_ep_result": s.hip_ep_result,
                "compile_blockers": s.compile_blockers,
                "shape_blockers": s.shape_blockers,
                "variant_blockers": s.variant_blockers,
                "dtype_blockers": s.dtype_blockers,
                "runtime_blockers": s.runtime_blockers,
                "partial": s.partial,
                "incomplete": s.incomplete,
                "unsupported": s.unsupported,
                "all_blocking_ops": s.all_blocking_ops,
            }
            for s in summaries
            if s.all_blocking_ops or s.status != "ok"
        ],
        "by_op": build_batch_op_index(with_issues),
    }


def format_batch_rollup(reports: list[ModelReport]) -> str:
    """Single consolidated view of blocking ops across a model directory run."""
    if len(reports) <= 1:
        return ""

    summaries = collect_batch_issue_summaries(reports)
    with_issues = [s for s in summaries if s.all_blocking_ops or s.status != "ok"]
    ok_count = len(reports) - len(with_issues)
    lines: list[str] = [
        "=" * 72,
        "BATCH ROLLUP — all blocking ops across models",
        "=" * 72,
        f"Models tested : {len(reports)}",
        f"  OK          : {ok_count}",
        f"  With issues : {len(with_issues)}",
        "",
    ]

    if with_issues:
        lines.append("By model:")
        model_w = max(len("Model"), *(len(s.model_short) for s in with_issues))
        for summary in with_issues:
            parts: list[str] = []
            if summary.compile_blockers:
                parts.append("compile: " + ", ".join(summary.compile_blockers))
            if summary.shape_blockers:
                parts.append("shape: " + ", ".join(summary.shape_blockers))
            if summary.variant_blockers:
                parts.append("variant: " + ", ".join(summary.variant_blockers))
            if summary.dtype_blockers:
                parts.append("dtype: " + ", ".join(summary.dtype_blockers))
            if summary.runtime_blockers:
                parts.append("runtime: " + ", ".join(summary.runtime_blockers))
            if summary.partial:
                parts.append("partial: " + ", ".join(summary.partial))
            if summary.incomplete:
                parts.append("incomplete: " + ", ".join(summary.incomplete))
            if summary.unsupported:
                parts.append("unsupported: " + ", ".join(summary.unsupported))
            if not parts:
                parts.append(
                    f"{summary.hip_ep_result} — no specific blocking ops identified"
                )
            lines.append(f"  {summary.model_short:<{model_w}}  {' | '.join(parts)}")
        lines.append("")

        op_index = build_batch_op_index(with_issues)
        if op_index:
            lines.append("By op (models that need each op):")
            op_w = max(len("Op"), *(len(op) for op in op_index))
            for op, models in op_index.items():
                lines.append(f"  {op:<{op_w}}  {', '.join(models)}")
            lines.append("")
    else:
        lines.append("All models passed — no blocking ops detected.")
        lines.append("")

    lines.append("=" * 72)
    return "\n".join(lines)


def format_missing_ops_output(reports: list[ModelReport]) -> str:
    """Minimal output: compile blocker names only (--quiet)."""
    if len(reports) == 1 and reports[0].op_inventory:
        blockers = sorted(
            {
                item.op
                for item in reports[0].op_inventory
                if item.status in BLOCKING_STATUSES
            }
        )
        return "\n".join(blockers)

    if len(reports) == 1:
        return "\n".join(reports[0].unique_missing_ops)

    lines: list[str] = []
    for report in reports:
        if report.op_inventory:
            ops = sorted(
                {
                    item.op
                    for item in report.op_inventory
                    if item.status in BLOCKING_STATUSES
                }
            )
        else:
            ops = report.unique_missing_ops
        if not ops and report.status == "ok":
            continue
        label = _model_short_label(report.model)
        lines.append(f"{label}: {', '.join(ops) if ops else report.hip_ep_result}")

    rollup = format_batch_rollup(reports)
    if rollup and lines:
        return rollup + "\n\n" + "\n".join(lines)
    if rollup:
        return rollup
    return "\n".join(lines)


def _format_op_table(findings: list[OpFinding], *, title: str) -> list[str]:
    grouped = _group_findings_by_op(findings)
    if not grouped:
        return [f"{title}:", "  (none identified)"]

    ops = sorted(grouped)
    op_w = max(len("Op"), *(len(op) for op in ops))
    count_w = max(len("Count"), *(len(str(grouped[op]["count"])) for op in ops))

    lines = [f"{title}:"]
    lines.append(f"  {'Op':<{op_w}} {'Count':>{count_w}}  {'Detected via'}")
    lines.append(f"  {'-' * op_w} {'-' * count_w}  {'-' * 28}")
    for op in ops:
        info = grouped[op]
        sources = ", ".join(SOURCE_LABELS.get(s, s) for s in sorted(info["sources"]))
        lines.append(f"  {op:<{op_w}} {info['count']:>{count_w}}  {sources}")
        for detail in info["details"][:2]:
            lines.append(f"  {'':<{op_w}} {'':>{count_w}}  note: {detail}")
    return lines


def _gap_status_flags(g: OpGapLayers) -> tuple[str, str, str, str]:
    schema = "yes" if g.schema.startswith("yes") else "no"
    conv = "yes" if g.onnx_converter.startswith("yes") else "no"
    if g.onnx_converter.startswith("no (Custom"):
        conv = "partial"
    fusion = (
        "yes"
        if g.fusion.startswith("yes")
        else ("partial" if g.fusion.startswith("partial") else "no")
    )
    runtime = (
        "yes"
        if g.runtime.startswith("yes")
        else ("partial" if g.runtime.startswith("partial") else "no")
    )
    return schema, conv, fusion, runtime


def _format_gap_table(
    items: list[OpInventoryItem],
    *,
    title: str,
    show_next_step: bool = False,
) -> list[str]:
    """Gap analysis: schema / converter / fusion / runtime (+ optional next step)."""
    gap_items = [i for i in items if i.gap is not None]
    if not gap_items:
        return _format_inventory_table(items, title=title)

    sorted_items = sorted(gap_items, key=lambda i: (-i.count, i.op))
    op_w = max(len("Op"), *(len(i.op) for i in sorted_items))
    cnt_w = max(len("Cnt"), *(len(str(i.count)) for i in sorted_items))
    schema_w, conv_w, fusion_w, runtime_w = 6, 9, 8, 8

    lines = [f"{title}:"]
    header = (
        f"  {'Op':<{op_w}} {'Cnt':>{cnt_w}}  "
        f"{'Schema':<{schema_w}} {'OnnxToHip':<{conv_w}} "
        f"{'Fusion':<{fusion_w}} {'Runtime':<{runtime_w}}"
    )
    separator = (
        f"  {'-' * op_w} {'-' * cnt_w}  "
        f"{'-' * schema_w} {'-' * conv_w} "
        f"{'-' * fusion_w} {'-' * runtime_w}"
    )
    if show_next_step:
        header += "  Next step"
        separator += f"  {'-' * 20}"
    lines.extend([header, separator])

    for item in sorted_items:
        g = item.gap
        assert g is not None
        schema, conv, fusion, runtime = _gap_status_flags(g)
        row = (
            f"  {item.op:<{op_w}} {item.count:>{cnt_w}}  "
            f"{schema:<{schema_w}} {conv:<{conv_w}} "
            f"{fusion:<{fusion_w}} {runtime:<{runtime_w}}"
        )
        if show_next_step:
            row += f"  {g.next_step}"
        lines.append(row)
    return lines


def _format_gap_legend(*, show_next_step: bool = False) -> list[str]:
    lines = [
        "Gap layer key:",
        "  Schema    — ONNX / morphizen op definition (graph import)",
        "  OnnxToHip — MLIR RewritePattern or Custom converter in lib/Conversion/OnnxToHip",
        "  Fusion    — pre-lowering fusion pass (e.g. FastGeluFusion on inlined chains)",
        "  Runtime   — wrap_* dispatch + HIP kernel in lib/Runtime",
    ]
    if show_next_step:
        lines.append(
            "  Next step — recommended work item(s) to enable this op on hip-ep"
        )
    else:
        lines.append(
            "  (Pass --show-next-step to print recommended work items per op.)"
        )
    lines.append("")
    return lines


def _format_inventory_table(
    items: list[OpInventoryItem],
    *,
    title: str,
) -> list[str]:
    if not items:
        return [f"{title}:", "  (none)"]

    sorted_items = sorted(items, key=lambda i: (-i.count, i.op))
    op_w = max(len("Op"), *(len(i.op) for i in sorted_items))
    count_w = max(len("Count"), *(len(str(i.count)) for i in sorted_items))

    lines = [f"{title}:"]
    lines.append(f"  {'Op':<{op_w}} {'Count':>{count_w}}  {'hip-ep handling'}")
    lines.append(f"  {'-' * op_w} {'-' * count_w}  {'-' * 40}")
    for item in sorted_items:
        lines.append(
            f"  {item.op:<{op_w}} {item.count:>{count_w}}  {item.hip_ep_handling}"
        )
    return lines


def format_summary_report(
    reports: list[ModelReport],
    *,
    show_next_step: bool = False,
) -> str:
    """Structured human-readable compatibility summary."""
    blocks: list[str] = []
    show_gap_legend = False

    if len(reports) > 1:
        rollup = format_batch_rollup(reports)
        if rollup:
            blocks.append(rollup)

    for report in reports:
        lines: list[str] = []
        model_label = _model_display_path(report.model)

        lines.append("=" * 72)
        lines.append("hip-ep compatibility report")
        lines.append("=" * 72)
        lines.append(f"Model             : {model_label}")
        lines.append(f"hip-ep result     : {report.hip_ep_result}")
        lines.append(
            "Failure stage     : "
            + FAILURE_STAGE_LABELS.get(report.failure_level, report.failure_level)
        )
        if report.failure_detail:
            lines.append(f"Failure reason    : {report.failure_detail}")
        if report.qps is not None:
            lines.append(f"Inference QPS     : {report.qps:.3f}")
        if report.exit_code is not None:
            lines.append(f"perf_test exit    : {report.exit_code}")
        if report.mlir_dump_path:
            lines.append(f"MLIR dump dir     : {report.mlir_dump_path}")
        lines.append("")

        if report.op_inventory:
            blockers = [i for i in report.op_inventory if i.status == "compile_blocker"]
            shape_blockers = [
                i for i in report.op_inventory if i.status == "shape_blocker"
            ]
            variant_blockers = [
                i for i in report.op_inventory if i.status == "variant_blocker"
            ]
            dtype_blockers = [
                i for i in report.op_inventory if i.status == "dtype_blocker"
            ]
            runtime_blockers = [
                i for i in report.op_inventory if i.status == "runtime_blocker"
            ]
            partial_ops = [i for i in report.op_inventory if i.status == "partial"]
            fused = [i for i in report.op_inventory if i.status == "fused"]
            lowered = [i for i in report.op_inventory if i.status == "lowered"]
            lowered_inc = [
                i for i in report.op_inventory if i.status == "lowered_incomplete"
            ]
            lowered_uv = [
                i for i in report.op_inventory if i.status == "lowered_unverified"
            ]
            unsupported = [i for i in report.op_inventory if i.status == "unsupported"]
            has_blockers = any(
                (
                    blockers,
                    shape_blockers,
                    variant_blockers,
                    dtype_blockers,
                    runtime_blockers,
                    partial_ops,
                    unsupported,
                )
            )

            if blockers:
                lines.extend(
                    _format_gap_table(
                        blockers,
                        title="Compile blockers (no converter — survived MLIR lowering)",
                        show_next_step=show_next_step,
                    )
                )
                lines.append("")

            if shape_blockers:
                lines.extend(
                    _format_gap_table(
                        shape_blockers,
                        title=(
                            "Shape blockers (converter exists; dynamic spatial "
                            "dims block lowering)"
                        ),
                        show_next_step=show_next_step,
                    )
                )
                lines.append("")

            if variant_blockers:
                lines.extend(
                    _format_gap_table(
                        variant_blockers,
                        title=(
                            "Variant blockers (converter exists; unsupported "
                            "attributes / rank / mode)"
                        ),
                        show_next_step=show_next_step,
                    )
                )
                lines.append("")

            if dtype_blockers:
                lines.extend(
                    _format_gap_table(
                        dtype_blockers,
                        title=(
                            "Dtype blockers (converter exists; unsupported "
                            "element type in graph)"
                        ),
                        show_next_step=show_next_step,
                    )
                )
                lines.append("")

            if runtime_blockers:
                lines.extend(
                    _format_gap_table(
                        runtime_blockers,
                        title="Runtime blockers (compiled; kernel rejected variant)",
                        show_next_step=show_next_step,
                    )
                )
                lines.append("")

            if partial_ops:
                lines.extend(
                    _format_gap_table(
                        partial_ops,
                        title="Partial support (no converter; some runtime pieces exist)",
                        show_next_step=show_next_step,
                    )
                )
                lines.append("")

            if (
                report.failure_level in {"mlir_bufferize", "mlir_compile"}
                and not has_blockers
            ):
                lines.append("Compile blockers:")
                lines.append(
                    "  Compile failed but no op names in MLIR dump. "
                    "Re-run with --dump-mlir."
                )
                lines.append("")

            if fused:
                lines.extend(
                    _format_inventory_table(
                        fused,
                        title="Fused by hip-ep (fusion pass in repo)",
                    )
                )
                lines.append("")

            if lowered:
                lines.extend(
                    _format_inventory_table(
                        lowered,
                        title="Lowered by hip-ep (confirmed — not in bufferize dump)",
                    )
                )
                lines.append("")

            if lowered_inc:
                lines.extend(
                    _format_inventory_table(
                        lowered_inc,
                        title=(
                            "Incomplete lowering (converter in repo; still in MLIR "
                            "at failure — check variant/shape/dtype or pass ordering)"
                        ),
                    )
                )
                lines.append("")

            if lowered_uv:
                lines.extend(
                    _format_inventory_table(
                        lowered_uv,
                        title="Lowered by hip-ep (repo converter; re-run --dump-mlir to confirm)",
                    )
                )
                lines.append("")

            if unsupported:
                lines.extend(
                    _format_gap_table(
                        unsupported,
                        title="Unsupported (no hip-ep converter in repo)",
                        show_next_step=show_next_step,
                    )
                )
                lines.append("")

            if has_blockers:
                show_gap_legend = True

            lines.append(
                f"Graph summary: {len(report.op_inventory)} op types | "
                f"{len(blockers)} compile | "
                f"{len(shape_blockers)} shape | "
                f"{len(variant_blockers)} variant | "
                f"{len(dtype_blockers)} dtype | "
                f"{len(runtime_blockers)} runtime | "
                f"{len(partial_ops)} partial | "
                f"{len(lowered_inc)} incomplete | "
                f"{len(fused)} fused | "
                f"{len(lowered) + len(lowered_uv)} lowered | "
                f"{len(unsupported)} unsupported"
            )
        else:
            grouped = _group_findings_by_op(report.missing_ops)
            compile_ops = [
                f
                for f in report.missing_ops
                if f.source in {"mlir_dump", "perf_test_log", "log"}
                and "runtime unsupported" not in f.detail
            ]
            runtime_ops = [
                f for f in report.missing_ops if "runtime unsupported" in f.detail
            ]

            if compile_ops:
                lines.extend(
                    _format_op_table(
                        compile_ops,
                        title="Compile blockers (confirmed in MLIR / compile log)",
                    )
                )
                lines.append("")
            elif report.failure_level in {"mlir_bufferize", "mlir_compile"}:
                lines.append("Compile blockers:")
                lines.append(
                    "  Log shows compile failure but no specific op names. "
                    "Re-run with --dump-mlir."
                )
                lines.append("")

            if runtime_ops:
                lines.extend(
                    _format_op_table(
                        runtime_ops,
                        title="Runtime gaps (kernel unsupported variant)",
                    )
                )
                lines.append("")

            if not grouped and report.hip_ep_result == "OK":
                lines.append("Missing ops : (none - model is compatible with hip-ep)")

        lines.append("")
        if report.op_inventory:
            summary_map = {
                "compile_blocker": "compile blocked by",
                "shape_blocker": "shape blockers",
                "variant_blocker": "variant blockers",
                "dtype_blocker": "dtype blockers",
                "runtime_blocker": "runtime blockers",
                "partial": "partial support",
                "lowered_incomplete": "incomplete lowering",
                "unsupported": "unsupported (need converters)",
            }
            grouped_summary: dict[str, list[str]] = {key: [] for key in summary_map}
            for item in report.op_inventory:
                if item.status in grouped_summary:
                    grouped_summary[item.status].append(item.op)
            summary_parts: list[str] = []
            for status, label in summary_map.items():
                ops = sorted(set(grouped_summary[status]))
                if ops:
                    summary_parts.append(f"{label}: " + ", ".join(ops))
            if summary_parts:
                lines.append("Summary: " + "; ".join(summary_parts))
            elif report.hip_ep_result == "OK":
                lines.append("Summary: all graph ops lowered/fused by hip-ep.")
            else:
                lines.append("Summary: see inventory above.")
        elif report.unique_missing_ops:
            lines.append(
                "Summary: hip-ep cannot compile/run this model until converters "
                "exist for: " + ", ".join(report.unique_missing_ops)
            )
        elif report.hip_ep_result != "OK":
            lines.append(
                "Summary: hip-ep rejected this model at the "
                f"{FAILURE_STAGE_LABELS.get(report.failure_level, report.failure_level)} "
                "stage, but no specific missing op was identified."
            )
        else:
            lines.append("Summary: no missing ops detected for hip-ep.")

        lines.append("=" * 72)
        blocks.append("\n".join(lines))

    if show_gap_legend:
        blocks.append("\n".join(_format_gap_legend(show_next_step=show_next_step)))

    return "\n\n".join(blocks)


def format_report_text(reports: list[ModelReport]) -> str:
    lines: list[str] = []
    if len(reports) > 1:
        rollup = format_batch_rollup(reports)
        if rollup:
            lines.append(rollup)
            lines.append("")

    lines.append("=" * 72)
    lines.append("hip-ep missing ops report")
    lines.append("=" * 72)

    failed = [r for r in reports if r.status != "ok"]
    ok = [r for r in reports if r.status == "ok"]

    lines.append(f"Models tested : {len(reports)}")
    lines.append(f"  OK          : {len(ok)}")
    lines.append(f"  Failed      : {len(failed)}")
    lines.append("")

    all_missing: dict[str, list[str]] = {}
    for report in reports:
        for op in report.unique_missing_ops:
            label = _model_short_label(report.model)
            all_missing.setdefault(op, []).append(label)

    if all_missing:
        lines.append("Missing ops (aggregated across models):")
        for op, models in sorted(
            all_missing.items(), key=lambda kv: (-len(kv[1]), kv[0])
        ):
            lines.append(
                f"  {op:32s}  ({len(models)} model(s): {', '.join(sorted(set(models)))})"
            )
        lines.append("")

    for report in reports:
        lines.append("-" * 72)
        lines.append(f"Model  : {_model_display_path(report.model)}")
        lines.append(f"Result : {report.hip_ep_result}")
        lines.append(
            "Stage  : "
            + FAILURE_STAGE_LABELS.get(report.failure_level, report.failure_level)
        )
        if report.failure_detail:
            lines.append(f"Reason : {report.failure_detail}")
        lines.append(f"Status : {report.status}")
        if report.exit_code is not None:
            lines.append(f"Exit   : {report.exit_code}")
        if report.qps is not None:
            lines.append(f"QPS    : {report.qps:.3f}")
        if report.log_path:
            lines.append(f"Log    : {report.log_path}")

        if report.unique_missing_ops:
            lines.append("Missing ops:")
            for finding in report.missing_ops:
                suffix = f" x{finding.count}" if finding.count > 1 else ""
                detail = f" - {finding.detail}" if finding.detail else ""
                lines.append(f"  - {finding.op} [{finding.source}]{suffix}{detail}")
        elif report.status != "ok":
            lines.append("Missing ops: (none identified - see errors/log)")

        if report.errors:
            lines.append("Errors:")
            for err in report.errors[:8]:
                lines.append(f"  - {err}")
            if len(report.errors) > 8:
                lines.append(f"  ... and {len(report.errors) - 8} more")

        lines.append("")

    lines.append("=" * 72)
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Identify missing ONNX ops when running models with hip-ep."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--parse-log",
        nargs="+",
        metavar="LOG",
        help="Parse existing perf_test / EP log file(s).",
    )
    mode.add_argument(
        "--parse-mlir-dir",
        nargs="+",
        metavar="DIR",
        help="Parse MLIR IR dump directory (HIPDNN_EP_IR_DUMP_TREE output).",
    )
    mode.add_argument(
        "--scan-onnx",
        nargs="+",
        metavar="MODEL",
        help="Static ONNX scan (requires pip install onnx).",
    )

    parser.add_argument(
        "model",
        nargs="*",
        metavar="MODEL",
        help="ONNX model file(s) or directory to run with onnxruntime_perf_test.",
    )

    parser.add_argument(
        "--glob",
        default="*.onnx",
        help='When MODEL is a directory, glob pattern (default: "*.onnx").',
    )
    parser.add_argument(
        "--package-dir",
        type=Path,
        default=Path(os.environ.get("PACKAGE_DIR", "gpu-test-package")),
        help="gpu-test-package root (default: gpu-test-package or $PACKAGE_DIR).",
    )
    parser.add_argument(
        "--perf-exe",
        type=Path,
        default=None,
        help="Path to onnxruntime_perf_test binary (default: <package>/bin/...).",
    )
    parser.add_argument(
        "--plugin-eps",
        default=None,
        help="EP plugin name for --plugin_eps (default: auto-detect).",
    )
    parser.add_argument(
        "--plugin-ep-libs",
        default=None,
        help="EP plugin libs for --plugin_ep_libs (default: auto-detect).",
    )
    parser.add_argument(
        "--plugin-ep-options",
        default=None,
        help='Optional --plugin_ep_options value (e.g. "profile|llm").',
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="onnxruntime_perf_test -r value (default: 1 for fast smoke).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=int(os.environ.get("TEST_TIMEOUT", "600")),
        help="Subprocess timeout in seconds (default: 600).",
    )
    parser.add_argument(
        "--allow-cpu-fallback",
        action="store_true",
        help="Do NOT set session.disable_cpu_ep_fallback|1 (not recommended).",
    )
    parser.add_argument(
        "--dump-mlir",
        action="store_true",
        help="Set HIPDNN_EP_IR_DUMP_TREE=1 to capture MLIR pass dumps on failure.",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Print only missing op names (one per line).",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print full diagnostic report with raw errors.",
    )
    parser.add_argument(
        "--save-log",
        action="store_true",
        help="Write perf_test stdout/stderr to _missing_ops_logs/ (off by default).",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=SCRIPT_DIR / "_missing_ops_logs",
        help="Directory for logs when --save-log is set.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Write JSON report to this path.",
    )
    parser.add_argument(
        "--show-next-step",
        action="store_true",
        help="Include recommended work items (Next step column) in gap analysis tables.",
    )
    parser.add_argument(
        "--extra-perf-arg",
        action="append",
        default=[],
        metavar="ARG",
        help="Extra argument forwarded to onnxruntime_perf_test (repeatable).",
    )
    args = parser.parse_args()

    known_ops = _repo_known_onnx_ops()
    reports: list[ModelReport] = []

    if args.model:
        if args.parse_log or args.parse_mlir_dir or args.scan_onnx:
            parser.error(
                "MODEL positional argument cannot be combined with "
                "--parse-log, --parse-mlir-dir, or --scan-onnx"
            )
        package_dir = resolve_package_dir(args.package_dir)
        models = collect_models([Path(p) for p in args.model], args.glob)
        if not models:
            print("[ERROR] No .onnx models found.", file=sys.stderr)
            return 2

        for model in models:
            report = run_perf_test(
                model,
                package_dir=package_dir,
                perf_exe=args.perf_exe,
                plugin_eps=args.plugin_eps,
                plugin_ep_libs=args.plugin_ep_libs,
                plugin_ep_options=args.plugin_ep_options,
                repeats=args.repeats,
                disable_cpu_fallback=not args.allow_cpu_fallback,
                dump_mlir=args.dump_mlir,
                save_log=args.save_log,
                log_dir=args.log_dir,
                timeout_sec=args.timeout,
                extra_args=args.extra_perf_arg,
                known_ops=known_ops,
            )
            reports.append(report)

    elif args.parse_log:
        for log_path in args.parse_log:
            path = Path(log_path)
            text = path.read_text(encoding="utf-8", errors="ignore")
            findings, errors, qps = parse_text_for_missing_ops(
                text, source_label="log", known_ops=known_ops
            )
            status = "ok"
            if findings or errors:
                status = "compile_failed" if findings else "runtime_failed"
            reports.append(
                finalize_model_report(
                    ModelReport(
                        model=_model_display_path(path),
                        status=status,
                        missing_ops=findings,
                        errors=sorted(set(errors)),
                        qps=qps,
                        log_path=str(path),
                    ),
                    output=text,
                )
            )

    elif args.parse_mlir_dir:
        for dump_dir in args.parse_mlir_dir:
            path = Path(dump_dir)
            findings, errors = parse_mlir_tree(path, known_ops=known_ops)
            reports.append(
                finalize_model_report(
                    ModelReport(
                        model=_model_display_path(path),
                        status="compile_failed" if findings else "ok",
                        missing_ops=findings,
                        errors=sorted(set(errors)),
                    )
                )
            )

    elif args.scan_onnx:
        for model in args.scan_onnx:
            path = Path(model)
            try:
                op_inventory = build_op_inventory(
                    path,
                    known_ops=known_ops,
                    compile_failed=False,
                    report_shape_blockers=True,
                )
                findings = [
                    OpFinding(
                        op=item.op,
                        source="onnx_static",
                        detail=item.hip_ep_handling,
                        count=item.count,
                    )
                    for item in op_inventory
                    if item.status in BLOCKING_STATUSES
                ]
            except RuntimeError as exc:
                print(f"[ERROR] {exc}", file=sys.stderr)
                return 2
            reports.append(
                finalize_model_report(
                    ModelReport(
                        model=_model_display_path(path),
                        status="compile_failed" if findings else "ok",
                        missing_ops=findings,
                        op_inventory=op_inventory,
                    )
                )
            )

    else:
        parser.print_help()
        return 2

    if args.verbose:
        print(format_report_text(reports))
    elif args.quiet:
        summary = format_missing_ops_output(reports)
        if summary:
            print(summary)
    else:
        print(format_summary_report(reports, show_next_step=args.show_next_step))

    if args.json_out:
        report_dicts = [r.to_dict() for r in reports]
        if not args.show_next_step:
            for report in report_dicts:
                for item in report.get("op_inventory", []):
                    gap = item.get("gap")
                    if gap:
                        gap.pop("next_step", None)
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "known_converter_ops": sorted(known_ops),
            "reports": report_dicts,
        }
        if len(reports) > 1:
            payload["batch_summary"] = batch_summary_to_dict(reports)
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        if args.verbose:
            print(f"JSON report: {args.json_out}")

    failed = sum(1 for r in reports if r.status != "ok" or r.unique_missing_ops)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
