#!/usr/bin/env python3
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Count operators in the EP input MLIR.

Writes ep_input_ops.json. The shape matches step1_onnx_parser.py's
step1_onnx_ops.json so the distribution comparison and the report builder
work unchanged, but the name differs on purpose: both files are in play at
once -- this one describes the graph the EP compiles, step1's describes the
original ONNX -- and the two are easy to mix up when they share a name.

step1 still runs against the original .onnx to provide the comparison
baseline and the fallback path when the dump is unavailable.

Four things this has to get right, all of them because MLIR and ONNX do not
describe the same graph in the same way:

- `onnx.Custom` carries the real operator in its `function_name` attribute.
  Without normalizing it back to (MatMulNBits, com.microsoft) the comparison
  against the original ONNX misaligns every com.microsoft row.
- MorphiZen and onnx-mlir attach bookkeeping attributes that are not ONNX
  operator attributes. They are listed in
  morphizen/mlir-imp/src/mlir-constants.hpp and must not reach the schema
  checks downstream.
- ONNX initializers become `onnx.Constant` operations carrying
  location/offset/size. Counting them would inflate the denominator of the
  support rate by more than the compute operators themselves (805 of 809 in
  a 1.8B model), so weight constants are tallied separately.
- Loop/If bodies are nested regions at this stage -- they only become
  separate func.func ops after onnx-loop-outline. Line-based scanning
  therefore includes subgraph operators by default, matching step1's
  include_subgraphs behaviour; indentation depth separates the two for
  count_top_level.

Usage:
  python mlir_op_parser.py <ep_input.mlir> <output_dir>   # -> ep_input_ops.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

# Bookkeeping attributes injected by MorphiZen / onnx-mlir. Source of truth:
# morphizen/mlir-imp/src/mlir-constants.hpp. `morphizen.*` is matched by
# prefix, as that header instructs.
METADATA_ATTRS = frozenset(
    {
        "onnx_node_name",  # NodeProto.name
        "node.outputs",  # NodeProto.output
        "function_name",  # onnx.Custom: real op_type
        "domain_name",  # onnx.Custom: real domain
        "onnx.graph.name",  # GraphProto.name
        "onnx.name",  # NodeArg name, on func args/results
        "location",  # onnx.Constant external data reference
        "offset",
        "size",
    }
)
METADATA_PREFIX = "morphizen."

# Terminators and placeholders; not operators. mlir-constants.hpp states
# outright that onnx.Constant and onnx.Return are not morphizen::Node.
NON_COMPUTE_OPS = frozenset(
    {"onnx.Return", "onnx.Yield", "onnx.NoValue", "func.return"}
)

# MLIR element type -> the ONNX spelling step1 emits.
_ELEM_TYPES = {
    "f16": "float16",
    "f32": "float32",
    "f64": "float64",
    "bf16": "bfloat16",
    "i1": "bool",
    "i8": "int8",
    "i16": "int16",
    "i32": "int32",
    "i64": "int64",
    "ui8": "uint8",
    "ui16": "uint16",
    "ui32": "uint32",
    "ui64": "uint64",
}

_OP_NAME = re.compile(r'"(onnx\.[A-Za-z0-9_]+)"\s*\(')
_FUNC_DEF = re.compile(r"^(\s*)func\.func\b")
_TENSOR = re.compile(r"tensor<([^<>]*)>")


def _split_top_level(text: str) -> list[str]:
    """Split on commas that are not inside brackets or quotes."""
    parts: list[str] = []
    buf: list[str] = []
    depth = 0
    in_str = False
    i = 0
    while i < len(text):
        ch = text[i]
        if in_str:
            if ch == "\\":
                buf.append(ch)
                i += 1
                if i < len(text):
                    buf.append(text[i])
                i += 1
                continue
            if ch == '"':
                in_str = False
            buf.append(ch)
        elif ch == '"':
            in_str = True
            buf.append(ch)
        elif ch in "[<({":
            depth += 1
            buf.append(ch)
        elif ch in "]>)}":
            depth -= 1
            buf.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
        i += 1
    if buf:
        parts.append("".join(buf).strip())
    return [p for p in parts if p]


def _attr_body(line: str) -> str:
    """Return the operation's attribute dictionary body, or ''.

    Both `<{...}>` (inherent attributes) and `{...}` (discardable) forms
    appear; the operands list `(...)` and the trailing type signature must
    not be mistaken for either.
    """
    close_paren = line.find(")")
    if close_paren < 0:
        return ""
    rest = line[close_paren + 1 :]
    out: list[str] = []
    depth = 0
    for ch in rest:
        if ch == "{":
            depth += 1
            if depth == 1:
                continue
        elif ch == "}":
            depth -= 1
            if depth == 0:
                out.append(",")
                continue
        if depth > 0:
            out.append(ch)
    return "".join(out)


def parse_attributes(line: str) -> dict[str, str | None]:
    """Parse `key = value` pairs, dropping the MLIR type suffix.

    `saturate = 1 : si64` yields `1`. A bare name (a unit attribute such as
    `value` on onnx.NoValue) maps to None.
    """
    attrs: dict[str, str | None] = {}
    for item in _split_top_level(_attr_body(line)):
        if "=" not in item:
            attrs[item.strip()] = None
            continue
        key, value = item.split("=", 1)
        key, value = key.strip(), value.strip()
        m = re.match(r"^(.*?)\s*:\s*[a-z]?[iuf]\d+$", value)
        if m:
            value = m.group(1).strip()
        attrs[key] = value
    return attrs


def real_attributes(attrs: dict[str, str | None]) -> dict[str, str | None]:
    """Drop bookkeeping attributes, keeping real ONNX operator attributes."""
    return {
        k: v
        for k, v in attrs.items()
        if k not in METADATA_ATTRS and not k.startswith(METADATA_PREFIX)
    }


def _type_signature(line: str) -> str:
    """The trailing `: (...) -> ...` part of an operation line."""
    idx = line.rfind("} :")
    if idx < 0:
        idx = line.rfind(") :")
    return line[idx + 2 :] if idx >= 0 else ""


def tensor_facts(line: str) -> tuple[set[str], set[str]]:
    """Element types and static/dynamic classification from the signature."""
    dtypes: set[str] = set()
    shapes: set[str] = set()
    for body in _TENSOR.findall(_type_signature(line)):
        parts = body.split("x")
        elem = parts[-1].strip()
        dtypes.add(_ELEM_TYPES.get(elem, elem))
        dims = parts[:-1]
        if not dims:
            # Rank 0. step1 reports "unknown" for a shape with no dims.
            shapes.add("unknown")
        elif any(d.strip() in ("?", "*") for d in dims):
            shapes.add("dynamic")
        else:
            shapes.add("static")
    return dtypes, shapes


def parse_mlir(path: Path) -> dict:
    ops: dict[str, dict] = defaultdict(
        lambda: {
            "count": 0,
            "count_top_level": 0,
            "domain": set(),
            "data_types": set(),
            "shape_types": set(),
            "scopes": set(),
            "instances": [],
        }
    )

    weight_constants = 0
    total = 0
    top_level = 0
    func_indent: int | None = None
    # Innermost region-bearing operator, for naming subgraph scopes.
    region_stack: list[tuple[int, str]] = []

    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fn = _FUNC_DEF.match(raw)
        if fn:
            func_indent = len(fn.group(1))
            region_stack.clear()
            continue

        m = _OP_NAME.search(raw)
        if not m:
            continue
        mlir_op = m.group(1)
        if mlir_op in NON_COMPUTE_OPS:
            continue

        attrs = parse_attributes(raw)

        # Weight constants are ONNX initializers materialized as operations.
        if mlir_op == "onnx.Constant" and "location" in attrs:
            weight_constants += 1
            continue

        indent = len(raw) - len(raw.lstrip())
        body_indent = (func_indent + 2) if func_indent is not None else indent
        while region_stack and indent <= region_stack[-1][0]:
            region_stack.pop()
        if indent <= body_indent:
            scope = "main"
            is_top = True
        else:
            scope = region_stack[-1][1] if region_stack else "subgraph"
            is_top = False
        if raw.rstrip().endswith("({"):
            region_stack.append((indent, f"{mlir_op}.body"))

        # onnx.Custom is a container; the real operator is in function_name.
        op_type = mlir_op.split(".", 1)[1]
        domain = "onnx"
        if mlir_op == "onnx.Custom":
            op_type = (attrs.get("function_name") or '"Custom"').strip('"')
        raw_domain = attrs.get("domain_name")
        if raw_domain:
            d = raw_domain.strip('"')
            domain = "onnx" if d in ("", "ai.onnx") else d

        dtypes, shapes = tensor_facts(raw)
        node_name = (attrs.get("onnx_node_name") or "").strip('"')

        entry = ops[op_type]
        entry["count"] += 1
        total += 1
        if is_top:
            entry["count_top_level"] += 1
            top_level += 1
        entry["domain"].add(domain)
        entry["data_types"].update(dtypes)
        entry["shape_types"].update(shapes)
        entry["scopes"].add(scope)
        # Record the source line number, not the line itself: the probe reads
        # the MLIR anyway, and `--mlir-print-debuginfo` reports each lowered
        # operation's origin as loc("<file>":<line>:<col>), so this is the key
        # that pairs an ONNX operation with what it became.
        entry["instances"].append(
            {
                "name": node_name,
                "line_no": line_no,
                "attributes": real_attributes(attrs),
            }
        )

    result: dict = {
        "_analysis_meta": {
            "include_subgraphs": True,
            "total_nodes": total,
            "top_level_nodes": top_level,
            "subgraph_nodes": total - top_level,
            "weight_constants": weight_constants,
            "source": "mlir",
        }
    }
    for op_type, info in sorted(ops.items(), key=lambda kv: -kv[1]["count"]):
        result[op_type] = {
            "count": info["count"],
            "count_top_level": info["count_top_level"],
            "domain": sorted(info["domain"]),
            "data_types": sorted(info["data_types"]),
            "shape_types": sorted(info["shape_types"]),
            "scopes": sorted(info["scopes"])[:20],
            "instances": info["instances"],
        }
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Operator distribution from EP input MLIR")
    ap.add_argument("mlir_path", type=Path)
    ap.add_argument("output_dir", type=Path)
    ap.add_argument(
        "--max-instances-per-op",
        type=int,
        default=0,
        metavar="N",
        help="Cap stored instances per op type (0 = unlimited; default 0). "
        "Instances carry the parsed attributes the probe needs, so the "
        "default keeps them all.",
    )
    args = ap.parse_args()

    result = parse_mlir(args.mlir_path)

    if args.max_instances_per_op > 0:
        for key, value in result.items():
            if not key.startswith("_"):
                value["instances"] = value["instances"][: args.max_instances_per_op]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "ep_input_ops.json"
    out_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    meta = result["_analysis_meta"]
    print(f"Source: {args.mlir_path}")
    print(f"  compute operator instances : {meta['total_nodes']}")
    print(f"  operator types             : {len(result) - 1}")
    print(
        f"  top-level / subgraph       : {meta['top_level_nodes']} / {meta['subgraph_nodes']}"
    )
    print(f"  weight constants (excluded): {meta['weight_constants']}")
    print(f"[OK] {out_path}")


if __name__ == "__main__":
    main()
