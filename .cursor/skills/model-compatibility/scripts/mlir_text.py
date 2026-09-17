#!/usr/bin/env python3
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""
Minimal reader for the textual MLIR the compatibility pipeline works with.

Two forms appear in the dumps:

  generic (unregistered onnx dialect, as the MorphiZen importer writes it)
    %1 = "onnx.Add"(%arg0, %arg0) {onnx_node_name = "a"} : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>

  custom (registered hip dialect, after convert-onnx-to-hip)
    %9 = hip.rms_norm(%arg0) ins(%8, %2 : tensor<?xf32>, tensor<2048xf32>) outs(%7 : tensor<?xf32>) {axis = -1 : i64} : tensor<?xf32>

Both are single-line per operation, so a line scanner is enough; a real
parse would need MLIR itself, which is not available from Python here.
Ops that open a region (func.func, module) are recognized by name only.

Positions come from `--mlir-print-debuginfo`: re-parsing a text file gives
every op a FileLineCol location into that file, and the conversion
propagates it to the ops it creates. That makes the location string a
usable join key between the pre- and post-conversion dumps.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# Carrier / terminator ops that exist in the MLIR form but are not graph
# compute nodes. Initializers in particular become one onnx.Constant each,
# which would swamp an operator distribution taken from the protobuf.
NON_COMPUTE_ONNX_OPS = frozenset(
    {"onnx.Constant", "onnx.NoValue", "onnx.Return", "onnx.EntryPoint"}
)

_LOC_ALIAS_RE = re.compile(r"^#(loc\w*)\s*=\s*(.+?)\s*$")
_TRAILING_LOC_RE = re.compile(r"\s+loc\((#?\S+?)\)\s*$")
_GENERIC_OP_RE = re.compile(r'^(?:%\S+(?::\s*\d+)?\s*=\s*)?"([A-Za-z_][\w.]*)"\s*\(')
_CUSTOM_OP_RE = re.compile(r"^(?:%\S+(?::\s*\d+)?\s*=\s*)?([a-z][\w]*\.[\w.]+)\b")

_ELEM_TYPE_NAMES = {
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


@dataclass
class MlirOp:
    """One operation line."""

    name: str
    attrs: Dict[str, str] = field(default_factory=dict)
    type_signature: str = ""
    loc: str = ""
    line_no: int = 0
    scope: str = "main"

    @property
    def dialect(self) -> str:
        return self.name.split(".", 1)[0]

    def onnx_key(self) -> Tuple[str, str]:
        """(op_type, domain) for an onnx.* op, matching ONNX protobuf naming.

        onnx.Custom is a container whose real operator is in `function_name`,
        so report that instead of collapsing every custom op into "Custom".
        """
        bare = self.name.split(".", 1)[1] if "." in self.name else self.name
        if self.name != "onnx.Custom":
            return bare, "ai.onnx"
        op_type = strip_quotes(self.attrs.get("function_name", "")) or "Custom"
        domain = strip_quotes(self.attrs.get("domain_name", "")) or "com.microsoft"
        return op_type, domain

    def onnx_attr_names(self) -> List[str]:
        """Attribute names that came from the ONNX node.

        `function_name` / `domain_name` select the operator for onnx.Custom and
        `node.outputs` / `onnx_node_name` are importer bookkeeping, so none of
        them is an operator attribute that a HIP op could carry.
        """
        skip = {"function_name", "domain_name", "node.outputs", "onnx_node_name"}
        return sorted(k for k in self.attrs if k not in skip)


def strip_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
        return value[1:-1]
    return value


def _split_top_level(text: str, separator: str = ",") -> List[str]:
    """Split on `separator` outside quotes, brackets, braces and angle types."""
    parts, depth, start, in_string, escaped = [], 0, 0, False, False
    for i, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "([{<":
            depth += 1
        elif ch in ")]}>":
            depth -= 1
        elif ch == separator and depth == 0:
            parts.append(text[start:i])
            start = i + 1
    parts.append(text[start:])
    return [p.strip() for p in parts if p.strip()]


def _find_attr_dict(line: str) -> Tuple[Optional[str], int]:
    """Return the operation's attribute dictionary body and where it started.

    The dictionary is the last brace group that the type signature follows,
    which distinguishes it from braces inside an attribute value.
    """
    idx = len(line)
    while True:
        close = line.rfind("}", 0, idx)
        if close < 0:
            return None, -1
        after = line[close + 1 :].lstrip()
        if after.startswith(":"):
            open_idx = _match_backwards(line, close)
            if open_idx >= 0:
                return line[open_idx + 1 : close], open_idx
        idx = close


def _match_backwards(line: str, close_idx: int) -> int:
    depth = 0
    in_string = False
    for i in range(close_idx, -1, -1):
        ch = line[i]
        if in_string:
            if ch == '"' and (i == 0 or line[i - 1] != "\\"):
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "}":
            depth += 1
        elif ch == "{":
            depth -= 1
            if depth == 0:
                return i
    return -1


def _parse_attrs(body: Optional[str]) -> Dict[str, str]:
    if not body:
        return {}
    attrs = {}
    for entry in _split_top_level(body):
        key, sep, value = entry.partition("=")
        key = key.strip()
        if not key:
            continue
        # A bare key is a unit attribute (`{value}` on onnx.NoValue).
        attrs[key] = value.strip() if sep else ""
    return attrs


def _type_signature(line: str, attr_start: int) -> str:
    """Text after the operation's final top-level colon."""
    search_from = attr_start if attr_start >= 0 else 0
    depth = 0
    in_string = False
    for i in range(search_from, len(line)):
        ch = line[i]
        if in_string:
            if ch == '"' and line[i - 1] != "\\":
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "([{<":
            depth += 1
        elif ch in ")]}>":
            depth -= 1
        elif ch == ":" and depth == 0:
            return line[i + 1 :].strip()
    return ""


def tensor_types(type_signature: str) -> List[str]:
    return re.findall(r"tensor<([^<>]*)>", type_signature)


def element_type_names(type_signature: str) -> List[str]:
    """Readable element-type names for every tensor in a type signature."""
    names = []
    for body in tensor_types(type_signature):
        elem = body.rsplit("x", 1)[-1].strip() if "x" in body else body.strip()
        names.append(_ELEM_TYPE_NAMES.get(elem, elem))
    return [n for n in names if n]


def shape_kind(type_signature: str) -> Optional[str]:
    bodies = tensor_types(type_signature)
    if not bodies:
        return None
    return "dynamic" if any("?" in b for b in bodies) else "static"


class MlirModule:
    """Operations of one textual MLIR module, in source order."""

    def __init__(self, ops: List[MlirOp], loc_aliases: Dict[str, str]):
        self.ops = ops
        self.loc_aliases = loc_aliases

    def resolve_loc(self, op: MlirOp) -> str:
        if not op.loc:
            return ""
        if op.loc.startswith("#"):
            return self.loc_aliases.get(op.loc[1:], op.loc)
        return op.loc

    def onnx_ops(self, include_non_compute: bool = False) -> List[MlirOp]:
        return [
            op
            for op in self.ops
            if op.dialect == "onnx"
            and (include_non_compute or op.name not in NON_COMPUTE_ONNX_OPS)
        ]


def parse_mlir(text: str) -> MlirModule:
    loc_aliases: Dict[str, str] = {}
    ops: List[MlirOp] = []
    scope = "main"

    for line_no, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line in {"}", "})", "{"}:
            continue

        if line.startswith("#"):
            alias = _LOC_ALIAS_RE.match(line)
            if alias:
                loc_aliases[alias.group(1)] = alias.group(2)
            continue

        loc = ""
        trailing = _TRAILING_LOC_RE.search(line)
        if trailing:
            loc = trailing.group(1)
            line = line[: trailing.start()].rstrip()

        # Track the enclosing function so nested (outlined loop/if body) ops
        # are attributed to it rather than to the main graph.
        func = re.match(r"^func\.func\s+(?:private\s+)?@([\w$.]+)", line)
        if func:
            scope = func.group(1)
            continue
        if line.startswith("module"):
            continue

        match = _GENERIC_OP_RE.match(line) or _CUSTOM_OP_RE.match(line)
        if not match:
            continue
        name = match.group(1)
        if "." not in name:
            continue

        attr_body, attr_start = _find_attr_dict(line)
        ops.append(
            MlirOp(
                name=name,
                attrs=_parse_attrs(attr_body),
                type_signature=_type_signature(line, attr_start),
                loc=loc,
                line_no=line_no,
                scope=scope,
            )
        )

    return MlirModule(ops, loc_aliases)


def parse_mlir_file(path) -> MlirModule:
    from pathlib import Path

    return parse_mlir(Path(path).read_text(encoding="utf-8"))
